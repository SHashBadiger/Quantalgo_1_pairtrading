import numpy as np
import pandas as pd
import yfinance as yf
from backtest_report import kalman_filter_beta
from sharpe_booster import calculate_ou_half_life, calculate_atr

def get_regime(returns, window=20):
    """
    Classifies Market Regime based on volatility.
    """
    vol = returns.rolling(window).std() * np.sqrt(252)
    regime = "Steady"
    if vol.iloc[-1] > 0.30: # 30% Vol threshold for 'High Vol'
        regime = "High Vol"
    return regime, vol.iloc[-1]

def backtest_with_friction(s1, s2, start_date, end_date, slippage_bps=5, execution_bps=2):
    try:
        data = yf.download([s1, s2], start=start_date, end=end_date, progress=False)
        if data.empty or len(data) < 40: return None
        prices = data['Close']
        vols = data['Volume']
        common_idx = prices.index.intersection(vols.index)
        prices, vols = prices.loc[common_idx], vols.loc[common_idx]
        
        x, y = prices[s2].values, prices[s1].values
        betas, intercepts = kalman_filter_beta(x, y)
        spread = pd.Series(y - (betas * x + intercepts))
        zscore = (spread - spread.mean()) / spread.std()
        
        # Pre-calc ATR and Vol MA
        spread_high = data['High'][s1].loc[common_idx] - betas * data['Low'][s2].loc[common_idx]
        spread_low = data['Low'][s1].loc[common_idx] - betas * data['High'][s2].loc[common_idx]
        spread_atr = calculate_atr(spread_high, spread_low, spread)
        vol_ma_s1 = vols[s1].rolling(20).mean()
        
        market_rets = prices[s1].pct_change()
        
        pos = 0
        gross_returns = []
        net_returns = []
        cost_per_trade = (slippage_bps + execution_bps) / 10000
        
        for i in range(25, len(zscore)):
            # 1. Regime Classifier
            regime, _ = get_regime(market_rets.iloc[:i])
            z_threshold = 2.5 if regime == "High Vol" else 2.0
            
            vol_condition = (vols[s1].iloc[i] > vol_ma_s1.iloc[i])
            
            if pos == 0:
                if (zscore[i] > z_threshold or zscore[i] < -z_threshold) and vol_condition:
                    pos = -1 if zscore[i] > z_threshold else 1
                    entry_price = spread.iloc[i]
                    stop_loss = entry_price + (1.5 * spread_atr.iloc[i]) if pos == -1 else entry_price - (1.5 * spread_atr.iloc[i])
                    # Deduct cost on entry
                    net_returns.append(-cost_per_trade)
                else:
                    net_returns.append(0)
                gross_returns.append(0)
            else:
                ret = pos * ((y[i]/y[i-1]-1) - betas[i]*(x[i]/x[i-1]-1))
                gross_returns.append(ret)
                
                # Check exit
                if (pos == 1 and (zscore[i] >= 0 or spread.iloc[i] < stop_loss)) or \
                   (pos == -1 and (zscore[i] <= 0 or spread.iloc[i] > stop_loss)):
                    pos = 0
                    net_returns.append(ret - cost_per_trade) # Deduct cost on exit
                else:
                    net_returns.append(ret)

        net_rets = np.array(net_returns)
        if len(net_rets) == 0 or np.std(net_rets) == 0: return None
        
        sharpe = np.sqrt(252) * np.mean(net_rets) / np.std(net_rets)
        cum_ret = np.prod(1 + net_rets) - 1
        return {'sharpe': sharpe, 'cum_ret': cum_ret}
    except:
        return None

# --- WALK-FORWARD OPTIMIZATION ---
print("Running Walk-Forward Optimization + TCA for Tier 1...")
tier1_pairs = [('PFE', 'TGT'), ('HD', 'MA'), ('AVGO', 'JPM')]
wf_results = []

# 12 months, rolling windows
start_base = pd.Timestamp('2024-01-01')
for s1, s2 in tier1_pairs:
    windows_passed = 0
    total_sharpe = 0
    print(f"Testing {s1}-{s2}...")
    for m in range(12):
        window_start = start_base + pd.DateOffset(months=m)
        window_end = window_start + pd.DateOffset(months=1)
        res = backtest_with_friction(s1, s2, window_start, window_end)
        if res:
            total_sharpe += res['sharpe']
            if res['sharpe'] >= 1.2: windows_passed += 1
            
    wf_results.append({
        'Pair': f"{s1}-{s2}",
        'Avg_Net_Sharpe': total_sharpe / 12,
        'Stability_Score': f"{windows_passed}/12 Months > 1.2 Sharpe"
    })

print("\n--- INSTITUTIONAL WALK-FORWARD & TCA REPORT ---")
print(pd.DataFrame(wf_results).to_markdown(index=False))
