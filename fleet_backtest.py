import numpy as np
import pandas as pd
import yfinance as yf
from backtest_report import kalman_filter_beta
from sharpe_booster import calculate_ou_half_life, calculate_atr

def optimized_backtest(s1, s2, period='1y'):
    try:
        data = yf.download([s1, s2], period=period, progress=False)
        if data.empty: return None
        prices = data['Close']
        vols = data['Volume']
        
        # Sync indices
        common_idx = prices.index.intersection(vols.index)
        prices = prices.loc[common_idx]
        vols = vols.loc[common_idx]
        
        x, y = prices[s2].values, prices[s1].values
        betas, intercepts = kalman_filter_beta(x, y)
        
        spread = pd.Series(y - (betas * x + intercepts))
        zscore = (spread - spread.mean()) / spread.std()
        
        half_life = calculate_ou_half_life(spread)
        if half_life > 10:
            return {'status': 'Filtered (Slow)', 'half_life': half_life}
        
        spread_high = data['High'][s1].loc[common_idx] - betas * data['Low'][s2].loc[common_idx]
        spread_low = data['Low'][s1].loc[common_idx] - betas * data['High'][s2].loc[common_idx]
        spread_atr = calculate_atr(spread_high, spread_low, spread)
        vol_ma_s1 = vols[s1].rolling(20).mean()
        vol_ma_s2 = vols[s2].rolling(20).mean()

        opt_returns = []
        pos_opt = 0
        stop_loss = 0
        
        for i in range(20, len(zscore)):
            vol_condition = (vols[s1].iloc[i] > vol_ma_s1.iloc[i]) and (vols[s2].iloc[i] > vol_ma_s2.iloc[i])
            
            if pos_opt == 0:
                if (zscore[i] > 2.0 or zscore[i] < -2.0) and vol_condition:
                    pos_opt = -1 if zscore[i] > 2.0 else 1
                    entry_price = spread.iloc[i]
                    stop_loss = entry_price + (1.5 * spread_atr.iloc[i]) if pos_opt == -1 else entry_price - (1.5 * spread_atr.iloc[i])
                opt_returns.append(0)
            elif pos_opt == 1:
                ret = (y[i]/y[i-1]-1) - betas[i]*(x[i]/x[i-1]-1)
                opt_returns.append(ret)
                if zscore[i] >= 0 or spread.iloc[i] < stop_loss: pos_opt = 0
            elif pos_opt == -1:
                ret = -(y[i]/y[i-1]-1) + betas[i]*(x[i]/x[i-1]-1)
                opt_returns.append(ret)
                if zscore[i] <= 0 or spread.iloc[i] > stop_loss: pos_opt = 0

        returns = np.array(opt_returns)
        if len(returns) == 0 or np.std(returns) == 0: return None
        
        sharpe = np.sqrt(252) * np.mean(returns) / np.std(returns)
        dd = (np.maximum.accumulate(1+returns)-(1+returns))/np.maximum.accumulate(1+returns)
        max_dd = np.max(dd)
        
        return {
            'status': 'Success',
            'sharpe': sharpe,
            'max_dd': max_dd,
            'half_life': half_life
        }
    except Exception as e:
        return {'status': f'Error: {str(e)}'}

watchlist = [
    ('AVGO', 'JPM'), ('AVGO', 'WFC'), ('AXP', 'TSM'), ('BAC', 'TSM'),
    ('BP', 'ORCL'), ('AVGO', 'GS'), ('AMZN', 'AVGO'), ('BP', 'UNH'),
    ('COST', 'NVDA'), ('AVGO', 'COST'), ('BP', 'PYPL'), ('MA', 'META'),
    ('BP', 'LOW'), ('HD', 'MA'), ('BP', 'MS'), ('BP', 'HD'),
    ('ORCL', 'WMT'), ('PFE', 'TGT'), ('CVX', 'INTC'), ('LOW', 'ORCL')
]

results = []
for s1, s2 in watchlist:
    print(f"Processing {s1}-{s2}...")
    res = optimized_backtest(s1, s2)
    if res and res['status'] == 'Success':
        results.append({
            'Pair': f"{s1}-{s2}",
            'Sharpe': res['sharpe'],
            'Max_DD': res['max_dd'],
            'Half_Life': res['half_life']
        })

df_results = pd.DataFrame(results).sort_values(by='Sharpe', ascending=False)
print("\n--- OPTIMIZED LEADERBOARD ---")
print(df_results.to_markdown(index=False))

# Export Tier 1
tier1 = df_results[(df_results['Sharpe'] > 1.5) & (df_results['Max_DD'] < 0.10)]
tier1.to_csv('pairs-trader/TIER1_PORTFOLIO.csv', index=False)
