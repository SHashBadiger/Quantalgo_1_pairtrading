import numpy as np
import pandas as pd
import yfinance as yf
import statsmodels.api as sm

def calculate_ou_half_life(spread):
    """
    Calculates the half-life of mean reversion using an OU process (Ornstein-Uhlenbeck).
    """
    spread_lag = spread.shift(1)
    spread_diff = spread - spread_lag
    spread_lag = spread_lag.dropna()
    spread_diff = spread_diff.dropna()
    
    # Regression: dy = (lambda * (mu - y)) * dt
    model = sm.OLS(spread_diff, sm.add_constant(spread_lag)).fit()
    # Check if we have enough parameters and avoid KeyError
    if len(model.params) < 2:
        return np.inf
    lambda_param = -model.params.iloc[1]
    
    if lambda_param <= 0:
        return np.inf # No mean reversion
        
    half_life = np.log(2) / lambda_param
    return half_life

def calculate_atr(high, low, close, period=14):
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def refined_strategy_backtest(s1, s2, period='2y'):
    print(f"Refining Strategy for {s1}-{s2}...")
    data = yf.download([s1, s2], period=period)
    # Using MultiIndex columns from yfinance
    prices = data['Close']
    vols = data['Volume']
    
    # Kalman for Beta
    x, y = prices[s2].values, prices[s1].values
    from backtest_report import kalman_filter_beta # Reuse previous logic
    betas, intercepts = kalman_filter_beta(x, y)
    
    spread = pd.Series(y - (betas * x + intercepts))
    zscore = (spread - spread.mean()) / spread.std()
    
    # 1. OU Half-Life Check
    half_life = calculate_ou_half_life(spread)
    print(f"Calculated Half-Life: {half_life:.2f} days")
    
    if half_life > 10:
        print("CRITICAL: Half-life > 10 days. Spread is too slow. Filtering for speed...")
        # Note: In a real bot, we would stop here. For backtest, we'll continue to show the diff.

    # 2. ATR for Trailing Stop
    # Approximate spread high/low for ATR
    spread_high = data['High'][s1] - betas * data['Low'][s2]
    spread_low = data['Low'][s1] - betas * data['High'][s2]
    spread_atr = calculate_atr(spread_high, spread_low, spread)
    
    # 3. Volume Moving Average
    vol_ma_s1 = vols[s1].rolling(20).mean()
    vol_ma_s2 = vols[s2].rolling(20).mean()
    
    pos = 0
    returns = []
    entry_price = 0
    stop_loss = 0
    
    for i in range(20, len(zscore)):
        # Refined Entry Logic
        vol_condition = (vols[s1].iloc[i] > vol_ma_s1.iloc[i]) and (vols[s2].iloc[i] > vol_ma_s2.iloc[i])
        
        if pos == 0:
            if vol_condition and half_life <= 10:
                if zscore[i] > 2.0:
                    pos = -1 # Short spread
                    entry_price = spread.iloc[i]
                    stop_loss = entry_price + (1.5 * spread_atr.iloc[i])
                elif zscore[i] < -2.0:
                    pos = 1 # Long spread
                    entry_price = spread.iloc[i]
                    stop_loss = entry_price - (1.5 * spread_atr.iloc[i])
            returns.append(0)
            
        elif pos == 1: # Long
            ret = (y[i]/y[i-1]-1) - betas[i]*(x[i]/x[i-1]-1)
            returns.append(ret)
            # Exit conditions: Mean reversion (Z=0) OR ATR Stop Loss
            if zscore[i] >= 0 or spread.iloc[i] < stop_loss:
                pos = 0
                
        elif pos == -1: # Short
            ret = -(y[i]/y[i-1]-1) + betas[i]*(x[i]/x[i-1]-1)
            returns.append(ret)
            if zscore[i] <= 0 or spread.iloc[i] > stop_loss:
                pos = 0
                
    returns = np.array(returns)
    sharpe = np.sqrt(252) * np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
    max_dd = np.max((np.maximum.accumulate(1+returns)-(1+returns))/np.maximum.accumulate(1+returns))
    
    print(f"\n--- OPTIMIZED REPORT: {s1}-{s2} ---")
    print(f"Optimized Sharpe: {sharpe:.2f}")
    print(f"Optimized Max DD: {max_dd:.2%}")
    print(f"Filters applied: OU Half-Life, 1.5x ATR Stop, Vol-Weighted Entry.")

refined_strategy_backtest('AVGO', 'JPM')
