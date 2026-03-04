import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize

def kalman_filter_beta(x, y):
    delta = 1e-5
    trans_cov = delta / (1 - delta) * np.eye(2)
    obs_mat = np.expand_dims(np.vstack([[x], [np.ones(len(x))]]).T, axis=1)
    state_means = np.zeros((len(y), 2))
    R = 1.0
    current_state_mean = np.zeros(2)
    current_state_cov = np.ones((2, 2))
    for i in range(len(y)):
        prediction_state_cov = current_state_cov + trans_cov
        y_pred = np.dot(obs_mat[i], current_state_mean)
        res = y[i] - y_pred
        res_cov = np.dot(np.dot(obs_mat[i], prediction_state_cov), obs_mat[i].T) + R
        K = np.dot(np.dot(prediction_state_cov, obs_mat[i].T), np.linalg.inv(res_cov))
        current_state_mean = current_state_mean + np.dot(K, res)
        current_state_cov = prediction_state_cov - np.dot(np.dot(K, obs_mat[i]), prediction_state_cov)
        state_means[i] = current_state_mean
    return state_means[:, 0], state_means[:, 1]

def backtest(s1_ticker, s2_ticker, period='2y'):
    print(f"Backtesting {s1_ticker}-{s2_ticker} for {period}...")
    df = yf.download([s1_ticker, s2_ticker], period=period)['Close']
    x, y = df[s2_ticker].values, df[s1_ticker].values
    
    betas, intercepts = kalman_filter_beta(x, y)
    spread = y - (betas * x + intercepts)
    zscore = (spread - np.mean(spread)) / np.std(spread)
    
    # Strategy: Entry at +/- 2.0, Exit at 0
    pos = 0
    returns = []
    for i in range(1, len(zscore)):
        if pos == 0:
            if zscore[i] > 2.0: pos = -1 # Short spread
            elif zscore[i] < -2.0: pos = 1 # Long spread
            returns.append(0)
        elif pos == 1:
            # PnL: (S1_t/S1_t-1 - 1) - beta*(S2_t/S2_t-1 - 1)
            ret = (y[i]/y[i-1]-1) - betas[i]*(x[i]/x[i-1]-1)
            returns.append(ret)
            if zscore[i] >= 0: pos = 0
        elif pos == -1:
            ret = -(y[i]/y[i-1]-1) + betas[i]*(x[i]/x[i-1]-1)
            returns.append(ret)
            if zscore[i] <= 0: pos = 0
            
    returns = np.array(returns)
    cum_ret = np.prod(1 + returns) - 1
    sharpe = np.sqrt(252) * np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
    drawdown = (np.maximum.accumulate(1+returns) - (1+returns)) / np.maximum.accumulate(1+returns)
    max_dd = np.max(drawdown)
    
    print(f"\n--- REPORT: {s1_ticker}-{s2_ticker} ---")
    print(f"Cumulative Return: {cum_ret:.2%}")
    print(f"Sharpe Ratio: {sharpe:.2f}")
    print(f"Max Drawdown: {max_dd:.2%}")
    print(f"Total Trades: {np.sum(np.diff((np.array(returns)!=0).astype(int)) > 0)}")

backtest('AVGO', 'JPM', '2y')
