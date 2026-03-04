import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from main_pairfinder import data

def kalman_filter_beta(x, y):
    """
    Uses a Kalman Filter to estimate the dynamic hedge ratio (beta) between two assets.
    """
    n_series = len(y)
    
    # State space model parameters
    # delta is the process noise covariance (how fast beta changes)
    delta = 1e-5
    trans_cov = delta / (1 - delta) * np.eye(2)
    
    obs_mat = np.expand_dims(np.vstack([[x], [np.ones(len(x))]]).T, axis=1)
    
    # Initial state estimates
    state_means = np.zeros((n_series, 2))
    state_covs = np.zeros((n_series, 2, 2))
    
    # R is the measurement noise covariance
    R = 1
    
    # Delta deals with how fast the beta should change while R deals with how accurate the daily data is (how much noise exists)
    # to tune the sensitivity, tinker with delta and R. for more sensitivity increase delta and reduce R. vice versa for smoother curves.


    # Kalman Loop
    current_state_mean = np.zeros(2)
    current_state_cov = np.ones((2, 2))
    
    for i in range(n_series):
        # Prediction step
        # (State doesn't change, just the covariance increases)
        prediction_state_mean = current_state_mean
        prediction_state_cov = current_state_cov + trans_cov
        
        # Update step
        # Measurement residual
        y_pred = np.dot(obs_mat[i], prediction_state_mean)
        res = y[i] - y_pred
        
        # Residual covariance
        res_cov = np.dot(np.dot(obs_mat[i], prediction_state_cov), obs_mat[i].T) + R
        
        # Kalman Gain
        K = np.dot(np.dot(prediction_state_cov, obs_mat[i].T), np.linalg.inv(res_cov))
        
        # New State
        current_state_mean = prediction_state_mean + np.dot(K, res)
        current_state_cov = prediction_state_cov - np.dot(np.dot(K, obs_mat[i]), prediction_state_cov)
        
        state_means[i] = current_state_mean
        state_covs[i] = current_state_cov
        
    return state_means[:, 0], state_means[:, 1] # Beta and Intercept

def run_kalman_logic(s1_ticker, s2_ticker, data):
    print(f"Fetching data for {s1_ticker} and {s2_ticker}...")
    #df = yf.download([s1_ticker, s2_ticker], start='2024-01-01', progress=False)['Close']
    df = data[[s1_ticker, s2_ticker]].loc['2024-01-01':]
    x, y = df[s1_ticker].values, df[s2_ticker].values

    print("Running Kalman Filter to track Dynamic Beta...")
    betas, intercepts = kalman_filter_beta(x, y)

    # Calculate dynamic spread and z-score
    spread = y - (betas * x + intercepts)
    zscore_raw = (spread - np.mean(spread)) / np.std(spread)
    zscore = pd.Series(zscore_raw, index=df.index)

    # INSTITUTIONAL UPGRADE: Minimum Profit Threshold (15bps)
    # Only allow signal if expected reversion gain covers 7bps friction + buffer
    expected_reversion_gain = np.abs(spread / y)
    min_threshold = 0.0015 # 15 bps
    
    # Suppress signals that don't meet the profit threshold
    zscore[expected_reversion_gain < min_threshold] = 0
    
    return df, betas, zscore

if __name__ == "__main__":
    symbol_a = 'EVRG'
    symbol_b = 'NI'
    
    # 2. Pass variables into your function
    df, betas, zscore = run_kalman_logic(symbol_a, symbol_b, data)
    
    # Visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    ax1.plot(df.index, betas, label='Dynamic Beta (Hedge Ratio)', color='orange')
    ax1.set_title(f'Kalman Filter: Dynamic Hedge Ratio ({symbol_a} and {symbol_b})')
    ax1.legend()
    ax2.plot(df.index, zscore, label='Dynamic Z-Score (Thresholded)', color='purple')
    ax2.axhline(0, color='black', alpha=0.5)
    ax2.axhline(2, color='red', linestyle='--')
    ax2.axhline(-2, color='green', linestyle='--')
    ax2.set_title('Dynamic Z-Score (Signal Engine w/ 15bps Min Profit)')
    ax2.legend()
    plt.tight_layout()
    plt.savefig('kalman_results.png')
    print("Institutional-grade math complete with 15bps Min Profit Threshold.")
