# TO FIND THE DATA ON A PARTICULAR PAIR OF STOCKS


import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from main_pairfinder import data
import matplotlib.pyplot as plt

def run_custom_analysis(ticker1, ticker2, data):

    if ticker1 not in data.columns or ticker2 not in data.columns:
        print(f"Error: One or both tickers not found in dataset.")
        return

    s1, s2 = data[ticker1], data[ticker2]
    
    # 1. Correlation
    corr = s1.corr(s2)
    
    # 2. ADF & Regression
    X = sm.add_constant(s1)
    model = sm.OLS(s2, X, missing='drop').fit()
    spread = model.resid.dropna()
    adf_result = adfuller(spread)

    spread_mean = spread.mean()
    spread_std = spread.std()
    z_score = (spread-spread_mean) / spread_std
    # Output Results
    print(f"Correlation: {corr:.4f}")
    print(f"Beta (Hedge Ratio): {model.params.iloc[1]:.4f}")
    print(f"ADF Statistic: {adf_result[0]:.4f}")
    print(f"P-Value: {adf_result[1]:.4e}")
    print(f"1% Critical Value: {adf_result[4]['1%']:.4f}")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    recent_z_score = z_score.iloc[-1000:]
    ax1.plot(recent_z_score)
    ax1.axhline(0, color='black') # Mean line
    ax1.axhline(1.75, color='red', linestyle='--')  # 2 Std Dev (Sell Signal)
    ax1.axhline(-1.75, color='green', linestyle='--') # -2 Std Dev (Buy Signal)
    ax1.set_ylabel('Standard Deviations (Z-Score)')
    ax1.set_title("Z score")

    recent_spread = spread.iloc[-1000:]
    ax2.plot(recent_spread)
    ax2.set_title(f"Spread between {ticker1} and {ticker2}")
    ax2.set_xlabel("Time")
    ax2.axhline(spread.mean(), color='red', linestyle='--')

    plt.gcf().autofmt_xdate()
    plt.show()

# Compute for specific pairs
run_custom_analysis('BKNG', 'MA', data)
