# TO FIND THE DATA ON A PARTICULAR PAIR OF STOCKS


import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from main_pairfinder import data
import matplotlib.pyplot as plt

def run_custom_analysis(ticker1, ticker2, data, start_date='2024-01-01'):

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

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    recent_z_score = z_score.loc[start_date:]
    ax2.plot(recent_z_score,label= "Static Z-score",  color="purple")
    ax2.axhline(0, color='black') # Mean line
    ax2.axhline(2.0, color='red', linestyle='--')  # 2 Std Dev (Sell Signal)
    ax2.axhline(-2.0, color='green', linestyle='--') # -2 Std Dev (Buy Signal)
    ax2.set_ylabel('Standard Deviations (Z-Score)')
    ax2.set_title(f"Z score between {ticker1} and {ticker2} (Hedge ratio:{model.params.iloc[1]:.4f})")
    ax2.legend()

    recent_spread = spread.loc[start_date:]
    ax1.plot(recent_spread, label= f"{ticker2}-({model.params.iloc[1]:.4f}*{ticker1})")
    ax1.set_title(f"Spread between {ticker1} and {ticker2}")
    ax1.set_ylabel("Spread")
    ax1.axhline(spread.mean(), color='red', linestyle='--')
    ax1.legend()

    plt.tight_layout()
    plt.savefig('Spread & Zscore results.png')

# Compute for specific pairs
run_custom_analysis('EVRG', 'NI', data)
