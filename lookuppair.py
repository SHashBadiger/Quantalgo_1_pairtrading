import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from main_pairfinder import data

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

    # Output Results
    print(f"Correlation: {corr:.4f}")
    print(f"Beta (Hedge Ratio): {model.params.iloc[1]:.4f}")
    print(f"ADF Statistic: {adf_result[0]:.4f}")
    print(f"P-Value: {adf_result[1]:.4e}")
    print(f"1% Critical Value: {adf_result[4]['1%']:.4f}")

# Compute for specific pairs
run_custom_analysis('BKNG', 'MA', data)