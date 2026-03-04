import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set(style='darkgrid')

def get_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date)['Close']
    return data

def find_cointegrated_pairs(data):
    n = data.shape[1]
    keys = data.keys()
    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            S1 = data[keys[i]]
            S2 = data[keys[j]]
            result = sm.tsa.stattools.coint(S1, S2)
            pvalue = result[1]
            if pvalue < 0.05:
                pairs.append((keys[i], keys[j], pvalue))
    return pairs

# Tech sector sample
tickers = ['AAPL', 'MSFT', 'GOOGL', 'META', 'AMZN', 'NVDA', 'ADBE', 'CRM', 'INTC', 'CSCO']
print(f"Fetching data for: {tickers}")
df = get_data(tickers, '2023-01-01', '2024-01-01')

print("\nScanning for cointegrated pairs (p-value < 0.05)...")
pairs = find_cointegrated_pairs(df)

for p in pairs:
    print(f"Pair: {p[0]} - {p[1]} | P-Value: {p[2]:.4f}")

if pairs:
    # Pick the strongest pair
    s1, s2 = pairs[0][0], pairs[0][1]
    print(f"\nVisualizing top pair: {s1} & {s2}")
    
    plt.figure(figsize=(12, 6))
    plt.plot(df[s1] / df[s1].iloc[0], label=f'{s1} (Normalized)')
    plt.plot(df[s2] / df[s2].iloc[0], label=f'{s2} (Normalized)')
    plt.title(f'Normalized Price: {s1} vs {s2}')
    plt.legend()
    plt.savefig('pair_comparison.png')
    
    # Plot the spread
    model = sm.OLS(df[s1], sm.add_constant(df[s2])).fit()
    spread = df[s1] - model.params[s2] * df[s2] - model.params['const']
    zscore = (spread - spread.mean()) / spread.std()
    
    plt.figure(figsize=(12, 4))
    zscore.plot()
    plt.axhline(zscore.mean(), color='black')
    plt.axhline(2.0, color='red', linestyle='--')
    plt.axhline(-2.0, color='green', linestyle='--')
    plt.title(f'Z-Score of Spread ({s1}/{s2})')
    plt.savefig('pair_zscore.png')
    print("Visuals saved to pair_comparison.png and pair_zscore.png")
else:
    print("No pairs found in this sample.")
