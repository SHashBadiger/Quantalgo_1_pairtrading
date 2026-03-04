import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint

def get_sector_data():
    # Tech, Finance, Energy sectors
    tickers = [
        'AAPL', 'MSFT', 'GOOGL', 'META', 'AMZN', 'NVDA', 'ADBE', # Tech
        'JPM', 'BAC', 'GS', 'MS', 'WFC', # Finance
        'XOM', 'CVX', 'BP', 'SHEL', # Energy
        'KO', 'PEP', # Consumer
        'V', 'MA' # Payments
    ]
    print(f"Fetching data for {len(tickers)} tickers...")
    data = yf.download(tickers, start='2023-01-01', end='2025-01-01')['Close']
    return data.dropna(axis=1)

def find_best_pairs(data):
    n = data.shape[1]
    keys = data.keys()
    results = []
    
    print("Running cointegration matrix...")
    for i in range(n):
        for j in range(i+1, n):
            S1 = data[keys[i]]
            S2 = data[keys[j]]
            score, pvalue, _ = coint(S1, S2)
            
            # Correlation for extra confirmation
            correlation = S1.corr(S2)
            
            if pvalue < 0.02: # Stricter threshold for "best"
                results.append({
                    'Pair': f"{keys[i]} - {keys[j]}",
                    'P-Value': pvalue,
                    'Correlation': correlation,
                    'S1': keys[i],
                    'S2': keys[j]
                })
    
    # Sort by p-value (lower is better cointegration)
    return sorted(results, key=lambda x: x['P-Value'])

df = get_sector_data()
best_pairs = find_best_pairs(df)

print("\n--- TOP COINTEGRATED PAIRS ---")
for i, p in enumerate(best_pairs[:10]):
    print(f"{i+1}. {p['Pair']} | P-Value: {p['P-Value']:.5f} | Corr: {p['Correlation']:.2f}")

# Save for next step (visuals)
import json
with open('best_pairs.json', 'w') as f:
    json.dump(best_pairs[:5], f)
