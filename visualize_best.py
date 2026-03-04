import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Set style
sns.set(style='darkgrid')

def visualize_pair(s1, s2, data):
    # Normalized Price
    plt.figure(figsize=(12, 6))
    plt.plot(data[s1] / data[s1].iloc[0], label=f'{s1} (Normalized)')
    plt.plot(data[s2] / data[s2].iloc[0], label=f'{s2} (Normalized)')
    plt.title(f'Normalized Comparison: {s1} vs {s2}')
    plt.legend()
    filename_p = f'comparison_{s1}_{s2}.png'
    plt.savefig(filename_p)
    plt.close()

    # Z-Score Spread
    model = sm.OLS(data[s1], sm.add_constant(data[s2])).fit()
    spread = data[s1] - model.params[s2] * data[s2] - model.params['const']
    zscore = (spread - spread.mean()) / spread.std()
    
    plt.figure(figsize=(12, 4))
    zscore.plot(color='purple')
    plt.axhline(0, color='black', alpha=0.5)
    plt.axhline(2.0, color='red', linestyle='--')
    plt.axhline(-2.0, color='green', linestyle='--')
    plt.title(f'Z-Score Spread: {s1} / {s2}')
    filename_z = f'zscore_{s1}_{s2}.png'
    plt.savefig(filename_z)
    plt.close()
    
    return filename_p, filename_z

# Load best pairs
with open('best_pairs.json', 'r') as f:
    pairs = json.load(f)

# Fetch data for visual
tickers = list(set([p['S1'] for p in pairs] + [p['S2'] for p in pairs]))
df = yf.download(tickers, start='2024-01-01')['Close']

# Gen visuals for top 2
results = []
for p in pairs[:2]:
    p1, p2 = visualize_pair(p['S1'], p['S2'], df)
    results.append((p['Pair'], p1, p2))

print(json.dumps(results))
