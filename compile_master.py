import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
import json

def get_extensive_data():
    # Diversified list across Tech, Finance, Energy, Retail, Healthcare, Payments, and Semi
    sectors = {
        'TECH': ['AAPL', 'MSFT', 'GOOGL', 'META', 'AMZN', 'ADBE', 'CRM', 'ORCL'],
        'FINANCE': ['JPM', 'BAC', 'GS', 'MS', 'WFC', 'C'],
        'ENERGY': ['XOM', 'CVX', 'BP', 'SHEL', 'TTE'],
        'RETAIL': ['WMT', 'TGT', 'COST', 'HD', 'LOW'],
        'HEALTH': ['JNJ', 'PFE', 'UNH', 'ABBV', 'MRK'],
        'PAYMENTS': ['V', 'MA', 'PYPL', 'AXP'],
        'SEMI': ['NVDA', 'AMD', 'INTC', 'TSM', 'AVGO']
    }
    
    all_tickers = [ticker for sector in sectors.values() for ticker in sector]
    print(f"Fetching data for {len(all_tickers)} tickers across {len(sectors)} sectors...")
    
    data = yf.download(all_tickers, start='2023-01-01', end='2025-02-21')['Close']
    return data.dropna(axis=1), sectors

def analyze_all_pairs(data, sectors):
    tickers = data.columns
    n = len(tickers)
    results = []
    
    print(f"Analyzing {n*(n-1)//2} potential combinations...")
    
    for i in range(n):
        for j in range(i+1, n):
            t1, t2 = tickers[i], tickers[j]
            s1, s2 = data[t1], data[t2]
            
            # Cointegration Test
            score, pvalue, _ = coint(s1, s2)
            
            # Correlation
            correlation = s1.corr(s2)
            
            # Only keep statistically significant pairs (p < 0.05)
            if pvalue < 0.05:
                # Determine if it's Intra-sector or Inter-sector
                sec1 = next(k for k, v in sectors.items() if t1 in v)
                sec2 = next(k for k, v in sectors.items() if t2 in v)
                pair_type = "Intra-sector" if sec1 == sec2 else "Inter-sector"
                
                results.append({
                    'Pair': f"{t1} - {t2}",
                    'P-Value': float(pvalue),
                    'Correlation': float(correlation),
                    'Type': pair_type,
                    'Sector1': sec1,
                    'Sector2': sec2
                })
    
    # Sort by strongest cointegration (lowest p-value)
    return sorted(results, key=lambda x: x['P-Value'])

df, sectors = get_extensive_data()
pairs_results = analyze_all_pairs(df, sectors)

# Create the final Master List file
output_path = 'pairs-trader/MASTER_PAIRS_LIST.md'
with open(output_path, 'w') as f:
    f.write("# Master Pairs Trading Candidates\n")
    f.write(f"Generated on: 2025-02-21\n")
    f.write(f"Analysis Period: 2023-01-01 to Present\n\n")
    f.write("| Rank | Pair | P-Value | Correlation | Type | Sectors |\n")
    f.write("|------|------|---------|-------------|------|---------|\n")
    
    for i, p in enumerate(pairs_results[:20]): # Top 20 for the file
        f.write(f"| {i+1} | {p['Pair']} | {p['P-Value']:.5f} | {p['Correlation']:.2f} | {p['Type']} | {p['Sector1']}/{p['Sector2']} |\n")

print(f"Master list compiled to {output_path}")
