import pandas as pd
import numpy as np
import yfinance as yf
from statsmodels.tsa.vector_ar.vecm import coint_johansen

def johansen_basket(tickers):
    print(f"Initializing Johansen Basket for: {tickers}")
    df = yf.download(tickers, period='2y')['Close'].dropna()
    
    # det_order=0 (constant), k_ar_diff=1 (lag)
    result = coint_johansen(df, 0, 1)
    
    # Eigenvectors (first one is the most stationary combination)
    weights = result.evec[:, 0]
    # Normalize weights
    weights = weights / weights[0]
    
    print("\n--- STATIONARY BASKET WEIGHTS ---")
    for i, t in enumerate(tickers):
        print(f"{t}: {weights[i]:.4f}")
    
    print("\nTrace Statistic / Critical Values (95%):")
    print(f"{result.lr1[0]:.2f} / {result.cvt[0, 1]:.2f}")
    if result.lr1[0] > result.cvt[0, 1]:
        print("Result: Cointegration confirmed at 95% confidence.")
    else:
        print("Result: Weak cointegration in this cluster.")

johansen_basket(['EVRG', 'NI', 'ATO', 'XEL'])
