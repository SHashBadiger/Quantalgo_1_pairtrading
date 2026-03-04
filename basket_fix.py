import pandas as pd
import numpy as np
import yfinance as yf
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import itertools

def solve_basket():
    tickers = ['AVGO', 'TSM', 'NVDA', 'JPM', 'GS', 'MS', 'WFC']
    print(f"Phase 2: Pruning Johansen Baskets for: {tickers}")
    df = yf.download(tickers, period='2y')['Close'].dropna()
    
    best_ratio = 0
    best_subset = None
    best_weights = None

    # Test all sub-combinations of minimum 4 stocks
    for r in range(4, len(tickers) + 1):
        for subset in itertools.combinations(tickers, r):
            subset = list(subset)
            try:
                # det_order=0, k_ar_diff=1
                res = coint_johansen(df[subset], 0, 1)
                
                # We look at the first Trace Statistic (r=0) vs the 99% Critical Value (index 2)
                trace_stat = res.lr1[0]
                crit_99 = res.cvt[0, 2]
                ratio = trace_stat / crit_99
                
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_subset = subset
                    best_weights = res.evec[:, 0] / res.evec[0, 0]
            except:
                continue

    print(f"\n--- WINNING BASKET (99% Confidence Ratio: {best_ratio:.2f}) ---")
    print(f"Stocks: {best_subset}")
    for i, t in enumerate(best_subset):
        print(f"{t}: {best_weights[i]:.4f}")
    
    return best_subset, best_weights

def sentiment_guard(score, position_open):
    """
    Implements the 'Kill Switch' logic.
    """
    print(f"Phase 3: Sentiment Sentinel Check (Score: {score})")
    if position_open and score < -0.6:
        print("!!! SENTIMENT KILL SWITCH ACTIVATED: FORCING EXIT !!!")
        return "EXIT"
    return "HOLD"

if __name__ == "__main__":
    best_subset, best_weights = solve_basket()
    # Dummy score check for demonstration
    sentiment_guard(-0.75, True)
