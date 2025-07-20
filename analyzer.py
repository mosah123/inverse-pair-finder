from scipy.stats import pearsonr
import pandas as pd
import numpy as np

def compute_returns(prices):
    """Convert price Series dict→dict of returns Series."""
    return {t: p.pct_change().dropna() for t, p in prices.items()}

def find_inverse_pair(anchor, returns):
    """
    Given anchor ticker and dict[ticker→returns Series],
    return (best_ticker, correlation) with the lowest Pearson r.
    """
    if anchor not in returns:
        raise ValueError(f"Anchor ticker '{anchor}' not found in returns data")
        
    anchor_r = returns[anchor]
    best = (None, 1.0)
    
    results = []
    for t, r in returns.items():
        if t == anchor: 
            continue
            
        # align dates
        common = anchor_r.index.intersection(r.index)
        if len(common) < 30: 
            print(f"Skipping {t}: insufficient overlapping data points ({len(common)})")
            continue  # skip too-few points
            
        # Calculate correlation
        corr, p_value = pearsonr(anchor_r.loc[common], r.loc[common])
        results.append((t, corr, p_value, len(common)))
        
        if corr < best[1]:
            best = (t, corr)
    
    # Print top 3 most inverse correlations (changed from 5)
    results.sort(key=lambda x: x[1])
    print("\nTop 3 most inverse correlations with", anchor)
    print("Ticker    Correlation   P-Value   Data Points")
    print("----------------------------------------")
    for t, corr, p, n in results[:3]:
        print(f"{t:8s}   {corr:.4f}       {p:.4e}   {n}")
        
    return best
