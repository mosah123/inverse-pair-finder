import json
import os
import argparse
import sys
import pandas as pd
import numpy as np
import time
import itertools
from scipy.stats import pearsonr
from fetcher import load_all_prices
from analyzer import compute_returns
import concurrent.futures
from tqdm import tqdm  # For a better progress bar. If not installed, run: pip install tqdm

def convert_to_biweekly(prices_dict):
    """Convert daily price series to bi-weekly by taking the last price every two weeks."""
    biweekly_prices = {}
    for ticker, daily_series in prices_dict.items():
        try:
            # Make sure the index is a datetime index
            if not isinstance(daily_series.index, pd.DatetimeIndex):
                print(f"Converting index to DatetimeIndex for {ticker}")
                daily_series.index = pd.to_datetime(daily_series.index)
                
            # Resample to bi-weekly (every 2 weeks) and take the last price
            # '2W' means two-week frequency
            biweekly_series = daily_series.resample('2W').last()
            
            # Only keep series with sufficient data points
            if len(biweekly_series) < 50:  # Require at least 50 bi-weekly points
                print(f"Skipping {ticker}: insufficient data points ({len(biweekly_series)})")
                continue
                
            biweekly_prices[ticker] = biweekly_series
        except Exception as e:
            print(f"Error converting {ticker} to bi-weekly: {e}")
            # Skip this ticker if there's an issue
            continue
    
    print(f"Converted daily prices to bi-weekly for {len(biweekly_prices)} tickers")
    return biweekly_prices

def process_chunk(pairs_chunk, returns_dict, min_data_points):
    """Calculates correlations for a chunk of ticker pairs."""
    chunk_results = []
    for ticker1, ticker2 in pairs_chunk:
        r1 = returns_dict[ticker1]
        r2 = returns_dict[ticker2]
        
        # Align dates
        common = r1.index.intersection(r2.index)
        if len(common) < min_data_points:
            continue
            
        # Calculate correlation
        try:
            r1_aligned = r1.loc[common]
            r2_aligned = r2.loc[common]
            corr, p_value = pearsonr(r1_aligned, r2_aligned)
            chunk_results.append((ticker1, ticker2, corr, p_value, len(common)))
        except Exception:
            # Skip pair if calculation fails
            continue
    return chunk_results

def calculate_all_correlations(returns, min_data_points=50):
    """Calculate correlations between all possible pairs of stocks using a chunk-based multiprocessing approach."""
    start_time = time.time()
    
    # First filter out tickers with insufficient data
    filtered_returns = {}
    for ticker, series in returns.items():
        if len(series) >= min_data_points:
            filtered_returns[ticker] = series
        else:
            print(f"Excluding {ticker} from correlation analysis: only {len(series)} data points")
    
    tickers = list(filtered_returns.keys())
    n_tickers = len(tickers)
    
    # Generate all unique pairs of tickers
    pairs = list(itertools.combinations(tickers, 2))
    total_possible_pairs = len(pairs)
    
    print(f"Calculating correlations for {n_tickers} stocks with sufficient data ({total_possible_pairs} possible pairs)...")
    
    results = []
    
    # Determine number of workers and chunk size
    num_workers = os.cpu_count() or 4  # Default to 4 if cpu_count is None
    chunk_size = max(1, len(pairs) // (num_workers * 4)) # Create many small chunks for better load balancing
    chunks = [pairs[i:i + chunk_size] for i in range(0, len(pairs), chunk_size)]
    
    # Use ProcessPoolExecutor for parallel processing
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Map the processing function to all chunks
        future_to_chunk = {executor.submit(process_chunk, chunk, filtered_returns, min_data_points): chunk for chunk in chunks}
        
        # Use tqdm for a progress bar over the chunks
        for future in tqdm(concurrent.futures.as_completed(future_to_chunk), total=len(chunks), desc="Calculating Correlations"):
            chunk_result = future.result()
            if chunk_result:
                results.extend(chunk_result)

    elapsed = time.time() - start_time
    print(f"Completed all correlations in {elapsed:.2f} seconds")
    print(f"Attempted {total_possible_pairs} comparisons, successfully calculated {len(results)} correlations")
    print(f"Filtered out {total_possible_pairs - len(results)} pairs due to insufficient overlapping data points")
    
    return results, total_possible_pairs

def run(data_path=None, output_path=None, min_data_points=50):
    """
    Run the inverse pair finder pipeline for all possible stock pairs.
    
    Args:
        data_path: Path to the directory containing CSV files
        output_path: Path to save the output JSON file
        min_data_points: Minimum number of data points required for correlation
    """
    total_start_time = time.time()
    
    # Calculate default paths relative to the script location
    if data_path is None:
        # Use absolute path based on script location
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(os.path.dirname(script_dir), "data")
    
    if output_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(os.path.dirname(script_dir), "correlations")
    
    # Print absolute paths for clarity
    print(f"Using data directory: {os.path.abspath(data_path)}")
    
    # Ensure output directory exists
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, "correlations.json")
    
    # --- EDA Step 1: Load Raw Data ---
    # Load price data
    load_start = time.time()
    print(f"Loading price data from {data_path}...")
    daily_prices = load_all_prices(data_path)
    load_end = time.time()
    print(f"Loading data took {load_end - load_start:.2f} seconds")
    
    if not daily_prices:
        print("No price data found. Exiting.")
        return
    
    # --- EDA Step 2: Data Cleaning & Transformation ---
    # Convert daily prices to bi-weekly and filter out series with insufficient data
    convert_start = time.time()
    print("Converting daily prices to bi-weekly...")
    prices = convert_to_biweekly(daily_prices)
    convert_end = time.time()
    print(f"Converting to bi-weekly took {convert_end - convert_start:.2f} seconds")
    
    # Compute returns from the cleaned price data
    returns_start = time.time()
    print(f"Computing bi-weekly returns...")
    returns = compute_returns(prices)
    returns_end = time.time()
    print(f"Computing returns took {returns_end - returns_start:.2f} seconds")
    
    # Print data consistency parameters
    print(f"Using minimum data points threshold: {min_data_points}")
    
    # --- EDA Step 3: Statistical Analysis & Calculation ---
    # Calculate all correlations with minimum data point requirement
    corr_start = time.time()
    print(f"Calculating correlations for all pairs (minimum {min_data_points} data points)...")
    all_correlations, comparison_attempts = calculate_all_correlations(returns, min_data_points=min_data_points)
    corr_end = time.time()
    print(f"Calculating all correlations took {corr_end - corr_start:.2f} seconds")
    
    # --- EDA Step 4: Summarize and Save Results ---
    # Find and print the most inverse pairs
    all_correlations.sort(key=lambda x: x[2])  # Sort by correlation (ascending)
    
    # Print top 10 most inverse pairs
    print("\nTop 10 most inverse correlations:")
    print("Ticker1   Ticker2   Correlation   P-Value        Data Points")
    print("-" * 65)
    for t1, t2, corr, p, n in all_correlations[:10]:
        print(f"{t1:8s}  {t2:8s}  {corr:11.4f}   {p:.8f}   {n}")
    
    # Save all results to JSON
    output_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_tickers": len(returns),
        "total_comparisons_attempted": comparison_attempts,
        "total_pairs_calculated": len(all_correlations),
        "minimum_data_points": min_data_points,
        "correlations": [
            {
                "ticker1": t1,
                "ticker2": t2,
                "correlation": corr,
                "p_value": p,
                "data_points": n
            }
            for t1, t2, corr, p, n in all_correlations
        ]
    }
    
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)
    
    total_end_time = time.time()
    print(f"\nTotal runtime: {total_end_time - total_start_time:.2f} seconds")
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find correlations between all possible stock pairs")
    parser.add_argument("--data-path", help="Path to directory with CSV files")
    parser.add_argument("--output-path", help="Path to save output JSON")
    parser.add_argument("--min-data-points", type=int, default=50, 
                        help="Minimum number of data points required for correlation")
    
    args = parser.parse_args()
    run(data_path=args.data_path, output_path=args.output_path, 
        min_data_points=args.min_data_points)
