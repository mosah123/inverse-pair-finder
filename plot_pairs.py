import json
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import numpy as np
import argparse

def plot_price_pair(ticker1, ticker2, correlation, p_value, data_dir=None, output_path=None, min_data_points=30):
    """Generate a normalized price chart for a pair of stocks."""
    # Calculate default paths based on script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    
    if data_dir is None:
        data_dir = os.path.join(project_dir, "data")
    if output_path is None:
        output_path = os.path.join(project_dir, "result")
    
    print(f"Plotting normalized prices for {ticker1} vs {ticker2}...")
    
    try:
        # load prices
        df_a = pd.read_csv(os.path.join(data_dir, f"{ticker1}.csv"), parse_dates=["Date"])
        df_b = pd.read_csv(os.path.join(data_dir, f"{ticker2}.csv"), parse_dates=["Date"])
        
        # Set index and align dates
        df_a.set_index("Date", inplace=True)
        df_b.set_index("Date", inplace=True)
        
        # Get common date range
        common_dates = df_a.index.intersection(df_b.index)
        
        # Check if we have enough data points
        if len(common_dates) < min_data_points:
            print(f"Skipping price plot: Only {len(common_dates)} common dates between {ticker1} and {ticker2}, minimum required is {min_data_points}")
            return False
            
        # Subset to common dates and normalize
        df_a = df_a.loc[common_dates]["Close"]
        df_b = df_b.loc[common_dates]["Close"]
        
        # normalize to 100 at start
        df_a_norm = df_a / df_a.iloc[0] * 100
        df_b_norm = df_b / df_b.iloc[0] * 100
        
        # Create plot
        plt.figure(figsize=(12, 6))
        plt.plot(common_dates, df_a_norm, label=f"{ticker1}", linewidth=2)
        plt.plot(common_dates, df_b_norm, label=f"{ticker2}", linewidth=2)
        
        plt.title(f"Normalized Prices: {ticker1} vs {ticker2} (r={correlation:.4f}, p={p_value:.4g})", fontsize=14)
        plt.xlabel("Date", fontsize=12)
        plt.ylabel("Normalized Price (Start=100)", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        
        # Format dates on x-axis
        if len(common_dates) > 500:
            plt.xticks(rotation=45)
            plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m'))
        else:
            plt.xticks(rotation=45)
            
        plt.tight_layout()
        os.makedirs(output_path, exist_ok=True)
        output_file = os.path.join(output_path, f"price_plot_{ticker1}_{ticker2}.png")
        plt.savefig(output_file, dpi=300)
        print(f"Price chart saved to {output_file}")
        return True
        
    except Exception as e:
        print(f"Error generating price plot: {e}")
        return False

def plot_return_scatter(ticker1, ticker2, correlation, p_value, data_dir=None, output_path=None, min_data_points=30):
    """Generate a scatter plot of returns."""
    # Calculate default paths based on script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    
    if data_dir is None:
        data_dir = os.path.join(project_dir, "data")
    if output_path is None:
        output_path = os.path.join(project_dir, "result")
    
    print(f"Plotting return scatter for {ticker1} vs {ticker2}...")
    
    try:
        # load data
        df_a = pd.read_csv(os.path.join(data_dir, f"{ticker1}.csv"), parse_dates=["Date"])
        df_b = pd.read_csv(os.path.join(data_dir, f"{ticker2}.csv"), parse_dates=["Date"])
        
        # Calculate returns
        df_a.set_index("Date", inplace=True)
        df_b.set_index("Date", inplace=True)
        
        r_a = df_a["Close"].pct_change().dropna()
        r_b = df_b["Close"].pct_change().dropna()
        
        # Get common dates
        common_dates = r_a.index.intersection(r_b.index)
        
        # Check if we have enough data points
        if len(common_dates) < min_data_points:
            print(f"Skipping return scatter: Only {len(common_dates)} common return data points between {ticker1} and {ticker2}, minimum required is {min_data_points}")
            return False
            
        r_a = r_a.loc[common_dates]
        r_b = r_b.loc[common_dates]
        
        # Create scatter plot
        plt.figure(figsize=(10, 8))
        plt.scatter(r_a, r_b, alpha=0.5, s=30, color='blue')
        
        # Add regression line
        m, b0 = np.polyfit(r_a, r_b, 1)
        x_range = np.linspace(r_a.min(), r_a.max(), 100)
        plt.plot(x_range, m*x_range+b0, color="red", linewidth=2, 
                label=f"Slope={m:.4f}, r={correlation:.4f}, p={p_value:.4g}")
        
        plt.title(f"Daily Return Scatter: {ticker1} vs {ticker2}", fontsize=14)
        plt.xlabel(f"{ticker1} Daily Return", fontsize=12)
        plt.ylabel(f"{ticker2} Daily Return", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        plt.legend(fontsize=12)
        
        plt.tight_layout()
        os.makedirs(output_path, exist_ok=True)
        output_file = os.path.join(output_path, f"scatter_plot_{ticker1}_{ticker2}.png") 
        plt.savefig(output_file, dpi=300)
        print(f"Scatter plot saved to {output_file}")
        return True
        
    except Exception as e:
        print(f"Error generating scatter plot: {e}")
        return False

def plot_top_inverse_pairs(json_path=None, data_dir=None, output_path=None, num_pairs=3, min_data_points=30):
    """Plot the top N most inverse pairs from the correlations.json file."""
    # Calculate default paths based on script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    
    if json_path is None:
        json_path = os.path.join(project_dir, "correlations", "correlations.json")
    if data_dir is None:
        data_dir = os.path.join(project_dir, "data")
    if output_path is None:
        output_path = os.path.join(project_dir, "result")
    
    print(f"Using JSON file: {json_path}")
    
    try:
        # Load JSON
        with open(json_path, 'r') as f:
            correlation_data = json.load(f)
        
        # Get correlations and sort by correlation (ascending)
        correlations = correlation_data["correlations"]
        
        # First filter by minimum data points
        filtered_correlations = [pair for pair in correlations if pair.get("data_points", 0) >= min_data_points]
        
        if len(filtered_correlations) == 0:
            print(f"No pairs found with at least {min_data_points} data points")
            return
            
        filtered_correlations.sort(key=lambda x: x["correlation"])
        
        # Print summary of filtered data
        print(f"Found {len(filtered_correlations)} pairs with at least {min_data_points} data points")
        print(f"Original correlation data had {len(correlations)} pairs")
        
        # Plot top N most inverse pairs
        successful_plots = 0
        pair_index = 0
        
        while successful_plots < num_pairs and pair_index < len(filtered_correlations):
            pair_data = filtered_correlations[pair_index]
            ticker1 = pair_data["ticker1"]
            ticker2 = pair_data["ticker2"]
            correlation = pair_data["correlation"]
            p_value = pair_data["p_value"]
            data_points = pair_data.get("data_points", "unknown")
            
            print(f"Attempting to plot pair {pair_index+1}: {ticker1} vs {ticker2} (r={correlation:.4f}, p={p_value:.4g}, points={data_points})")
            
            # Try to plot both charts
            price_success = plot_price_pair(ticker1, ticker2, correlation, p_value, data_dir, output_path, min_data_points)
            scatter_success = plot_return_scatter(ticker1, ticker2, correlation, p_value, data_dir, output_path, min_data_points)
            
            if price_success and scatter_success:
                successful_plots += 1
                print(f"Successfully plotted for pair {successful_plots}/{num_pairs}")
            
            pair_index += 1
            
        if successful_plots < num_pairs:
            print(f"Warning: Only able to plot {successful_plots} pairs with the given data constraints")
            
    except Exception as e:
        print(f"Error processing correlation data: {e}")

def plot_consolidated_top_pairs(json_path=None, data_dir=None, output_path=None, num_pairs=3, min_data_points=30):
    """Create a single consolidated plot showing the top N most inverse pairs."""
    # Calculate default paths based on script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    
    if json_path is None:
        json_path = os.path.join(project_dir, "correlations", "correlations.json")
    if data_dir is None:
        data_dir = os.path.join(project_dir, "data")
    if output_path is None:
        output_path = os.path.join(project_dir, "result")
    
    print(f"Creating consolidated plot for top {num_pairs} inverse correlations...")
    print(f"Using JSON file: {json_path}")
    
    try:
        # Load JSON
        with open(json_path, 'r') as f:
            correlation_data = json.load(f)
        
        # Get correlations and sort by correlation (ascending)
        correlations = correlation_data["correlations"]
        
        # Filter by minimum data points
        filtered_correlations = [pair for pair in correlations if pair.get("data_points", 0) >= min_data_points]
        
        if len(filtered_correlations) == 0:
            print(f"No pairs found with at least {min_data_points} data points")
            return
            
        filtered_correlations.sort(key=lambda x: x["correlation"])
        
        # Get top N pairs
        top_pairs = filtered_correlations[:num_pairs]
        
        # Create a figure with 2 rows: first for prices, second for scatters
        fig, axs = plt.subplots(2, num_pairs, figsize=(6*num_pairs, 10))
        
        for i, pair_data in enumerate(top_pairs):
            ticker1 = pair_data["ticker1"]
            ticker2 = pair_data["ticker2"]
            correlation = pair_data["correlation"]
            p_value = pair_data["p_value"]
            
            print(f"Processing pair {i+1}: {ticker1} vs {ticker2} (r={correlation:.4f}, p={p_value:.4g})")
            
            try:
                # Load price data
                df_a = pd.read_csv(os.path.join(data_dir, f"{ticker1}.csv"), parse_dates=["Date"])
                df_b = pd.read_csv(os.path.join(data_dir, f"{ticker2}.csv"), parse_dates=["Date"])
                
                # Set index and align dates
                df_a.set_index("Date", inplace=True)
                df_b.set_index("Date", inplace=True)
                
                # Get common date range
                common_dates = df_a.index.intersection(df_b.index)
                
                if len(common_dates) < min_data_points:
                    print(f"Skipping pair: insufficient data points ({len(common_dates)})")
                    continue
                
                # Price plot (top row)
                df_a_close = df_a.loc[common_dates]["Close"]
                df_b_close = df_b.loc[common_dates]["Close"]
                
                # Normalize to 100 at start
                df_a_norm = df_a_close / df_a_close.iloc[0] * 100
                df_b_norm = df_b_close / df_b_close.iloc[0] * 100
                
                axs[0, i].plot(common_dates, df_a_norm, label=ticker1)
                axs[0, i].plot(common_dates, df_b_norm, label=ticker2)
                axs[0, i].set_title(f"{ticker1} vs {ticker2}\nr={correlation:.4f}, p={p_value:.4g}")
                axs[0, i].set_xlabel("Date")
                axs[0, i].set_ylabel("Normalized Price")
                axs[0, i].grid(True, alpha=0.3)
                axs[0, i].legend()
                axs[0, i].tick_params(axis='x', rotation=45)
                
                # Scatter plot (bottom row)
                r_a = df_a["Close"].pct_change().dropna()
                r_b = df_b["Close"].pct_change().dropna()
                
                common_dates_returns = r_a.index.intersection(r_b.index)
                r_a = r_a.loc[common_dates_returns]
                r_b = r_b.loc[common_dates_returns]
                
                axs[1, i].scatter(r_a, r_b, alpha=0.5, s=15)
                
                # Add regression line
                m, b0 = np.polyfit(r_a, r_b, 1)
                x_range = np.linspace(r_a.min(), r_a.max(), 100)
                axs[1, i].plot(x_range, m*x_range+b0, color="red", linewidth=2)
                
                axs[1, i].set_title(f"Return Scatter")
                axs[1, i].set_xlabel(f"{ticker1} Return")
                axs[1, i].set_ylabel(f"{ticker2} Return")
                axs[1, i].grid(True, alpha=0.3)
                axs[1, i].axhline(y=0, color='black', linestyle='-', alpha=0.3)
                axs[1, i].axvline(x=0, color='black', linestyle='-', alpha=0.3)
                
            except Exception as e:
                print(f"Error processing pair {ticker1}-{ticker2}: {e}")
                continue
        
        plt.tight_layout()
        os.makedirs(output_path, exist_ok=True)
        output_file = os.path.join(output_path, "top_inverse_pairs.png")
        plt.savefig(output_file, dpi=300)
        print(f"Consolidated plot saved to {output_file}")
        
    except Exception as e:
        print(f"Error creating consolidated plot: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate plots for inverse pairs")
    parser.add_argument("--json", help="Path to the correlations JSON file")
    parser.add_argument("--data-dir", help="Directory with CSV files")
    parser.add_argument("--output", help="Directory to save plots")
    parser.add_argument("--pairs", type=int, default=10, help="Number of most inverse pairs to plot")
    parser.add_argument("--ticker1", help="First ticker for manual pair plotting")
    parser.add_argument("--ticker2", help="Second ticker for manual pair plotting")
    parser.add_argument("--min-points", type=int, default=30, 
                        help="Minimum number of overlapping data points required")
    parser.add_argument("--consolidated", action="store_true", 
                        help="Create a single consolidated plot instead of individual files")
    
    args = parser.parse_args()
    
    if args.ticker1 and args.ticker2:
        # Load JSON to find correlation
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.dirname(script_dir)
        json_path = args.json or os.path.join(project_dir, "correlations", "correlations.json")
        
        try:
            with open(json_path, 'r') as f:
                correlation_data = json.load(f)
                
            # Find the correlation for this pair
            correlation = None
            p_value = None
            for pair in correlation_data["correlations"]:
                if ((pair["ticker1"] == args.ticker1 and pair["ticker2"] == args.ticker2) or
                    (pair["ticker1"] == args.ticker2 and pair["ticker2"] == args.ticker1)):
                    correlation = pair["correlation"]
                    p_value = pair["p_value"]
                    break
                    
            if correlation is None:
                print(f"No correlation data found for {args.ticker1}-{args.ticker2}, using 0")
                correlation = 0
                p_value = 0
                
            plot_price_pair(args.ticker1, args.ticker2, correlation, p_value, args.data_dir, args.output, args.min_points)
            plot_return_scatter(args.ticker1, args.ticker2, correlation, p_value, args.data_dir, args.output, args.min_points)
            
        except Exception as e:
            print(f"Error: {e}")
            # Plot anyway without correlation value
            plot_price_pair(args.ticker1, args.ticker2, 0, 0, args.data_dir, args.output, args.min_points)
            plot_return_scatter(args.ticker1, args.ticker2, 0, 0, args.data_dir, args.output, args.min_points)
    elif args.consolidated:
        # Use the consolidated plotting function
        plot_consolidated_top_pairs(args.json, args.data_dir, args.output, args.pairs, args.min_points)
    else:
        # Plot top N most inverse pairs as individual files
        plot_top_inverse_pairs(args.json, args.data_dir, args.output, args.pairs, args.min_points)
