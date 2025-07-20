import os
import pandas as pd

def load_all_prices(path="../data"):
    """Return dict[ticker→Series of daily 'Close' prices indexed by date]."""
    prices = {}
    if not os.path.exists(path):
        print(f"Warning: Directory {path} does not exist!")
        return prices
        
    for fn in os.listdir(path):
        if fn.upper().endswith(".CSV"):
            ticker = fn[:-4].upper()
            try:
                # Explicitly parse dates and set as index, with utc=True to fix warnings
                df = pd.read_csv(os.path.join(path, fn))
                if "Date" in df.columns:
                    df["Date"] = pd.to_datetime(df["Date"], utc=True)
                    df.set_index("Date", inplace=True)
                    if "Close" in df.columns:
                        prices[ticker] = df["Close"].sort_index()
                    else:
                        print(f"Warning: No 'Close' column in {fn}")
                else:
                    print(f"Warning: No 'Date' column in {fn}")
            except Exception as e:
                print(f"Error loading {fn}: {e}")
    
    if not prices:
        print(f"Warning: No valid CSV files found in {path}")
    else:
        print(f"Loaded price data for {len(prices)} tickers")
    
    return prices
