# src/data_fetcher.py

import os
import pandas as pd
import yfinance as yf
from datetime import datetime
try:
    from fredapi import Fred # type: ignore
except ImportError:
    Fred = None  # Optional if user doesn't have FRED API key

RAW_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
PROCESSED_DIR = os.path.join(RAW_DIR, 'processed')

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

class DataFetcher:
    def __init__(self, start_date='2010-01-01', end_date=None, fred_api_key=None):
        self.start_date = start_date
        self.end_date = end_date if end_date else datetime.today().strftime('%Y-%m-%d')
        self.fred = Fred(api_key=fred_api_key) if fred_api_key and Fred else None

   
    # Equity, ETF, Commodity, FX
   
    def fetch_equities(self, symbols):
        """
        Fetch historical OHLCV data from yfinance for multiple symbols.
        Returns dict of DataFrames.
        """
        data_dict = {}
        for symbol in symbols:
            print(f"Fetching {symbol}...")
            df = yf.download(symbol, start=self.start_date, end=self.end_date)
            if not df.empty: # type: ignore
                data_dict[symbol] = df
                self.save_raw(df, symbol)
        return data_dict

    
    # Macro Indicators via FRED
    
    def fetch_macro(self, indicators):
        """
        Fetch macro indicators from FRED.
        Returns dict of DataFrames.
        """
        if not self.fred:
            raise ValueError("FRED API key not provided or fredapi not installed.")
        data_dict = {}
        for indicator in indicators:
            print(f"Fetching {indicator} from FRED...")
            series = self.fred.get_series(indicator, observation_start=self.start_date, observation_end=self.end_date)
            df = pd.DataFrame(series, columns=[indicator])
            df.index = pd.to_datetime(df.index)
            data_dict[indicator] = df
            self.save_raw(df, indicator)
        return data_dict

    
    # Save / Load
    
    def save_raw(self, df, name):
        """
        Save raw data to data/raw/processed
        """
        path = os.path.join(PROCESSED_DIR, f"{name}.parquet")
        df.to_parquet(path)
        print(f"Saved {name} to {path}")

    def load_processed(self, name):
        """
        Load processed data from data/raw/processed
        """
        path = os.path.join(PROCESSED_DIR, f"{name}.parquet")
        if os.path.exists(path):
            return pd.read_parquet(path)
        else:
            raise FileNotFoundError(f"{path} does not exist.")


# Example usage

if __name__ == "__main__":
    fetcher = DataFetcher(start_date='2015-01-01', end_date='2025-01-01', fred_api_key=None)

    # Example symbols
    equities = ['AAPL', 'SPY', 'GLD']
    fetcher.fetch_equities(equities)

    # Example macro indicators (only works if FRED API key is set)
    # macro_indicators = ['CPIAUCSL', 'UNRATE', 'FEDFUNDS']
    # fetcher.fetch_macro(macro_indicators)