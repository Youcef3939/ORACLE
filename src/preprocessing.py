# src/preprocessing.py

import os
import pandas as pd
import numpy as np

# Paths

RAW_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
os.makedirs(RAW_DIR, exist_ok=True)  # ensure the folder exists

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
os.makedirs(PROCESSED_DIR, exist_ok=True)

PROCESSED_MATRIX_PATH = os.path.join(PROCESSED_DIR, 'processed_matrix.parquet')

# Preprocessor Class
class Preprocessor:
    
    def __init__(self, asset_files=None):
        """
        asset_files: list of filenames (without extension) to process.
        If None, process all parquet files in RAW_DIR.
        """
        if asset_files:
            self.asset_files = [f"{f}.parquet" for f in asset_files]
        else:
            self.asset_files = [f for f in os.listdir(RAW_DIR) if f.endswith('.parquet')]

    def load_data(self):
        """
        Load all parquet files and return a dict of DataFrames
        """
        data_dict = {}
        for f in self.asset_files:
            name = f.replace('.parquet','')
            path = os.path.join(RAW_DIR, f)
            df = pd.read_parquet(path)
            if 'Close' in df.columns:
                df = df[['Close']].rename(columns={'Close': name})
            else:
                # In case it's a macro indicator
                df = df.rename(columns={df.columns[0]: name})
            data_dict[name] = df
        return data_dict

    def align_data(self, data_dict):
        """
        Align all DataFrames on the same date index
        """
        combined = pd.concat(data_dict.values(), axis=1)
        combined = combined.sort_index()
        combined = combined.ffill().bfill()  # Fill missing values
        return combined

    def compute_returns(self, df, log_returns=False):
        """
        Compute daily returns
        """
        if log_returns:
            returns = np.log(df / df.shift(1))
        else:
            returns = df.pct_change()
        returns = returns.dropna()
        return returns

    def normalize(self, df):
        """
        Z-score normalization per column
        """
        return (df - df.mean()) / df.std()

    def save_matrix(self, df):
        df.to_parquet(PROCESSED_MATRIX_PATH)
        print(f"Processed matrix saved to {PROCESSED_MATRIX_PATH}")

    def run(self, log_returns=True):
        data_dict = self.load_data()
        combined = self.align_data(data_dict)
        returns = self.compute_returns(combined, log_returns=log_returns)
        normalized = self.normalize(returns)
        self.save_matrix(normalized)
        return normalized


# Example usage

if __name__ == "__main__":
    preprocessor = Preprocessor()
    matrix = preprocessor.run(log_returns=True)
    print(matrix.head())
