import os
import pandas as pd
import numpy as np

PROCESSED_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "processed")
LATENT_EMBEDDINGS_PATH = os.path.join(PROCESSED_DIR, "latent_embeddings.parquet")
FEATURES_OUTPUT_PATH = os.path.join(PROCESSED_DIR, "features.parquet")

class FeatureEngineer:
    def __init__(self, window=5):
        self.window = window
        self.latent = pd.read_parquet(LATENT_EMBEDDINGS_PATH)
        self.features = pd.DataFrame(index=self.latent.index)

    def add_moving_average(self):
        for col in self.latent.columns:
            self.features[f"{col}_ma{self.window}"] = self.latent[col].rolling(self.window).mean()

    def add_volatility(self):
        for col in self.latent.columns:
            self.features[f"{col}_vol{self.window}"] = self.latent[col].rolling(self.window).std()

    def add_returns(self):
        for col in self.latent.columns:
            self.features[f"{col}_ret"] = self.latent[col].pct_change()

    def add_lags(self, n_lags=3):
        for col in self.latent.columns:
            for lag in range(1, n_lags+1):
                self.features[f"{col}_lag{lag}"] = self.latent[col].shift(lag)

    def create_features(self):
        self.add_moving_average()
        self.add_volatility()
        self.add_returns()
        self.add_lags()
        self.features.dropna(inplace=True)
        self.features.to_parquet(FEATURES_OUTPUT_PATH)
        print(f"Features saved to {FEATURES_OUTPUT_PATH}")
        return self.features

if __name__ == "__main__":
    fe = FeatureEngineer(window=5)
    features = fe.create_features()
    print(features.head())
