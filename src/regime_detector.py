# src/regime_detector.py
import sys
import os
import pandas as pd
import hdbscan
from sklearn.preprocessing import StandardScaler
from model_core import LATENT_EMBEDDINGS_PATH, PROCESSED_DIR

REGIME_OUTPUT_PATH = os.path.join(PROCESSED_DIR, "regime_labels.parquet")

REGIME_LABELS = {0: "Bull", 1: "Bear", 2: "Crisis", 3: "Recovery"}


def detect_regimes(latent_df, min_cluster_size=30, min_samples=10):
    """
    Cluster latent embeddings to detect market regimes.
    
    Args:
        latent_df (pd.DataFrame): latent embeddings from VAE
        min_cluster_size (int): minimum size of clusters for HDBSCAN
        min_samples (int): controls how conservative clustering is
    
    Returns:
        pd.DataFrame: latent embeddings with regime labels
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(latent_df.values)

    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
    cluster_labels = clusterer.fit_predict(X_scaled)

    unique_labels = sorted(set(cluster_labels))
    label_map = {old: REGIME_LABELS.get(i % len(REGIME_LABELS), f"Regime_{i}") for i, old in enumerate(unique_labels)}
    regimes = [label_map[label] for label in cluster_labels]

    latent_df_copy = latent_df.copy()
    latent_df_copy["regime"] = regimes

    latent_df_copy.to_parquet(REGIME_OUTPUT_PATH)
    print(f"Regime labels saved to {REGIME_OUTPUT_PATH}")

    return latent_df_copy


if __name__ == "__main__":
    df_latent = pd.read_parquet(LATENT_EMBEDDINGS_PATH)
    print(f"Loaded {df_latent.shape[0]} days of latent embeddings.")

    # Detect regimes
    df_regimes = detect_regimes(df_latent)
    print(df_regimes.head())
