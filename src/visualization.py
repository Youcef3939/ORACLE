# src/visualization.py

import os
import pandas as pd
import plotly.express as px
import umap
from model_core import LATENT_EMBEDDINGS_PATH, PROCESSED_DIR
from regime_detector import REGIME_OUTPUT_PATH

ASSETS_DIR = os.path.join(PROCESSED_DIR, "raw_assets")  # assuming you have processed prices here

def plot_regime_timeline():
    # Load regime labels
    df_regimes = pd.read_parquet(REGIME_OUTPUT_PATH)
    df_regimes = df_regimes.reset_index()  # ensure 'Date' is a column

    # Plot regime timeline
    fig = px.scatter(df_regimes, x="Date", y=[0]*len(df_regimes),
                     color="regime",
                     labels={"y": "Market Regime"},
                     title="Market Regime Timeline")
    fig.update_yaxes(showticklabels=False)
    fig.show()


def plot_latent_umap():
    df_latent = pd.read_parquet(LATENT_EMBEDDINGS_PATH)
    df_latent = df_latent.reset_index()
    
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    embedding_2d = reducer.fit_transform(df_latent.iloc[:, 1:])  # skip Date
    
    df_latent["UMAP_1"] = embedding_2d[:, 0] # type: ignore
    df_latent["UMAP_2"] = embedding_2d[:, 1] # type: ignore

    # Load regimes for color
    df_regimes = pd.read_parquet(REGIME_OUTPUT_PATH).reset_index()
    df_latent["regime"] = df_regimes["regime"]

    # Plot
    fig = px.scatter(df_latent, x="UMAP_1", y="UMAP_2", color="regime",
                     hover_data={"Date": df_latent["Date"]},
                     title="Latent Embeddings Projection (UMAP)")
    fig.show()


if __name__ == "__main__":
    print("Plotting market regime timeline...")
    plot_regime_timeline()
    
    print("Plotting latent embeddings UMAP...")
    plot_latent_umap()
