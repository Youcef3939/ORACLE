# dashboard/app.py
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
from sklearn.decomposition import PCA

# Paths
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
LATENT_PATH = DATA_DIR / "latent_embeddings.parquet"
REGIME_PATH = DATA_DIR / "regime_labels.parquet"
PRED_PATH = DATA_DIR / "predicted_regimes.parquet"

# Load data safely
@st.cache_data
def load_data():
    latent = pd.read_parquet(LATENT_PATH)
    regimes = pd.read_parquet(REGIME_PATH)
    try:
        predicted = pd.read_parquet(PRED_PATH)
    except FileNotFoundError:
        predicted = pd.DataFrame(index=latent.index)
    return latent, regimes, predicted

latent, regimes, predicted = load_data()

# Sidebar controls
st.sidebar.title("ORACLE Market Regime Dashboard")
assets = st.sidebar.multiselect("Select assets", latent.columns, default=list(latent.columns))
start_date = st.sidebar.date_input("Start date", latent.index.min())
end_date = st.sidebar.date_input("End date", latent.index.max())
show_pred = st.sidebar.checkbox("Show predicted regimes", value=True)
resample_timeline = st.sidebar.checkbox("Resample timeline weekly", value=True)

# Filter and handle missing data
latent_filtered = latent.loc[start_date:end_date, assets].ffill().bfill()
regimes_filtered = regimes.loc[start_date:end_date].ffill().bfill()
predicted_filtered = predicted.loc[start_date:end_date] if not predicted.empty else pd.DataFrame(index=latent_filtered.index)

# Timeline plot
st.header("Market Regime Timeline")
timeline_df = regimes_filtered.reset_index()[["Date", "regime"]]

# Add predicted regimes safely
if show_pred and not predicted_filtered.empty:
    pred_col = predicted_filtered.columns[0] if len(predicted_filtered.columns) > 0 else None
    if pred_col:
        timeline_df = timeline_df.set_index("Date")
        timeline_df["Predicted"] = predicted_filtered[pred_col]
        timeline_df = timeline_df.reset_index()

# Optional resampling
if resample_timeline:
    timeline_df.set_index("Date", inplace=True)
    timeline_df = timeline_df.resample("W").last().reset_index()

# Dynamic color map for regimes
all_labels = timeline_df["regime"].unique()
color_map = {label: px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]
             for i, label in enumerate(all_labels)}

fig = px.scatter(
    timeline_df,
    x="Date",
    y="regime",
    color="regime",
    symbol="Predicted" if show_pred and "Predicted" in timeline_df.columns else None,
    title="Market Regimes Over Time",
    color_discrete_map=color_map
)
st.plotly_chart(fig, use_container_width=True)

# Latent embeddings projection
st.header("Latent Embeddings Projection")
pca = PCA(n_components=2)
latent_2d = pca.fit_transform(latent_filtered)
latent_2d_df = pd.DataFrame(latent_2d, columns=["PC1", "PC2"], index=latent_filtered.index)
latent_2d_df["Regime"] = regimes_filtered["regime"].values

fig2 = px.scatter(
    latent_2d_df,
    x="PC1",
    y="PC2",
    color="Regime",
    title="Latent Space Projection",
    color_discrete_map=color_map
)
st.plotly_chart(fig2, use_container_width=True)

# Forecast probabilities (if available)
st.header("Predicted Regime Probabilities")
if {"Bull_prob", "Bear_prob", "Crisis_prob"}.issubset(predicted_filtered.columns):
    prob_df = predicted_filtered[["Bull_prob", "Bear_prob", "Crisis_prob"]].loc[start_date:end_date]
    st.line_chart(prob_df)
else:
    st.info("Prediction probabilities not available. Only regime labels are shown.")

# Metrics
st.header("Regime Counts")
regime_counts = regimes_filtered["regime"].value_counts()
st.bar_chart(regime_counts)

if show_pred and "regime" in predicted_filtered.columns:
    pred_counts = predicted_filtered["regime"].value_counts()
    st.bar_chart(pred_counts)

st.success("ORACLE dashboard loaded! 🔮")
