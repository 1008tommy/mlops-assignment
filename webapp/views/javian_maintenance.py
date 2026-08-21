"""Javian Ng: predictive maintenance page.

Single and batch prediction with the xgboost classifier trained on the smart
manufacturing sensor data. The model needs 19 engineered features built from
per-machine history, so this page rebuilds those features before predicting.
"""

from io import BytesIO
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
RAW_CSV = ROOT / "javian" / "data" / "raw" / "smart_manufacturing_data.csv"
MODEL_PKL = ROOT / "javian" / "models" / "final_XGBClassifier.pkl"

SENSORS = ["temperature", "vibration", "humidity", "pressure", "energy_consumption"]
REQUIRED = ["timestamp", "machine_id"] + SENSORS

FEATURES = [
    "temperature", "vibration", "humidity", "pressure", "energy_consumption",
    "temperature_lag1", "vibration_lag1", "humidity_lag1", "pressure_lag1",
    "energy_consumption_lag1",
    "temperature_roll_mean", "temperature_roll_std", "vibration_roll_mean",
    "vibration_roll_std", "energy_consumption_roll_mean",
    "energy_consumption_roll_std",
    "temperature_delta", "vibration_delta", "temp_x_vib",
]


@st.cache_resource
def load_maintenance_model():
    return joblib.load(MODEL_PKL)


@st.cache_data
def clip_bounds():
    """1st/99th percentile clip limits, fixed to the full training data."""
    raw = pd.read_csv(RAW_CSV)
    return {
        col: tuple(raw[col].quantile([0.01, 0.99]).tolist())
        for col in ["vibration", "temperature"]
    }


def engineer_features(df):
    """Build the 19 temporal features from raw per-machine sensor rows.

    Replicates the notebook feature engineering exactly: winsorize vibration
    and temperature, then per machine add lag1, rolling mean/std (window 5),
    deltas, and the temp x vibration interaction.
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["machine_id", "timestamp"]).reset_index(drop=True)

    bounds = clip_bounds()
    for col in ["vibration", "temperature"]:
        lo, hi = bounds[col]
        df[col] = df[col].clip(lo, hi)

    g = df.groupby("machine_id")
    for s in SENSORS:
        df[f"{s}_lag1"] = g[s].shift(1)
    for s in ["temperature", "vibration", "energy_consumption"]:
        df[f"{s}_roll_mean"] = g[s].transform(
            lambda x: x.rolling(5, min_periods=1).mean()
        )
        df[f"{s}_roll_std"] = g[s].transform(
            lambda x: x.rolling(5, min_periods=1).std().fillna(0)
        )
    for s in ["temperature", "vibration"]:
        df[f"{s}_delta"] = df[s] - df[f"{s}_lag1"]
    df["temp_x_vib"] = df["temperature"] * df["vibration"]

    return df.dropna(subset=FEATURES).reset_index(drop=True)


@st.cache_data
def training_data():
    return engineer_features(pd.read_csv(RAW_CSV))


model = load_maintenance_model()

st.title("Predictive maintenance")
st.caption("Javian's model: xgboost classifier on smart manufacturing sensor data")

tab_single, tab_batch = st.tabs(["Single prediction", "Batch prediction"])

with tab_single:
    st.markdown("Pick a machine and predict whether it needs maintenance from its latest sensor reading.")
    data = training_data()
    machines = sorted(data["machine_id"].unique())
    machine = st.selectbox("Machine", machines, format_func=lambda m: f"Machine {m}")
    row = data[data["machine_id"] == machine].sort_values("timestamp").iloc[-1]

    st.caption(f"Latest reading timestamp: {row['timestamp']}")
    cols = st.columns(5)
    for c, s in zip(cols, SENSORS):
        c.metric(s, f"{row[s]:.2f}")

    if st.button("Predict maintenance", type="primary"):
        X = row[FEATURES].to_frame().T
        risk = float(model.predict_proba(X)[0, 1])
        if risk >= 0.5:
            st.error(f"Maintenance required (risk {risk:.1%})")
        else:
            st.success(f"No maintenance required (risk {risk:.1%})")

with tab_batch:
    st.markdown(
        "Upload a CSV of raw sensor readings and predict all rows at once. "
        f"Required columns: `{', '.join(REQUIRED)}`. Rows are grouped per "
        "machine and ordered by time, and each machine needs at least two "
        "rows so the lag features exist."
    )
    with st.expander("Download a sample input file"):
        sample = pd.read_csv(RAW_CSV)[REQUIRED].head(200)
        buf = BytesIO()
        sample.to_csv(buf, index=False)
        st.download_button("Download sample CSV", buf.getvalue(), "sample_input.csv", "text/csv")

    uploaded = st.file_uploader("Upload CSV", type="csv")
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        missing = [c for c in REQUIRED if c not in df.columns]
        if missing:
            st.error(f"Missing columns: {', '.join(missing)}")
        else:
            featured = engineer_features(df)
            if featured.empty:
                st.warning("No rows could be predicted. Each machine needs at least two time ordered rows.")
            else:
                pred, proba = model.predict(featured[FEATURES]).astype(int), model.predict_proba(featured[FEATURES])[:, 1]
                result = pd.DataFrame(
                    {
                        "machine_id": featured["machine_id"],
                        "timestamp": featured["timestamp"],
                        **{s: featured[s] for s in SENSORS},
                        "maintenance_probability": proba.round(4),
                        "prediction": pred,
                    }
                )
                st.dataframe(result, width="stretch")

                c1, c2 = st.columns(2)
                c1.metric("Rows predicted", len(result))
                c2.metric("Maintenance predicted", int((result["prediction"] == 1).sum()))

                buf = BytesIO()
                result.to_csv(buf, index=False)
                st.download_button("Download predictions", buf.getvalue(), "predictions.csv", "text/csv")
