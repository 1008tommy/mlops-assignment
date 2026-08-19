"""Shared Streamlit web app (starter).

This is a placeholder so the Docker image has something to serve. Replace the
body with your real logic:

  1. dvc pull (or load a baked-in model) to get trained artifacts
  2. joblib.load / mlflow load the model(s)
  3. build input widgets and call model.predict(...)

Run locally:
    streamlit run webapp/streamlit_app.py
"""

import streamlit as st

st.set_page_config(page_title="IT3385 MLOps", layout="wide")

st.title("IT3385 MLOps — Prediction Web App")

st.info(
    "Starter app. TODO: load your trained model/artifacts "
    "(e.g. `dvc pull` + `joblib.load`), add input widgets, and call "
    "`model.predict(...)`."
)

with st.form("predict_form"):
    val = st.number_input("Enter a feature value", value=0.0)
    submitted = st.form_submit_button("Predict")

    if submitted:
        # Replace with a real model call, e.g.:
        #   pred = model.predict([[val]])
        #   st.success(f"Prediction: {pred[0]}")
        st.write("Prediction: (TODO — replace with your model call)")
