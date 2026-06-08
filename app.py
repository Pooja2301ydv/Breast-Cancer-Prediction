import streamlit as st
import pandas as pd
import numpy as np
from src.utils import load_model
from sklearn.datasets import load_breast_cancer

st.set_page_config(page_title="Breast Cancer Predictor", page_icon="🩺")
st.title("🩺 Breast Cancer Diagnosis Predictor")
st.write("Predict whether a tumor is **malignant (0)** or **benign (1)**.")

@st.cache_resource
def get_model():
    return load_model("models/best_model.joblib")

@st.cache_resource
def get_features():
    ds = load_breast_cancer()
    return list(ds.feature_names)

model = get_model()
feature_names = get_features()

mode = st.radio("Input mode", ["Manual entry", "Upload CSV"], horizontal=True)

def predict_df(df: pd.DataFrame):
    probs = model.predict_proba(df)[:, 1]
    preds = (probs >= 0.5).astype(int)
    out = df.copy()
    out["prob_benign"] = probs
    out["prediction"] = preds
    return out

if mode == "Manual entry":
    st.subheader("Enter feature values")
    cols = st.columns(3)
    values = []
    for i, f in enumerate(feature_names):
        with cols[i % 3]:
            val = st.number_input(f, value=0.0, step=0.1, format="%.4f")
        values.append(val)
    if st.button("Predict"):
        df = pd.DataFrame([values], columns=feature_names)
        res = predict_df(df)
        st.success(f"Predicted class: {int(res['prediction'].iloc[0])}  |  Probability benign: {res['prob_benign'].iloc[0]:.3f}")
        st.dataframe(res)

else:
    st.subheader("Upload CSV")
    st.caption("CSV must have exactly these columns (in any order):")
    st.code(", ".join(feature_names))
    up = st.file_uploader("Choose CSV", type=["csv"])
    if up:
        df = pd.read_csv(up)
        missing = [c for c in feature_names if c not in df.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            res = predict_df(df[feature_names])
            st.success("Predictions ready!")
            st.dataframe(res)
            st.download_button("Download results CSV", res.to_csv(index=False), file_name="predictions.csv")
