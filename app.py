import streamlit as st
import numpy as np
import joblib
import os
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostRegressor

# -------------------------------
# PATH TO DATA (FIXED)
# -------------------------------
DATA_PATH = "ml_ready_dataset.xlsx"   # keep file in same folder as app.py
MODEL_DIR = "models"

model_path = os.path.join(MODEL_DIR, "trained_model.pkl")
scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")

# -------------------------------
# LOAD DATA
# -------------------------------
def load_data():
    return pd.read_excel(DATA_PATH)

# -------------------------------
# TRAIN MODEL
# -------------------------------
def train_and_save():
    df = load_data()

    X = df.drop(columns=["Manning_n"])
    y = df["Manning_n"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = AdaBoostRegressor(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)

    os.makedirs(MODEL_DIR, exist_ok=True)

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    return model, scaler

# -------------------------------
# LOAD OR TRAIN
# -------------------------------
if not os.path.exists(model_path):
    st.warning("Training model from dataset...")
    model, scaler = train_and_save()
else:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

# -------------------------------
# UI
# -------------------------------
st.title("AI-Based Hydraulic Roughness Predictor")

Fr = st.number_input("Froude Number", value=0.5)
Re = st.number_input("Reynolds Number", value=10000.0)
HD = st.number_input("H/D", value=1.0)
LD = st.number_input("λ/D", value=2.0)
Slope = st.number_input("Slope", value=0.01)
u_star = st.number_input("Shear Velocity", value=0.05)

if st.button("Predict"):
    input_data = np.array([[Fr, Re, HD, LD, Slope, u_star]])
    input_scaled = scaler.transform(input_data)

    pred = model.predict(input_scaled)

    st.success(f"Predicted Manning's n: {pred[0]:.5f}")
