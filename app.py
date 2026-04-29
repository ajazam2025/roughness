import streamlit as st
import numpy as np
import joblib
import os
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostRegressor

# -------------------------------
# PATH SETUP
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_PATH = os.path.join(BASE_DIR, "data", "ml_ready_dataset.xlsx")

model_path = os.path.join(MODEL_DIR, "trained_model.pkl")
scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")

# -------------------------------
# AUTO TRAIN IF MODEL MISSING
# -------------------------------
def train_and_save():
    df = pd.read_excel(DATA_PATH)

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
# LOAD MODEL
# -------------------------------
if not os.path.exists(model_path) or not os.path.exists(scaler_path):
    st.warning("Model not found. Training automatically...")
    model, scaler = train_and_save()
else:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

# -------------------------------
# STREAMLIT UI
# -------------------------------
st.title("AI-Based Hydraulic Roughness Predictor")

st.write("Predict Manning’s roughness coefficient (n)")

# Inputs
Fr = st.number_input("Froude Number (Fr)", value=0.5)
Re = st.number_input("Reynolds Number (Re)", value=10000.0)
HD = st.number_input("H/D (Relative Submergence)", value=1.0)
LD = st.number_input("λ/D (Spacing Ratio)", value=2.0)
Slope = st.number_input("Slope (S)", value=0.01)
u_star = st.number_input("Shear Velocity (u*)", value=0.05)

# Prediction
if st.button("Predict"):
    input_data = np.array([[Fr, Re, HD, LD, Slope, u_star]])
    input_scaled = scaler.transform(input_data)

    pred = model.predict(input_scaled)

    st.success(f"Predicted Manning's n: {pred[0]:.5f}")
