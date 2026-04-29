import streamlit as st
import numpy as np
import joblib
import os

# Load model + scaler
model = joblib.load(os.path.join("models", "trained_model.pkl"))
scaler = joblib.load(os.path.join("models", "scaler.pkl"))

st.title("Surface Roughness Predictor")

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
