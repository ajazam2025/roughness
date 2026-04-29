import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model
model = joblib.load("models/trained_model.pkl")

st.title("Surface Roughness Predictor (AI-based)")
st.write("Predict Manning’s roughness coefficient using ML")

# Input fields
Fr = st.number_input("Froude Number", value=0.5)
Re = st.number_input("Reynolds Number", value=10000.0)
HD = st.number_input("H/D (Relative Submergence)", value=1.0)
LD = st.number_input("λ/D (Spacing Ratio)", value=2.0)
Slope = st.number_input("Slope", value=0.01)
u_star = st.number_input("Shear Velocity (u*)", value=0.05)

# Prediction
if st.button("Predict Roughness"):
    input_data = np.array([[Fr, Re, HD, LD, Slope, u_star]])
    pred = model.predict(input_data)

    st.success(f"Predicted Manning's n: {pred[0]:.5f}")
