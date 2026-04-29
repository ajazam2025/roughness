import streamlit as st
import numpy as np
import joblib
import os
import pandas as pd
from io import StringIO

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostRegressor

# -------------------------------
# 🔥 EMBEDDED DATASET
# -------------------------------
DATA_CSV = """Fr,Re,H_D,LD,Slope_S,u_star,Manning_n
0.50158836,543478.2609,1.678571429,2,0.0005,0.339510677,0.00368199
0.209453438,217391.3043,1.642857143,2.5,0.0005,0.335879443,0.008880878
0.553773604,652173.913,1.75,2.5,0.0005,0.346659054,0.003289009
0.523633595,543478.2609,1.642857143,2.5,0.0005,0.335879443,0.003552351
0.314180157,326086.9565,1.642857143,2.5,0.0003,0.260171098,0.004586066
0.418906876,434782.6087,1.642857143,2.5,0.0003,0.260171098,0.003439549
0.523633595,543478.2609,1.642857143,2.5,0.0003,0.260171098,0.002751639
0.209453438,217391.3043,1.642857143,2.5,0.0003,0.260171098,0.006879098
0.314180157,326086.9565,1.642857143,2.5,0.00015,0.183968747,0.003242838
0.523633595,543478.2609,1.642857143,2.5,0.00015,0.183968747,0.001945703
0.314180157,326086.9565,1.642857143,2.5,1.00E-05,0.047500526,0.000837297
0.599247532,543478.2609,1.535714286,2,0.0005,0.324742205,0.003174682
0.276886802,326086.9565,1.75,1.5,0.0005,0.346659054,0.006578018
0.461478004,543478.2609,1.75,1.5,0.0005,0.346659054,0.003946811
0.300953016,326086.9565,1.678571429,1.5,0.0005,0.339510677,0.00613665
"""

def load_data():
    return pd.read_csv(StringIO(DATA_CSV))

# -------------------------------
# PATHS
# -------------------------------
MODEL_DIR = "models"
model_path = os.path.join(MODEL_DIR, "trained_model.pkl")
scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")

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
# LOAD MODEL
# -------------------------------
if not os.path.exists(model_path):
    st.warning("Training model from embedded dataset...")
    model, scaler = train_and_save()
else:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

# -------------------------------
# UI
# -------------------------------
st.title("AI-Based Hydraulic Roughness Predictor")

Fr = st.number_input("Froude Number", value=0.5)
Re = st.number_input("Reynolds Number", value=500000.0)
HD = st.number_input("H/D", value=1.6)
LD = st.number_input("λ/D", value=2.5)
Slope = st.number_input("Slope", value=0.0005)
u_star = st.number_input("Shear Velocity", value=0.3)

if st.button("Predict"):
    input_data = np.array([[Fr, Re, HD, LD, Slope, u_star]])
    input_scaled = scaler.transform(input_data)

    pred = model.predict(input_scaled)

    st.success(f"Predicted Manning's n: {pred[0]:.6f}")
