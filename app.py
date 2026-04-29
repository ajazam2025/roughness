import streamlit as st
import numpy as np
import joblib
import os
import pandas as pd
import base64
from io import BytesIO

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostRegressor

# -------------------------------
# EMBEDDED DATASET (YOUR FILE)
# -------------------------------
DATA_BASE64 = """
UEsDBBQAAAAIAF1rnVxGx01IlQAAAM0AAAAQAAAAZG9jUHJvcHMvYXBwLnhtbE3PTQvCMAwG4L9SdreZih6kDkQ9ip68zy51hbYpbYT67+0EP255ecgboi6JIia2mEXxLuRtMzLHDUDWI/o+y8qhiqHke64x3YGMsRoPpB8eA8OibdeAhTEMOMzit7Dp1C5GZ3XPlkJ3sjpRJsPiWDQ6sScfq9wcChDneiU+ixNLOZcrBf+LU8sVU57mym/8ZAW/B7oXUEsDBBQAAAAIAF1rnVyc7ERk8AAAACsCAAARAAAAZG9jUHJvcHMvY29yZS54bWzNksFOwzAMhl8F5d66TaGCqOtlEyeQkJgE4hYl3hataaLEqN3b04atE4IH4Bj7z+fPkhvlhXIBX4LzGMhgvBlt10eh/IodiLwAiOqAVsZ8SvRTc+eClTQ9wx68VEe5R+BFUYNFklqShBmY+YXI2kYroQJKcuGM12rB+8/QJZhWgB1a7ClCmZfA2nmiP41dA1fADCMMNn4XUC/EVP0TmzrAzskxmiU1DEM+VCk37VDC+/PTa1o3M30k2SucfkUj6ORxxS6T36r1ZvvIWl7wOituM/6wLSvBa3F3/zG7/vC7Clunzc78Y+OLYNvAr7tovwBQSwMEFAAAAAgAXWudXJlcnCMQBgAAnCcAABMAAAB4bC90aGVtZS90aGVtZTEueG1s7Vpbc9o4FH7vr9B4Z/ZtC8Y2gba0E3Npdtu0mYTtTh+FEViNbHlkkYR/v0c2EMuWDe2STbqbPAQs6fvORUfn6Dh58+4uYuiGiJTyeGDZL9vWu7cv3uBXMiQRQTAZp6/wwAqlTF61WmkAwzh9yRMSw9yCiwhLeBTL1lzgWxovI9bqtNvdVoRpbKEYR2RgfV4saEDQVFFab18gtOUfM/gVy1SNZaMBE1dBJrmItPL5bMX82t4+Zc/pOh0ygW4wG1ggf85vp+ROWojhVMLEwGpnP1Zrx9HSSICCyX2UBbpJ9qPTFQgyDTs6nVjOdnz2xO2fjMradDRtGuDj8Xg4tsvSi3AcBOBRu57CnfRsv6RBCbSjadBk2PbarpGmqo1TT9P3fd/rm2icCo1bT9Nrd93TjonGrdB4Db7xT4fDronGq9B4...
"""

# -------------------------------
# LOAD DATA
# -------------------------------
def load_data():
    decoded = base64.b64decode(DATA_BASE64)
    return pd.read_excel(BytesIO(decoded))

# -------------------------------
# PATH SETUP
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

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
# LOAD OR TRAIN
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
