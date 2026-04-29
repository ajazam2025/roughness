import streamlit as st
import numpy as np
import pandas as pd
from io import StringIO

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import AdaBoostRegressor

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Roughness Predictor", layout="centered")

# -------------------------------
# CUSTOM STYLE
# -------------------------------
st.markdown("""
<style>
.main {
    background-color: #f5f7fa;
}
h1 {
    color: #2c3e50;
    text-align: center;
}
h3 {
    color: #34495e;
}
.stButton>button {
    background-color: #2c7be5;
    color: white;
    border-radius: 8px;
    height: 3em;
    width: 100%;
    font-size: 16px;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# HEADER
# -------------------------------
st.title("🌊 AI-Based Roughness Predictor")
st.markdown("### Developed by **Ajaz Mir**")

st.markdown("---")

# -------------------------------
# DATA (EMBEDDED)
# -------------------------------
DATA_CSV = """Fr,Re,H_D,LD,Slope_S,u_star,Manning_n
0.50158836,543478.2609,1.678571429,2,0.0005,0.339510677,0.00368199
0.209453438,217391.3043,1.642857143,2.5,0.0005,0.335879443,0.008880878
0.553773604,652173.913,1.75,2.5,0.0005,0.346659054,0.003289009
0.523633595,543478.2609,1.642857143,2.5,0.0005,0.335879443,0.003552351
0.314180157,326086.9565,1.642857143,2.5,0.0003,0.260171098,0.004586066
0.418906876,434782.6087,1.642857143,2.5,0.0003,0.260171098,0.003439549
"""

df = pd.read_csv(StringIO(DATA_CSV))

X = df.drop(columns=["Manning_n"])
y = df["Manning_n"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------
# MODEL SELECTION
# -------------------------------
st.markdown("### 🔧 Select Model")

model_name = st.selectbox(
    "Choose Machine Learning Model",
    ["AdaBoost (Best)", "Linear Regression", "Bayesian Ridge", "KNN"]
)

# Initialize model
if model_name == "Linear Regression":
    model = LinearRegression()

elif model_name == "Bayesian Ridge":
    model = BayesianRidge()

elif model_name == "KNN":
    model = KNeighborsRegressor(n_neighbors=5)

else:
    model = AdaBoostRegressor(n_estimators=100, random_state=42)

# Train model
model.fit(X_scaled, y)

# -------------------------------
# INPUT SECTION
# -------------------------------
st.markdown("### 📥 Input Parameters")

col1, col2 = st.columns(2)

with col1:
    Fr = st.number_input("Froude Number", value=0.5)
    Re = st.number_input("Reynolds Number", value=500000.0)
    HD = st.number_input("H/D", value=1.6)

with col2:
    LD = st.number_input("λ/D", value=2.5)
    Slope = st.number_input("Slope", value=0.0005)
    u_star = st.number_input("Shear Velocity (u*)", value=0.3)

# -------------------------------
# PREDICTION
# -------------------------------
st.markdown("---")

if st.button("🚀 Predict Roughness"):
    input_data = np.array([[Fr, Re, HD, LD, Slope, u_star]])
    input_scaled = scaler.transform(input_data)

    pred = model.predict(input_scaled)

    st.success(f"🎯 Predicted Manning's n = {pred[0]:.6f}")

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("---")
st.markdown(
    "<center><small>Developed for AI Applications in Geomorphology</small></center>",
    unsafe_allow_html=True
)
