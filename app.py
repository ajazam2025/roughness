import streamlit as st
import numpy as np
import pandas as pd
from io import StringIO

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import AdaBoostRegressor

# ------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Roughness Predictor", layout="centered")

# -------------------------------
# CUSTOM CSS (FULL STYLING)
# -------------------------------
st.markdown("""
<style>

/* Background */
body {
    background-color: #f4f7fb;
}

/* Top Banner */
.banner {
    background: linear-gradient(90deg, #1f77b4, #2ca02c);
    padding: 15px;
    border-radius: 10px;
    text-align: center;
    color: white;
    font-size: 18px;
    font-weight: bold;
    margin-bottom: 20px;
}

/* Section Box */
.box {
    background-color: white;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
    margin-bottom: 20px;
}

/* Button */
.stButton>button {
    background: linear-gradient(90deg, #ff7f0e, #d62728);
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 16px;
    font-weight: bold;
}

/* Footer */
.footer {
    background: linear-gradient(90deg, #2c3e50, #34495e);
    padding: 15px;
    border-radius: 10px;
    text-align: center;
    color: white;
    font-size: 14px;
    margin-top: 30px;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------
# TOP BANNER
# -------------------------------
st.markdown('<div class="banner">Developed for AI Applications in Geomorphology</div>', unsafe_allow_html=True)

# -------------------------------
# TITLE
# -------------------------------
st.markdown("<h1 style='text-align:center; color:#2c3e50;'>🌊 Hydraulic Roughness Predictor</h1>", unsafe_allow_html=True)

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
# MODEL SELECTION BOX
# -------------------------------
st.markdown('<div class="box">', unsafe_allow_html=True)

st.subheader("🔧 Select Machine Learning Model")

model_name = st.selectbox(
    "",
    ["AdaBoost (Recommended)", "Linear Regression", "Bayesian Ridge", "KNN"]
)

st.markdown('</div>', unsafe_allow_html=True)

# Initialize model
if model_name == "Linear Regression":
    model = LinearRegression()

elif model_name == "Bayesian Ridge":
    model = BayesianRidge()

elif model_name == "KNN":
    model = KNeighborsRegressor(n_neighbors=5)

else:
    model = AdaBoostRegressor(n_estimators=100, random_state=42)

model.fit(X_scaled, y)

# -------------------------------
# INPUT BOX
# -------------------------------
st.markdown('<div class="box">', unsafe_allow_html=True)

st.subheader("📥 Input Parameters")

col1, col2 = st.columns(2)

with col1:
    Fr = st.number_input("Froude Number", value=0.5)
    Re = st.number_input("Reynolds Number", value=500000.0)
    HD = st.number_input("H/D", value=1.6)

with col2:
    LD = st.number_input("λ/D", value=2.5)
    Slope = st.number_input("Slope", value=0.0005)
    u_star = st.number_input("Shear Velocity (u*)", value=0.3)

st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------
# PREDICTION BOX
# -------------------------------
st.markdown('<div class="box">', unsafe_allow_html=True)

if st.button("🚀 Predict Roughness"):
    input_data = np.array([[Fr, Re, HD, LD, Slope, u_star]])
    input_scaled = scaler.transform(input_data)

    pred = model.predict(input_scaled)

    st.success(f"🎯 Predicted Manning's n = {pred[0]:.6f}")

st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------
# FOOTER (YOUR NAME HERE)
# -------------------------------
st.markdown("""
<div class="footer">
Developed by <b>Ajaz Mir</b><br>
Research Scholar<br>
Dr. B R Ambedkar National Institute of Technology, Jalandhar
</div>
""", unsafe_allow_html=True)
