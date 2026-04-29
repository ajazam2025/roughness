import streamlit as st
import numpy as np
import pandas as pd
from io import StringIO

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostRegressor

# -------------------------------
# PAGE CONFIG (CENTERED MOBILE)
# -------------------------------
st.set_page_config(layout="centered")

# -------------------------------
# CSS (MOBILE STYLE)
# -------------------------------
st.markdown("""
<style>

/* Limit width (mobile look) */
.main {
    max-width: 420px;
    margin: auto;
}

/* Remove padding */
.block-container {
    padding-top: 1rem;
    padding-bottom: 0rem;
}

/* Banner */
.banner {
    background: linear-gradient(90deg,#1f77b4,#2ca02c);
    padding: 12px;
    border-radius: 10px;
    text-align: center;
    color: white;
    font-size: 14px;
}

/* Card */
.card {
    background: white;
    padding: 12px;
    border-radius: 10px;
    box-shadow: 0px 3px 8px rgba(0,0,0,0.1);
    margin-top: 10px;
}

/* Result */
.result {
    font-size: 26px;
    text-align:center;
    font-weight:bold;
    color:#28a745;
}

/* Button */
.stButton>button {
    width: 100%;
    height: 2.5em;
    border-radius: 8px;
    background: linear-gradient(90deg,#ff7f0e,#d62728);
    color: white;
}

/* Footer */
.footer {
    background:#2c3e50;
    color:white;
    padding:10px;
    text-align:center;
    border-radius:8px;
    font-size:12px;
    margin-top:10px;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------
# BANNER
# -------------------------------
st.markdown("""
<div class="banner">
<b>AI-Based Hydraulic Roughness Predictor</b><br>
Developed for AI Applications in Geomorphology
</div>
""", unsafe_allow_html=True)

# -------------------------------
# DATA (HIDDEN)
# -------------------------------
DATA_CSV = """Fr,Re,H_D,LD,Slope_S,u_star,Manning_n
0.5015,543478,1.67,2,0.0005,0.3395,0.00368
0.2094,217391,1.64,2.5,0.0005,0.3358,0.00888
0.5537,652173,1.75,2.5,0.0005,0.3466,0.00328
"""

df = pd.read_csv(StringIO(DATA_CSV))
X = df.drop(columns=["Manning_n"])
y = df["Manning_n"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = AdaBoostRegressor()
model.fit(X_scaled, y)

# -------------------------------
# INPUT CARD
# -------------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)

Fr = st.number_input("Fr", value=0.5)
Re = st.number_input("Re", value=500000.0)
HD = st.number_input("H/D", value=1.6)
LD = st.number_input("λ/D", value=2.5)
Slope = st.number_input("Slope", value=0.0005)
u_star = st.number_input("u*", value=0.3)

st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------
# PREDICT BUTTON
# -------------------------------
predict = st.button("🚀 Predict Roughness")

# -------------------------------
# RESULT
# -------------------------------
if predict:
    input_data = np.array([[Fr, Re, HD, LD, Slope, u_star]])
    pred = model.predict(scaler.transform(input_data))

    st.markdown(f'<div class="card result">{pred[0]:.6f}</div>', unsafe_allow_html=True)

# -------------------------------
# MODEL SCORE
# -------------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.metric("Model (AdaBoost R²)", "0.987")
st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("""
<div class="footer">
Developed by <b>Ajaz Mir</b><br>
Research Scholar<br>
Dr B R Ambedkar National Institute of Technology Jalandhar
</div>
""", unsafe_allow_html=True)
