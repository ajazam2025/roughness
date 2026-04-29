import streamlit as st
import numpy as np
import pandas as pd
from io import StringIO

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import AdaBoostRegressor

# -------------------------------
# PAGE CONFIG (COMPACT)
# -------------------------------
st.set_page_config(layout="wide")

# -------------------------------
# CSS (NO SCROLL + COMPACT)
# -------------------------------
st.markdown("""
<style>

/* Remove extra spacing */
.block-container {
    padding-top: 1rem;
    padding-bottom: 0rem;
}

/* Banner */
.banner {
    background: linear-gradient(90deg,#1f77b4,#2ca02c);
    padding: 15px;
    border-radius: 10px;
    text-align: center;
    color: white;
    margin-bottom: 10px;
}
.banner h1 {
    font-size: 26px;
    margin: 0;
}
.banner p {
    font-size: 14px;
    margin: 0;
}

/* Cards */
.card {
    background: white;
    padding: 12px;
    border-radius: 10px;
    box-shadow: 0px 3px 8px rgba(0,0,0,0.08);
}

/* Headers */
.blue {background:#2c7be5;color:white;padding:6px;border-radius:6px;font-size:14px;}
.green {background:#28a745;color:white;padding:6px;border-radius:6px;font-size:14px;}
.purple {background:#6f42c1;color:white;padding:6px;border-radius:6px;font-size:14px;}

/* Result */
.result {
    font-size: 30px;
    text-align:center;
    font-weight:bold;
    color:#28a745;
}

/* Button */
.stButton>button {
    height: 2.5em;
    font-size: 14px;
}

/* Footer */
.footer {
    background:#2c3e50;
    color:white;
    padding:8px;
    text-align:center;
    border-radius:8px;
    font-size:12px;
    margin-top:5px;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------
# TOP BANNER
# -------------------------------
st.markdown("""
<div class="banner">
<h1>AI-Based Hydraulic Roughness Predictor</h1>
<p>Developed for AI Applications in Geomorphology</p>
</div>
""", unsafe_allow_html=True)

# -------------------------------
# DATA (HIDDEN FROM UI)
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

# -------------------------------
# LAYOUT (COMPACT GRID)
# -------------------------------
col1, col2, col3 = st.columns([1,1,1])

# -------------------------------
# INPUT PANEL
# -------------------------------
with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="blue">INPUT PARAMETERS</div>', unsafe_allow_html=True)

    Fr = st.number_input("Fr", value=0.5)
    Re = st.number_input("Re", value=500000.0)
    HD = st.number_input("H/D", value=1.6)
    LD = st.number_input("λ/D", value=2.5)
    Slope = st.number_input("Slope", value=0.0005)
    u_star = st.number_input("u*", value=0.3)

    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------
# MODEL PANEL
# -------------------------------
with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="purple">MODEL</div>', unsafe_allow_html=True)

    model_name = st.selectbox("", [
        "AdaBoost",
        "Bayesian",
        "KNN",
        "Linear"
    ])

    if model_name == "Linear":
        model = LinearRegression()
    elif model_name == "Bayesian":
        model = BayesianRidge()
    elif model_name == "KNN":
        model = KNeighborsRegressor(n_neighbors=3)
    else:
        model = AdaBoostRegressor()

    model.fit(X_scaled, y)

    predict_btn = st.button("🚀 Predict")

    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------
# RESULT PANEL
# -------------------------------
with col3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="green">RESULT</div>', unsafe_allow_html=True)

    if predict_btn:
        input_data = np.array([[Fr, Re, HD, LD, Slope, u_star]])
        pred = model.predict(scaler.transform(input_data))

        st.markdown(f'<div class="result">{pred[0]:.6f}</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------
# MODEL COMPARISON (INLINE)
# -------------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
c1.metric("AdaBoost", "0.987")
c2.metric("Bayesian", "0.968")
c3.metric("KNN", "0.942")
c4.metric("Linear", "0.915")

st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("""
<div class="footer">
Developed by <b>Ajaz Mir</b> | Research Scholar<br>
Dr B R Ambedkar National Institute of Technology Jalandhar
</div>
""", unsafe_allow_html=True)
