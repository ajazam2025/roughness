import streamlit as st
import numpy as np
import pandas as pd
from io import StringIO
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostRegressor

# =================================================
# Page configuration
# =================================================
st.set_page_config(
    page_title="Roughness Predictor",
    layout="centered"
)

# =================================================
# Custom CSS
# =================================================
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(to right, #e3f2fd, #ffffff);
        font-family: 'Segoe UI', sans-serif;
    }

    h1 {
        color: #0d47a1;
        text-align: center;
    }

    h2, h3 {
        color: #1565c0;
    }

    label {
        font-weight: bold;
        color: #263238;
    }

    div.stButton > button {
        background-color: #1976d2;
        color: white;
        border-radius: 12px;
        height: 3em;
        width: 100%;
        font-size: 18px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =================================================
# Title
# =================================================
st.title("🌊 GUI based Roughness Prediction Tool in Steep Channels")

st.markdown("""
### ML-based estimation of Manning’s roughness coefficient (n)  
**Model used:** AdaBoost Regressor
""")

# =================================================
# DATA
# =================================================
DATA_CSV = """Fr,Re,H_D,LD,Slope_S,u_star,Manning_n
0.5,500000,1.6,2.5,0.0005,0.3,0.0035
0.4,400000,1.5,2.0,0.0004,0.25,0.0045
"""

df = pd.read_csv(StringIO(DATA_CSV))
X = df.drop(columns=["Manning_n"])
y = df["Manning_n"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = AdaBoostRegressor()
model.fit(X_scaled, y)

# =================================================
# Input parameters
# =================================================
st.header("🔹 Input Parameters")

col1, col2 = st.columns(2)

with col1:
    Fr = st.number_input("⚡ Froude Number, Fr", min_value=0.01, value=0.5)
    Re = st.number_input("🔁 Reynolds Number, Re", min_value=1000.0, value=500000.0)
    HD = st.number_input("📏 Relative Submergence, H/D", min_value=0.1, value=1.6)

with col2:
    LD = st.number_input("📐 Spacing Ratio, λ/D", min_value=0.1, value=2.5)
    Slope = st.number_input("📉 Channel Slope, S", min_value=0.00001, value=0.0005)
    u_star = st.number_input("🌪 Shear Velocity, u*", min_value=0.01, value=0.3)

# =================================================
# Governing equations
# =================================================
with st.expander("📐 Governing Equations"):
    st.latex(r"Fr = \frac{U}{\sqrt{gH}}")
    st.latex(r"Re = \frac{UR}{\nu}")
    st.latex(r"n = \frac{R^{2/3} S^{1/2}}{U}")

# =================================================
# Prediction
# =================================================
st.markdown("---")

if st.button("🚀 Predict Manning’s Roughness (n)"):

    X_input = np.array([[Fr, Re, HD, LD, Slope, u_star]])
    X_scaled_input = scaler.transform(X_input)

    pred = model.predict(X_scaled_input)[0]

    st.markdown(
        f"""
        <div style="
            background:#e8f5e9;
            padding:20px;
            border-radius:15px;
            text-align:center;
            font-size:22px;
            color:#1b5e20;
            margin-top:15px;
        ">
        🌊 <b>Predicted Manning’s n</b><br>
        <span style="font-size:32px;">{pred:.6f}</span>
        </div>
        """,
        unsafe_allow_html=True
    )

# =================================================
# Footer
# =================================================
st.markdown("---")

st.caption(
    "🔬 Developed by Ajaz Mir, Research Scholar, "
    "Dr B R Ambedkar National Institute of Technology Jalandhar | "
    "AI-based geomorphology tool for hydraulic roughness prediction"
)
