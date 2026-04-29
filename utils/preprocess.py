import joblib
import numpy as np

scaler = joblib.load("models/scaler.pkl")

def preprocess_input(data):
    return scaler.transform(data)
