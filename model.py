import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostRegressor

# Load data
df = pd.read_excel("data/ml_ready_dataset.xlsx")

X = df.drop(columns=["Manning_n"])
y = df["Manning_n"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)

# Train model
model = AdaBoostRegressor(n_estimators=100, random_state=42)
model.fit(X_train_s, y_train)

# Save model + scaler
joblib.dump(model, "models/trained_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("Model saved successfully!")
