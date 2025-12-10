import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

DATA_PATH = "data/processed/processed_churn.csv"

def train_model():
    print("Loading processed dataset...")
    df = pd.read_csv(DATA_PATH)
    print(f"Dataset shape: {df.shape}")
    X = df.drop("Churn", axis=1)
    y = df["Churn"]
    feature_list = X.columns.tolist()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=200, max_depth=10,random_state=42)
    print("Training model...")
    model.fit(X_train, y_train)
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/rf_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(feature_list, "models/features.pkl")
    print("Model, scaler, and feature list saved successfully!")

if __name__ == "__main__":
    train_model()
