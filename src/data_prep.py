import pandas as pd
import numpy as np
import os

RAW_PATH = "data/raw/telco_churn.csv"
PROCESSED_PATH = "data/processed/processed_churn.csv"

def preprocess():
    print("Starting preprocessing...")
    df = pd.read_csv(RAW_PATH)
    print(f"Loaded raw data: {df.shape}")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)
    df["avg_charge"] = df["TotalCharges"] / (df["tenure"] + 1)
    bins = [0, 12, 24, 48, 72]
    labels = ["0-12", "12-24", "24-48", "48-72"]
    df["tenure_group"] = pd.cut(df["tenure"], bins=bins, labels=labels, right=False)
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    categorical_cols.remove("customerID")  
    categorical_cols.remove("Churn")       
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    df.drop(columns=["customerID"], inplace=True)
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)
    print(f"Processed data saved to {PROCESSED_PATH}")
    print("Preprocessing completed successfully!")

if __name__ == "__main__":
    preprocess()
