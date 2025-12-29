import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)

DATA_PATH = "data/processed/processed_churn.csv"
MODEL_PATH = "models/rf_model.pkl"
SCALER_PATH = "models/scaler.pkl"
FEATURES_PATH = "models/features.pkl"

OUTPUT_DIR = "reports/evaluation"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def evaluate():
    print("Loading artifacts...")

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    features = joblib.load(FEATURES_PATH)

    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    X = X.reindex(columns=features, fill_value=0)

    scaler_features = scaler.feature_names_in_
    X[scaler_features] = scaler.transform(X[scaler_features])

    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    print("\nEvaluation Metrics")
    print("------------------")
    print("Accuracy :", accuracy_score(y, y_pred))
    print("Precision:", precision_score(y, y_pred))
    print("Recall   :", recall_score(y, y_pred))
    print("F1 Score :", f1_score(y, y_pred))
    print("ROC AUC  :", roc_auc_score(y, y_proba))

    cm = confusion_matrix(y, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(f"{OUTPUT_DIR}/confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.show()

    fpr, tpr, _ = roc_curve(y, y_proba)
    plt.figure(figsize=(6,4))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y, y_proba):.3f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig(f"{OUTPUT_DIR}/roc_curve.png", dpi=300, bbox_inches="tight")
    plt.show()

    print("\nEvaluation completed successfully.")
    print(f"Plots saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    evaluate()
