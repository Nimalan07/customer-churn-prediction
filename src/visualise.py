import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

DATA_PATH = "data/processed/processed_churn.csv"
MODEL_PATH = "models/rf_model.pkl"
FEATURES_PATH = "models/features.pkl"

sns.set(style="whitegrid")
os.makedirs("reports/visuals", exist_ok=True)

def save_fig(name):
    """Utility to save figures cleanly."""
    path = f"reports/visuals/{name}.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print(f"✔ Saved: {path}")

def visualize():
    print("Loading dataframe...")
    df = pd.read_csv(DATA_PATH)
    print(df.shape)
    # ========== 1. CHURN DISTRIBUTION ==========
    plt.figure(figsize=(5,4))
    df['Churn'].value_counts().plot(kind='bar', color=['steelblue','salmon'])
    plt.title("Churn Distribution")
    plt.xticks(rotation=0)
    save_fig("churn_distribution")
    plt.show()
    # ========== 2. TENURE vs CHURN ==========
    plt.figure(figsize=(6,4))
    sns.boxplot(data=df, x='Churn', y='tenure')
    plt.title("Tenure vs Churn")
    save_fig("tenure_vs_churn")
    plt.show()
    # ========== 3. MONTHLY CHARGES vs CHURN ==========
    plt.figure(figsize=(6,4))
    sns.boxplot(data=df, x='Churn', y='MonthlyCharges')
    plt.title("Monthly Charges vs Churn")
    save_fig("monthlycharges_vs_churn")
    plt.show()
    # ========== 4. CONTRACT TYPE vs CHURN ==========
    contract_cols = [c for c in df.columns if "Contract_" in c]
    if contract_cols:
        plt.figure(figsize=(6,4))
        contract_means = [df[col].mean() for col in contract_cols]
        sns.barplot(x=contract_cols, y=contract_means, color='skyblue')
        plt.title("Contract Type Churn Rates")
        plt.ylabel("Percentage")
        plt.xticks(rotation=45)
        save_fig("contract_type_churn")
        plt.show()
    # ========== 5. CORRELATION HEATMAP ==========
    plt.figure(figsize=(10,6))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    save_fig("correlation_heatmap")
    plt.show()
    # ========== 6. FEATURE IMPORTANCE ==========
    try:
        print("Loading model for feature importance...")
        model = joblib.load(MODEL_PATH)
        features = joblib.load(FEATURES_PATH)

        importances = pd.Series(model.feature_importances_, index=features)
        top20 = importances.sort_values(ascending=False).head(20)

        plt.figure(figsize=(7,6))
        top20.plot(kind='barh', color='teal')
        plt.title("Top 20 Feature Importances")
        plt.gca().invert_yaxis()
        save_fig("feature_importance_top20")
        plt.show()
    except:
        print("Model or feature list missing — skipping feature importance chart.")

if __name__ == "__main__":
    visualize()
