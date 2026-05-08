# 🚀 Customer Churn Prediction System (ML + Power BI)

An end-to-end Machine Learning and Business Intelligence project that predicts customer churn using classification models and visualizes business insights through an interactive Power BI dashboard.

The system helps businesses identify customers likely to leave and supports data-driven retention strategies.

---

# 🧠 Project Overview

Customer churn is one of the biggest business challenges in subscription-based industries.

This project combines:
- Machine Learning
- Data Analytics
- Business Intelligence

to predict customer churn and analyze customer behavior patterns.

The project includes:
- data preprocessing
- feature engineering
- ML model training
- churn risk prediction
- Power BI dashboard visualization

---

# 🔥 Features

✅ End-to-end ML pipeline  
✅ Customer churn prediction using Random Forest  
✅ Feature engineering & preprocessing  
✅ One-hot encoding & scaling  
✅ Model evaluation metrics  
✅ Customer risk segmentation  
✅ Saved ML artifacts (`.pkl` files)  
✅ Interactive Power BI dashboard  
✅ Business KPI analysis  
✅ Modular project architecture

---

# 🏗️ Project Workflow

```text
Raw Customer Data
        ↓
Data Cleaning
        ↓
Feature Engineering
        ↓
Encoding & Scaling
        ↓
Model Training
        ↓
Churn Prediction
        ↓
Risk Segmentation
        ↓
Power BI Dashboard
```

---

# 📂 Project Structure

```text
customer-churn-prediction/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── src/
│   ├── data_prep.py
│   ├── train_model.py
│   └── evaluate.py
│
├── models/
│   ├── rf_model.pkl
│   ├── scaler.pkl
│   └── features.pkl
│
├── dashboard/
│   └── churn_dashboard.pbix
│
├── reports/
│   └── visuals/
│
├── README.md
└── requirements.txt
```

---

# ⚙️ Tech Stack

| Area | Technology |
|---|---|
| Programming Language | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-Learn |
| Visualization | Matplotlib, Seaborn |
| BI Dashboard | Power BI |
| Model Serialization | Joblib |

---

# 📊 Dataset

The dataset contains customer-related information such as:
- tenure
- contract type
- internet service
- monthly charges
- total charges
- payment methods
- churn status

Processed dataset contains:
- 7000+ customer records
- engineered features
- encoded variables
- scaled numerical features

---

# 🚀 Installation

## 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/customer-churn-prediction.git
```

```bash
cd customer-churn-prediction
```

---

## 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 3️⃣ Run Data Preparation

```bash
python src/data_prep.py
```

---

## 4️⃣ Train Model

```bash
python src/train_model.py
```

---

## 5️⃣ Evaluate Model

```bash
python src/evaluate.py
```

---

# 🧠 Machine Learning Pipeline

## Data Cleaning

Handles:
- missing values
- invalid entries
- datatype corrections

---

## Feature Engineering

Creates meaningful features for better prediction performance.

Examples:
- tenure groups
- contract categories
- service combinations

---

## Encoding & Scaling

Includes:
- one-hot encoding
- numerical scaling
- feature transformation

---

## Model Training

Random Forest Classifier used for churn prediction.

The trained model is saved as:

```text
models/rf_model.pkl
```

---

## Model Evaluation

Evaluation metrics include:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

---

# 📈 Power BI Dashboard

The Power BI dashboard provides interactive business insights.

### Dashboard Includes

✅ Churn Rate KPI  
✅ Total Customers  
✅ Churned Customers  
✅ Tenure Analysis  
✅ Monthly Charges Distribution  
✅ Contract-Based Churn Analysis  
✅ Internet Service Insights  
✅ Customer Segmentation Filters

---

# 📤 Example Prediction Output

```json
{
    "customer_id": "7590-VHVEG",
    "churn_prediction": "Yes",
    "risk_level": "High",
    "probability": 0.87
}
```

---

# 🎯 Business Impact

This system helps businesses:
- identify high-risk customers
- improve customer retention
- reduce churn rates
- optimize marketing strategies
- improve customer lifetime value

---

# 📊 Key Insights

- Customers with month-to-month contracts show higher churn probability
- Higher monthly charges correlate with increased churn risk
- Long-tenure customers are less likely to churn
- Fiber optic internet users show higher churn trends

---

# 🚀 Future Improvements

- XGBoost & LightGBM models
- Hyperparameter tuning
- Real-time prediction API
- Streamlit web app
- Customer recommendation engine
- Cloud deployment
- Automated retraining pipeline

---

# 🧠 Business Intelligence Integration

Power BI dashboard enables:
- executive-level KPI monitoring
- customer segmentation analysis
- churn trend visualization
- interactive filtering & drilldowns

---

# 👨‍💻 Author

Nimalan Mani M

---

# ⭐ If you found this project useful

Give this repository a star ⭐
