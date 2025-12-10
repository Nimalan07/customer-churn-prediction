# Customer Churn Prediction (ML + Power BI)

An end-to-end Customer Churn Prediction project using Python (ML) and Power BI.
The goal is to identify customers likely to churn and provide business insights for decision-making.

# Project Highlights

Full ML pipeline: preprocessing → modeling → evaluation

Random Forest model with saved artifacts (model.pkl, scaler.pkl, features.pkl)

Customer risk segmentation (Low / Medium / High)

Power BI dashboard with KPIs and visuals

7000+ processed customer records with engineered features

# Folder Structure
| Folder / File                    |
| -------------------------------- |
| `data/`                          |
| `data/raw/`                      |
| `data/processed/`                |
| `src/`                           |
| `src/data_prep.py`               |
| `src/train_model.py`             |
| `src/evaluate.py`                |
| `models/`                        |
| `models/rf_model.pkl`            |
| `models/scaler.pkl`              |
| `models/features.pkl`            |
| `reports/`                       |
| `reports/visuals/`               |
| `dashboard/`                     |
| `dashboard/churn_dashboard.pbix` |
| `README.md`                      |

     

# ML Workflow

. Data cleaning & feature engineering

. One-hot encoding

. Scaling numeric features

. Random Forest training

### Classification metrics (AUC, F1, recall, precision)

# Power BI Dashboard

Includes:

*Churn Rate KPI

*Total Customers & Churned

*Tenure insights

*Charges distribution

* Filters (Contract, Gender, Internet, Tenure Group)

# Tech Stack

Python, Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn, Joblib, Power BI

# Outcome

A complete business-focused churn prediction solution combining ML + Analytics, ideal for resumes and portfolios.
