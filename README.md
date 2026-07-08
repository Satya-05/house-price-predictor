# 🏠 House Price Predictor

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange?logo=scikit-learn)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-red?logo=streamlit)
![Status](https://img.shields.io/badge/Status-Live-brightgreen)

> An end-to-end machine learning web app that predicts house sale prices using an ensemble of **Ridge**, **Lasso**, and **Gradient Boosting** regressors — trained on the Ames Housing dataset with 230 engineered features.

🔗 **Live App:** [house-price-predictor.streamlit.app](https://house-price-predictor-9qp6vnemjcaqpkxo8mjsvg.streamlit.app/)

---

## 📌 Problem Statement

Predicting residential house prices is a classic regression problem with real-world impact — for buyers, sellers, and financial institutions alike. This project builds a production-ready price prediction system by combining regularized linear models with gradient boosting, and deploying the result as an interactive web application.

---

## 🎯 Project Highlights

- 🔢 **230 features** engineered from 79 raw input variables
- 🧹 **Lasso automatically eliminated 134/230 features** (58%) — demonstrating automatic feature selection
- 📉 **Ensemble RMSE of 0.1136** (log scale) — outperforming every individual model
- 🚀 **Deployed live** on Streamlit Cloud with real-time predictions
- 📊 **Three-model comparison** with coefficient analysis and feature importance visualization

---

## 🧠 Methodology

### 1. Exploratory Data Analysis (`notebooks/01_eda.ipynb`)
- Analyzed 1,460 houses across 79 features
- Identified **right-skewed target** (`SalePrice` skewness: 1.88) → applied `log1p` transform (skewness: 0.12)
- Removed 2 extreme outliers (large area, anomalously low price) documented in original dataset
- Top correlations with `SalePrice`: `OverallQual` (0.79), `GrLivArea` (0.71), `GarageCars` (0.64)

### 2. Feature Engineering (`notebooks/02_feature_engineering.ipynb`)

| Technique | Details |
|---|---|
| Missing value imputation | "None"/0 for structural absences, neighborhood median for `LotFrontage`, mode for categoricals |
| Ordinal encoding | 17 quality/condition columns mapped to numeric scales (e.g. Ex→5, Gd→4, TA→3) |
| New features | `TotalSF`, `TotalBath`, `HouseAge`, `YearsSinceRemodel`, `HasGarage`, `HasPool`, etc. |
| One-hot encoding | 27 nominal categorical columns expanded to binary features |
| Skew correction | `log1p` applied to 36 numeric features with skewness > 0.75 |
| **Final feature count** | **230 features** (from 79 raw inputs) |

### 3. Modeling (`notebooks/03_modeling.ipynb`)

All models evaluated using **5-fold cross-validated RMSE** on log-transformed prices.

| Model | Best Hyperparameters | CV RMSE |
|---|---|---|
| Ridge | α = 200 | 0.1196 |
| Lasso | α = 0.003 | 0.1156 |
| Gradient Boosting | lr=0.05, depth=4, n=200 | 0.1233 |
| **Ensemble (avg)** | — | **0.1136 ✅ Best** |

> The ensemble averages predictions from all three models, leveraging their complementary strengths — Ridge handles multicollinearity, Lasso performs feature selection, and GBR captures nonlinear interactions.

---

## 🔍 Key Insights

**All three models agree on the most important features:**

| Feature | Why it matters |
|---|---|
| `OverallQual` | Single strongest predictor (corr: 0.79 with price) |
| `TotalSF` | Engineered feature combining all floor areas — more predictive than individual floor areas |
| `GrLivArea` | Above-ground living area drives perceived value |
| `OverallCond` | Condition independently adds value beyond quality |
| `Neighborhood` | Location effects captured through one-hot encoding |

**Lasso zeroed out 134 of 230 features** — confirming that most of the one-hot encoded dummy variables carry little predictive signal. The model selected just 96 truly meaningful features.

---

## 🌐 Streamlit App

The app takes key house characteristics as input and returns real-time price predictions from all three models plus the ensemble.

**Input features:** Overall quality & condition, living area, basement, garage, bathrooms, bedrooms, fireplaces, kitchen quality, neighborhood, year built/remodeled, central air.

**Output:** Predicted sale price from Ridge, Lasso, GBR, and ensemble — with model performance metrics and feature importance charts.

---

## 📁 Project Structure

house-price-predictor/
│
├── app/
│   ├── app.py                  # Streamlit app
│   └── feature_importance.png  # Model comparison chart
│
├── data/
│   ├── train.csv               # Ames Housing training data
│   ├── test.csv                # Test data
│   └── submission.csv          # Kaggle-format predictions
│
├── models/
│   ├── ridge_model.pkl
│   ├── lasso_model.pkl
│   ├── gbr_model.pkl
│   ├── scaler.pkl
│   ├── feature_columns.pkl
│   ├── feature_medians.pkl
│   └── skewed_cols.pkl
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_modeling.ipynb
│
├── requirements.txt
└── README.md

---

## 🚀 Run Locally

```bash
# Clone the repo
git clone https://github.com/Satya-05/house-price-predictor.git
cd house-price-predictor

# Create virtual environment
python -m venv venv
venv\Scripts\Activate.ps1   # Windows PowerShell

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app/app.py
```

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.11 |
| ML | scikit-learn (Ridge, Lasso, GradientBoostingRegressor) |
| Data | pandas, numpy, scipy |
| Visualization | matplotlib, seaborn |
| App | Streamlit |
| Deployment | Streamlit Cloud |
| Version Control | Git, GitHub |

---

## 📊 Dataset

**Ames Housing Dataset** — compiled by Dean De Cock, used as a modern alternative to the Boston Housing dataset.
- 1,460 training samples, 79 features
- Source: [Kaggle House Prices Competition](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)
