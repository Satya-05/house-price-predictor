import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="wide")

st.title("🏠 House Price Predictor")
st.markdown("Predict house sale prices using an ensemble of **Ridge, Lasso, and Gradient Boosting** models trained on the Ames Housing dataset.")

# ---------- Load models and artifacts ----------
@st.cache_resource
def load_artifacts():
    ridge = joblib.load('models/ridge_model.pkl')
    lasso = joblib.load('models/lasso_model.pkl')
    gbr = joblib.load('models/gbr_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    feature_columns = joblib.load('models/feature_columns.pkl')
    feature_medians = joblib.load('models/feature_medians.pkl')
    skewed_cols = joblib.load('models/skewed_cols.pkl')
    return ridge, lasso, gbr, scaler, feature_columns, feature_medians, skewed_cols

ridge, lasso, gbr, scaler, feature_columns, feature_medians, skewed_cols = load_artifacts()

qual_map = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1}

NEIGHBORHOODS = sorted([
    'Blmngtn', 'Blueste', 'BrDale', 'BrkSide', 'ClearCr', 'CollgCr', 'Crawfor',
    'Edwards', 'Gilbert', 'IDOTRR', 'MeadowV', 'Mitchel', 'NAmes', 'NoRidge',
    'NPkVill', 'NridgHt', 'NWAmes', 'OldTown', 'SWISU', 'Sawyer', 'SawyerW',
    'Somerst', 'StoneBr', 'Timber', 'Veenker'
])

# ---------- Sidebar inputs ----------
st.sidebar.header("Enter House Details")

overall_qual = st.sidebar.slider("Overall Quality (1=Poor, 10=Excellent)", 1, 10, 6)
overall_cond = st.sidebar.slider("Overall Condition (1=Poor, 10=Excellent)", 1, 10, 5)
gr_liv_area = st.sidebar.number_input("Above Ground Living Area (sq ft)", 300, 6000, 1500)
total_bsmt_sf = st.sidebar.number_input("Total Basement Area (sq ft)", 0, 3000, 800)
first_flr_sf = st.sidebar.number_input("1st Floor Area (sq ft)", 300, 4000, 1000)
second_flr_sf = st.sidebar.number_input("2nd Floor Area (sq ft)", 0, 2000, 0)
lot_area = st.sidebar.number_input("Lot Area (sq ft)", 1000, 50000, 9000)
year_built = st.sidebar.number_input("Year Built", 1870, 2026, 2000)
year_remod = st.sidebar.number_input("Year Remodeled", 1870, 2026, 2005)
year_sold = st.sidebar.number_input("Year Sold", 2006, 2026, 2024)
garage_cars = st.sidebar.slider("Garage Capacity (cars)", 0, 4, 2)
garage_area = st.sidebar.number_input("Garage Area (sq ft)", 0, 1500, 400)
full_bath = st.sidebar.slider("Full Bathrooms", 0, 4, 2)
half_bath = st.sidebar.slider("Half Bathrooms", 0, 2, 0)
bedroom = st.sidebar.slider("Bedrooms Above Ground", 0, 8, 3)
fireplaces = st.sidebar.slider("Fireplaces", 0, 4, 0)
kitchen_qual = st.sidebar.selectbox("Kitchen Quality", ['Ex', 'Gd', 'TA', 'Fa', 'Po'], index=2)
neighborhood = st.sidebar.selectbox("Neighborhood", NEIGHBORHOODS, index=NEIGHBORHOODS.index('NAmes'))
central_air = st.sidebar.selectbox("Central Air", ['Y', 'N'])

predict_btn = st.sidebar.button("Predict Price", type="primary")

# ---------- Build feature row ----------
def set_feature(row, col, raw_value):
    """Assigns a raw value to a column, applying log1p if that column was log-transformed during training."""
    if col in skewed_cols:
        row[col] = np.log1p(max(raw_value, 0))
    else:
        row[col] = raw_value

def build_row():
    row = feature_medians.copy()

    set_feature(row, 'OverallQual', overall_qual)
    set_feature(row, 'OverallCond', overall_cond)
    set_feature(row, 'GrLivArea', gr_liv_area)
    set_feature(row, 'TotalBsmtSF', total_bsmt_sf)
    set_feature(row, '1stFlrSF', first_flr_sf)
    set_feature(row, '2ndFlrSF', second_flr_sf)
    set_feature(row, 'LotArea', lot_area)
    set_feature(row, 'YearBuilt', year_built)
    set_feature(row, 'YearRemodAdd', year_remod)
    set_feature(row, 'YrSold', year_sold)
    set_feature(row, 'GarageCars', garage_cars)
    set_feature(row, 'GarageArea', garage_area)
    set_feature(row, 'FullBath', full_bath)
    set_feature(row, 'HalfBath', half_bath)
    set_feature(row, 'BedroomAbvGr', bedroom)
    set_feature(row, 'Fireplaces', fireplaces)
    set_feature(row, 'KitchenQual', qual_map[kitchen_qual])

    # Derived features
    total_sf_raw = total_bsmt_sf + first_flr_sf + second_flr_sf
    set_feature(row, 'TotalSF', total_sf_raw)
    set_feature(row, 'TotalBath', full_bath + 0.5 * half_bath)
    set_feature(row, 'HouseAge', year_sold - year_built)
    set_feature(row, 'YearsSinceRemodel', year_sold - year_remod)
    set_feature(row, 'WasRemodeled', int(year_built != year_remod))
    set_feature(row, 'HasGarage', int(garage_area > 0))
    set_feature(row, 'HasFireplace', int(fireplaces > 0))
    set_feature(row, 'Has2ndFloor', int(second_flr_sf > 0))
    set_feature(row, 'HasBasement', int(total_bsmt_sf > 0))

    # One-hot: reset all Neighborhood/CentralAir dummies, then set the chosen one
    for col in feature_columns:
        if col.startswith('Neighborhood_') or col.startswith('CentralAir_'):
            row[col] = 0

    neigh_col = f'Neighborhood_{neighborhood}'
    if neigh_col in feature_columns:
        row[neigh_col] = 1

    air_col = f'CentralAir_{central_air}'
    if air_col in feature_columns:
        row[air_col] = 1

    return pd.DataFrame([row])[feature_columns]

# ---------- Predict ----------
if predict_btn:
    input_df = build_row()
    scaled = scaler.transform(input_df)

    ridge_pred = ridge.predict(scaled)[0]
    lasso_pred = lasso.predict(scaled)[0]
    gbr_pred = gbr.predict(scaled)[0]

    ensemble_pred = (ridge_pred + lasso_pred + gbr_pred) / 3
    final_price = np.expm1(ensemble_pred)

    st.subheader("💰 Predicted Sale Price")
    st.metric(label="Ensemble Estimate", value=f"${final_price:,.0f}")

    col1, col2, col3 = st.columns(3)
    col1.metric("Ridge", f"${np.expm1(ridge_pred):,.0f}")
    col2.metric("Lasso", f"${np.expm1(lasso_pred):,.0f}")
    col3.metric("Gradient Boosting", f"${np.expm1(gbr_pred):,.0f}")
else:
    st.info("Adjust the inputs in the sidebar and click **Predict Price** to get an estimate.")

st.markdown("---")
st.subheader("📊 Model Performance (Cross-Validated RMSE, log scale)")
perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
perf_col1.metric("Ridge", "0.1196")
perf_col2.metric("Lasso", "0.1156")
perf_col3.metric("Gradient Boosting", "0.1233")
perf_col4.metric("Ensemble", "0.1136", delta="Best", delta_color="normal")

st.subheader("🔍 Top Feature Importance")
st.image('app/feature_importance.png', caption="Top 10 features per model")