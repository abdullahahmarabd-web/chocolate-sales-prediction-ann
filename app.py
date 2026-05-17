import streamlit as st
import numpy as np
import pandas as pd

# Title of the Application
st.title("🍫 Chocolate Sales Prediction Dashboard")
st.write("Enter the transaction details below to forecast the expected sales volume (Units).")

# 1. Load data safely
df = pd.read_csv("choclate protfolio project - 11.csv")

# 2. Extract Options for Dropdowns
sales_persons = sorted(df['Sales Person'].unique().tolist())
geographies = sorted(df['Geography'].unique().tolist())
products = sorted(df['Product'].unique().tolist())

# 3. Pure Mathematical Regression (No scikit-learn dependency!)
# Mapping strings to numerical values for math processing
df_encoded = df.copy()
df_encoded['SP_idx'] = df_encoded['Sales Person'].map({name: i for i, name in enumerate(sales_persons)})
df_encoded['Geo_idx'] = df_encoded['Geography'].map({geo: i for i, geo in enumerate(geographies)})
df_encoded['Prod_idx'] = df_encoded['Product'].map({prod: i for i, prod in enumerate(products)})

# Extract feature matrix and target vector using numpy
X = df_encoded[['SP_idx', 'Geo_idx', 'Prod_idx', 'cost per unit']].values
# Adding a bias column (column of 1s) for the intercept
X_bias = np.c_[np.ones(X.shape[0]), X]
y = df_encoded['Units'].values

# Calculate weights using Normal Equation: Beta = (X^T * X)^(-1) * X^T * y
try:
    beta = np.linalg.inv(X_bias.T @ X_bias) @ X_bias.T @ y
except np.linalg.LinAlgError:
    # Fallback to pseudo-inverse if matrix is singular
    beta = np.linalg.pinv(X_bias) @ y

# 4. Creating User Input Interface
st.subheader("Transaction Inputs")

sales_person_input = st.selectbox("Select Sales Person", sales_persons)
geography_input = st.selectbox("Select Geography/Country", geographies)
product_input = st.selectbox("Select Chocolate Product", products)
cost_per_unit = st.number_input("Cost Per Unit ($)", min_value=1, max_value=50, value=12)

# Prediction execution button
if st.button("Predict Units Sold"):
    # Convert user selection into numerical index map
    sp_mapped = sales_persons.index(sales_person_input)
    geo_mapped = geographies.index(geography_input)
    prod_mapped = products.index(product_input)
    
    # Structure input vector with the bias term (1)
    user_features = np.array([1, sp_mapped, geo_mapped, prod_mapped, cost_per_unit])
    
    # Dot product calculation for prediction
    prediction = np.dot(user_features, beta)
    
    # Ensure prediction value limits are positive integers
    final_units = max(0, int(np.round(prediction)))
    
    st.success(f"🔮 Estimated Sales Demand: **{final_units} Units**")
