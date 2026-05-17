import streamlit as st
import numpy as np
import pandas as pd
import os

# Title of the Application
st.title("🍫 Chocolate Sales Prediction Dashboard")
st.write("Enter the transaction details below to forecast the expected sales volume (Units).")

FILE_NAME = "choclate protfolio project - 11.csv"

# Check if file exists before running the app logic
if not os.path.exists(FILE_NAME):
    st.error(f"❌ File '{FILE_NAME}' GitHub par nahi mili! Meharbani karke check karein ke aapne exact is naam ki CSV file upload ki hui hai.")
else:
    # 1. Load data safely
    df = pd.read_csv(FILE_NAME)

    # 2. Extract Options for Dropdowns
    sales_persons = sorted(df['Sales Person'].unique().tolist())
    geographies = sorted(df['Geography'].unique().tolist())
    products = sorted(df['Product'].unique().tolist())

    # 3. Pure Mathematical Regression
    df_encoded = df.copy()
    df_encoded['SP_idx'] = df_encoded['Sales Person'].map({name: i for i, name in enumerate(sales_persons)})
    df_encoded['Geo_idx'] = df_encoded['Geography'].map({geo: i for i, geo in enumerate(geographies)})
    df_encoded['Prod_idx'] = df_encoded['Product'].map({prod: i for i, prod in enumerate(products)})

    X = df_encoded[['SP_idx', 'Geo_idx', 'Prod_idx', 'cost per unit']].values
    X_bias = np.c_[np.ones(X.shape[0]), X]
    y = df_encoded['Units'].values

    try:
        beta = np.linalg.inv(X_bias.T @ X_bias) @ X_bias.T @ y
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(X_bias) @ y

    # 4. Creating User Input Interface
    st.subheader("Transaction Inputs")

    sales_person_input = st.selectbox("Select Sales Person", sales_persons)
    geography_input = st.selectbox("Select Geography/Country", geographies)
    product_input = st.selectbox("Select Chocolate Product", products)
    cost_per_unit = st.number_input("Cost Per Unit ($)", min_value=1, max_value=50, value=12)

    # Prediction execution button
    if st.button("Predict Units Sold"):
        sp_mapped = sales_persons.index(sales_person_input)
        geo_mapped = geographies.index(geography_input)
        prod_mapped = products.index(product_input)
        
        user_features = np.array([1, sp_mapped, geo_mapped, prod_mapped, cost_per_unit])
        prediction = np.dot(user_features, beta)
        final_units = max(0, int(np.round(prediction)))
        
        st.success(f"🔮 Estimated Sales Demand: **{final_units} Units**")
