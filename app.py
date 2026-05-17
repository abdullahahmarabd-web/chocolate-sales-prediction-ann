import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Title of the Application
st.title("🍫 Chocolate Sales Prediction Dashboard")
st.write("Enter the transaction details below to forecast the expected sales volume (Units).")

# 1. Load and clean data safely
@st.cache_data
def load_and_clean_data():
    df = pd.read_csv("choclate protfolio project - 11.csv")
    return df

df = load_and_clean_data()

# 2. Extract Options for Dropdowns
sales_persons = sorted(df['Sales Person'].unique().tolist())
geographies = sorted(df['Geography'].unique().tolist())
products = sorted(df['Product'].unique().tolist())

# 3. Bulletproof Manual Mapping (No pd.get_dummies matrix tracking needed!)
df_encoded = df.copy()
df_encoded['Sales Person'] = df_encoded['Sales Person'].map({name: i for i, name in enumerate(sales_persons)})
df_encoded['Geography'] = df_encoded['Geography'].map({geo: i for i, geo in enumerate(geographies)})
df_encoded['Product'] = df_encoded['Product'].map({prod: i for i, prod in enumerate(products)})

# Features and Target selection
X = df_encoded[['Sales Person', 'Geography', 'Product', 'cost per unit']]
y = df_encoded['Units']

# Train the model smoothly with standard columns
model = LinearRegression()
model.fit(X, y)

# 4. Creating User Input Interface
st.subheader("Transaction Inputs")

sales_person_input = st.selectbox("Select Sales Person", sales_persons)
geography_input = st.selectbox("Select Geography/Country", geographies)
product_input = st.selectbox("Select Chocolate Product", products)
cost_per_unit = st.number_input("Cost Per Unit ($)", min_value=1, max_value=50, value=12)

# Prediction execution button
if st.button("Predict Units Sold"):
    # Convert user selection into the exact same numeric map values
    sp_mapped = sales_persons.index(sales_person_input)
    geo_mapped = geographies.index(geography_input)
    prod_mapped = products.index(product_input)
    
    # Structure input as a simple matching dataframe to completely satisfy feature names
    user_data = pd.DataFrame([{
        'Sales Person': sp_mapped,
        'Geography': geo_mapped,
        'Product': prod_mapped,
        'cost per unit': cost_per_unit
    }])
    
    # Predict directly using matching names framework
    prediction = model.predict(user_data)[0]
    
    # Ensure prediction value limits are positive integers
    final_units = max(0, int(np.round(prediction)))
    
    st.success(f"🔮 Estimated Sales Demand: **{final_units} Units**")
