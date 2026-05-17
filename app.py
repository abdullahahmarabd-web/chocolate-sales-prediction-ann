import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Title of the Application
st.title("🍫 Chocolate Sales Prediction Dashboard")
st.write("Enter the transaction details below to forecast the expected sales volume (Units).")

# 1. Load and clean data
@st.cache_data
def load_and_clean_data():
    df = pd.read_csv("choclate protfolio project - 11.csv")
    def clean_currency(val):
        if isinstance(val, str):
            val = val.replace('$', '').replace(',', '').replace('%', '').strip()
            if '(' in val and ')' in val:
                val = '-' + val.replace('(', '').replace(')', '')
            try: return float(val)
            except ValueError: return 0.0
        return val

    for col in ['Amount', 'Cost', 'Profit', 'profit %']:
        df[col] = df[col].apply(clean_currency)
    return df

df = load_and_clean_data()

# 2. Features and Target Formatting (Safe Structure)
X_raw = df[['Sales Person', 'Geography', 'Product', 'cost per unit']]
X_encoded = pd.get_dummies(X_raw, columns=['Sales Person', 'Geography', 'Product'])
feature_columns = X_encoded.columns  

y = df['Units']

# Training on pure values array to avoid Feature Names Warning
model = LinearRegression()
model.fit(X_encoded.values, y.values)

# 3. Creating User Input Interface
st.subheader("Transaction Inputs")

sales_person = st.selectbox("Select Sales Person", df['Sales Person'].unique())
geography = st.selectbox("Select Geography/Country", df['Geography'].unique())
product = st.selectbox("Select Chocolate Product", df['Product'].unique())
cost_per_unit = st.number_input("Cost Per Unit ($)", min_value=1, max_value=50, value=12)

# Prediction execution button
if st.button("Predict Units Sold"):
    # Structure user input into exactly the same structure
    input_df = pd.DataFrame([{
        'Sales Person': sales_person,
        'Geography': geography,
        'Product': product,
        'cost per unit': cost_per_unit
    }])
    
    # Matching user options with dataset matrix
    input_encoded = pd.get_dummies(input_df, columns=['Sales Person', 'Geography', 'Product'])
    input_encoded = input_encoded.reindex(columns=feature_columns, fill_value=0)
    
    # Predict passing absolute numpy arrays (.values)
    prediction = model.predict(input_encoded.values)[0]
    
    # Ensure prediction is non-negative
    final_units = max(0, int(np.round(prediction)))
    
    st.success(f"🔮 Estimated Sales Demand: **{final_units} Units**")
