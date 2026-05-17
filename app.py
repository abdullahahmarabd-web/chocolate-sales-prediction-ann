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

# 2. Extract Options for Dropdowns BEFORE any formatting
sales_persons = sorted(df['Sales Person'].unique().tolist())
geographies = sorted(df['Geography'].unique().tolist())
products = sorted(df['Product'].unique().tolist())

# 3. Direct Matrix One-Hot Encoding to ensure 100% data compatibility
X_raw = df[['Sales Person', 'Geography', 'Product', 'cost per unit']]
X_encoded = pd.get_dummies(X_raw, columns=['Sales Person', 'Geography', 'Product'])

# Saving exact numpy columns structure and array representations
feature_columns = list(X_encoded.columns)
X_matrix = X_encoded.values
y_matrix = df['Units'].values

# Fit training model instantly on absolute arrays
model = LinearRegression()
model.fit(X_matrix, y_matrix)

# 4. Creating User Input Interface
st.subheader("Transaction Inputs")

sales_person = st.selectbox("Select Sales Person", sales_persons)
geography = st.selectbox("Select Geography/Country", geographies)
product = st.selectbox("Select Chocolate Product", products)
cost_per_unit = st.number_input("Cost Per Unit ($)", min_value=1, max_value=50, value=12)

# Prediction execution button
if st.button("Predict Units Sold"):
    # Rebuilding single matrix layout manually via clean logic framework
    user_row = pd.DataFrame([{
        'Sales Person': sales_person,
        'Geography': geography,
        'Product': product,
        'cost per unit': cost_per_unit
    }])
    
    # Process user input data framing matching trained matrix
    user_encoded = pd.get_dummies(user_row, columns=['Sales Person', 'Geography', 'Product'])
    user_encoded = user_encoded.reindex(columns=feature_columns, fill_value=0)
    
    # FORCING absolute numpy conversion to fix data dimension errors once and for all
    user_input_array = np.array(user_encoded.values, dtype=np.float64)
    
    # Predict passing safe multi-dimensional array bounds
    prediction = model.predict(user_input_array)[0]
    
    # Ensure prediction value limits are positive integers
    final_units = max(0, int(np.round(prediction)))
    
    st.success(f"🔮 Estimated Sales Demand: **{final_units} Units**")
