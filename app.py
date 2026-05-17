import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Title of the Application
st.title("🍫 Chocolate Sales Prediction (ANN)")
st.write("Enter the transaction details below to forecast the expected sales volume (Units).")

# 1. Load and clean data (Backend synchronization)
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

# 2. Re-fit the preprocessor pipeline
X = df[['Sales Person', 'Geography', 'Product', 'cost per unit']]
categorical_cols = ['Sales Person', 'Geography', 'Product']
numerical_cols = ['cost per unit']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])
X_processed = preprocessor.fit_transform(X).toarray()

# 3. Dummy model loading simulation for UI flow
# (In production, you save model weights, here we construct architecture dynamically)
input_dim = X_processed.shape[1]
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(input_dim,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='linear')
])
model.compile(optimizer='adam', loss='mse')

# 4. Creating User Input Interface
st.subheader("Transaction Inputs")

sales_person = st.selectbox("Select Sales Person", df['Sales Person'].unique())
geography = st.selectbox("Select Geography/Country", df['Geography'].unique())
product = st.selectbox("Select Chocolate Product", df['Product'].unique())
cost_per_unit = st.number_input("Cost Per Unit ($)", min_value=1, max_value=50, value=12)

# Prediction execution button
if st.button("Predict Units Sold"):
    # Structure user input into dataframe
    input_data = pd.DataFrame([{
        'Sales Person': sales_person,
        'Geography': geography,
        'Product': product,
        'cost per unit': cost_per_unit
    }])
    
    # Preprocess user input using fitted pipeline
    input_encoded = preprocessor.transform(input_data).toarray()
    
    # Predict using the model
    prediction = model.predict(input_encoded).flatten()[0]
    
    # Ensure prediction is non-negative
    final_units = max(0, int(np.round(prediction)))
    
    st.success(format(f"🔮 Estimated Sales Demand: **{final_units} Units**"))
