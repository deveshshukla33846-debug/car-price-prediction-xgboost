import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb

# Page configuration
st.set_page_config(
    page_title="Used Car Price Predictor",
    page_icon="🚗",
    layout="centered"
)

st.title("🚗 Used Car Price Predictor App")
st.write("Enter the vehicle details below to estimate the accurate market resale value instantly.")

# 1. Load the trained XGBoost model
@st.cache_resource
def load_model():
    model = xgb.Booster()
    model.load_model("xgb_model.json")
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# 2. User Input UI Components
col1, col2 = st.columns(2)

with col1:
    present_price = st.number_input("Present Showroom Price of Car (in Lakhs)", min_value=0.1, max_value=100.0, value=5.0, step=0.5)
    kms_driven = st.number_input("Distance Driven (in Kilometers)", min_value=0, max_value=500000, value=45000, step=1000)
    fuel_type = st.selectbox("Fuel Type", options=["Petrol", "Diesel", "CNG"])

with col2:
    seller_type = st.selectbox("Seller Type", options=["Dealer", "Individual"])
    transmission = st.selectbox("Transmission Type", options=["Manual", "Automatic"])
    owner = st.selectbox("Owner Type (Previous Owners)", options=[0, 1, 3]) # 0 for First, 1 for Second, 3 for Third
    age = st.number_input("Vehicle Age (in Years)", min_value=0, max_value=30, value=4, step=1)

# 3. Preprocess Inputs to match Training Data
if st.button("🔥 Calculate Estimated Price", type="primary"):
    
    # Map categorical variables to numeric values based on standard encoding
    fuel_mapping = {"Petrol": 0, "Diesel": 1, "CNG": 2}
    seller_mapping = {"Dealer": 0, "Individual": 1}
    transmission_mapping = {"Manual": 0, "Automatic": 1}
    
    # Create input dictionary exactly as expected by your model
    input_data = {
        'Present_Price': [present_price],
        'Kms_Driven': [kms_driven],
        'Fuel_Type': [fuel_mapping[fuel_type]],
        'Seller_Type': [seller_mapping[seller_type]],
        'Transmission': [transmission_mapping[transmission]],
        'Owner': [owner],
        'Age': [age]
    }
    
    # Convert to DataFrame with exact column names and order
    input_df = pd.DataFrame(input_data)
    
    # Convert to XGBoost DMatrix
    dmatrix_input = xgb.DMatrix(input_df)
    
    try:
        # Predict price
        predicted_price = model.predict(dmatrix_input)[0]
        
        # Display the result (Handle negative values if model fluctuates)
        if predicted_price < 0:
            predicted_price = 0
            
        st.success(f"### 🎉 Estimated Resale Value: ₹ {predicted_price:.2f} Lakhs")
    except Exception as prediction_error:
        st.error(f"Prediction Error: {prediction_error}")
