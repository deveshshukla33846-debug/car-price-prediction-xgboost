import streamlit as st
import xgboost as xgb
import numpy as np

# 1. Load the pre-trained XGBoost Model
@st.cache_resource
def load_my_model():
    model = xgb.Booster()
    model.load_model("xgb_model.json")
    return model

model = load_my_model()

# 2. Web Application Page Configuration & Title
st.set_page_config(page_title="Car Price Predictor", page_icon="🚗")
st.title("🚗 Used Car Price Predictor App")
st.write("Enter the vehicle details below to estimate the accurate market resale value instantly.")

st.divider()

# 3. User Input Fields (Dropdowns and Sliders)
col1, col2 = st.columns(2)

with col1:
    car_age = st.number_input("Vehicle Age (in Years)", min_value=0, max_value=20, value=4)
    fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])

with col2:
    mileage = st.slider("Distance Driven (in Kilometers)", min_value=0, max_value=200000, value=45000, step=1000)
    owner_type = st.selectbox("Owner Type", ["First Owner", "Second Owner", "Third Owner"])

st.divider()

# 4. Predict Button and Model Inference
if st.button("🔥 Calculate Estimated Price", use_container_width=True):
    # Encoding categorical inputs to match the model features
    fuel_encoded = 0 if fuel_type == "Petrol" else (1 if fuel_type == "Diesel" else 2)
    owner_encoded = 1 if owner_type == "First Owner" else (2 if owner_type == "Second Owner" else 3)
    
    # Preparing data format for XGBoost
    input_data = np.array([[car_age, mileage, fuel_encoded, owner_encoded]], dtype=float)
    dmatrix = xgb.DMatrix(input_data)
    
    # Making prediction
    prediction = model.predict(dmatrix)[0]
    
    # Displaying results with animation
    st.balloons()
    st.success(f"### 🎉 The estimated resale value of this car is: *₹ {round(prediction, 2)} Lakh*")
