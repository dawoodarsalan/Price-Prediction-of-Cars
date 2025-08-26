import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained model and related data
with open("best_car_price_model.pkl", "rb") as f:
    data = pickle.load(f)

model = data["model"]
label_encoders = data["label_encoders"]
feature_columns = data["feature_columns"]

# App title
st.set_page_config(page_title="Car Price Predictor", layout="centered")
st.title("🚗 Car Selling Price Predictor")
st.markdown("Enter the car details below to predict its selling price (in lakhs).")

# Input fields
present_price = st.number_input("💰 Present Price (in lakhs)", min_value=0.0, step=0.1)
kms_driven = st.number_input("🛣️ Kilometers Driven", min_value=0)
owner = st.selectbox("👤 Number of Previous Owners", [0, 1, 2, 3])
fuel_type = st.selectbox("⛽ Fuel Type", label_encoders["Fuel_Type"].classes_)
seller_type = st.selectbox("🏢 Seller Type", label_encoders["Seller_Type"].classes_)
transmission = st.selectbox("⚙️ Transmission", label_encoders["Transmission"].classes_)
car_age = st.slider("📅 Car Age (Years)", min_value=0, max_value=25, value=5)

# Encode categorical values
fuel_type_encoded = label_encoders["Fuel_Type"].transform([fuel_type])[0]
seller_type_encoded = label_encoders["Seller_Type"].transform([seller_type])[0]
transmission_encoded = label_encoders["Transmission"].transform([transmission])[0]

# Prepare input dataframe
input_data = pd.DataFrame([[
    present_price, kms_driven, owner,
    fuel_type_encoded, seller_type_encoded,
    transmission_encoded, car_age
]], columns=feature_columns)

# Check for valid input and make prediction
if st.button("🔍 Predict Selling Price"):
    # Convert all columns to numeric in case of any parsing issues
    input_data = input_data.apply(pd.to_numeric, errors='coerce')

    # Debug line: Show input data (optional)
    # st.write("Input Data for Model:", input_data)

    if input_data.isnull().any().any():
        st.error("❌ Invalid input detected. Please fill all fields with valid values.")
    else:
        try:
            prediction = model.predict(input_data)[0]
            st.success(f"✅ Estimated Selling Price: ₹ {prediction:.2f} lakhs")
        except ValueError as ve:
            st.error(f"❌ Prediction error: {ve}")
