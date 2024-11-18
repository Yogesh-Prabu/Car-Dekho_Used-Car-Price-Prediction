import streamlit as st
import pandas as pd
import joblib

# Load the trained Gradient Boosting model and necessary encoders
model = joblib.load('PKL_Files/tuned_gradient_boosting.pkl')
brand_encoder = joblib.load('PKL_Files/brand.pkl')
variant_mapping = joblib.load('PKL_Files/variant_name_mapping.pkl')
model_mapping = joblib.load('PKL_Files/model_mapping.pkl')
scaler = joblib.load('PKL_Files/scaler.pkl')  # Updated scaler (without 'price')

# Define a function for feature engineering
def feature_engineering(inputs):
    inputs['car_age'] = max(2024 - inputs['model_year'], 0)  # Ensure car_age is non-negative
    inputs['price_per_km'] = inputs['kms_driven'] / 1000 if inputs['kms_driven'] > 0 else 0
    inputs['high_mileage'] = 1 if inputs['kms_driven'] > 150000 else 0
    inputs['multiple_owners'] = 1 if inputs['owner_no'] > 1 else 0
    inputs['mileage_normalized'] = inputs['mileage_kmpl'] / 100
    brand_popularity_mapping = {0: 0.8, 1: 0.7, 2: 0.6}  # Replace with real mappings
    inputs['brand_popularity'] = brand_popularity_mapping.get(inputs['brand'], 0.5)
    # Adjust car_age influence
    inputs['car_age'] = inputs['car_age'] * 0.5  # Reduce its impact dynamically
    return inputs

# Define a function to scale the input features
def apply_scaling(input_df):
    columns_to_scale = ['kms_driven', 'engine_cc', 'mileage_kmpl', 
                        'car_age', 'mileage_normalized', 'brand_popularity', 'price_per_km']
    input_df[columns_to_scale] = scaler.transform(input_df[columns_to_scale])
    return input_df

# Streamlit app
def main():
    st.title("Used Car Price Prediction App")
    st.write("Enter the car details below to predict the price.")

    # Collect user inputs
    brand = st.selectbox("Brand", list(brand_encoder.classes_))
    variant_name = st.selectbox("Variant", list(variant_mapping.keys()))
    model_name = st.selectbox("Model", list(model_mapping.keys()))
    owner_no = st.slider("Number of Owners", 1, 10, 1)
    model_year = st.slider("Model Year", 1990,2024,2020)
    kms_driven = st.number_input("Kilometers Driven", min_value=0, value=50000)
    engine_cc = st.number_input("Engine CC", min_value=500, value=1000)
    mileage_kmpl = st.number_input("Mileage (kmpl)", min_value=5.0, value=15.0)

    city = st.selectbox("City", ['Chennai', 'Delhi', 'Hyderabad', 'Jaipur', 'Kolkata'])
    transmission = st.selectbox("Transmission", ['Manual', 'Automatic'])
    fuel_type = st.selectbox("Fuel Type", ['Diesel', 'Electric', 'LPG', 'Petrol'])

    # Predict button
    if st.button("Predict Price"):
        brand_encoded = brand_encoder.transform([brand])[0]
        variant_encoded = variant_mapping[variant_name]
        model_encoded = model_mapping[model_name]

        # Prepare input data
        input_data = {
            'brand': brand_encoded,
            'variant_name_encoded': variant_encoded,
            'model_encoded': model_encoded,
            'owner_no': owner_no,
            'model_year': model_year,
            'kms_driven': kms_driven,
            'engine_cc': engine_cc,
            'mileage_kmpl': mileage_kmpl,
            'city_chennai': int(city == 'Chennai'),
            'city_delhi': int(city == 'Delhi'),
            'city_hyderabad': int(city == 'Hyderabad'),
            'city_jaipur': int(city == 'Jaipur'),
            'city_kolkata': int(city == 'Kolkata'),
            'transmission_manual': int(transmission == 'Manual'),
            'fuel_type_diesel': int(fuel_type == 'Diesel'),
            'fuel_type_electric': int(fuel_type == 'Electric'),
            'fuel_type_lpg': int(fuel_type == 'LPG'),
            'fuel_type_petrol': int(fuel_type == 'Petrol')
        }

        # Apply feature engineering and scaling
        input_data = feature_engineering(input_data)
        input_df = pd.DataFrame([input_data])
        input_df = input_df[model.feature_names_in_]  # Reorder to match model features
        input_df = apply_scaling(input_df)

        # Predict price
        predicted_price = model.predict(input_df)[0]
        st.success(f"The predicted price of the car is ₹{predicted_price:.2f}")

if __name__ == '__main__':
    main()
