import streamlit as st
import numpy as np
import joblib

# Load the model and preprocessing artifacts
model = joblib.load('optimized_random_forest_model.pkl')  # Trained model
columns = joblib.load(r'PKL_Files/model_columns.pkl')  # Model column names
brand_encoder = joblib.load(r'PKL_Files/brand.pkl')  # Fitted LabelEncoder for brand
variant_name_mapping = joblib.load(r'PKL_Files/variant_name_mapping.pkl')  # Dictionary for variant name mapping
model_mapping = joblib.load(r'PKL_Files/model_mapping.pkl')  # Dictionary for model mapping

# Streamlit Application
st.title("Used Car Price Prediction App")

# User Inputs
owner_no = st.number_input("Number of Previous Owners", min_value=1, max_value=5, step=1)
model_year = st.number_input("Year of Manufacture", min_value=2000, max_value=2023, step=1)
registered_year = st.number_input("Registration Year", min_value=2000, max_value=2023, step=1)
kms_driven = st.number_input("Kilometers Driven", min_value=0)
mileage_kmpl = st.number_input("Mileage (kmpl)", min_value=0.0, step=0.1)
engine_cc = st.number_input("Engine Capacity (cc)", min_value=500, step=50)
fuel_type = st.selectbox("Fuel Type", ['CNG', 'Diesel', 'LPG', 'Petrol'])
transmission = st.selectbox("Transmission Type", ['Automatic', 'Manual'])
city = st.selectbox("City", ['Bangalore', 'Chennai', 'Delhi', 'Hyderabad', 'Jaipur', 'Kolkata'])

# Dropdowns for encoded features
brand = st.selectbox("Car Brand", list(brand_encoder.classes_))  # Use .classes_ for LabelEncoder
variant_name = st.selectbox("Variant Name", list(variant_name_mapping.keys()))  # Use .keys() for dictionary
model_name = st.selectbox("Model Name", list(model_mapping.keys()))  # Use .keys() for dictionary

# Derived Features
car_age = model_year - registered_year
model_age = 2023 - model_year  # Replace 2023 with dynamic year
registration_lag = registered_year - model_year
price_per_km = 0 if kms_driven == 0 else 1 / kms_driven
mileage_normalized = mileage_kmpl / 30  # Normalize using benchmark
kms_per_year = 0 if car_age == 0 else kms_driven / car_age
high_mileage = 1 if mileage_kmpl > 20 else 0
multiple_owners = 1 if owner_no > 1 else 0
brand_popularity = 0.5  # Placeholder; update if necessary

# Prepare input array
input_data = np.zeros(len(columns))

# Map user inputs to model columns
input_data[columns.index('owner_no')] = owner_no
input_data[columns.index('model_year')] = model_year
input_data[columns.index('registered_year')] = registered_year
input_data[columns.index('kms_driven')] = kms_driven
input_data[columns.index('mileage_kmpl')] = mileage_kmpl
input_data[columns.index('engine_cc')] = engine_cc
input_data[columns.index('car_age')] = car_age
input_data[columns.index('model_age')] = model_age
input_data[columns.index('registration_lag')] = registration_lag
input_data[columns.index('price_per_km')] = price_per_km
input_data[columns.index('mileage_normalized')] = mileage_normalized
input_data[columns.index('kms_per_year')] = kms_per_year
input_data[columns.index('high_mileage')] = high_mileage
input_data[columns.index('multiple_owners')] = multiple_owners
input_data[columns.index('brand_popularity')] = brand_popularity

# Encode brand using LabelEncoder
input_data[columns.index('brand_encoded')] = brand_encoder.transform([brand])[0]

# Encode variant and model using dictionary mappings
input_data[columns.index('variant_name_encoded')] = variant_name_mapping[variant_name]
input_data[columns.index('model_encoded')] = model_mapping[model_name]

# One-hot encoding for categorical features
if f'fuel_type_{fuel_type.lower()}' in columns:
    input_data[columns.index(f'fuel_type_{fuel_type.lower()}')] = 1

if f'transmission_{transmission.lower()}' in columns:
    input_data[columns.index(f'transmission_{transmission.lower()}')] = 1

if f'city_{city.lower()}' in columns:
    input_data[columns.index(f'city_{city.lower()}')] = 1

# Prediction
if st.button("Predict Price"):
    prediction = model.predict([input_data])
    st.success(f"Estimated Price: ₹{prediction[0]:,.2f}")
