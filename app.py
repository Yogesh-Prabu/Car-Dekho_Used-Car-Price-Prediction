import streamlit as st
import numpy as np
import joblib
import base64

# Load the model and preprocessing artifacts
model = joblib.load('optimized_random_forest_model.pkl')
columns = joblib.load(r'PKL_Files/model_columns.pkl')
brand_encoder = joblib.load(r'PKL_Files/brand.pkl')
variant_name_mapping = joblib.load(r'PKL_Files/variant_name_mapping.pkl')
model_mapping = joblib.load(r'PKL_Files/model_mapping.pkl')

# Function to load the background image
def get_base64_image(image_file):
    with open(image_file, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Set the background image
bg_image_base64 = get_base64_image(r"ASSETS/car_image.jpg")
page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] {{
    background-image: url("data:image/jpg;base64,{bg_image_base64}");
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: fixed;
    color: white;
}}
h1 {{
    font-size: 50px !important;
    font-weight: bold;
    color: #FFD700 !important;
    text-shadow: 3px 3px 8px black;
    text-align: center;
}}
label {{
    font-size: 16px !important;
    font-weight: bold;
    color: white !important;
    text-shadow: 1px 1px 2px black;
}}
.stButton>button {{
    background-color: #4CAF50 !important;
    color: white !important;
    border-radius: 8px !important;
    font-size: 18px !important;
    padding: 8px 16px !important;
    border: none !important;
    box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.2);
}}
.stButton>button:hover {{
    background-color: #45a049 !important;
    color: white !important;
    border: 2px solid #45a049 !important;
}}
.stButton>button:active {{
    background-color: #3e8e41 !important;
    color: white !important;
}}
.stSelectbox, .stNumberInput {{
    background-color: rgba(0, 0, 0, 0.6) !important;
    color: white !important;
    border-radius: 5px !important;
    padding: 8px !important;
    border: 1px solid rgba(255, 255, 255, 0.4);
    width: 100%;
}}
.stSuccess {{
    font-size: 20px !important;
    font-weight: bold !important;
    color: #4CAF50 !important;
    background-color: rgba(255, 255, 255, 0.8) !important;
    padding: 10px !important;
    border-radius: 8px !important;
    text-align: center;
    box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.2);
}}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# Title
st.markdown("<h1>USED CAR PRICE VALUATION</h1>", unsafe_allow_html=True)

# User Inputs
col1, col2 = st.columns(2)

# Left column inputs
with col1:
    city = st.selectbox("City", ['Bangalore', 'Chennai', 'Delhi', 'Hyderabad', 'Jaipur', 'Kolkata'])
    brand = st.selectbox("Car Brand", list(brand_encoder.classes_))
    model_name = st.selectbox("Model Name", list(model_mapping.keys()))
    variant_name = st.selectbox("Variant Name", list(variant_name_mapping.keys()))
    fuel_type = st.selectbox("Fuel Type", ['CNG', 'Diesel', 'LPG', 'Petrol'])
    mileage_kmpl = st.selectbox("Mileage (kmpl)", [round(x, 1) for x in np.linspace(0, 30, 61)])

# Right column inputs
with col2:
    owner_no = st.selectbox("Number of Previous Owners", list(range(1, 6)))
    model_year = st.selectbox("Year of Manufacture", list(range(2000, 2024)))
    registered_year = st.selectbox("Registration Year", list(range(2000, 2024)))
    kms_driven = st.selectbox("Kilometers Driven", list(range(0, 300001, 5000)))
    engine_cc = st.selectbox("Engine Capacity (cc)", list(range(500, 5001, 100)))
    transmission = st.selectbox("Transmission Type", ['Automatic', 'Manual'])

# Align the button to the right side
st.markdown("<div style='text-align: right; padding-right: 20px;'>", unsafe_allow_html=True)
if st.button("Predict Price"):
    # Derived Features
    car_age = model_year - registered_year
    model_age = 2023 - model_year
    registration_lag = registered_year - model_year
    price_per_km = 0 if kms_driven == 0 else 1 / kms_driven
    mileage_normalized = mileage_kmpl / 30
    kms_per_year = 0 if car_age == 0 else kms_driven / car_age
    high_mileage = 1 if mileage_kmpl > 20 else 0
    multiple_owners = 1 if owner_no > 1 else 0
    brand_popularity = 0.5

    # Prepare input array
    input_data = np.zeros(len(columns))
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

    # Encoding categorical features
    input_data[columns.index('brand_encoded')] = brand_encoder.transform([brand])[0]
    input_data[columns.index('variant_name_encoded')] = variant_name_mapping[variant_name]
    input_data[columns.index('model_encoded')] = model_mapping[model_name]
    if f'fuel_type_{fuel_type.lower()}' in columns:
        input_data[columns.index(f'fuel_type_{fuel_type.lower()}')] = 1
    if f'transmission_{transmission.lower()}' in columns:
        input_data[columns.index(f'transmission_{transmission.lower()}')] = 1
    if f'city_{city.lower()}' in columns:
        input_data[columns.index(f'city_{city.lower()}')] = 1

    # Prediction
    prediction = model.predict([input_data])
    st.markdown(f"<div class='stSuccess'>🚘 Estimated Price: ₹{prediction[0]:,.2f}</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
