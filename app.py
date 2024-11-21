import streamlit as st
import numpy as np
import joblib
import base64

# Load the model and encoders
model = joblib.load('optimized_random_forest_model.pkl')
brand_encoder = joblib.load('PKL_Files/brand.pkl')
variant_name_mapping = joblib.load('PKL_Files/variant_name_mapping.pkl')
model_mapping = joblib.load('PKL_Files/model_mapping.pkl')

# Define all expected one-hot encoded categories
categories = {
    'city': ['bangalore', 'chennai', 'delhi', 'hyderabad', 'jaipur', 'kolkata'],
    'transmission': ['automatic', 'manual'],
    'fuel_type': ['cng', 'diesel', 'lpg', 'petrol'],
    'kms_bins': ['High', 'Low', 'Moderate', 'Unused', 'Very_Low']
}

# Helper function to derive Kilometers Driven Bin from kms_driven
def get_kms_bin(kms_driven):
    if kms_driven == 0:
        return 'Unused'
    elif kms_driven <= 5000:
        return 'Very_Low'
    elif kms_driven <= 50000:
        return 'Low'
    elif kms_driven <= 100000:
        return 'Moderate'
    else:
        return 'High'

# Load the background and logo images
def get_base64_image(image_file):
    with open(image_file, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

bg_image_base64 = get_base64_image("ASSETS/car_image.jpg")
logo_image_base64 = get_base64_image("ASSETS/logo.png")

page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] {{
    background-image: url("data:image/jpg;base64,{bg_image_base64}");
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: fixed;
    color: white;
}}

[data-testid="stHeader"] {{
    background: rgba(0, 0, 0, 0);
}}

h1 {{
    font-size: 45px !important;
    font-weight: bold;
    text-align: center;
    color: #FFD700;
    text-shadow: 3px 3px 8px rgba(0, 0, 0, 0.8);
    margin-top: 20px;
}}

h2 {{
    font-size: 22px;
    font-weight: bold;
    color: white;
    text-shadow: 2px 2px 6px black;
    text-align: center;
    margin-bottom: 20px;
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

.stSelectbox, .stNumberInput {{
    background-color: rgba(0, 0, 0, 0.6) !important;
    color: white !important;
    border-radius: 5px !important;
    padding: 8px !important;
    border: 1px solid rgba(255, 255, 255, 0.4);
    width: 100%;
}}

.logo {{
    position: fixed;
    top: 10px;
    left: 10px;
    height: 60px;
    z-index: 100;
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

[data-testid="stVerticalBlock"] {{
    margin-left: auto;
    margin-right: auto;
    width: 95%;
    max-width: 800px;
}}
</style>

<img class="logo" src="data:image/png;base64,{logo_image_base64}" />
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# Title
st.markdown("<h1>USED CAR PRICE VALUATION</h1>", unsafe_allow_html=True)
st.markdown(
    """
    <h2 style="font-size: 22px; font-weight: bold; color: white; text-shadow: 2px 2px 6px black; text-align: center; margin-bottom: 20px;">
    Provide the following details:
    </h2>
    """,
    unsafe_allow_html=True,
)

# User Inputs
col1, col2 = st.columns(2, gap="large")

# Left column inputs
with col1:
    city = st.selectbox("City", categories['city'])
    brand = st.selectbox("Car Brand", list(brand_encoder.classes_))
    model_name = st.selectbox("Model Name", list(model_mapping.keys()))
    variant_name = st.selectbox("Variant Name", list(variant_name_mapping.keys()))
    mileage_kmpl = st.number_input("Mileage (kmpl)", min_value=0.0, step=0.1, format="%.1f")
    engine_cc = st.number_input("Engine Capacity (cc)", min_value=500, max_value=5000, step=100)

# Right column inputs
with col2:
    fuel_type = st.selectbox("Fuel Type", categories['fuel_type'])
    transmission = st.selectbox("Transmission Type", categories['transmission'])
    owner_no = st.number_input("Number of Previous Owners", min_value=1, max_value=5, step=1)
    kms_driven = st.number_input("Kilometers Driven", min_value=0, step=1000)
    model_year = st.number_input("Year of Manufacture", min_value=2000, max_value=2024, step=1)
    registered_year = st.number_input("Registration Year", min_value=2000, max_value=2024, step=1)

# Derive Kilometers Driven Bin
kms_bin = get_kms_bin(kms_driven)

# Encode categorical features
city_encoded = [1 if city == cat else 0 for cat in categories['city']]
transmission_encoded = [1 if transmission == cat else 0 for cat in categories['transmission']]
fuel_type_encoded = [1 if fuel_type == cat else 0 for cat in categories['fuel_type']]
kms_bin_encoded = [1 if kms_bin == cat else 0 for cat in categories['kms_bins']]
brand_encoded_val = brand_encoder.transform([brand])[0]
variant_name_encoded = variant_name_mapping[variant_name]
model_encoded = model_mapping[model_name]

# Compute derived features
car_age = registered_year - model_year
model_age = 2023 - model_year
registration_lag = registered_year - model_year
normalized_model_age = car_age / (model_age + 1) if model_age > 0 else 0
mileage_normalized = mileage_kmpl / 30
high_mileage = 1 if kms_driven > 150000 else 0
multiple_owners = 1 if owner_no > 1 else 0
brand_popularity = 0.5  # Placeholder value
kms_per_year = kms_driven / (car_age + 1) if car_age > 0 else 0

# Combine all features in the correct order
input_features = [
    owner_no, model_year, registered_year, kms_driven, mileage_kmpl, engine_cc,
    car_age, model_age, registration_lag, normalized_model_age, mileage_normalized,
    high_mileage, multiple_owners, brand_popularity, kms_per_year
] + city_encoded + transmission_encoded + fuel_type_encoded + kms_bin_encoded + [
    brand_encoded_val, variant_name_encoded, model_encoded
]

# Predict and display the result
if st.button("Predict Price"):
    prediction = model.predict([input_features])
    st.markdown(f"<div class='stSuccess'>🚘 Estimated Price: ₹{prediction[0]:,.2f}</div>", unsafe_allow_html=True)
