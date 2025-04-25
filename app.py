import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import requests
import shutil

# Set page config
st.set_page_config(layout="wide")
st.title("NYC Taxi Fare Prediction")

# Center content using custom CSS
st.markdown("""
    <style>
        .main {
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            flex-direction: column;
        }
        .block-container {
            padding-top: 2rem;
        }
    </style>
""", unsafe_allow_html=True)

# === Load model and preprocessor ===
model_path = os.path.join(os.path.dirname(__file__), 'best_gradient_boosting_model.pkl')
preprocessor_path = os.path.join(os.path.dirname(__file__), 'preprocessor.pkl')

# ✅ FIXED: Load correct model and preprocessor using defined paths
model = joblib.load(model_path)
preprocessor = joblib.load(preprocessor_path)

# === Load dataset ===
@st.cache_data
def load_data():
    local_path = "fixed_nyc_taxi_sample_2M.csv.gz"
    one_drive_url = "https://onedrive.live.com/download?resid=D8331802E3D7620B%21254&authkey=!AFmF9wiQ4d1E1hM"

    if not os.path.exists(local_path):
        st.info("📥 Downloading dataset from OneDrive...")
        with requests.get(one_drive_url, stream=True) as r:
            with open(local_path, 'wb') as f:
                shutil.copyfileobj(r.raw, f)
        st.success("✅ Download complete.")

    columns_needed = ['tpep_pickup_datetime', 'tpep_dropoff_datetime', 'trip_distance', 'fare_amount', 'tip_amount']
    df = pd.read_csv(local_path, compression='gzip', usecols=columns_needed, parse_dates=['tpep_pickup_datetime', 'tpep_dropoff_datetime'])

    df['pickup_hour'] = df['tpep_pickup_datetime'].dt.hour
    df['pickup_day'] = df['tpep_pickup_datetime'].dt.dayofweek
    df['trip_duration'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds() / 60
    return df.dropna()

df = load_data()

# === Sidebar: Prediction Section ===
st.sidebar.header("📈 Predict Taxi Fare")

trip_distance = st.sidebar.number_input("Trip Distance (miles)", min_value=0.0, value=2.0, step=0.1)
pickup_hour = st.sidebar.selectbox("Pickup Hour", list(range(24)), index=0)
pickup_day = st.sidebar.selectbox("Pickup Day (0=Monday, 6=Sunday)", list(range(7)), index=0)
passenger_count = st.sidebar.number_input("Passenger Count", min_value=1, max_value=6, value=1)

if st.sidebar.button("Predict Fare"):
    input_df = pd.DataFrame([[trip_distance, pickup_hour, pickup_day, passenger_count]],
                             columns=['trip_distance', 'pickup_hour', 'pickup_day', 'passenger_count'])
    
    # ✅ Make sure the input matches what the preprocessor expects
    try:
        scaled_input = preprocessor.transform(input_df)
        prediction = model.predict(scaled_input)[0]
        st.sidebar.success(f"💰 Predicted Fare: ${prediction:.2f}")
    except Exception as e:
        st.sidebar.error(f"⚠️ Prediction failed: {e}")

# Center the main content
st.markdown("""
    <div class="main">
        <h3>📊 Predict Your NYC Taxi Fare</h3>
        <p>Fill out the options in the sidebar to predict your taxi fare based on trip details.</p>
    </div>
""", unsafe_allow_html=True)
