import asyncio
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import plotly.express as px
import os
import requests
import gzip
import shutil

# ===== FIX EVENT LOOP ISSUE =====
def fix_event_loop():
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

fix_event_loop()

# ===== SETUP =====
st.set_page_config(layout="wide")
st.title("NYC Taxi Fare Prediction & Data Insights")

# ===== MODEL LOADING =====
@st.cache_resource
def load_model_artifacts():
    try:
        # Use relative paths
        model_path = os.path.join(os.path.dirname(__file__), 'best_gradient_boosting_model.pkl')
        preprocessor_path = os.path.join(os.path.dirname(__file__), 'preprocessor.pkl')
        
        model = joblib.load(model_path)
        preprocessor = joblib.load(preprocessor_path)
        return model, preprocessor
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

model, preprocessor = load_model_artifacts()

# ===== DATA LOADING =====
@st.cache_data
def load_data():
    local_path = "fixed_nyc_taxi_sample_2M.csv.gz"
    one_drive_url = "https://onedrive.live.com/download?resid=D8331802E3D7620B%21254&authkey=!AFmF9wiQ4d1E1hM"

    if not os.path.exists(local_path):
        st.info("📥 Downloading dataset from OneDrive...")
        try:
            with requests.get(one_drive_url, stream=True) as r:
                with open(local_path, 'wb') as f:
                    shutil.copyfileobj(r.raw, f)
            st.success("✅ Download complete.")
        except Exception as e:
            st.error(f"Download failed: {str(e)}")
            return None

    columns_needed = ['tpep_pickup_datetime', 'tpep_dropoff_datetime', 
                     'trip_distance', 'fare_amount', 'tip_amount']
    try:
        df = pd.read_csv(local_path, compression='gzip', 
                        usecols=columns_needed,
                        parse_dates=['tpep_pickup_datetime', 'tpep_dropoff_datetime'])
        
        # Feature engineering
        df['pickup_hour'] = df['tpep_pickup_datetime'].dt.hour
        df['pickup_day'] = df['tpep_pickup_datetime'].dt.dayofweek
        df['trip_duration'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds() / 60
        
        return df.dropna()
    except Exception as e:
        st.error(f"Data loading failed: {str(e)}")
        return None

df = load_data()
if df is None:
    st.stop()

# ===== SIDEBAR PREDICTION =====
st.sidebar.header("📈 Predict Taxi Fare")

with st.sidebar.form("prediction_form"):
    trip_distance = st.number_input("Trip Distance (miles)", min_value=0.0, value=2.0, step=0.1)
    pickup_hour = st.selectbox("Pickup Hour", list(range(24)), index=12)
    pickup_day = st.selectbox("Pickup Day (0=Monday, 6=Sunday)", list(range(7)), index=0)
    passenger_count = st.number_input("Passenger Count", min_value=1, max_value=6, value=1)
    
    submitted = st.form_submit_button("Predict Fare")
    
    if submitted:
        try:
            input_df = pd.DataFrame({
                'trip_distance': [trip_distance],
                'pickup_hour': [pickup_hour],
                'pickup_day': [pickup_day],
                'passenger_count': [passenger_count]
            })
            
            scaled_input = preprocessor.transform(input_df)
            prediction = model.predict(scaled_input)[0]
            st.sidebar.success(f"💰 Predicted Fare: ${prediction:.2f}")
        except Exception as e:
            st.sidebar.error(f"Prediction failed: {str(e)}")

# ===== MAIN VISUALIZATIONS =====
st.subheader("🔍 Data Insights & Visualizations")

# Data cleaning
clean_df = df[
    (df['trip_distance'] > 0) &
    (df['trip_duration'] > 0) &
    (df['fare_amount'] > 0)
].copy()

# --- Visualization 1 ---
st.markdown("### 1. Trip Distance vs Duration by Hour")
corr = clean_df['trip_distance'].corr(clean_df['trip_duration'])
fig1 = px.scatter(
    clean_df.sample(n=10000, random_state=42),  # Sample for performance
    x='trip_distance',
    y='trip_duration',
    color='pickup_hour',
    trendline="lowess",
    title=f"Duration vs Distance | Hourly Patterns (Corr: {corr:.2f})",
    labels={'trip_distance': 'Distance (miles)', 'trip_duration': 'Duration (mins)'}
)
st.plotly_chart(fig1, use_container_width=True)

# --- Visualization 2 ---
st.markdown("### 2. Average Fare and Tip by Trip Distance")
clean_df['distance_bin'] = pd.cut(clean_df['trip_distance'], 
                                bins=[0, 1, 3, 5, 10, 20], 
                                labels=['0-1', '1-3', '3-5', '5-10', '10-20'])
grouped = clean_df.groupby('distance_bin')[['fare_amount', 'tip_amount']].mean().reset_index()

fig2, ax2 = plt.subplots(figsize=(10, 5))
grouped.plot(x='distance_bin', kind='bar', colormap='Set2', ax=ax2)
ax2.set_title('Average Fare and Tip Amount by Trip Distance')
ax2.set_xlabel('Trip Distance (miles)')
ax2.set_ylabel('Average Amount ($)')
ax2.grid(axis='y', linestyle='--', alpha=0.7)
st.pyplot(fig2)

st.success("✅ App loaded successfully!")
