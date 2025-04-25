# ===== EVENT LOOP FIX (MUST BE AT VERY TOP) =====
import asyncio
import nest_asyncio
nest_asyncio.apply()  # Patch the event loop
asyncio.set_event_loop(asyncio.new_event_loop())

# ===== IMPORTS =====
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

# ===== STREAMLIT CONFIG =====
st.set_page_config(layout="wide")
st.set_option('server.enableCORS', False)  # Disable CORS for health checks
st.title("NYC Taxi Fare Prediction & Data Insights")

# ===== MODEL LOADING (WITH CACHING) =====
@st.cache_resource
def load_models():
    try:
        model = joblib.load('best_gradient_boosting_model.pkl')
        preprocessor = joblib.load('preprocessor.pkl')
        return model, preprocessor
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        st.stop()

model, preprocessor = load_models()

# ===== DATA LOADING =====
@st.cache_data
def load_dataset():
    local_path = "fixed_nyc_taxi_sample_2M.csv.gz"
    if not os.path.exists(local_path):
        with st.spinner("Downloading dataset..."):
            one_drive_url = "https://onedrive.live.com/download?resid=D8331802E3D7620B%21254&authkey=!AFmF9wiQ4d1E1hM"
            with requests.get(one_drive_url, stream=True) as r:
                with open(local_path, 'wb') as f:
                    shutil.copyfileobj(r.raw, f)
    
    cols = ['tpep_pickup_datetime', 'tpep_dropoff_datetime', 
            'trip_distance', 'fare_amount', 'tip_amount']
    df = pd.read_csv(local_path, compression='gzip', usecols=cols,
                    parse_dates=['tpep_pickup_datetime', 'tpep_dropoff_datetime'])
    
    # Feature engineering
    df['pickup_hour'] = df['tpep_pickup_datetime'].dt.hour
    df['pickup_day'] = df['tpep_pickup_datetime'].dt.dayofweek
    df['trip_duration'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds() / 60
    
    return df.dropna()

df = load_dataset()

# ===== SIDEBAR PREDICTION =====
st.sidebar.header("📈 Predict Taxi Fare")
with st.sidebar.form("prediction_form"):
    trip_distance = st.number_input("Trip Distance (miles)", 0.1, 30.0, 2.0, 0.1)
    pickup_hour = st.selectbox("Pickup Hour", range(24), index=12)
    pickup_day = st.selectbox("Pickup Day (0=Monday)", range(7), index=0)
    passenger_count = st.number_input("Passenger Count", 1, 6, 1)
    
    if st.form_submit_button("Predict"):
        input_data = pd.DataFrame([[
            trip_distance, pickup_hour, pickup_day, passenger_count
        ]], columns=['trip_distance', 'pickup_hour', 'pickup_day', 'passenger_count'])
        
        try:
            features = preprocessor.transform(input_data)
            fare = model.predict(features)[0]
            st.sidebar.success(f"Predicted Fare: ${fare:.2f}")
        except Exception as e:
            st.sidebar.error(f"Prediction failed: {str(e)}")

# ===== VISUALIZATIONS =====
st.subheader("🔍 Data Insights")

# Clean data
clean_df = df[
    (df['trip_distance'] > 0) & 
    (df['trip_duration'] > 0) & 
    (df['fare_amount'] > 0)
].copy()

# Plot 1: Distance vs Duration
st.markdown("### Trip Duration Patterns")
fig1 = px.scatter(
    clean_df.sample(10000, random_state=42),
    x='trip_distance',
    y='trip_duration',
    color='pickup_hour',
    trendline="lowess",
    labels={'trip_distance': 'Distance (miles)', 'trip_duration': 'Duration (mins)'}
)
st.plotly_chart(fig1, use_container_width=True)

# Plot 2: Fare by Distance
st.markdown("### Fare Distribution")
clean_df['distance_bin'] = pd.cut(clean_df['trip_distance'], 
                                bins=[0, 1, 3, 5, 10, 20],
                                labels=['0-1', '1-3', '3-5', '5-10', '10-20'])

fig2, ax = plt.subplots(figsize=(10,5))
clean_df.groupby('distance_bin')['fare_amount'].mean().plot.bar(ax=ax, color='skyblue')
ax.set_title("Average Fare by Distance")
ax.set_xlabel("Distance Bin (miles)")
ax.set_ylabel("Average Fare ($)")
st.pyplot(fig2)

st.success("App running successfully!")
