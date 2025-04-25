import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import plotly.express as px

# Set page config
st.set_page_config(layout="wide")
st.title("NYC Taxi Fare Prediction & Data Insights")

# === Load model and preprocessor ===
model_path = "/Users/rue/Desktop/Rawan_Yasser_202201681_Data_Science/best_gradient_boosting_model.pkl"
preprocessor_path = "/Users/rue/Desktop/Rawan_Yasser_202201681_Data_Science/preprocessor.pkl"
model = joblib.load(model_path)
preprocessor = joblib.load(preprocessor_path)

# === Load dataset ===
import os
import requests
import gzip
import shutil

@st.cache_data
def load_data():
    local_path = "fixed_nyc_taxi_sample_2M.csv.gz"
    one_drive_url = "https://onedrive.live.com/download?resid=D8331802E3D7620B%21254&authkey=!AFmF9wiQ4d1E1hM"

    # If file not found locally, download it
    if not os.path.exists(local_path):
        st.info("📥 Downloading dataset from OneDrive...")
        with requests.get(one_drive_url, stream=True) as r:
            with open(local_path, 'wb') as f:
                shutil.copyfileobj(r.raw, f)
        st.success("✅ Download complete.")

    # Read gzip CSV only necessary columns for efficiency
    columns_needed = ['tpep_pickup_datetime', 'tpep_dropoff_datetime', 'trip_distance', 'fare_amount', 'tip_amount']
    df = pd.read_csv(local_path, compression='gzip', usecols=columns_needed, parse_dates=['tpep_pickup_datetime', 'tpep_dropoff_datetime'])

    # Preprocess and create new columns
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
    scaled_input = preprocessor.transform(input_df)
    prediction = model.predict(scaled_input)[0]
    st.sidebar.success(f"💰 Predicted Fare: ${prediction:.2f}")

# === Main: Visuals Section ===
st.subheader("🔍 Data Insights & Visualizations")

# --- 1. Distance vs Duration Scatter (with trendline) ---
st.markdown("### 1. Trip Distance vs Duration by Hour")

# Clean and filter data for valid values
clean_df = df[
    (df['trip_distance'] > 0) &
    (df['trip_duration'] > 0) &
    (df['fare_amount'] > 0)
].dropna()

# Correlation after cleaning
corr = clean_df['trip_distance'].corr(clean_df['trip_duration'])

# Plot
fig1 = px.scatter(
    clean_df,
    x='trip_distance',
    y='trip_duration',
    color='pickup_hour',
    trendline="lowess",
    title=f"Duration vs Distance | Hourly Patterns (Corr: {corr:.2f})",
    labels={'trip_distance': 'Distance (miles)', 'trip_duration': 'Duration (mins)'}
)
st.plotly_chart(fig1, use_container_width=True)

# --- 2. Average Fare & Tip by Distance Bin ---
st.markdown("### 2. Average Fare and Tip by Trip Distance")
df['distance_bin'] = pd.cut(df['trip_distance'], bins=[0, 1, 3, 5, 10, 20], labels=['0-1', '1-3', '3-5', '5-10', '10-20'])
grouped = df.groupby('distance_bin')[['fare_amount', 'tip_amount']].mean().reset_index()

fig2, ax2 = plt.subplots(figsize=(10, 5))
grouped.plot(x='distance_bin', kind='bar', colormap='Set2', ax=ax2)
ax2.set_title('Average Fare and Tip Amount by Trip Distance')
ax2.set_xlabel('Trip Distance (miles)')
ax2.set_ylabel('Average Amount ($)')
ax2.grid(axis='y', linestyle='--', alpha=0.7)
st.pyplot(fig2)
