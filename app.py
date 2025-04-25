#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
NYC Taxi Fare Prediction App - Stable Version
"""

# ===== CRITICAL EVENT LOOP FIX =====
import asyncio
import nest_asyncio
import threading

def fix_event_loop():
    try:
        # Check if we're in the main thread
        if threading.current_thread() is threading.main_thread():
            nest_asyncio.apply()
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                asyncio.set_event_loop(asyncio.new_event_loop())
    except:
        asyncio.set_event_loop(asyncio.new_event_loop())

fix_event_loop()

# ===== IMPORTS =====
import streamlit as st
from streamlit.runtime.scriptrunner import add_script_run_ctx
import pandas as pd
import joblib
import plotly.express as px
import matplotlib.pyplot as plt
import os
import sys
from io import BytesIO

# ===== STREAMLIT CONFIGURATION =====
st.set_page_config(
    layout="wide",
    page_title="NYC Taxi Fare Prediction",
    page_icon="🚕"
)

# ===== MODEL LOADING =====
@st.cache_resource(ttl=3600, show_spinner="Loading ML model...")
def load_model_artifacts():
    """Load model and preprocessor with enhanced error handling"""
    try:
        model = joblib.load('best_gradient_boosting_model.pkl')
        preprocessor = joblib.load('preprocessor.pkl')
        
        # Verify the preprocessor is fitted
        if not hasattr(preprocessor, 'transform'):
            raise ValueError("Preprocessor not fitted properly")
            
        return model, preprocessor
    except Exception as e:
        st.error(f"""**Model Loading Failed**  
                Error: {str(e)}  
                Please check:  
                1. Model files exist in deployment  
                2. File paths are correct  
                3. Python versions match""")
        st.stop()

model, preprocessor = load_model_artifacts()

# ===== DATA LOADING =====
@st.cache_data(ttl=600, show_spinner="Loading dataset...")
def load_data():
    """Load and preprocess data with robust error handling"""
    try:
        # Sample data - replace with your actual data loading logic
        data = {
            'trip_distance': [2.5, 3.0, 1.5],
            'pickup_hour': [12, 18, 8],
            'pickup_day': [3, 5, 1],
            'fare_amount': [15.5, 22.0, 9.5]
        }
        df = pd.DataFrame(data)
        
        # Add your actual feature engineering here
        df['trip_duration'] = df['trip_distance'] * 5  # Example calculation
        
        return df
    except Exception as e:
        st.error(f"Data loading failed: {str(e)}")
        return pd.DataFrame()  # Return empty dataframe to prevent crashes

df = load_data()

# ===== APP LAYOUT =====
def main():
    """Main application function"""
    st.title("🚕 NYC Taxi Fare Prediction")
    
    # === PREDICTION SECTION ===
    with st.sidebar:
        st.header("Fare Prediction")
        with st.form("prediction_form"):
            distance = st.slider("Trip Distance (miles)", 0.1, 30.0, 2.5)
            hour = st.selectbox("Hour", options=list(range(24)), index=12)
            day = st.selectbox("Day (0=Mon)", options=list(range(7)), index=0)
            
            if st.form_submit_button("Predict Fare"):
                try:
                    input_df = pd.DataFrame([{
                        'trip_distance': distance,
                        'pickup_hour': hour,
                        'pickup_day': day
                    }])
                    
                    # Transform and predict
                    features = preprocessor.transform(input_df)
                    prediction = model.predict(features)[0]
                    
                    st.success(f"**Predicted Fare:** ${prediction:.2f}")
                except Exception as e:
                    st.error(f"Prediction failed: {str(e)}")

    # === VISUALIZATIONS ===
    st.header("Data Insights")
    
    if not df.empty:
        tab1, tab2 = st.tabs(["Duration Analysis", "Fare Distribution"])
        
        with tab1:
            fig = px.scatter(
                df,
                x='trip_distance',
                y='trip_duration',
                color='pickup_hour',
                title="Trip Duration vs Distance"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            fig, ax = plt.subplots()
            df['fare_amount'].hist(ax=ax, bins=20)
            ax.set_title("Fare Amount Distribution")
            st.pyplot(fig)
    else:
        st.warning("No data available for visualizations")

# ===== MAIN EXECUTION =====
if __name__ == "__main__":
    # Add thread context for Streamlit
    if hasattr(st.runtime.scriptrunner, 'add_script_run_ctx'):
        add_script_run_ctx(threading.current_thread())
    
    main()
