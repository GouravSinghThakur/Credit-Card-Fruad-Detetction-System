import streamlit as st
import requests
import numpy as np

API_URL = "https://credit-card-fruad-detetction-system.onrender.com/predict"

# Page configuration
st.set_page_config(page_title="Fraud Detection", layout="wide")
st.title("💳 Credit Card Fraud Detection System")
st.write("Enter transaction feature values to predict whether a transaction is fraudulent.")

# --- STEP 1: Main Transaction Metadata ---
col_meta1, col_meta2 = st.columns(2)
with col_meta1:
    time_val = st.number_input("Transaction Time (Seconds elapsed since first transaction)", min_value=0.0, value=0.0, step=1.0)
with col_meta2:
    amount_val = st.number_input("Transaction Amount ($)", min_value=0.0, value=10.0, step=0.01)

# --- STEP 2: Hidden Grid for PCA Features V1 to V28 ---
features = [time_val] # Start your features array with Time

with st.expander("🔍 Adjust PCA Anonymous Features (V1 - V28)", expanded=True):
    # Create 4 columns to make 28 sliders compact and neat
    cols = st.columns(4)
    
    for i in range(29):
        with cols[i % 4]:
            # Added float type defaults (0.0) and unique keys to prevent streamlit rendering crashes
            value = st.slider(
                label=f"Feature V{i+1}", 
                min_value=-50.0, 
                max_value=50.0, 
                value=0.0, 
                step=0.1,
                key=f"slider_v{i+1}"
            )
            features.append(value)

# Append Amount at the end to match standard 31-feature model input shape
features.append(amount_val) 

# --- STEP 3: Prediction Trigger ---
if st.button("Predict Transaction", type="primary"):
    try:
        # Convert list to pure Python floats to prevent JSON serialization errors
        payload = {"features": [float(x) for x in features]}
        
        response = requests.post(API_URL, json=payload, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            st.subheader("Prediction Result")
            
            # Check if prediction exists in your JSON structure
            if "prediction" in result:
                if result["prediction"] == "Fraud" or result.get("prediction") == 1:
                    st.error("Fraud Transaction ⚠️")
                else:
                    st.success("Normal Transaction ✅")
            
            if "probability" in result:
                st.metric(label="Fraud Probability", value=f"{result['probability']:.4f}")
        else:
            st.error(f"Backend API error code: {response.status_code}")
            
    except requests.exceptions.RequestException:
        st.warning("API not running or unreachable. Please start your FastAPI backend on Render.")

# Footer layout placement
st.divider()
st.caption("Developed BY Gourav Singh Thakur")
