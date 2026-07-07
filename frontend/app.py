import streamlit as st
import requests
import numpy as np

API_URL = "https://onrender.com"

# Page configuration
st.set_page_config(page_title="Fraud Detection", layout="wide")
st.title("💳 Credit Card Fraud Detection System")
st.write("Enter transaction feature values to predict whether a transaction is fraudulent.")

# --- NEW STEP: Preset Sample Loaders ---
st.subheader("🧪 Quick Test Presets")
col_btn1, col_btn2, _ = st.columns([1, 1, 2])

# Initialize session state variables for inputs if they don't exist
if "time_input" not in st.session_state: st.session_state.time_input = 0.0
if "amount_input" not in st.session_state: st.session_state.amount_input = 10.0
for i in range(29):
    if f"v_{i+1}" not in st.session_state:
        st.session_state[f"v_{i+1}"] = 0.0

with col_btn1:
    if st.button("🟢 Load Normal Case", use_container_width=True):
        st.session_state.time_input = 45000.0
        st.session_state.amount_input = 25.50
        for i in range(29):
            st.session_state[f"v_{i+1}"] = 0.0  # Safe, average benchmark features

with col_btn2:
    if st.button("🔴 Load Fraud Case", use_container_width=True):
        st.session_state.time_input = 120000.0
        st.session_state.amount_input = 999.99
        # Typical highly volatile distributions seen in fraud vectors (V1 to V5 usually drop significantly)
        for i in range(29):
            if i < 5:
                st.session_state[f"v_{i+1}"] = -15.5  # Heavy negative anomalies
            elif i == 10 or i == 11:
                st.session_state[f"v_{i+1}"] = -8.0   # Secondary indicator features
            else:
                st.session_state[f"v_{i+1}"] = 1.2

st.divider()

# --- STEP 1: Main Transaction Metadata ---
col_meta1, col_meta2 = st.columns(2)
with col_meta1:
    time_val = st.number_input("Transaction Time (Seconds elapsed)", min_value=0.0, value=st.session_state.time_input, step=1.0, key="time_widget")
with col_meta2:
    amount_val = st.number_input("Transaction Amount ($)", min_value=0.0, value=st.session_state.amount_input, step=0.01, key="amount_widget")

# --- STEP 2: Hidden Grid for PCA Features V1 to V29 ---
features = [time_val] 

with st.expander("🔍 Adjust PCA Anonymous Features (V1 - V29)", expanded=True):
    cols = st.columns(4)
    for i in range(29):
        with cols[i % 4]:
            value = st.slider(
                label=f"Feature V{i+1}", 
                min_value=-50.0, 
                max_value=50.0, 
                value=st.session_state[f"v_{i+1}"], 
                step=0.1, 
                key=f"slider_v{i+1}"
            )
            features.append(value)

features.append(amount_val) 

# --- STEP 3: Prediction Trigger ---
if st.button("Predict Transaction", type="primary", use_container_width=True):
    try:
        payload = {"features": [float(x) for x in features]}
        
        # Render free-tier cold-start visual cue
        with st.spinner("Analyzing transaction data via Render API (May take 30+ seconds if backend is sleeping)..."):
            response = requests.post(API_URL, json=payload, timeout=45)
            
        if response.status_code == 200:
            result = response.json()
            st.subheader("Prediction Result")
            
            if "prediction" in result:
                # Handle both string types ("Fraud") and integer outputs (1)
                if result["prediction"] in ["Fraud", 1, "1"]:
                    st.error("Fraud Transaction ⚠️")
                else:
                    st.success("Normal Transaction ✅")
            if "probability" in result:
                st.metric(label="Fraud Probability Score", value=f"{result['probability']:.4f}")
        else:
            st.error(f"Backend API error code: {response.status_code}")
    except requests.exceptions.Timeout:
        st.error("The request timed out. Your Render backend might still be waking up from an idle state.")
    except requests.exceptions.RequestException:
        st.warning("API not running or unreachable. Please verify your FastAPI backend hosting state.")

# Footer layout placement
st.divider()
st.caption("Developed BY Gourav Singh Thakur")
