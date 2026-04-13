import streamlit as st
import requests
import numpy as np

# Page configuration
st.set_page_config(page_title="Fraud Detection", layout="wide")

st.title("Credit Card Fraud Detection System")

st.write("Enter transaction feature values to predict whether a transaction is fraudulent.")

# Create input fields
features = []

for i in range(30):
    value = st.number_input(f"Feature V{i+1}", value=0.0)
    features.append(value)

# Prediction button
if st.button("Predict Transaction"):

    try:
        response = requests.post(
            "http://127.0.0.1:8000/predict",
            json={"features": features}
        )

        result = response.json()

        st.subheader("Prediction Result")

        if result["prediction"] == "Fraud":
            st.error(f"Fraud Transaction ⚠️")
        else:
            st.success("Normal Transaction ✅")

        st.write(f"Fraud Probability: {result['probability']:.4f}")

    except:
        st.warning("API not running. Please start the FastAPI backend.")