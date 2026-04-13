from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Initialize API
app = FastAPI(
    title="Fraud Detection API",
    description="API for credit card fraud prediction",
    version="1.0"
)

# Load model and scaler
model = load_model("../models/fraud_detection_model.keras")
scaler = joblib.load("../models/scaler.pkl")


# Input schema
class Transaction(BaseModel):
    features: list


# Home route
@app.get("/")
def home():
    return {"message": "Fraud Detection API is running"}


# Prediction endpoint
@app.post("/predict")
def predict(transaction: Transaction):

    features = np.array(transaction.features).reshape(1, -1)

    # scale features
    features_scaled = scaler.transform(features)

    # predict probability
    probability = model.predict(features_scaled)[0][0]

    prediction = "Fraud" if probability > 0.5 else "Normal"

    return {
        "prediction": prediction,
        "probability": float(probability)
    }