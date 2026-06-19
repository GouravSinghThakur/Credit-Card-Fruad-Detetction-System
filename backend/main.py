from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib

# Initialize API
app = FastAPI(
    title="Fraud Detection API",
    description="API for credit card fraud prediction"
)

# Load model and scaler
model = joblib.load("../models/fraud_detection_pipeline.pkl")
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
def predict(data: dict):

    features = np.array(
        data["features"]
    ).reshape(1, -1)

    features_scaled = scaler.transform(features)

    prediction = int(
        model.predict(features_scaled)[0]
    )

    probability = float(
        model.predict_proba(features_scaled)[0][1]
    )

    return {
        "prediction": prediction,
        "probability": probability
    }
