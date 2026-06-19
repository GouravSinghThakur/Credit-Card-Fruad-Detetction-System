# 🔐 Credit Card Fraud Detection System

![Python](https://img.shields.io/badge/python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-EC6B23?style=for-the-badge)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)
![SHAP](https://img.shields.io/badge/SHAP-FF6F00?style=for-the-badge)
![FastAPI](https://img.shields.io/badge/fastapi-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Uvicorn](https://img.shields.io/badge/Uvicorn-4051B5?style=for-the-badge)
![Pandas](https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge)
![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0?style=for-the-badge)
![SMOTETomek](https://img.shields.io/badge/SMOTETomek-8A2BE2?style=for-the-badge)

## Project Overview

This project is an end-to-end Credit Card Fraud Detection System designed to identify fraudulent transactions using machine learning. The solution includes data preprocessing, imbalance handling, model training, experiment tracking, model explainability, API deployment, and an interactive dashboard.

The system leverages XGBoost with SMOTETomek resampling to address severe class imbalance and improve fraud detection performance.

### Key Features

* Fraud detection using XGBoost Classifier
* Class imbalance handling with SMOTETk
* Feature scaling using StandardScaler
* Experiment tracking using MLflow
* Model explainability using SHAP
* REST API deployment with FastAPI
* Interactive dashboard with Streamlit
* Real-time fraud prediction

---

## Dataset

**Dataset:** Credit Card Fraud Detection Dataset

### Dataset Statistics

* Total Transactions: 284,807
* Fraudulent Transactions: 492
* Fraud Rate: 0.17%
* Features: 30
* Target Variable:

  * 0 = Normal Transaction
  * 1 = Fraudulent Transaction

### Features

* V1 – V28: PCA-transformed transaction features
* Amount: Transaction amount
* Time: Seconds elapsed between transactions
* Class: Target variable

---

## Machine Learning Pipeline

### Data Preprocessing

* Data validation and quality checks
* Train-test split
* Feature scaling using StandardScaler

### Imbalance Handling

* SMOTETomek

  * SMOTE for minority class oversampling
  * Tomek Links for noise reduction

### Model

* XGBoost Classifier

### Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC
* Precision-Recall AUC
* Confusion Matrix

### Model Explainability

SHAP (SHapley Additive Explanations)

Generated Visualizations:

* ROC Curve
* Precision-Recall Curve
* Precision-Recall Tradeoff Curve
* Confusion Matrix
* Feature Importance Plot
* SHAP Summary Plot

---

## Experiment Tracking

MLflow is used for:

* Parameter tracking
* Metric tracking
* Experiment comparison
* Model versioning
* Artifact management

Tracked Metrics:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC
* PR-AUC
* Training Time

---

## Tech Stack

### Machine Learning

* XGBoost
* Scikit-learn
* Imbalanced-learn
* SHAP

### Data Processing

* Pandas
* NumPy

### Experiment Tracking

* MLflow

### Backend

* FastAPI
* Uvicorn

### Frontend

* Streamlit

### Visualization

* Matplotlib
* Seaborn

---

## Future Improvements

* Hyperparameter optimization using Optuna
* Real-time streaming predictions
* Automated retraining pipeline
* Data drift monitoring
* Model performance monitoring
* Cloud deployment (AWS, Azure, GCP)
* Ensemble learning approaches

---

## Author

**Gourav Singh Thakur**

**Data Scientist
**
**⭐ If you find this project helpful, please star it on GitHub!**
