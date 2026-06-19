# 🔐 [Credit Card Fraud Detection System](https://fruad-detetction-system.streamlit.app/)

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
  <img width="300" height="100" alt="roc_curve" src="https://github.com/user-attachments/assets/62a31621-3644-4269-ac05-7d43a761c6c0" />


* Precision-Recall Curve
  <img width="600" height="400" alt="pr_curve" src="https://github.com/user-attachments/assets/2a302c1c-3cdf-4448-bb52-a76bbe61d23c" />


* Precision-Recall Tradeoff Curve
  <img width="800" height="400" alt="pr_tradeoff" src="https://github.com/user-attachments/assets/57f4235c-af68-48ce-a8a3-44fed9d6fc1c" />

* Confusion Matrix
  <img width="600" height="500" alt="confusion_matrix" src="https://github.com/user-attachments/assets/3a6c59be-b354-426a-aff7-97276537ce5c" />

* Feature Importance Plot\
  <img width="640" height="480" alt="feature_importance" src="https://github.com/user-attachments/assets/210c9a3f-b05e-498d-af34-5acdb1276d66" />

* SHAP Summary Plot
  <img width="518" height="624" alt="image" src="https://github.com/user-attachments/assets/6e8a0c37-1f49-46d4-86bc-388a4ea8aece" />
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
