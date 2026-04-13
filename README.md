# 🔐 Credit Card Fraud Detection System

![Python](https://img.shields.io/badge/python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/fastapi-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![TensorFlow](https://img.shields.io/badge/tensorflow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Joblib](https://img.shields.io/badge/joblib-FFD43B?style=for-the-badge)

An end-to-end **Deep Learning project** that analyzes **credit card transaction patterns** and builds a machine learning system to **predict fraudulent transactions in real-time** based on transaction features.

---

## Project Overview

Credit card fraud costs billions globally and impacts both consumers and financial institutions. Understanding **transaction patterns and anomalies** enables early detection and prevention of fraudulent activities.

This project includes:

- Deep neural network model for fraud detection
- Real-time prediction API using FastAPI
- Interactive web dashboard using Streamlit
- High-accuracy fraud classification system
---

## Dataset

Dataset: **Credit Card Fraud Detection Dataset**

Key features:

- `V1 - V28`: Principal component analysis (PCA) transformed features
- `Amount`: Transaction amount
- `Time`: Seconds elapsed from first transaction
- `Class`: Binary target (0 = Normal, 1 = Fraud)

Statistics:

- **284,807** total transactions
- **492** fraudulent transactions (0.17% - highly imbalanced)
- **30** input features
- **2 classes** (Normal & Fraud)

---

## Project Workflow

### 1. Data Preprocessing

- Handling class imbalance
- Feature scaling and normalization
- Data cleaning and validation
- Train-test split
- Feature standardization using `StandardScaler`

---

### 2. Exploratory Data Analysis (EDA)

Analysis performed:

- Fraud vs Normal transaction distribution
- Amount distribution analysis
- Feature correlation analysis
- Class imbalance visualization
- Temporal patterns

Visualization tools:

- Matplotlib
- Seaborn

---

### 3. Machine Learning Model

**Architecture:**

Deep Neural Network with:
- **Input Layer:** 30 transaction features
- **Hidden Layers:** 
  - Dense(1024) → BatchNormalization → Dense(512) → BatchNormalization
  - Dense(256) → Dropout(0.3) → BatchNormalization
  - Dense(128) → BatchNormalization → Dense(64) → Dropout(0.3)
  - Dense(32) → BatchNormalization → Dense(16)
- **Output Layer:** Dense(1) with Sigmoid activation
- **Total Parameters:** 2,211,077 (2.2M)
- **Regularization:** Batch Normalization + Dropout for improved generalization

**Evaluation Metrics:**

- Accuracy
- Precision & Recall
- F1-Score
- ROC-AUC Score

---

## Tech Stack

### Programming Language
- Python 3.8+

### Data Processing
- Pandas
- NumPy

### Deep Learning
- TensorFlow/Keras

### Backend & API
- FastAPI
- Uvicorn

### Web Application
- Streamlit

### Model Serialization
- Joblib

### Utilities
- Scikit-learn

---

## Project Structure

```
fraud-detection-system/
│
├── backend/
│   └── main.py                    # FastAPI application & endpoints
│
├── frontend/
│   └── app.py                     # Streamlit web interface
│
├── models/
│   ├── best_fraud_model_2.keras   # Trained neural network
│   └── scaler.pkl                 # Feature scaler
│
├── data/
│   └── creditcard.csv             # Training dataset
│
├── reports/
│   ├── model_summary.txt
├── requirements.txt               # Python dependencies
└── README.md                      # Project documentation
```

---

## Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip or conda package manager

### Step 1: Clone Repository
```bash
git clone https://github.com/yourusername/fraud-detection-system.git
cd fraud-detection-system
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Quick Start

### Option 1: Full Application (API + Dashboard)

**Terminal 1 - Start FastAPI Backend:**
```bash
cd backend
uvicorn main:app --reload
```
API available at: `http://127.0.0.1:8000`
API docs at: `http://127.0.0.1:8000/docs`

**Terminal 2 - Start Streamlit Dashboard:**
```bash
cd frontend
streamlit run app.py
```
Dashboard available at: `http://localhost:8501`

### Option 2: API Only
```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000
```

---

## API Endpoints

### GET `/`
Health check endpoint.

**Response:**
```json
{
  "message": "Fraud Detection API is running"
}
```

### POST `/predict`
Predict fraud probability for a transaction.

**Request:**
```json
{
  "features": [0.5, -0.3, 0.2, ..., 0.1]  // 30 features
}
```

**Response:**
```json
{
  "prediction": "Normal",
  "probability": 0.0234
}
```

**Classification:**
- Probability > 0.5: **Fraud** ⚠️
- Probability ≤ 0.5: **Normal** ✅

---

## Web Dashboard Features

The Streamlit interface provides:

- **Input Fields:** 30 feature inputs (V1-V30)
- **Real-time Predictions:** Instant fraud probability calculation
- **Visual Feedback:** ✅ Normal / ⚠️ Fraud indicators
- **Probability Score:** Precise 4-decimal probability display
- **Error Handling:** Graceful API connection error messages

---

## Model Performance

**Key Metrics:**

- High sensitivity to detect frauds
- Low false positive rate
- Balanced precision-recall trade-off
- Optimized for production deployment

**Results stored in `reports/model_summary.txt`**

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| fastapi | Latest | REST API framework |
| uvicorn | Latest | ASGI web server |
| streamlit | Latest | Web UI framework |
| tensorflow | Latest | Deep learning framework |
| numpy | Latest | Numerical computing |
| pandas | Latest | Data processing |
| joblib | Latest | Model serialization |
| scikit-learn | Latest | ML utilities |
| requests | Latest | HTTP client |

---

## Configuration

### Adjust Fraud Threshold

Edit `backend/main.py`:
```python
# Default: > 0.5 is fraud
prediction = "Fraud" if probability > 0.5 else "Normal"

# Adjust to your needs (e.g., 0.3 for stricter detection)
prediction = "Fraud" if probability > 0.3 else "Normal"
```

### Change API Port
```bash
uvicorn main:app --port 8080
```

### Change Streamlit Port
```bash
streamlit run app.py --server.port 8502
```

---

## Use Cases

### Example 1: Real-time Fraud Detection
```
Transaction Amount: $500
Features: [0.5, -0.3, 0.2, ..., 0.1]
Model Output: Normal Transaction (probability: 0.02)
```

### Example 2: High-Risk Transaction Alert
```
Transaction Amount: $5,000
Features: [2.5, 1.8, -0.9, ..., 2.1]
Model Output: Fraud Alert (probability: 0.87)
```

---

## Key Insights

- Strong predictive power for fraud detection
- Neural network captures complex transaction patterns
- Batch normalization improves model stability
- Dropout regularization prevents overfitting
- Scalable architecture for high-volume predictions
- Real-time processing capability

---

## Future Improvements

- Deploy to cloud (AWS, GCP, Azure)
- Add model explainability (SHAP values)
- Implement continuous model retraining
- Add historical prediction dashboard
- Real-time data ingestion pipeline
- Multi-model ensemble approach
- Mobile app integration

---

## License

This project is licensed under the MIT License.

---

## Support & Contributing

For issues or questions, please open an issue on GitHub.

Contributions are welcome! Feel free to submit pull requests.

---

## Author

**Gourav Singh Thakur**

Data Scientist | Machine Learning Engineer | AI

---

## Acknowledgments

- Dataset: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- Built with [FastAPI](https://fastapi.tiangolo.com/), [Streamlit](https://streamlit.io/), and [TensorFlow](https://www.tensorflow.org/)

---

**⭐ If you find this project helpful, please star it on GitHub!**
