# 🧠 Stroke Risk Prediction System (End-to-End ML SaaS)

## 🌍 Live Application

🔹 **Frontend Dashboard**  
https://stroke-prediction-frontend.onrender.com/

🔹 **Backend API (Swagger Docs)**  
https://stroke-prediction-4nkn.onrender.com/docs

---

## 📌 Project Overview

This project is a production-ready Machine Learning system designed to predict stroke risk based on patient clinical and lifestyle attributes.

It is built as a full-stack ML SaaS application including:

- End-to-end ML pipeline
- Medical threshold optimization
- FastAPI backend deployment
- Cloud-hosted frontend dashboard
- Real-time risk visualization
- Circular probability gauge
- Cross-origin API integration
- Full production deployment on Render

This is not a notebook demo — it is a deployed, interactive ML system.

---

## 🏗 System Architecture
User (Browser Dashboard)
↓
Frontend (HTML/CSS/JS - Render Static Site)
↓
FastAPI Backend (Render Web Service)
↓
Trained ML Model (Logistic Regression + SMOTE)
↓
Prediction + Risk Categorization
↓
Response → Risk Gauge Visualization


---

## 🧠 Machine Learning Pipeline

### 1️⃣ Problem Type
Binary Classification  
Predict:
- 1 → Stroke Risk
- 0 → No Stroke Risk

---

### 2️⃣ Data Preprocessing

- Handling missing values
- One-hot encoding categorical features
- Feature scaling
- Train-test split (Stratified)
- Class imbalance analysis

---

### 3️⃣ Imbalance Handling

Stroke datasets are highly imbalanced.

To address this:

- Applied **SMOTE (Synthetic Minority Oversampling Technique)**
- Used Stratified 5-Fold Cross Validation

This ensures stable performance and improved minority class detection.

---

### 4️⃣ Model Selection

Selected Model:
- **Logistic Regression**

Reasons:
- Interpretable
- Stable for medical domain
- Probabilistic output
- Works well with threshold tuning

---

### 5️⃣ Model Performance

| Metric | Value |
|--------|--------|
| ROC-AUC | 0.845 |
| Recall (Default Threshold 0.5) | 18% |
| Recall (Optimized Threshold 0.1) | 74% |

---

## 🎯 Medical Threshold Optimization

Default classification threshold is 0.5.

However, in healthcare systems:

- False Negatives are dangerous
- Missing high-risk patients is unacceptable
- High Recall is prioritized

Therefore, threshold was optimized to: 0.1


This increases recall from 18% → 74%.

This design decision improves patient safety.

---

## 📊 Features Used

- Age
- Gender
- Hypertension
- Heart Disease
- Average Glucose Level
- BMI
- Smoking Status
- Marital Status
- Work Type
- Residence Type

---

## 🚀 Backend Architecture

### Framework
- FastAPI

### Features
- REST API endpoint `/predict`
- JSON request/response
- Medical threshold logic
- Probability output
- Risk level categorization
- CORS enabled
- Error handling with HTTPException
- Production deployment on Render

---

## 🔄 API Usage

### Endpoint
POST / PREDICT


### Example Request

```json
{
  "gender": "Male",
  "age": 67,
  "hypertension": 1,
  "heart_disease": 1,
  "ever_married": "Yes",
  "work_type": "Private",
  "Residence_type": "Urban",
  "avg_glucose_level": 228.69,
  "bmi": 36.6,
  "smoking_status": "formerly smoked"
}

Example Response 

{
  "stroke_prediction": 1,
  "stroke_probability": 0.9027,
  "risk_level": "High Risk",
  "threshold_used": 0.1
}

Frontend Dashboard
Built With

HTML5

CSS3

JavaScript (Fetch API)

Features

Professional SaaS dashboard layout

Sidebar navigation

Stats input cards

Animated circular probability gauge

Real-time API integration

Responsive design

Clean modern UI

Cloud deployment on Render

🛠 Tech Stack
Machine Learning

Python

Pandas

NumPy

Scikit-learn

Imbalanced-learn (SMOTE)

Backend

FastAPI

Uvicorn

Pydantic

Joblib

Frontend

HTML

CSS

JavaScript

Deployment

Render (Backend Web Service)

Render (Static Site Frontend)

GitHub (Version Control)

📦 Project Structure
stroke_prediction_project/
│
├── models/
│   └── best_model.pkl
│
├── train.py
├── explain.py
├── main.py
├── requirements.txt
│
├── index.html
├── style.css
└── script.js
🔍 Engineering Highlights

Imbalance handling using SMOTE

Threshold optimization for medical safety

Cross-validation for stability

Production-grade REST API

Cross-Origin Resource Sharing (CORS) handling

Circular gauge visualization using CSS conic gradients

End-to-end cloud deployment

🧪 Future Improvements

XGBoost / Ensemble learning

MLflow experiment tracking

Docker containerization

CI/CD automation

Model monitoring

Authentication system

Patient report PDF export

Role-based access control
