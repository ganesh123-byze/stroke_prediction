# 🧠 Stroke Risk Prediction System (End-to-End ML Deployment)

## 📌 Overview

This project is a production-ready Machine Learning system for predicting stroke risk based on patient health parameters. 

The system includes:
- Data preprocessing pipeline
- Imbalance handling using SMOTE
- Cross-validation for robustness
- Threshold optimization for medical recall improvement
- SHAP explainability
- FastAPI REST API
- Public cloud deployment

---

## 🚀 Live API

🔗 [Live Swagger UI](https://your-deployed-url.onrender.com/docs)

---

## 🏗 Project Architecture


stroke_prediction_project/
│
├── data/
│ └── stroke_data.csv
│
├── models/
│ └── best_model.pkl
│
├── train.py
├── explain.py
├── main.py
├── requirements.txt
└── README.md


---

## ⚙️ Model Details

- Best Model: Logistic Regression
- Test ROC-AUC: 0.845
- Medical Optimized Threshold: 0.1
- Recall improved from 18% → 74%

### Why Threshold 0.1?

In healthcare applications:
- False Negatives are dangerous
- High Recall is prioritized
- Model optimized for safer screening

---

## 📊 Features Used

- Age
- Hypertension
- Heart Disease
- BMI
- Avg Glucose Level
- Smoking Status
- Work Type
- Residence Type
- Marital Status

---

## 🧪 Cross Validation

- 5-Fold Stratified Cross Validation
- Evaluated using ROC-AUC and Recall
- Ensured model stability

---

## 🧠 Explainability

SHAP was used to interpret feature contributions and improve model transparency.

---

## 🖥 API Usage

### POST /predict

Example Request:

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

Example Response:

{
  "stroke_prediction": 1,
  "stroke_probability": 0.90,
  "risk_level": "High Risk",
  "threshold_used": 0.1
}
🛠 Tech Stack

Python

Scikit-learn

Imbalanced-learn (SMOTE)

SHAP

FastAPI

Uvicorn

Render (Cloud Deployment)

🎯 Key Achievements

Built end-to-end ML pipeline

Optimized classification threshold for medical safety

Implemented production-grade REST API

Deployed publicly on cloud platform

📌 Future Improvements

XGBoost integration

MLflow experiment tracking

CI/CD automation

Docker containerization

Model monitoring system

👨‍💻 Author

Ganesh Pedagada
Aspiring ML Engineer