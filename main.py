from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import pandas as pd

# ==============================
# CONFIG
# ==============================
MODEL_PATH = "models/best_model.pkl"
MEDICAL_THRESHOLD = 0.1  # Optimized for high recall

# ==============================
# FASTAPI INIT
# ==============================
app = FastAPI(
    title="Stroke Risk Prediction API",
    description="Medical-grade stroke prediction system with optimized recall threshold.",
    version="1.1"
)

# ==============================
# CORS FIX (IMPORTANT)
# ==============================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================
# LOAD MODEL SAFELY
# ==============================
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")

# ==============================
# INPUT SCHEMA
# ==============================
class StrokeInput(BaseModel):
    gender: str = Field(..., example="Male")
    age: float = Field(..., example=67)
    hypertension: int = Field(..., example=1)
    heart_disease: int = Field(..., example=1)
    ever_married: str = Field(..., example="Yes")
    work_type: str = Field(..., example="Private")
    Residence_type: str = Field(..., example="Urban")
    avg_glucose_level: float = Field(..., example=228.69)
    bmi: float = Field(..., example=36.6)
    smoking_status: str = Field(..., example="formerly smoked")

# ==============================
# HEALTH CHECK
# ==============================
@app.get("/")
def home():
    return {
        "message": "Stroke Prediction API is running.",
        "medical_threshold": MEDICAL_THRESHOLD
    }

# ==============================
# PREDICTION ENDPOINT
# ==============================
@app.post("/predict")
def predict(data: StrokeInput):
    try:
        input_df = pd.DataFrame([data.dict()])

        probability = model.predict_proba(input_df)[0][1]

        # Medical optimized threshold
        prediction = 1 if probability >= MEDICAL_THRESHOLD else 0

        risk_level = (
            "High Risk" if probability >= 0.5
            else "Moderate Risk" if probability >= MEDICAL_THRESHOLD
            else "Low Risk"
        )

        return {
            "stroke_prediction": prediction,
            "stroke_probability": round(float(probability), 4),
            "risk_level": risk_level,
            "threshold_used": MEDICAL_THRESHOLD
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))