from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from xgboost import XGBClassifier

app = FastAPI(
    title="Silent Churn Prediction API",
    description="Predicts whether a telecom customer will churn",
    version="1.0.0"
)

model = XGBClassifier()
model.load_model('models/churn_model.json')

class CustomerData(BaseModel):
    gender: int
    SeniorCitizen: int
    Partner: int
    Dependents: int
    tenure: float
    PhoneService: int
    MultipleLines: int
    InternetService: int
    OnlineSecurity: int
    OnlineBackup: int
    DeviceProtection: int
    TechSupport: int
    StreamingTV: int
    StreamingMovies: int
    Contract: int
    PaperlessBilling: int
    PaymentMethod: int
    MonthlyCharges: float
    TotalCharges: float

@app.get("/")
def home():
    return {
        "message": "Welcome to Churn Prediction API!",
        "version": "1.0.0"
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "XGBoost Churn Predictor"
    }

@app.post("/predict")
def predict(customer: CustomerData):
    customer_df = pd.DataFrame([customer.dict()])
    prediction = model.predict(customer_df)[0]
    probability = model.predict_proba(customer_df)[0]
    churn_probability = round(float(probability[1]) * 100, 2)

    if churn_probability >= 70:
        risk_level = "HIGH RISK"
        action = "Immediately offer discount or better plan!"
    elif churn_probability >= 40:
        risk_level = "MEDIUM RISK"
        action = "Send personalized retention offer"
    else:
        risk_level = "LOW RISK"
        action = "Customer is happy — maintain service quality"

    return {
        "churn_prediction": bool(prediction),
        "churn_probability": f"{churn_probability}%",
        "risk_level": risk_level,
        "recommended_action": action
    }