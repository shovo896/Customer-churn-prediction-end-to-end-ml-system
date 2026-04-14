import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import dagshub
import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

load_dotenv()

dagshub.init(
    repo_owner="shovo896",
    repo_name="Customer-chunk-prediction-end-to-end-ml-system",
    mlflow=True
)   
def load_production_model():
    client=mlflow.tracking.MlflowClient()
    versions=client.get_latest_versions("Customer Churn Prediction Model", stages=["Production"])
    if not versions:
        raise RuntimeError("No production model found. Please run training and register a model first.")
    model_uri=f"runs:/{versions[0].run_id}/model"
    model=mlflow.sklearn.load_model(model_uri)
    print(f"Loaded model from {model_uri}")
    return model
model=load_production_model()

## fastapi app
app=FastAPI(
    title="Customer Churn Prediction API",
    description="API for predicting customer churn using the trained model.",
    version="1.0.0"
)

## input schema
class CustomerData(BaseModel):
    tenure:float
    MonthlyCharges:float
    TotalCharges:float
    SeniorCitizen:int
    Partner:int 
    gender:int
    Dependents:int
    PhoneService:int
    PaperlessBilling:int
    charge_ratio:float
    
    class Config:
        json_schema_extra={
            "example":{
                "tenure": 12,
                "MonthlyCharges": 70.35,
                "TotalCharges": 845.5,
                "SeniorCitizen": 0,
                "Partner": 1,
                "gender": 0,
                "Dependents": 0,
                "PhoneService": 1,
                "PaperlessBilling": 1,
                "charge_ratio": 0.083
            }
        }
        
        
# output schema
class PredictionResponse(BaseModel):
    churn_probability: float
    churn_prediction: int
    risk_level: str
    message: str
    
    
# routes 
    
@app.get("/")
def root():
    return {
        "message": "Welcome to the Customer Churn Prediction API. Use the /predict endpoint to get predictions."
        "docs" : "/docs"
    }
    
@app.get("/health")
def health():
    return {"status": "ok", "message": "API is healthy and ready to serve predictions."}

