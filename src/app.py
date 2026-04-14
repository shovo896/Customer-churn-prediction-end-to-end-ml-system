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
    try:
        # Try to load from model registry
        versions=client.get_latest_versions("Customer Churn Prediction Model", stages=["Production"])
        if versions:
            model_uri=f"runs:/{versions[0].run_id}/model"
            model=mlflow.sklearn.load_model(model_uri)
            print(f"Loaded model from registry: {model_uri}")
            return model
    except Exception as e:
        print(f"Could not load from registry: {e}")
    
    # Fallback: Load from latest run
    try:
        runs=client.search_runs(experiment_ids=["0"], max_results=1)
        if runs:
            model_uri=f"runs:/{runs[0].info.run_id}/model"
            model=mlflow.sklearn.load_model(model_uri)
            print(f"Loaded model from latest run: {model_uri}")
            return model
    except Exception as e:
        print(f"Could not load from latest run: {e}")
    
    raise RuntimeError("No production model found. Please run training and register a model first.")

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
        "message": "Welcome to the Customer Churn Prediction API. Use the /predict endpoint to get predictions.",
        "docs": "/docs"
    }
    
@app.get("/health")
def health():
    return {"status": "ok", "message": "API is healthy and ready to serve predictions."}


@app.post("/predict", response_model=PredictionResponse)
def predict(data: CustomerData):
    try: 
        input_df=pd.DataFrame([data.model_dump()])
        prediction_prob=model.predict(input_df)[0]
        probability=round(float(model.predict_proba(input_df)[0][1]),4)
        risk_level="High" if probability > 0.7 else "Medium" if probability > 0.4 else "Low"
        return PredictionResponse(
            churn_probability=probability,
            churn_prediction=int(prediction_prob),
            risk_level=risk_level,
            message="Prediction successful"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
@app.post("/predict_batch")
def predict_batch(data: list[CustomerData]):
    try:
        input_df=pd.DataFrame([item.model_dump() for item in data])
        predictions=model.predict(input_df)
        probabilities=model.predict_proba(input_df)[:,1]
        results=[]
        for pred, prob in zip(predictions, probabilities):
            risk_level="High" if prob > 0.7 else "Medium" if prob > 0.4 else "Low"
            results.append({
                "churn_probability": round(float(prob),4),
                "churn_prediction": int(pred),
                "risk_level": risk_level
            })
        return {"predictions": results, "message": "Batch prediction successful"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000,reload=True)
        