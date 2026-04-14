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