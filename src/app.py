import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import dagshub
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from sklearn.linear_model import LogisticRegression

load_dotenv()

def load_production_model():
    """Load production model with fallback to dummy model."""
    try:
        dagshub.init(
            repo_owner="shovo896",
            repo_name="Customer-chunk-prediction-end-to-end-ml-system",
            mlflow=True
        )
        client = mlflow.tracking.MlflowClient()
        runs = client.search_runs(experiment_ids=["0"], max_results=10)
        for run in runs:
            if run.info.status == "FINISHED":
                try:
                    model_uri = f"runs:/{run.info.run_id}/model"
                    model = mlflow.sklearn.load_model(model_uri)
                    print(f"✓ Loaded model: {run.info.run_id[:8]}...")
                    return model
                except Exception:
                    continue
    except Exception:
        pass
    
    print("⚠ Using dummy model for demo")
    return LogisticRegression(random_state=42)

model = load_production_model()

app = FastAPI(
    title="Customer Churn Prediction API",
    description="API for predicting customer churn.",
    version="1.0.0"
)

class CustomerData(BaseModel):
    tenure: float
    MonthlyCharges: float
    TotalCharges: float
    SeniorCitizen: int
    Partner: int 
    gender: int
    Dependents: int
    PhoneService: int
    PaperlessBilling: int
    charge_ratio: float

class PredictionResponse(BaseModel):
    churn_probability: float
    churn_prediction: int
    risk_level: str
    message: str

@app.get("/")
def root():
    return {
        "message": "Welcome to Customer Churn Prediction API",
        "docs": "/docs"
    }
    
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionResponse)
def predict(data: CustomerData):
    try: 
        input_df = pd.DataFrame([data.model_dump()])
        prediction_prob = model.predict(input_df)[0]
        probability = round(float(model.predict_proba(input_df)[0][1]), 4)
        risk_level = "High" if probability > 0.7 else "Medium" if probability > 0.4 else "Low"
        return PredictionResponse(
            churn_probability=probability,
            churn_prediction=int(prediction_prob),
            risk_level=risk_level,
            message="Prediction successful"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_batch")
def predict_batch(data: list[CustomerData]):
    try:
        input_df = pd.DataFrame([item.model_dump() for item in data])
        predictions = model.predict(input_df)
        probabilities = model.predict_proba(input_df)[:, 1]
        results = []
        for pred, prob in zip(predictions, probabilities):
            risk_level = "High" if prob > 0.7 else "Medium" if prob > 0.4 else "Low"
            results.append({
                "churn_probability": round(float(prob), 4),
                "churn_prediction": int(pred),
                "risk_level": risk_level
            })
        return {"predictions": results, "message": "Batch prediction successful"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("Starting API...")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)

        