import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import dagshub
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

load_dotenv()


class FallbackChurnModel:
    """Simple fallback model to keep API responsive when remote model isn't available."""

    def predict_proba(self, X):
        n = len(X)
        probs = np.full((n, 2), 0.0)
        # Default low-risk probability
        probs[:, 1] = 0.2
        probs[:, 0] = 0.8
        return probs

    def predict(self, X):
        proba = self.predict_proba(X)[:, 1]
        return (proba >= 0.5).astype(int)


def _load_remote_model():
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
                run_name = run.data.tags.get("mlflow.runName", "")
                top_artifacts = client.list_artifacts(run.info.run_id)
                top_paths = [a.path for a in top_artifacts]

                # Training script logged models under run-name folder, not always under "model"
                candidate_paths = []
                if run_name:
                    candidate_paths.append(run_name)
                candidate_paths.extend(["model", "Logistic Regression", "LightGBM", "XGBoost"])

                model_subpath = next((p for p in candidate_paths if p in top_paths), None)
                if not model_subpath:
                    continue

                model_uri = f"runs:/{run.info.run_id}/{model_subpath}"
                model = mlflow.sklearn.load_model(model_uri)
                print(f"✓ Loaded model: {run.info.run_id[:8]}... ({model_subpath})")
                return model
            except Exception:
                continue
    return None

def load_production_model():
    """Load production model with timeout and fallback model."""
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_load_remote_model)
            model = future.result(timeout=20)
            if model is not None:
                return model
    except TimeoutError:
        print(" Remote model download timed out. Using fallback model.")
    except Exception:
        pass

    print(" Using fallback model for demo")
    return FallbackChurnModel()

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

        