import pandas as pd
import numpy as np
import mlflow
import dagshub
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

load_dotenv()

PROCESSED_X_TRAIN = Path("data/processed/processed_data_X_train.csv")


def load_expected_columns():
    try:
        if PROCESSED_X_TRAIN.exists():
            return pd.read_csv(PROCESSED_X_TRAIN, nrows=1).columns.tolist()
    except Exception:
        pass
    return None


EXPECTED_COLUMNS = load_expected_columns()


def get_model_expected_columns(loaded_model):
    cols = getattr(loaded_model, "feature_names_in_", None)
    if cols is None:
        return None
    return [str(c) for c in cols]


def prepare_features(input_df: pd.DataFrame) -> pd.DataFrame:
    """Map API input into the training feature schema."""
    if not EXPECTED_COLUMNS:
        return input_df

    X = pd.DataFrame(0.0, index=input_df.index, columns=EXPECTED_COLUMNS)

    # Copy direct columns that already match training features
    for col in input_df.columns:
        if col in X.columns:
            X[col] = input_df[col]

    # Derive tenure group one-hot columns (same idea as training)
    if "tenure" in input_df.columns:
        tenure = input_df["tenure"].astype(float)
        if "tenure_group_13-24" in X.columns:
            X.loc[(tenure >= 13) & (tenure <= 24), "tenure_group_13-24"] = 1.0
        if "tenure_group_25-48" in X.columns:
            X.loc[(tenure >= 25) & (tenure <= 48), "tenure_group_25-48"] = 1.0
        if "tenure_group_49-60" in X.columns:
            X.loc[(tenure >= 49) & (tenure <= 60), "tenure_group_49-60"] = 1.0
        if "tenure_group_60+" in X.columns:
            X.loc[tenure > 60, "tenure_group_60+"] = 1.0

    # Ensure all are numeric
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return X


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
    disable_remote = os.getenv("DISABLE_REMOTE_MODEL", "").strip().lower()
    if disable_remote in {"1", "true", "yes"}:
        print("Remote model loading disabled by DISABLE_REMOTE_MODEL. Using fallback model.")
        return FallbackChurnModel()

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
if not EXPECTED_COLUMNS:
    EXPECTED_COLUMNS = get_model_expected_columns(model)

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
        "docs": "http://127.0.0.1:8001/docs"
    }
    
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionResponse)
def predict(data: CustomerData):
    try: 
        input_df = pd.DataFrame([data.model_dump()])
        model_input = prepare_features(input_df)
        prediction_prob = model.predict(model_input)[0]
        probability = round(float(model.predict_proba(model_input)[0][1]), 4)
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
        model_input = prepare_features(input_df)
        predictions = model.predict(model_input)
        probabilities = model.predict_proba(model_input)[:, 1]
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
    uvicorn.run(
        app,
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8001")),
        reload=False,
    )

        