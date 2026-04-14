import pandas as pd
import numpy as np
import os
from pathlib import Path
import mlflow 
import mlflow.sklearn
from dotenv import load_dotenv
import yaml 
import dagshub


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score,classification_report
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
load_dotenv()

## dagshub connect 
dagshub.init(
    repo_owner="shovo896",
    repo_name="Customer-chunk-prediction-end-to-end-ml-system",
    mlflow=True
)

## data load 
def load_data():
    processed_dir = Path("data/processed")

    candidates = {
        "X_train": [processed_dir / "processed_data_X_train.csv", processed_dir / "processed_data.csv_X_train.csv"],
        "X_test": [processed_dir / "processed_data_X_test.csv", processed_dir / "processed_data.csv_X_test.csv"],
        "y_train": [processed_dir / "processed_data_y_train.csv", processed_dir / "processed_data.csv_y_train.csv"],
        "y_test": [processed_dir / "processed_data_y_test.csv", processed_dir / "processed_data.csv_y_test.csv"],
    }

    resolved_paths = {}
    for name, options in candidates.items():
        found = next((p for p in options if p.exists()), None)
        if found is None:
            raise FileNotFoundError(
                f"Missing processed file for {name}. Checked: {[str(p) for p in options]}. "
                "Run `python src/feature_engineering.py` first."
            )
        resolved_paths[name] = found

    X_train = pd.read_csv(resolved_paths["X_train"])
    X_test = pd.read_csv(resolved_paths["X_test"])
    y_train = pd.read_csv(resolved_paths["y_train"])
    y_test = pd.read_csv(resolved_paths["y_test"])
    return X_train, X_test, y_train, y_test

# metrics helper 
def get_metrics(y_true, y_pred,y_prob):
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "f1_score": round(f1_score(y_true, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_true, y_prob), 4),
        
    }
    
    
## single train run 
def train_model(model,model_name,params,X_train,X_test,y_train,y_test):
    with mlflow.start_run(run_name=model_name):
        mlflow.log_params(params)
        model.fit(X_train, y_train.values.ravel())
        y_pred=model.predict(X_test)
        y_prob=model.predict_proba(X_test)[:,1]
        metrics=get_metrics(y_test,y_pred,y_prob)
        mlflow.log_metrics(metrics)
        mlflow.log_params(params)
        mlflow.sklearn.log_model(model,artifact_path=model_name)
        
        print(f"Model: {model_name}")
        print(f"Metrics: {metrics}")
        print(f"accuracy: {metrics['accuracy']}, f1_score: {metrics['f1_score']}, roc_auc: {metrics['roc_auc']}")
        print(f"{'='*30}")
        return metrics
    
if __name__ == "__main__":
    with open("params.yaml") as f:
        params=yaml.safe_load(f)
        
    X_train,X_test,y_train,y_test=load_data()
    mlflow.set_experiment("Customer Churn Prediction")
    
    ## model 1 
    lr_params=params["logistic_regression"]
    train_model(
        LogisticRegression(**lr_params),
        "Logistic Regression",
        lr_params,
        X_train,X_test,y_train,y_test
    )
    # model 2 
    lightgbm_params=params["lightgbm"]
    train_model(
        LGBMClassifier(**lightgbm_params),
        "LightGBM",
        lightgbm_params,
        X_train,X_test,y_train,y_test
    )
    # model 3 
    xgboost_params=params["xgboost"]
    train_model(
        XGBClassifier(**xgboost_params),
        "XGBoost",
        xgboost_params,
        X_train,X_test,y_train,y_test
    )
    
    
        