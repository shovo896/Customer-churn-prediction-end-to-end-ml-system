import pandas as pd 
import numpy as np
import os
import mlflow
import mlflow.sklearn
import dagshub
import optuna
import yaml
from dotenv import load_dotenv
from pathlib import Path

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score 
from sklearn.model_selection import cross_val_score

load_dotenv()
##dagshub + mlflow connect``

dagshub.init(
    repo_owner="shovo896",
    repo_name="Customer-chunk-prediction-end-to-end-ml-system",
    mlflow=True
)

def load_data():
    X_train=pd.read_csv("data/processed/processed_data_X_train.csv")
    X_test=pd.read_csv("data/processed/processed_data_X_test.csv")
    y_train=pd.read_csv("data/processed/processed_data_y_train.csv")
    y_test=pd.read_csv("data/processed/processed_data_y_test.csv")
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = load_data()

def xgb_objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "random_state": 42,
    }
    model = XGBClassifier(**params)
    scores = cross_val_score(model, X_train, y_train.values.ravel(), cv=5, scoring="roc_auc").mean()
    
    with mlflow.start_run(run_name=f"XGB Trial {trial.number}",nested=True):
        mlflow.log_params(params)
        mlflow.log_metric("roc_auc", round(scores, 4))
    return scores.mean()

## tune and save best model 
def tune(objective, model_name,n_trials=50):
    print(f"Starting hyperparameter tuning for {model_name} with {n_trials} trials...")
    with mlflow.start_run(run_name=f"{model_name} Hyperparameter Tuning"):
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials,show_progress_bar=True)
        
        best_params=study.best_params
        best_score=round(study.best_value,4)
        mlflow.log_params(best_params)
        mlflow.log_metric("best_roc_auc", best_score)
        print(f"Best ROC AUC for {model_name}: {best_score} with params: {best_params}")
    return study.best_params


if __name__ == "__main__":
    mlflow.set_experiment("Customer Churn Prediction")
    xgb_best_params = tune(xgb_objective, "XGBoost", n_trials=50)
    best_lgbm_params = {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
    }
    with open("best_lgbm_params.yaml", "w") as f:
        yaml.dump(best_lgbm_params, f)  
    print(f"Best LightGBM params saved to best_lgbm_params.yaml: {best_lgbm_params}")