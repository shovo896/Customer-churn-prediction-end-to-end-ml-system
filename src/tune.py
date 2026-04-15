import pandas as pd 
import mlflow
import mlflow.sklearn
import dagshub
import optuna
import yaml
from typing import Any
from dotenv import load_dotenv

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
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
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
        "random_state": 42,
        "n_jobs": -1,
        "eval_metric": "auc",
        "verbosity": 0,
    }
    model = XGBClassifier(**params)
    scores = cross_val_score(model, X_train, y_train.values.ravel(), cv=5, scoring="roc_auc").mean()  # type: ignore[arg-type]
    
    with mlflow.start_run(run_name=f"XGB Trial {trial.number}",nested=True):
        mlflow.log_params(params)
        mlflow.log_metric("roc_auc", round(scores, 4))
    return float(scores)


def lgbm_objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", -1, 12),
        "num_leaves": trial.suggest_int("num_leaves", 20, 150),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }
    model: Any = LGBMClassifier(**params)
    scores = cross_val_score(model, X_train, y_train.values.ravel(), cv=5, scoring="roc_auc").mean()

    with mlflow.start_run(run_name=f"LGBM Trial {trial.number}", nested=True):
        mlflow.log_params(params)
        mlflow.log_metric("roc_auc", round(scores, 4))
    return float(scores)

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
    return best_params, best_score


if __name__ == "__main__":
    mlflow.set_experiment("Customer Churn Prediction")
    xgb_best_params, xgb_best_score = tune(xgb_objective, "XGBoost", n_trials=50)
    lgbm_best_params, lgbm_best_score = tune(lgbm_objective, "LightGBM", n_trials=50)

    with open("best_xgb_params.yaml", "w") as f:
        yaml.dump(xgb_best_params, f)

    with open("best_lgbm_params.yaml", "w") as f:
        yaml.dump(lgbm_best_params, f)

    print(f"Best XGBoost params saved to best_xgb_params.yaml (ROC-AUC: {xgb_best_score}): {xgb_best_params}")
    print(f"Best LightGBM params saved to best_lgbm_params.yaml (ROC-AUC: {lgbm_best_score}): {lgbm_best_params}")