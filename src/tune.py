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
