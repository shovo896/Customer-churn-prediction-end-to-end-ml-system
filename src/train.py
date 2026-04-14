import pandas as pd
import numpy as np
import os
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
    X_train=pd.read_csv("data/processed/processed_data_X_train.csv")
    X_test=pd.read_csv("data/processed/processed_data_X_test.csv")
    y_train=pd.read_csv("data/processed/processed_data_y_train.csv")
    y_test=pd.read_csv("data/processed/processed_data_y_test.csv")
    return X_train,X_test,y_train,y_test

# metrics helper 
def get_metrics(y_true, y_pred,y_prob):
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "f1_score": round(f1_score(y_true, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_true, y_prob), 4),
        
    }
    
    
## single train run 