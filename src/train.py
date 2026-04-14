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
