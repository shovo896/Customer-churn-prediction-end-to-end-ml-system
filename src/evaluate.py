import pandas as pd 
import numpy as np
import os
import mlflow
import mlflow.sklearn
from dotenv import load_dotenv
import dagshub
import json 
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score,classification_report,ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from pathlib import Path

load_dotenv()
## dagshub connect
dagshub.init(
    repo_owner="shovo896",
    repo_name="Customer-chunk-prediction-end-to-end-ml-system",
    mlflow=True
)
def load_data():
    X_test=pd.read_csv("data/processed/processed_data_X_test.csv")
    y_test=pd.read_csv("data/processed/processed_data_y_test.csv")
    return X_test, y_test
def evaluate_and_register():
    X_test,y_test=load_data()
    client=mlflow.tracking.MlflowClient()
    exp=client.get_experiment_by_name("XGB Hyperparameter Tuning")
    runs=client.search_runs(experiment_ids=[exp.experiment_id],order_by=["metrics.roc_auc DESC"])
    best_run=runs[0]
    best_run_id=best_run.info.run_id
    best_roc_auc=best_run.data.metrics["roc_auc"]
    best_model_name=best_run.data.tags.get("mlflow.runName", "Best Model")
    print(f"Best Run ID: {best_run_id}")
    print(f"Best ROC AUC: {best_roc_auc}")
    print(f"Best Model Name: {best_model_name}")
    
    model_uri=f"\runs:/{best_run_id}/{best_model_name}"
    model=mlflow.sklearn.load_model(model_uri)
    
    y_pred=model.predict(X_test)
    y_prob=model.predict_proba(X_test)[:,1]
    
    final_metrics={
        "accuracy": float(round(float(accuracy_score(y_test, y_pred)), 4)),
        "f1_score": float(round(float(f1_score(y_test, y_pred)), 4)),
        "roc_auc": float(round(float(roc_auc_score(y_test, y_prob)), 4)),
    }
    
print(f"Final Evaluation Metrics: {final_metrics}")
for k,v in final_metrics.items():
    print(f"{k}: {v}")
    os.makedirs("evaluation_results", exist_ok=True)
    cm=confusion_matrix(y_test, y_pred)
    disp=ConfusionMatrixDisplay(cm,display_labels=["Not Churn","Churn"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix - {best_model_name}")
    plt.savefig(f"evaluation_results/confusion_matrix_{best_model_name}.png")
    plt.close()
    
    ## metrics save as json 
    with open(f"evaluation_results/metrics_{best_model_name}.json","w") as f:
        json.dump(final_metrics,f,indent=4)
    print(f"Evaluation results saved to evaluation_results/ directory.")
    
    
    
    
    