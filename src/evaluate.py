import json
import os
from typing import Any, cast

import dagshub
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

load_dotenv()
MODEL_RUN_NAMES = {"Logistic Regression", "LightGBM", "XGBoost"}

## dagshub connect
dagshub.init(
    repo_owner="shovo896",
    repo_name="Customer-chunk-prediction-end-to-end-ml-system",
    mlflow=True
)


def load_data():
    X_test = pd.read_csv("data/processed/processed_data_X_test.csv")
    y_test = pd.read_csv("data/processed/processed_data_y_test.csv")
    return X_test, y_test


def evaluate_and_register():
    X_test, y_test = load_data()
    y_test_series = y_test.values.ravel()

    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name("Customer Churn Prediction")
    if exp is None:
        raise RuntimeError("Experiment 'Customer Churn Prediction' not found. Run training first.")

    all_runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["metrics.roc_auc DESC"],
        max_results=500,
    )
    if not all_runs:
        raise RuntimeError("No runs found in experiment 'Customer Churn Prediction'.")

    model_runs = [
        run for run in all_runs
        if run.data.tags.get("mlflow.runName") in MODEL_RUN_NAMES
        and "roc_auc" in run.data.metrics
    ]
    if not model_runs:
        raise RuntimeError(
            "No trained model runs found (expected run names: Logistic Regression, LightGBM, XGBoost)."
        )

    best_run = max(model_runs, key=lambda run: run.data.metrics.get("roc_auc", float("-inf")))
    best_run_id = best_run.info.run_id
    best_roc_auc = best_run.data.metrics.get("roc_auc")
    best_model_name = best_run.data.tags.get("mlflow.runName", "Best Model")

    print(f"Best Run ID: {best_run_id}")
    print(f"Best ROC AUC: {best_roc_auc}")
    print(f"Best Model Name: {best_model_name}")

    model_uri = f"runs:/{best_run_id}/{best_model_name}"
    model = cast(Any, mlflow.sklearn.load_model(model_uri))

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    final_metrics = {
        "accuracy": float(round(float(accuracy_score(y_test_series, y_pred)), 4)),
        "f1_score": float(round(float(f1_score(y_test_series, y_pred)), 4)),
        "roc_auc": float(round(float(roc_auc_score(y_test_series, y_prob)), 4)),
    }

    print(f"Final Evaluation Metrics: {final_metrics}")
    for k, v in final_metrics.items():
        print(f"{k}: {v}")

    os.makedirs("evaluation_results", exist_ok=True)
    cm = confusion_matrix(y_test_series, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Not Churn", "Churn"])
    disp.plot(cmap="Blues")
    plt.title(f"Confusion Matrix - {best_model_name}")
    safe_name = best_model_name.replace(" ", "_")
    cm_path = f"evaluation_results/confusion_matrix_{safe_name}.png"
    metrics_path = f"evaluation_results/metrics_{safe_name}.json"
    plt.savefig(cm_path)
    plt.close()

    ## metrics save as json
    with open(metrics_path, "w") as f:
        json.dump(final_metrics, f, indent=4)

    print(f"Evaluation results saved to evaluation_results/ directory.")

    ## mlflow register model
    with mlflow.start_run(run_name="final_evaluation"):
        mlflow.log_metrics(final_metrics)
        mlflow.log_artifact(cm_path)
        mlflow.log_artifact(metrics_path)
        registered=mlflow.register_model(model_uri, "CustomerChurnModel")

        client.transition_model_version_stage(
            name="CustomerChurnModel",
            version=registered.version,
            stage="Production",
            archive_existing_versions=True
        )
        print(f"Model registered in MLflow Model Registry with version: {registered.version} and transitioned to Production stage.")
        print(f"Evaluation and model registration completed successfully.")


if __name__ == "__main__":
    evaluate_and_register()
