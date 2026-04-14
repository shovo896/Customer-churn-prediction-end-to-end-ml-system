import mlflow
import dagshub

dagshub.init(
    repo_owner="shovo896",
    repo_name="Customer-chunk-prediction-end-to-end-ml-system",
    mlflow=True
)

client = mlflow.tracking.MlflowClient()
try:
    versions = client.get_latest_versions('Customer Churn Prediction Model', stages=['Production'])
    print(f'Found {len(versions)} production versions')
    if versions:
        print(f'Model URI: runs:/{versions[0].run_id}/model')
        print(f'Version: {versions[0].version}')
except Exception as e:
    print(f'Error: {type(e).__name__}: {e}')
