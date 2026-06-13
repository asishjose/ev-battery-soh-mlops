import mlflow
from mlflow import MlflowClient

MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

client = MlflowClient()

client.set_registered_model_alias(
    name="battery-soh-predictor",
    alias="production",
    version="1",
)

v = client.get_model_version_by_alias("battery-soh-predictor", "production")
print(f"Registered : battery-soh-predictor v{v.version}")
print(f"Alias      : @production → v{v.version}")
print(f"Run ID     : {v.run_id[:8]}...")
print(f"\nLoad URI for FastAPI: models:/battery-soh-predictor@production")