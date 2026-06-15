import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from mlflow import MlflowClient
from pathlib import Path
import sys

MLFLOW_TRACKING_URI = "http://127.0.0.1:5001"
MODEL_NAME          = "battery-soh-predictor"
PROCESSED_DIR       = Path("data/processed")
FEATURE_COLS = [
    "capacity_fade_rate", "rolling_mean_capacity_10", "rolling_std_capacity_10",
    "voltage_at_end", "mean_discharge_voltage", "voltage_drop",
    "mean_temperature", "max_temperature", "discharge_duration",
]

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


def get_production_rmse(client: MlflowClient) -> float | None:
    try:
        v = client.get_model_version_by_alias(MODEL_NAME, "production")
        run = mlflow.get_run(v.run_id)
        return float(run.data.metrics["rmse"])
    except Exception:
        return None


def train_and_evaluate() -> tuple[float, float, float, object, mlflow.ActiveRun]:
    train_df = pd.read_csv(PROCESSED_DIR / "train.csv")
    test_df  = pd.read_csv(PROCESSED_DIR / "test.csv")
    X_train, y_train = train_df[FEATURE_COLS], train_df["soh"]
    X_test,  y_test  = test_df[FEATURE_COLS],  test_df["soh"]

    with mlflow.start_run(run_name="rf-retrain") as run:
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        rmse  = float(np.sqrt(mean_squared_error(y_test, preds)))
        mae   = float(mean_absolute_error(y_test, preds))
        r2    = float(r2_score(y_test, preds))

        mlflow.log_params({
            "model_type"      : "RandomForestRegressor",
            "n_estimators"    : 100,
            "train_batteries" : "B0005,B0006,B0007",
            "test_battery"    : "B0018",
        })
        mlflow.log_metrics({"rmse": rmse, "mae": mae, "r2": r2})
        mlflow.sklearn.log_model(
            model,
            name="model",
            registered_model_name=MODEL_NAME,
        )

        return rmse, mae, r2, model, run


def evaluate_and_register():
    client = MlflowClient()

    prod_rmse = get_production_rmse(client)
    print(f"Production RMSE : {prod_rmse:.4f}%" if prod_rmse else "No production model found — will register directly")

    new_rmse, mae, r2, model, run = train_and_evaluate()
    print(f"New model RMSE  : {new_rmse:.4f}%")
    print(f"New model MAE   : {mae:.4f}%")
    print(f"New model R²    : {r2:.4f}")

    # Quality gate
    if prod_rmse is None or new_rmse < prod_rmse:
        # Get the version that was just registered
        versions = client.search_model_versions(f"name='{MODEL_NAME}'")
        latest   = max(versions, key=lambda v: int(v.version))

        client.set_registered_model_alias(MODEL_NAME, "production", latest.version)
        print(f"\nQuality gate PASSED — new RMSE {new_rmse:.4f}% < prod RMSE {prod_rmse:.4f}%") if prod_rmse else print(f"\nNo previous model — registering directly")
        print(f"Promoted {MODEL_NAME} v{latest.version} → @production")
    else:
        print(f"\nQuality gate FAILED — new RMSE {new_rmse:.4f}% >= prod RMSE {prod_rmse:.4f}%")
        print("Production model unchanged")
        sys.exit(1)


if __name__ == "__main__":
    evaluate_and_register()