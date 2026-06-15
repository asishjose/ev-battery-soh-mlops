import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
#updated

PROCESSED_DIR = Path("data/processed")
FEATURE_COLS = [
    "capacity_fade_rate",
    "rolling_mean_capacity_10",
    "rolling_std_capacity_10",
    "voltage_at_end",
    "mean_discharge_voltage",
    "voltage_drop",
    "mean_temperature",
    "max_temperature",
    "discharge_duration",
]

def load_splits():
    train_df = pd.read_csv(PROCESSED_DIR / "train.csv")
    test_df  = pd.read_csv(PROCESSED_DIR / "test.csv")
    X_train  = train_df[FEATURE_COLS]
    y_train  = train_df["soh"]
    X_test   = test_df[FEATURE_COLS]
    y_test   = test_df["soh"]
    return X_train, y_train, X_test, y_test


def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    rmse  = np.sqrt(mean_squared_error(y_test, preds))
    mae   = mean_absolute_error(y_test, preds)
    r2    = r2_score(y_test, preds)
    return rmse, mae, r2, preds


def train():
    X_train, y_train, X_test, y_test = load_splits()

    #mlflow.set_tracking_uri(uri="http://127.0.0.1:5000/")
    mlflow.set_experiment("battery-soh-baseline")

    with mlflow.start_run(run_name="rf-baseline"):
        params = {
            "model_type":   "RandomForestRegressor",
            "n_estimators": 100,
            "max_depth":    None,
            "train_batteries": "B0005,B0006,B0007",
            "test_battery":    "B0018",
        }

        model = RandomForestRegressor(
            n_estimators=params["n_estimators"],
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        rmse, mae, r2, _ = evaluate(model, X_test, y_test)

        mlflow.log_params(params)
        mlflow.log_metrics({"rmse": rmse, "mae": mae, "r2": r2})
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name="battery-soh-predictor",
        )

        print(f"RMSE : {rmse:.4f}%")
        print(f"MAE  : {mae:.4f}%")
        print(f"R²   : {r2:.4f}")
        print(f"Model registered as battery-soh-predictor v1")


if __name__ == "__main__":
    train()