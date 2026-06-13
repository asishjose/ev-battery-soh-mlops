import mlflow
import mlflow.sklearn
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

# ── MLflow ────────────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI = "http://127.0.0.1:5001"
MODEL_URI           = "models:/battery-soh-predictor@production"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
model = mlflow.sklearn.load_model(MODEL_URI)

# ── App ───────────────────────────────────────────────────────────────────
app = FastAPI(
    title="EV Battery SoH Predictor",
    description="Predicts State-of-Health and Remaining Useful Life for EV batteries",
    version="1.0.0",
)

# ── Request schema ────────────────────────────────────────────────────────
class CycleFeatures(BaseModel):
    capacity_fade_rate:        float
    rolling_mean_capacity_10:  float
    rolling_std_capacity_10:   float
    voltage_at_end:            float
    mean_discharge_voltage:    float
    voltage_drop:              float
    mean_temperature:          float
    max_temperature:           float
    discharge_duration:        float

# ── Routes ────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_URI}


@app.post("/predict")
def predict(features: CycleFeatures):
    X = np.array([[
        features.capacity_fade_rate,
        features.rolling_mean_capacity_10,
        features.rolling_std_capacity_10,
        features.voltage_at_end,
        features.mean_discharge_voltage,
        features.voltage_drop,
        features.mean_temperature,
        features.max_temperature,
        features.discharge_duration,
    ]])

    soh_pct = float(model.predict(X)[0])

    # RUL: linear extrapolation of cycles remaining before SoH hits 80%
    # fade_rate is Ah/cycle — convert to SoH%/cycle using original capacity assumption
    fade_rate_pct = abs(features.capacity_fade_rate / features.rolling_mean_capacity_10 * 100)
    if fade_rate_pct > 0:
        rul_cycles = int((soh_pct - 80.0) / fade_rate_pct)
        rul_cycles = max(rul_cycles, 0)
    else:
        rul_cycles = -1  # degradation rate too flat to estimate

    return {
        "soh_pct":    round(soh_pct, 4),
        "rul_cycles": rul_cycles,
        "confidence": "high" if soh_pct > 85 else "medium" if soh_pct > 80 else "low",
    }