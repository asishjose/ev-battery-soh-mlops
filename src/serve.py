import mlflow
import mlflow.sklearn
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram, Gauge
from elasticsearch import Elasticsearch
from datetime import datetime, timezone
import time

# ── MLflow ────────────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI = "http://127.0.0.1:5001"
MODEL_URI           = "models:/battery-soh-predictor@production"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
model = mlflow.sklearn.load_model(MODEL_URI)

# ── Elasticsearch ─────────────────────────────────────────────────────────
es = Elasticsearch("http://localhost:9200")
ES_INDEX = "battery-predictions"

# ── App ───────────────────────────────────────────────────────────────────
app = FastAPI(
    title="EV Battery SoH Predictor",
    description="Predicts State-of-Health and Remaining Useful Life for EV batteries",
    version="1.0.0",
)

# ── Prometheus metrics ────────────────────────────────────────────────────
Instrumentator().instrument(app).expose(app)

prediction_counter = Counter(
    "battery_predictions_total",
    "Total number of SoH predictions made",
)
error_counter = Counter(
    "battery_prediction_errors_total",
    "Total number of failed predictions",
)
latency_histogram = Histogram(
    "battery_prediction_latency_seconds",
    "Time taken to generate a prediction",
)
soh_gauge = Gauge(
    "battery_soh_predicted",
    "Most recently predicted SoH value",
)
rul_gauge = Gauge(
    "battery_rul_predicted",
    "Most recently predicted RUL in cycles",
)

# ── Request schema ────────────────────────────────────────────────────────
class CycleFeatures(BaseModel):
    battery_id:                str   = "unknown"
    cycle_index:               int   = -1
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
    start = time.time()

    try:
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

        fade_rate_pct = abs(features.capacity_fade_rate / features.rolling_mean_capacity_10 * 100)
        if fade_rate_pct > 0:
            rul_cycles = int((soh_pct - 80.0) / fade_rate_pct)
            rul_cycles = max(rul_cycles, 0)
        else:
            rul_cycles = -1

        latency_ms = round((time.time() - start) * 1000, 2)

        # ── Prometheus ────────────────────────────────────────────────────
        prediction_counter.inc()
        soh_gauge.set(soh_pct)
        rul_gauge.set(rul_cycles)
        latency_histogram.observe(time.time() - start)

        # ── Elasticsearch log ─────────────────────────────────────────────
        doc = {
            "timestamp"              : datetime.now(timezone.utc).isoformat(),
            "battery_id"             : features.battery_id,
            "cycle_index"            : features.cycle_index,
            "soh_pct"                : round(soh_pct, 4),
            "rul_cycles"             : rul_cycles,
            "confidence"             : "high" if soh_pct > 85 else "medium" if soh_pct > 80 else "low",
            "latency_ms"             : latency_ms,
            "features": {
                "capacity_fade_rate"       : features.capacity_fade_rate,
                "rolling_mean_capacity_10" : features.rolling_mean_capacity_10,
                "rolling_std_capacity_10"  : features.rolling_std_capacity_10,
                "voltage_at_end"           : features.voltage_at_end,
                "mean_discharge_voltage"   : features.mean_discharge_voltage,
                "voltage_drop"             : features.voltage_drop,
                "mean_temperature"         : features.mean_temperature,
                "max_temperature"          : features.max_temperature,
                "discharge_duration"       : features.discharge_duration,
            }
        }
        es.index(index=ES_INDEX, document=doc)

        return {
            "soh_pct"   : round(soh_pct, 4),
            "rul_cycles": rul_cycles,
            "confidence": "high" if soh_pct > 85 else "medium" if soh_pct > 80 else "low",
        }

    except Exception as e:
        error_counter.inc()
        raise e