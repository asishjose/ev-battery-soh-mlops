import scipy.io
import numpy as np
import pandas as pd
from pathlib import Path


def load_mat_file(filepath: str) -> dict:
    mat = scipy.io.loadmat(filepath, simplify_cells=True)
    battery_id = Path(filepath).stem          # e.g. "B0005"
    cycles = mat[battery_id]["cycle"]         # array of all cycles
    return battery_id, cycles


def extract_discharge_cycles(battery_id: str, cycles: np.ndarray) -> list[dict]:
    discharge_cycles = []

    for i, cycle in enumerate(cycles):
        if cycle["type"] != "discharge":
            continue

        data = cycle["data"]
        discharge_cycles.append({
            "cycle_index":   i,
            "battery_id":    battery_id,
            "capacity":      float(data["Capacity"]),
            "voltage":       np.array(data["Voltage_measured"]),
            "temperature":   np.array(data["Temperature_measured"]),
            "time":          np.array(data["Time"]),
        })

    return discharge_cycles

def engineer_features(discharge_cycles: list[dict]) -> pd.DataFrame:
    records = []
    capacities = [c["capacity"] for c in discharge_cycles]
    original_capacity = capacities[0]

    for i, cycle in enumerate(discharge_cycles):
        cap = cycle["capacity"]
        voltage = cycle["voltage"]
        temp = cycle["temperature"]
        time = cycle["time"]

        # Rolling window features (use available history if i < 10)
        window = capacities[max(0, i - 10):i + 1]
        rolling_mean = np.mean(window)
        rolling_std  = np.std(window) if len(window) > 1 else 0.0

        # Capacity fade rate: slope over last 10 cycles
        if len(window) >= 2:
            x = np.arange(len(window))
            fade_rate = np.polyfit(x, window, 1)[0]
        else:
            fade_rate = 0.0

        records.append({
            "cycle_index":              cycle["cycle_index"],
            "battery_id":               cycle["battery_id"],
            "capacity":                 cap,
            "soh":                      (cap / original_capacity) * 100,
            "capacity_fade_rate":       fade_rate,
            "rolling_mean_capacity_10": rolling_mean,
            "rolling_std_capacity_10":  rolling_std,
            "voltage_at_end":           float(voltage[-1]),
            "mean_discharge_voltage":   float(np.mean(voltage)),
            "voltage_drop":             float(voltage[0] - voltage[-1]),
            "mean_temperature":         float(np.mean(temp)),
            "max_temperature":          float(np.max(temp)),
            "discharge_duration":       float(time[-1] - time[0]),
        })

    return pd.DataFrame(records)

if __name__ == "__main__":
    battery_id, cycles = load_mat_file("data/raw/B0005.mat")
    discharge = extract_discharge_cycles(battery_id, cycles)
    df = engineer_features(discharge)

    print(df.shape)
    print(df[["cycle_index", "capacity", "soh", "capacity_fade_rate"]].head(10))
    print(f"\nSoH range: {df['soh'].min():.2f}% — {df['soh'].max():.2f}%")