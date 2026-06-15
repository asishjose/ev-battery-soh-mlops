import pandas as pd
import sys
from pathlib import Path

PROCESSED_DIR = Path("data/processed")
EXPECTED_FEATURES = [
    "capacity_fade_rate", "rolling_mean_capacity_10", "rolling_std_capacity_10",
    "voltage_at_end", "mean_discharge_voltage", "voltage_drop",
    "mean_temperature", "max_temperature", "discharge_duration",
]
MIN_TRAIN_ROWS = 400
MIN_TEST_ROWS  = 100


def validate():
    errors = []

    # Check files exist
    for fname in ["train.csv", "test.csv"]:
        path = PROCESSED_DIR / fname
        if not path.exists():
            errors.append(f"Missing file: {path}")

    if errors:
        for e in errors: print(f"ERROR: {e}")
        sys.exit(1)

    train_df = pd.read_csv(PROCESSED_DIR / "train.csv")
    test_df  = pd.read_csv(PROCESSED_DIR / "test.csv")

    # Check row counts
    if len(train_df) < MIN_TRAIN_ROWS:
        errors.append(f"Train set too small: {len(train_df)} rows (min {MIN_TRAIN_ROWS})")
    if len(test_df) < MIN_TEST_ROWS:
        errors.append(f"Test set too small: {len(test_df)} rows (min {MIN_TEST_ROWS})")

    # Check required columns exist
    for col in EXPECTED_FEATURES + ["soh", "battery_id"]:
        if col not in train_df.columns:
            errors.append(f"Missing column in train: {col}")
        if col not in test_df.columns:
            errors.append(f"Missing column in test: {col}")

    # Check no nulls in features
    for col in EXPECTED_FEATURES:
        if col in train_df.columns and train_df[col].isnull().any():
            errors.append(f"Nulls found in train column: {col}")
        if col in test_df.columns and test_df[col].isnull().any():
            errors.append(f"Nulls found in test column: {col}")

    # Check SoH range is sensible
    if train_df["soh"].min() < 50 or train_df["soh"].max() > 101:
        errors.append(f"SoH range suspicious: {train_df['soh'].min():.1f}–{train_df['soh'].max():.1f}")

    # Check battery-level split is correct
    train_batteries = set(train_df["battery_id"].unique())
    test_batteries  = set(test_df["battery_id"].unique())
    if train_batteries & test_batteries:
        errors.append(f"Data leakage: batteries in both train and test: {train_batteries & test_batteries}")

    if errors:
        for e in errors: print(f"ERROR: {e}")
        sys.exit(1)

    print(f"Validation passed:")
    print(f"  Train: {len(train_df)} rows | batteries: {sorted(train_batteries)}")
    print(f"  Test : {len(test_df)} rows  | batteries: {sorted(test_batteries)}")
    print(f"  SoH range: {train_df['soh'].min():.1f}% – {train_df['soh'].max():.1f}%")
    print(f"  No nulls, no leakage, all features present")


if __name__ == "__main__":
    validate()