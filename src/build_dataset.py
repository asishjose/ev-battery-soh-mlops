import pandas as pd
from pathlib import Path
from data_loader import load_mat_file, extract_discharge_cycles, engineer_features

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
TRAIN_BATTERIES = ["B0005", "B0006", "B0007"]
TEST_BATTERY    = "B0018"


def build_dataset():
    all_dfs = []

    for battery_id in TRAIN_BATTERIES + [TEST_BATTERY]:
        path = RAW_DIR / f"{battery_id}.mat"
        bid, cycles = load_mat_file(str(path))
        discharge   = extract_discharge_cycles(bid, cycles)
        df          = engineer_features(discharge)
        all_dfs.append(df)
        print(f"{battery_id}: {len(df)} cycles, SoH {df['soh'].min():.1f}%–{df['soh'].max():.1f}%")

    full_df = pd.concat(all_dfs, ignore_index=True)

    train_df = full_df[full_df["battery_id"].isin(TRAIN_BATTERIES)]
    test_df  = full_df[full_df["battery_id"] == TEST_BATTERY]

    train_df.to_csv(PROCESSED_DIR / "train.csv", index=False)
    test_df.to_csv(PROCESSED_DIR  / "test.csv",  index=False)

    print(f"\nTrain set: {train_df.shape}")
    print(f"Test set:  {test_df.shape}")
    print(f"Saved to data/processed/")


if __name__ == "__main__":
    build_dataset()