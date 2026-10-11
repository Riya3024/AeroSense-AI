import pandas as pd
from pathlib import Path

INPUT = Path("data/processed/openaq_pm25_features.csv")
OUTPUT_DIR = Path("data/processed/splits")

df = pd.read_csv(INPUT)

df["period_from_utc"] = pd.to_datetime(
    df["period_from_utc"],
    utc=True
)

train = df[
    (df["period_from_utc"] >= "2025-01-01") &
    (df["period_from_utc"] < "2026-01-01")
].copy()

validation = df[
    (df["period_from_utc"] >= "2026-01-01") &
    (df["period_from_utc"] < "2026-07-01")
].copy()

test = df[
    (df["period_from_utc"] >= "2026-07-01") &
    (df["period_from_utc"] < "2026-10-01")
].copy()

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

train.to_csv(
    OUTPUT_DIR / "train.csv",
    index=False,
    encoding="utf-8"
)

validation.to_csv(
    OUTPUT_DIR / "validation.csv",
    index=False,
    encoding="utf-8"
)

test.to_csv(
    OUTPUT_DIR / "test.csv",
    index=False,
    encoding="utf-8"
)

print("=" * 55)
print("AEROSENSE TEMPORAL DATA SPLIT")
print("=" * 55)

for name, data in [
    ("TRAIN", train),
    ("VALIDATION", validation),
    ("TEST", test)
]:
    print()
    print(name)
    print(f"Rows:       {len(data):,}")
    print(f"Sensors:    {data['sensor_id'].nunique()}")
    print(f"Start:      {data['period_from_utc'].min()}")
    print(f"End:        {data['period_from_utc'].max()}")
    print(f"Target MAE baseline candidate available: {data['target_1h'].notna().sum():,}")

print()
print("TOTAL:", len(train) + len(validation) + len(test))
print()
print(f"Output: {OUTPUT_DIR}")
print("=" * 55)
