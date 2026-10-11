import pandas as pd
import numpy as np
from pathlib import Path

INPUT = Path("data/processed/openaq_pm25_quality.csv")
OUTPUT = Path("data/processed/openaq_pm25_features.csv")

print("=" * 55)
print("AEROSENSE FEATURE ENGINEERING — EXACT 1H TARGET")
print("=" * 55)

df = pd.read_csv(INPUT)

df["period_from_utc"] = pd.to_datetime(
    df["period_from_utc"],
    utc=True
)

df = df.sort_values(
    ["sensor_id", "period_from_utc"]
).reset_index(drop=True)

# --------------------------------------------------
# Calendar features
# --------------------------------------------------

df["hour"] = df["period_from_utc"].dt.hour
df["day_of_week"] = df["period_from_utc"].dt.dayofweek
df["day_of_month"] = df["period_from_utc"].dt.day
df["month"] = df["period_from_utc"].dt.month
df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

# --------------------------------------------------
# Cyclic time features
# --------------------------------------------------

df["hour_sin"] = np.sin(
    2 * np.pi * df["hour"] / 24
)

df["hour_cos"] = np.cos(
    2 * np.pi * df["hour"] / 24
)

df["dow_sin"] = np.sin(
    2 * np.pi * df["day_of_week"] / 7
)

df["dow_cos"] = np.cos(
    2 * np.pi * df["day_of_week"] / 7
)

# --------------------------------------------------
# Continuous segments
# --------------------------------------------------

continuous = df["forecast_continuous"].fillna(False)

df["segment_id"] = (
    (~continuous)
    .groupby(df["sensor_id"])
    .cumsum()
)

group_cols = ["sensor_id", "segment_id"]

# --------------------------------------------------
# Lag features
# --------------------------------------------------

for lag in [1, 3, 6, 12, 24]:

    df[f"lag_{lag}h"] = (
        df.groupby(group_cols)["value"]
        .shift(lag)
    )

# --------------------------------------------------
# Rolling features
# --------------------------------------------------

for window in [3, 6, 24]:

    grouped = df.groupby(group_cols)["value"]

    df[f"rolling_mean_{window}h"] = (
        grouped
        .transform(
            lambda x: x.shift(1)
            .rolling(window, min_periods=window)
            .mean()
        )
    )

    df[f"rolling_std_{window}h"] = (
        grouped
        .transform(
            lambda x: x.shift(1)
            .rolling(window, min_periods=window)
            .std()
        )
    )

    df[f"rolling_min_{window}h"] = (
        grouped
        .transform(
            lambda x: x.shift(1)
            .rolling(window, min_periods=window)
            .min()
        )
    )

    df[f"rolling_max_{window}h"] = (
        grouped
        .transform(
            lambda x: x.shift(1)
            .rolling(window, min_periods=window)
            .max()
        )
    )

# --------------------------------------------------
# EXACT +1 HOUR TARGET
# --------------------------------------------------

target_lookup = (
    df[["sensor_id", "period_from_utc", "value"]]
    .rename(
        columns={
            "period_from_utc": "target_time",
            "value": "target_1h"
        }
    )
)

target_lookup["target_time"] = (
    target_lookup["target_time"]
    - pd.Timedelta(hours=1)
)

df = df.merge(
    target_lookup,
    left_on=["sensor_id", "period_from_utc"],
    right_on=["sensor_id", "target_time"],
    how="left"
)

df = df.drop(columns=["target_time"])

# --------------------------------------------------
# Verify exact target horizon
# --------------------------------------------------

df["target_available_1h"] = (
    df["target_1h"].notna()
)

# --------------------------------------------------
# Keep only rows with complete forecasting history
# AND exact next-hour target
# --------------------------------------------------

required = [
    "lag_1h",
    "lag_3h",
    "lag_6h",
    "lag_12h",
    "lag_24h",
    "rolling_mean_3h",
    "rolling_mean_6h",
    "rolling_mean_24h",
    "target_1h"
]

before = len(df)

df = df.dropna(
    subset=required
).copy()

after = len(df)

# --------------------------------------------------
# Remove helper
# --------------------------------------------------

df = df.drop(columns=["segment_id"])

# --------------------------------------------------
# Save
# --------------------------------------------------

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    OUTPUT,
    index=False,
    encoding="utf-8"
)

print()
print(f"Original rows:          {before:,}")
print(f"Exact 1h target rows:   {after:,}")
print(f"Rows removed:           {before-after:,}")
print(f"Sensors:                {df['sensor_id'].nunique()}")
print(f"Features:               {len(df.columns)}")

print()
print("TARGET STATISTICS")
print(df["target_1h"].describe().to_string())

print()
print("TARGET > 500:")
print((df["target_1h"] > 500).sum())

print("TARGET > 1000:")
print((df["target_1h"] > 1000).sum())

print("TARGET > 5000:")
print((df["target_1h"] > 5000).sum())

print()
print(f"Output: {OUTPUT}")
print("=" * 55)
