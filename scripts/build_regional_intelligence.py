import pandas as pd
import numpy as np
from pathlib import Path

INPUT = Path("data/processed/openaq_pm25_quality.csv")
OUTPUT = Path("data/processed/openaq_regional_intelligence.csv")

print("=" * 70)
print("AEROSENSE — REGIONAL INTELLIGENCE DATASET")
print("=" * 70)

print("\nLoading quality dataset...")

df = pd.read_csv(INPUT)

df["period_from_utc"] = pd.to_datetime(
    df["period_from_utc"],
    utc=True
)

df["value"] = pd.to_numeric(
    df["value"],
    errors="coerce"
)

df = df.dropna(
    subset=["sensor_id", "period_from_utc", "value"]
).copy()

df = df.sort_values(
    ["period_from_utc", "sensor_id"]
).reset_index(drop=True)

print(f"Rows:     {len(df):,}")
print(f"Sensors:  {df['sensor_id'].nunique():,}")
print(
    f"Start:    {df['period_from_utc'].min()}"
)
print(
    f"End:      {df['period_from_utc'].max()}"
)


# ---------------------------------------------------------
# 1. Regional statistics
# ---------------------------------------------------------

print("\nCalculating regional statistics...")

regional = (
    df.groupby("period_from_utc")["value"]
    .agg(
        regional_mean="mean",
        regional_median="median",
        regional_std="std",
        regional_min="min",
        regional_max="max",
        sensors_reporting="count",
    )
    .reset_index()
)

df = df.merge(
    regional,
    on="period_from_utc",
    how="left"
)


# ---------------------------------------------------------
# 2. Sensor deviation from regional conditions
# ---------------------------------------------------------

df["deviation_from_regional_median"] = (
    df["value"] - df["regional_median"]
)

df["deviation_ratio"] = (
    df["value"] /
    df["regional_median"].replace(0, np.nan)
)

df["regional_zscore"] = (
    (df["value"] - df["regional_mean"]) /
    df["regional_std"].replace(0, np.nan)
)


# ---------------------------------------------------------
# 3. Sensor-level temporal changes
# ---------------------------------------------------------

print("Calculating temporal changes...")

df = df.sort_values(
    ["sensor_id", "period_from_utc"]
).reset_index(drop=True)

df["change_1h"] = (
    df.groupby("sensor_id")["value"]
    .diff(1)
)

df["change_3h"] = (
    df.groupby("sensor_id")["value"]
    .diff(3)
)

df["pct_change_1h"] = (
    df["change_1h"] /
    df.groupby("sensor_id")["value"]
    .shift(1).replace(0, np.nan)
) * 100


# ---------------------------------------------------------
# 4. Rapid-change indicators
# ---------------------------------------------------------

df["rapid_rise_1h"] = (
    df["change_1h"] >= 100
)

df["rapid_rise_3h"] = (
    df["change_3h"] >= 200
)


# ---------------------------------------------------------
# 5. Network-level elevated sensors
# ---------------------------------------------------------

print("Calculating network agreement...")

# A sensor is considered elevated when PM2.5 exceeds
# both an absolute threshold and the regional median.

df["is_elevated"] = (
    (df["value"] > 100) &
    (df["value"] > df["regional_median"] * 1.5)
)

network = (
    df.groupby("period_from_utc")["is_elevated"]
    .agg(
        sensors_elevated="sum"
    )
    .reset_index()
)

df = df.merge(
    network,
    on="period_from_utc",
    how="left"
)

df["regional_elevated_fraction"] = (
    df["sensors_elevated"] /
    df["sensors_reporting"].replace(0, np.nan)
)


# ---------------------------------------------------------
# 6. Candidate anomaly/event rules
# ---------------------------------------------------------

df["sensor_anomaly_candidate"] = (
    (df["value"] > 500) &
    (
        (df["regional_elevated_fraction"] < 0.30) |
        (df["regional_zscore"] > 3)
    )
)

df["regional_event_candidate"] = (
    (df["sensors_reporting"] >= 8) &
    (df["regional_elevated_fraction"] >= 0.60) &
    (df["regional_mean"] > 100) &
    (df["regional_median"] > 80)
)


# ---------------------------------------------------------
# 7. Regional event confidence score
# ---------------------------------------------------------

# ---------------------------------------------------------
# 7. Regional event confidence score
# ---------------------------------------------------------

# Network coverage:
# 22 sensors exist in the dataset.
coverage_score = (
    df["sensors_reporting"] / 22.0
).clip(0, 1)


# Spatial agreement:
# Agreement is the most important signal for a regional event.
agreement_score = (
    df["regional_elevated_fraction"]
).fillna(0).clip(0, 1)


# Regional pollution intensity.
# Saturates at approximately 300 ug/m3.
intensity_score = (
    df["regional_median"] / 300.0
).clip(0, 1)


# Number of elevated sensors.
#
# 0-2  -> essentially no regional agreement
# 3-5  -> weak
# 6-9  -> moderate
# 10+  -> strong
#
# Normalize so 10+ reaches maximum strength.
elevated_count_score = (
    df["sensors_elevated"] / 10.0
).clip(0, 1)


# Final score.
#
# Spatial agreement and actual number of agreeing sensors
# dominate the score.
df["regional_event_score"] = (
    0.15 * coverage_score +
    0.50 * agreement_score +
    0.20 * elevated_count_score +
    0.15 * intensity_score
)


# Conservative high-confidence event definition.
df["regional_event_candidate"] = (
    (df["sensors_reporting"] >= 12) &
    (df["sensors_elevated"] >= 8) &
    (df["regional_elevated_fraction"] >= 0.65) &
    (df["regional_median"] > 100)
)


# Confidence categories.
df["regional_event_confidence"] = pd.cut(
    df["regional_event_score"],
    bins=[-0.01, 0.25, 0.50, 0.70, 1.00],
    labels=[
        "LOW",
        "MEDIUM",
        "HIGH",
        "VERY_HIGH"
    ]
)


# ---------------------------------------------------------
# 7. Restore chronological order
# ---------------------------------------------------------

df = df.sort_values(
    ["period_from_utc", "sensor_id"]
).reset_index(drop=True)


# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    OUTPUT,
    index=False
)


# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("REGIONAL INTELLIGENCE COMPLETE")
print("=" * 70)

print(f"Rows:                    {len(df):,}")
print(f"Sensors:                 {df['sensor_id'].nunique():,}")
print(
    f"Regional timestamps:     "
    f"{df['period_from_utc'].nunique():,}"
)

print(
    f"Sensor anomaly candidates: "
    f"{df['sensor_anomaly_candidate'].sum():,}"
)

print(
    f"High-confidence regional candidates: "
    f"{df['regional_event_candidate'].sum():,}"
)

print("\nRegional confidence distribution:")
print(
    df[
        [
            "period_from_utc",
            "regional_event_score",
            "regional_event_confidence"
        ]
    ]
    .drop_duplicates()
    ["regional_event_confidence"]
    .value_counts()
    .sort_index()
)

print("\nOutput:")
print(OUTPUT)