import pandas as pd
from pathlib import Path

INPUT = Path("data/processed/openaq_pm25_master.csv")
OUTPUT = Path("data/processed/openaq_pm25_quality.csv")

df = pd.read_csv(
    INPUT,
    parse_dates=["period_from_utc", "period_to_utc"]
)

df = df.sort_values(
    ["sensor_id", "period_from_utc"]
).reset_index(drop=True)

# Time gap between consecutive measurements
df["gap_hours"] = (
    df.groupby("sensor_id")["period_from_utc"]
      .diff()
      .dt.total_seconds()
      / 3600
)

# Basic quality indicators
df["is_flagged"] = df["has_flags"].astype(bool)

# Keep extreme values for anomaly analysis.
df["is_extreme_candidate"] = df["value"] > 500

# Very high values for separate investigation.
df["is_severe_candidate"] = df["value"] > 1000

# Extremely high values.
df["is_critical_candidate"] = df["value"] > 5000

# Large temporal gaps
df["has_large_gap_before"] = df["gap_hours"] > 6
df["has_major_gap_before"] = df["gap_hours"] > 24
df["has_long_gap_before"] = df["gap_hours"] > 168

# A record is suitable for normal forecasting continuity
# when it follows the previous observation by <= 6 hours.
df["forecast_continuous"] = (
    df["gap_hours"].isna() |
    (df["gap_hours"] <= 6)
)

df.to_csv(
    OUTPUT,
    index=False,
    encoding="utf-8"
)

print("========================================")
print("AEROSENSE QUALITY DATASET")
print("========================================")
print(f"Rows:                 {len(df):,}")
print(f"Sensors:              {df['sensor_id'].nunique():,}")
print(f"Flagged:              {df['is_flagged'].sum():,}")
print(f"Extreme >500:         {df['is_extreme_candidate'].sum():,}")
print(f"Severe >1000:         {df['is_severe_candidate'].sum():,}")
print(f"Critical >5000:       {df['is_critical_candidate'].sum():,}")
print(f"Gaps >6h:             {df['has_large_gap_before'].sum():,}")
print(f"Gaps >24h:            {df['has_major_gap_before'].sum():,}")
print(f"Gaps >7d:             {df['has_long_gap_before'].sum():,}")
print(f"Output:               {OUTPUT}")
