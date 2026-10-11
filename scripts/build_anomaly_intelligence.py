from pathlib import Path
import numpy as np
import pandas as pd


# =========================================================
# AEROSENSE — SENSOR ANOMALY INTELLIGENCE
# =========================================================

INPUT = Path("data/processed/openaq_regional_intelligence.csv")
OUTPUT = Path("data/processed/openaq_anomaly_intelligence.csv")

print("=" * 70)
print("AEROSENSE — SENSOR ANOMALY INTELLIGENCE")
print("=" * 70)

print("\nLoading regional intelligence dataset...")

df = pd.read_csv(INPUT, parse_dates=["period_from_utc"])

print(f"Rows:     {len(df):,}")
print(f"Sensors:  {df['sensor_id'].nunique():,}")

# ---------------------------------------------------------
# 1. Severity score
# ---------------------------------------------------------

# Absolute PM2.5 severity.
# 0 at 35 µg/m³ and approaches 1 at 500+.
df["severity_score"] = (
    (df["value"] - 35) / (500 - 35)
).clip(0, 1)

# Extreme pollution receives maximum severity.
df.loc[df["value"] >= 500, "severity_score"] = 1.0


# ---------------------------------------------------------
# 2. Rapid-rise score
# ---------------------------------------------------------

rise_1h = df["change_1h"].fillna(0)
rise_3h = df["change_3h"].fillna(0)

rapid_1h_score = (
    rise_1h / 200.0
).clip(0, 1)

rapid_3h_score = (
    rise_3h / 400.0
).clip(0, 1)

df["rapid_rise_score"] = (
    0.60 * rapid_1h_score +
    0.40 * rapid_3h_score
)


# ---------------------------------------------------------
# 3. Regional deviation score
# ---------------------------------------------------------

deviation_score = (
    df["deviation_ratio"].abs() / 2.0
).clip(0, 1)

zscore_score = (
    df["regional_zscore"].abs() / 4.0
).clip(0, 1)

df["regional_deviation_score"] = (
    0.60 * deviation_score +
    0.40 * zscore_score
)


# ---------------------------------------------------------
# 4. Network agreement score
# ---------------------------------------------------------

# Agreement is useful when many sensors are elevated.
df["network_agreement_score"] = (
    df["regional_elevated_fraction"]
).fillna(0).clip(0, 1)


# ---------------------------------------------------------
# 5. Data-quality score
# ---------------------------------------------------------

quality_score = pd.Series(1.0, index=df.index)

# Flagged observations receive a strong penalty.
quality_score = quality_score.where(
    ~df["is_flagged"].fillna(False),
    0.40
)

# Observations immediately after a large gap
# receive lower confidence.
quality_score = quality_score.where(
    ~df["has_large_gap_before"].fillna(False),
    0.60
)

# Observations after a major gap are even less trustworthy.
quality_score = quality_score.where(
    ~df["has_major_gap_before"].fillna(False),
    0.40
)

df["data_quality_score"] = quality_score


# ---------------------------------------------------------
# 6. Combined anomaly score
# ---------------------------------------------------------

df["anomaly_score"] = (
    0.25 * df["severity_score"] +
    0.25 * df["rapid_rise_score"] +
    0.20 * df["regional_deviation_score"] +
    0.20 * df["network_agreement_score"] +
    0.10 * df["data_quality_score"]
)


# ---------------------------------------------------------
# 7. Event classification
# ---------------------------------------------------------

# ---------------------------------------------------------
# 7. Event classification
# ---------------------------------------------------------

df["anomaly_class"] = "NORMAL"

# ---------------------------------------------------------
# Elevated pollution
# ---------------------------------------------------------

df.loc[
    (df["value"] >= 100),
    "anomaly_class"
] = "ELEVATED"


# ---------------------------------------------------------
# Rapid rise
# ---------------------------------------------------------

df.loc[
    (
        (df["change_1h"] >= 100) |
        (df["change_3h"] >= 200)
    ),
    "anomaly_class"
] = "RAPID_RISE"


# ---------------------------------------------------------
# Sensor-specific anomaly
# ---------------------------------------------------------

df.loc[
    (
        (df["anomaly_score"] >= 0.60) &
        (
            df["regional_elevated_fraction"].fillna(0) < 0.50
        )
    ),
    "anomaly_class"
] = "SENSOR_ANOMALY"


# ---------------------------------------------------------
# Potential regional event
# ---------------------------------------------------------

df.loc[
    (
        df["regional_event_candidate"] &
        (df["regional_event_score"] >= 0.50)
    ),
    "anomaly_class"
] = "REGIONAL_EVENT"


# ---------------------------------------------------------
# Extreme regional event
# ---------------------------------------------------------

df.loc[
    (
        (df["value"] >= 500) &
        (df["regional_elevated_fraction"].fillna(0) >= 0.65) &
        (df["sensors_elevated"] >= 8)
    ),
    "anomaly_class"
] = "EXTREME_EVENT"


# ---------------------------------------------------------
# Extreme sensor-specific anomaly
# ---------------------------------------------------------

df.loc[
    (
        (df["value"] >= 500) &
        (
            df["regional_elevated_fraction"].fillna(0) < 0.65
        )
    ),
    "anomaly_class"
] = "EXTREME_SENSOR_ANOMALY"


# ---------------------------------------------------------
# 8. Confidence label
# ---------------------------------------------------------

df["anomaly_confidence"] = pd.cut(
    df["anomaly_score"],
    bins=[-0.01, 0.25, 0.50, 0.70, 1.00],
    labels=[
        "LOW",
        "MEDIUM",
        "HIGH",
        "VERY_HIGH"
    ]
)


# ---------------------------------------------------------
# 9. Candidate flag
# ---------------------------------------------------------

df["anomaly_candidate"] = (
    (
        df["anomaly_score"] >= 0.50
    )
    |
    (
        df["value"] >= 500
    )
    |
    (
        df["regional_event_candidate"]
    )
)


# ---------------------------------------------------------
# 10. Save
# ---------------------------------------------------------

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

df.to_csv(
    OUTPUT,
    index=False
)


# ---------------------------------------------------------
# 11. Summary
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("ANOMALY INTELLIGENCE COMPLETE")
print("=" * 70)

print(f"Rows:                    {len(df):,}")
print(f"Sensors:                 {df['sensor_id'].nunique():,}")
print(
    f"Anomaly candidates:      "
    f"{df['anomaly_candidate'].sum():,}"
)

print("\nAnomaly class distribution:")
print(
    df["anomaly_class"]
    .value_counts()
    .to_string()
)

print("\nConfidence distribution:")
print(
    df["anomaly_confidence"]
    .value_counts()
    .sort_index()
    .to_string()
)

print("\nTop anomaly scores:")

top = (
    df[df["anomaly_candidate"]]
    .sort_values(
        "anomaly_score",
        ascending=False
    )
    [
        [
            "period_from_utc",
            "sensor_id",
            "station_id",
            "value",
            "anomaly_score",
            "anomaly_confidence",
            "anomaly_class",
            "change_1h",
            "change_3h",
            "regional_median",
            "regional_elevated_fraction",
            "regional_zscore",
        ]
    ]
    .head(20)
)

print(top.to_string(index=False))

print("\nOutput:")
print(OUTPUT)

print("=" * 70)