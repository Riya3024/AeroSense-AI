import numpy as np
import pandas as pd
from pathlib import Path


# =========================================================
# AEROSENSE — FINAL FORECAST-AWARE INTELLIGENCE
# =========================================================

ANOMALY_FILE = Path(
    "data/processed/openaq_anomaly_intelligence.csv"
)

RESIDUAL_FILE = Path(
    "data/processed/openaq_forecast_residuals.csv"
)

OUTPUT = Path(
    "data/processed/openaq_final_intelligence.csv"
)


print("=" * 70)
print("AEROSENSE — FINAL FORECAST-AWARE INTELLIGENCE")
print("=" * 70)


# ---------------------------------------------------------
# 1. Load anomaly intelligence
# ---------------------------------------------------------

print("\nLoading anomaly intelligence...")

anomaly = pd.read_csv(
    ANOMALY_FILE,
    parse_dates=["period_from_utc"]
)

print(f"Anomaly rows: {len(anomaly):,}")


# ---------------------------------------------------------
# 2. Load forecast residuals
# ---------------------------------------------------------

print("\nLoading forecast residuals...")

residual = pd.read_csv(
    RESIDUAL_FILE,
    parse_dates=["period_from_utc"]
)

print(f"Residual rows: {len(residual):,}")


# ---------------------------------------------------------
# 3. Select forecast intelligence columns
# ---------------------------------------------------------

forecast_columns = [
    "period_from_utc",
    "sensor_id",
    "target_1h",
    "expected_pm25_1h",
    "forecast_residual",
    "forecast_abs_residual",
    "forecast_surprise_score",
    "forecast_surprise_direction",
]

residual = residual[forecast_columns]


# ---------------------------------------------------------
# 4. Merge datasets
# ---------------------------------------------------------

print("\nMerging intelligence layers...")

df = anomaly.merge(
    residual,
    on=[
        "period_from_utc",
        "sensor_id",
    ],
    how="left",
    validate="one_to_one",
)

print(f"Merged rows: {len(df):,}")


# ---------------------------------------------------------
# 5. Improve forecast surprise signal
# ---------------------------------------------------------

# The existing score is intentionally heuristic.
# This secondary score gives larger residuals more separation.

df["forecast_surprise_strength"] = (
    df["forecast_abs_residual"] / 500.0
).clip(0, 1)


# ---------------------------------------------------------
# 6. Build final intelligence score
# ---------------------------------------------------------

df["final_intelligence_score"] = (
    0.25 * df["severity_score"].fillna(0)
    + 0.20 * df["rapid_rise_score"].fillna(0)
    + 0.20 * df["regional_deviation_score"].fillna(0)
    + 0.15 * df["network_agreement_score"].fillna(0)
    + 0.10 * df["forecast_surprise_strength"].fillna(0)
    + 0.10 * df["data_quality_score"].fillna(0)
)


# ---------------------------------------------------------
# 7. Final intelligence classification
# ---------------------------------------------------------

df["final_intelligence_class"] = "NORMAL"


# Elevated pollution state
df.loc[
    df["value"] >= 100,
    "final_intelligence_class"
] = "ELEVATED"


# Forecast surprise
df.loc[
    (
        (df["forecast_surprise_strength"] >= 0.50)
        & (df["final_intelligence_score"] >= 0.50)
    ),
    "final_intelligence_class"
] = "FORECAST_SURPRISE"


# Rapid unexpected increase
df.loc[
    (
        (df["forecast_surprise_direction"] == "UNEXPECTED_RISE")
        & (df["rapid_rise_score"] >= 0.50)
        & (df["final_intelligence_score"] >= 0.55)
    ),
    "final_intelligence_class"
] = "UNEXPECTED_RISE"


# Sensor-specific anomaly
df.loc[
    (
        (df["anomaly_class"].isin([
            "SENSOR_ANOMALY",
            "EXTREME_SENSOR_ANOMALY",
        ]))
        & (df["final_intelligence_score"] >= 0.60)
    ),
    "final_intelligence_class"
] = "SENSOR_ANOMALY"


# Regional event
df.loc[
    (
        (df["regional_event_candidate"] == True)
        & (df["final_intelligence_score"] >= 0.60)
    ),
    "final_intelligence_class"
] = "REGIONAL_EVENT"


# Extreme regional event
df.loc[
    (
        (df["value"] >= 500)
        & (df["regional_elevated_fraction"].fillna(0) >= 0.65)
        & (df["sensors_elevated"] >= 8)
        & (df["final_intelligence_score"] >= 0.70)
    ),
    "final_intelligence_class"
] = "EXTREME_REGIONAL_EVENT"


# ---------------------------------------------------------
# 8. Confidence level
# ---------------------------------------------------------

df["final_confidence"] = pd.cut(
    df["final_intelligence_score"],
    bins=[-0.01, 0.25, 0.50, 0.70, 1.00],
    labels=[
        "LOW",
        "MEDIUM",
        "HIGH",
        "VERY_HIGH",
    ],
)


# ---------------------------------------------------------
# 9. Final alert candidate
# ---------------------------------------------------------

df["final_alert_candidate"] = (
    (df["final_intelligence_score"] >= 0.50)
    | (df["value"] >= 500)
    | (df["forecast_surprise_strength"] >= 0.50)
)


# ---------------------------------------------------------
# 10. Save
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
# 11. Summary
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("FINAL INTELLIGENCE COMPLETE")
print("=" * 70)

print(
    f"Rows:                    {len(df):,}"
)

print(
    f"Alert candidates:        "
    f"{df['final_alert_candidate'].sum():,}"
)

print(
    f"Mean intelligence score: "
    f"{df['final_intelligence_score'].mean():.4f}"
)

print(
    f"Median intelligence:     "
    f"{df['final_intelligence_score'].median():.4f}"
)


print("\nFinal intelligence classes:")

print(
    df["final_intelligence_class"]
    .value_counts()
    .to_string()
)


print("\nFinal confidence:")

print(
    df["final_confidence"]
    .value_counts()
    .sort_index()
    .to_string()
)


print("\nTop final intelligence events:")

top = (
    df.sort_values(
        "final_intelligence_score",
        ascending=False
    )
    [
        [
            "period_from_utc",
            "sensor_id",
            "value",
            "expected_pm25_1h",
            "forecast_residual",
            "regional_median",
            "regional_elevated_fraction",
            "anomaly_score",
            "final_intelligence_score",
            "final_intelligence_class",
            "final_confidence",
        ]
    ]
    .head(25)
)

print(
    top.to_string(index=False)
)


print("\nOutput:")
print(OUTPUT)

print("=" * 70)