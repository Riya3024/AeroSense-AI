import pandas as pd
from pathlib import Path


# =========================================================
# AEROSENSE — FINAL INTELLIGENCE VALIDATION
# =========================================================

INPUT = Path(
    "data/processed/openaq_final_intelligence.csv"
)

OUTPUT = Path(
    "data/processed/aerosense_validation_report.txt"
)

print("=" * 70)
print("AEROSENSE — FINAL INTELLIGENCE VALIDATION")
print("=" * 70)


# ---------------------------------------------------------
# 1. Load dataset
# ---------------------------------------------------------

print("\nLoading final intelligence dataset...")

df = pd.read_csv(
    INPUT,
    parse_dates=["period_from_utc"]
)

print(f"Rows:    {len(df):,}")
print(f"Sensors: {df['sensor_id'].nunique():,}")


# ---------------------------------------------------------
# 2. Basic statistics
# ---------------------------------------------------------

print("\n" + "-" * 70)
print("DATASET SUMMARY")
print("-" * 70)

summary = []

summary.append("AEROSENSE — FINAL INTELLIGENCE VALIDATION")
summary.append("=" * 70)

summary.append(f"Total observations: {len(df):,}")
summary.append(f"Total sensors:      {df['sensor_id'].nunique():,}")
summary.append(
    f"Date range:         "
    f"{df['period_from_utc'].min()} → "
    f"{df['period_from_utc'].max()}"
)


# ---------------------------------------------------------
# 3. Classification distribution
# ---------------------------------------------------------

class_counts = (
    df["final_intelligence_class"]
    .value_counts()
)

print("\nFinal intelligence classes:")

print(
    class_counts.to_string()
)

summary.append("")
summary.append("FINAL INTELLIGENCE CLASSES")
summary.append("-" * 70)

for name, count in class_counts.items():
    percentage = count / len(df) * 100

    line = (
        f"{name:<25}"
        f"{count:>10,}"
        f" ({percentage:>6.2f}%)"
    )

    summary.append(line)


# ---------------------------------------------------------
# 4. Confidence distribution
# ---------------------------------------------------------

confidence_counts = (
    df["final_confidence"]
    .value_counts()
)

print("\nConfidence distribution:")

print(
    confidence_counts.to_string()
)

summary.append("")
summary.append("CONFIDENCE DISTRIBUTION")
summary.append("-" * 70)

for name, count in confidence_counts.items():

    percentage = count / len(df) * 100

    line = (
        f"{str(name):<15}"
        f"{count:>10,}"
        f" ({percentage:>6.2f}%)"
    )

    summary.append(line)


# ---------------------------------------------------------
# 5. Alert statistics
# ---------------------------------------------------------

alert_count = int(
    df["final_alert_candidate"].sum()
)

very_high_count = int(
    (df["final_confidence"] == "VERY_HIGH").sum()
)

high_count = int(
    (df["final_confidence"] == "HIGH").sum()
)

summary.append("")
summary.append("ALERT STATISTICS")
summary.append("-" * 70)
summary.append(
    f"Alert candidates: {alert_count:,} "
    f"({alert_count / len(df) * 100:.2f}%)"
)
summary.append(
    f"High confidence: {high_count:,}"
)
summary.append(
    f"Very high confidence: {very_high_count:,}"
)


# ---------------------------------------------------------
# 6. Extreme-value validation
# ---------------------------------------------------------

extreme = df[
    df["value"] >= 500
].copy()

extreme_sensor_anomaly = extreme[
    extreme["final_intelligence_class"]
    == "SENSOR_ANOMALY"
]

extreme_regional = extreme[
    extreme["final_intelligence_class"]
    == "EXTREME_REGIONAL_EVENT"
]

summary.append("")
summary.append("EXTREME VALUE VALIDATION")
summary.append("-" * 70)
summary.append(
    f"Observations >= 500: "
    f"{len(extreme):,}"
)
summary.append(
    f"Extreme → sensor anomaly: "
    f"{len(extreme_sensor_anomaly):,}"
)
summary.append(
    f"Extreme → regional event: "
    f"{len(extreme_regional):,}"
)


# ---------------------------------------------------------
# 7. Forecast surprise validation
# ---------------------------------------------------------

unexpected_rise = df[
    df["forecast_surprise_direction"]
    == "UNEXPECTED_RISE"
]

unexpected_drop = df[
    df["forecast_surprise_direction"]
    == "UNEXPECTED_DROP"
]

summary.append("")
summary.append("FORECAST SURPRISE VALIDATION")
summary.append("-" * 70)
summary.append(
    f"Unexpected rises: "
    f"{len(unexpected_rise):,}"
)
summary.append(
    f"Unexpected drops: "
    f"{len(unexpected_drop):,}"
)


# ---------------------------------------------------------
# 8. Known 7080 event validation
# ---------------------------------------------------------

known_event = df[
    (
        df["sensor_id"].astype(str)
        == "12234735"
    )
    &
    (
        df["value"] >= 7000
    )
].copy()

summary.append("")
summary.append("KNOWN EXTREME EVENT VALIDATION")
summary.append("-" * 70)

if len(known_event) == 0:

    summary.append(
        "WARNING: Known 7080 event not found."
    )

else:

    for _, row in known_event.iterrows():

        summary.append(
            f"Timestamp: {row['period_from_utc']}"
        )

        summary.append(
            f"Sensor: {row['sensor_id']}"
        )

        summary.append(
            f"Actual PM2.5: {row['value']:.2f}"
        )

        summary.append(
            f"Expected PM2.5: "
            f"{row['expected_pm25_1h']:.2f}"
        )

        summary.append(
            f"Forecast residual: "
            f"{row['forecast_residual']:.2f}"
        )

        summary.append(
            f"Regional median: "
            f"{row['regional_median']:.2f}"
        )

        summary.append(
            f"Regional elevated fraction: "
            f"{row['regional_elevated_fraction']:.4f}"
        )

        summary.append(
            f"Anomaly score: "
            f"{row['anomaly_score']:.4f}"
        )

        summary.append(
            f"Final score: "
            f"{row['final_intelligence_score']:.4f}"
        )

        summary.append(
            f"Final class: "
            f"{row['final_intelligence_class']}"
        )

        summary.append(
            f"Confidence: "
            f"{row['final_confidence']}"
        )

        summary.append("-" * 40)


# ---------------------------------------------------------
# 9. Top events
# ---------------------------------------------------------

summary.append("")
summary.append("TOP 20 INTELLIGENCE EVENTS")
summary.append("-" * 70)

top = (
    df.sort_values(
        "final_intelligence_score",
        ascending=False
    )
    .head(20)
)

for index, (_, row) in enumerate(
    top.iterrows(),
    start=1
):

    summary.append(
        f"{index:02d}. "
        f"{row['period_from_utc']} | "
        f"Sensor {row['sensor_id']} | "
        f"PM2.5={row['value']:.1f} | "
        f"Score={row['final_intelligence_score']:.4f} | "
        f"{row['final_intelligence_class']} | "
        f"{row['final_confidence']}"
    )


# ---------------------------------------------------------
# 10. Regional event check
# ---------------------------------------------------------

regional_events = df[
    df["final_intelligence_class"].isin(
        [
            "REGIONAL_EVENT",
            "EXTREME_REGIONAL_EVENT",
        ]
    )
]

summary.append("")
summary.append("REGIONAL EVENT CHECK")
summary.append("-" * 70)
summary.append(
    f"Regional events detected: "
    f"{len(regional_events):,}"
)

if len(regional_events) == 0:

    summary.append(
        "No high-confidence regional events "
        "were detected under the conservative gate."
    )


# ---------------------------------------------------------
# 11. Print report
# ---------------------------------------------------------

report = "\n".join(summary)

print("\n")
print(report)


# ---------------------------------------------------------
# 12. Save report
# ---------------------------------------------------------

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT.write_text(
    report,
    encoding="utf-8"
)

print("\n" + "=" * 70)
print("VALIDATION COMPLETE")
print("=" * 70)

print(f"Report: {OUTPUT}")