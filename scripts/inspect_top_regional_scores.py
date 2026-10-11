import pandas as pd
from pathlib import Path

INPUT = Path(
    "data/processed/openaq_regional_intelligence.csv"
)

df = pd.read_csv(INPUT)

df["period_from_utc"] = pd.to_datetime(
    df["period_from_utc"],
    utc=True
)

summary = (
    df[
        [
            "period_from_utc",
            "regional_event_score",
            "regional_event_confidence",
            "regional_mean",
            "regional_median",
            "regional_max",
            "sensors_reporting",
            "sensors_elevated",
            "regional_elevated_fraction",
        ]
    ]
    .drop_duplicates("period_from_utc")
    .sort_values(
        "regional_event_score",
        ascending=False
    )
    .head(20)
)

print("=" * 90)
print("AEROSENSE — TOP REGIONAL EVENT SCORES")
print("=" * 90)

print(
    summary.to_string(
        index=False,
        formatters={
            "regional_event_score": "{:.3f}".format,
            "regional_mean": "{:.2f}".format,
            "regional_median": "{:.2f}".format,
            "regional_max": "{:.2f}".format,
            "regional_elevated_fraction": "{:.2%}".format,
        }
    )
)

print("\n" + "=" * 90)
print("COMPLETE")
print("=" * 90)