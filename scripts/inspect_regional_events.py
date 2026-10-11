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

# Get timestamps where regional event was detected
events = (
    df.loc[
        df["regional_event_candidate"],
        "period_from_utc"
    ]
    .drop_duplicates()
    .sort_values()
)

print("=" * 75)
print("AEROSENSE — REGIONAL EVENT INVESTIGATION")
print("=" * 75)

print(f"\nCandidate regional event timestamps: {len(events)}")

for i, timestamp in enumerate(events, start=1):

    event = df[
        df["period_from_utc"] == timestamp
    ].copy()

    event = event.sort_values(
        "value",
        ascending=False
    )

    print("\n" + "=" * 75)
    print(f"EVENT {i}")
    print("=" * 75)

    summary = event.iloc[0]

    print(f"Timestamp:              {timestamp}")
    print(
        f"Regional mean:          "
        f"{summary['regional_mean']:.2f}"
    )
    print(
        f"Regional median:        "
        f"{summary['regional_median']:.2f}"
    )
    print(
        f"Regional std:           "
        f"{summary['regional_std']:.2f}"
    )
    print(
        f"Regional min:           "
        f"{summary['regional_min']:.2f}"
    )
    print(
        f"Regional max:           "
        f"{summary['regional_max']:.2f}"
    )
    print(
        f"Sensors reporting:      "
        f"{summary['sensors_reporting']}"
    )
    print(
        f"Sensors elevated:       "
        f"{summary['sensors_elevated']}"
    )
    print(
        f"Elevated fraction:      "
        f"{summary['regional_elevated_fraction']:.2%}"
    )

    print("\nSensor readings:")
    print("-" * 75)

    cols = [
        "sensor_id",
        "value",
        "regional_zscore",
        "deviation_from_regional_median",
        "is_elevated",
    ]

    print(
        event[cols].to_string(
            index=False
        )
    )

print("\n" + "=" * 75)
print("INVESTIGATION COMPLETE")
print("=" * 75)