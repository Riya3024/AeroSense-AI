import pandas as pd
import numpy as np
from pathlib import Path


# =========================================================
# AEROSENSE — EVENT ENGINE V3
# =========================================================
#
# Converts alert observations into meaningful pollution
# events and adds network-level context.
#
# V3 preserves the validated V2 event engine and adds:
#
#   - LOCALIZED / MIXED / REGIONAL context
#   - network coverage
#   - sensor specificity
#   - peak regional context
#   - event confidence reason
#   - regional support flag
#
# =========================================================


INPUT = Path(
    "data/processed/openaq_final_intelligence.csv"
)

OUTPUT = Path(
    "data/processed/aerosense_events.csv"
)

TOTAL_SENSORS = 22


print("=" * 70)
print("AEROSENSE — EVENT ENGINE V3")
print("=" * 70)


# =========================================================
# 1. LOAD FINAL INTELLIGENCE
# =========================================================

df = pd.read_csv(
    INPUT,
    parse_dates=["period_from_utc"]
)

df = df.sort_values(
    ["sensor_id", "period_from_utc"]
).reset_index(drop=True)

print(f"Observations: {len(df):,}")
print(f"Sensors:      {df['sensor_id'].nunique():,}")


# =========================================================
# 2. SELECT ALERT OBSERVATIONS
# =========================================================

alerts = df[
    df["final_alert_candidate"] == True
].copy()

print(f"Alert observations: {len(alerts):,}")


# =========================================================
# 3. TEMPORAL CONTINUITY
# =========================================================

alerts["previous_alert_time"] = (
    alerts
    .groupby("sensor_id")["period_from_utc"]
    .shift(1)
)

alerts["hours_since_previous"] = (
    (
        alerts["period_from_utc"]
        - alerts["previous_alert_time"]
    )
    .dt.total_seconds()
    / 3600
)


# =========================================================
# 4. START NEW EVENT
#
# A new event starts when:
#
# - there is no previous alert
# - OR previous alert is > 3 hours away
#
# =========================================================

alerts["new_event"] = (
    alerts["previous_alert_time"].isna()
    |
    (alerts["hours_since_previous"] > 3)
)


# =========================================================
# 5. ASSIGN EVENT GROUPS
# =========================================================

alerts["event_sequence"] = (
    alerts
    .groupby("sensor_id")["new_event"]
    .cumsum()
)

alerts["event_id"] = (
    "AS-"
    + alerts["period_from_utc"]
      .dt.strftime("%Y%m%d")
    + "-"
    + alerts["sensor_id"].astype(str)
    + "-"
    + alerts["event_sequence"]
      .astype(str)
      .str.zfill(2)
)


# =========================================================
# 6. EVENT CLASSIFICATION HELPERS
# =========================================================

CLASS_PRIORITY = {
    "NORMAL": 0,
    "ELEVATED": 1,
    "FORECAST_SURPRISE": 2,
    "UNEXPECTED_RISE": 3,
    "SENSOR_ANOMALY": 4,
    "REGIONAL_EVENT": 5,
    "EXTREME_REGIONAL_EVENT": 6,
}


def classify_event(group):

    classes = (
        group["final_intelligence_class"]
        .fillna("NORMAL")
        .astype(str)
    )

    classes = classes[
        classes != "NORMAL"
    ]

    if len(classes) == 0:
        return "ELEVATED"

    return max(
        classes,
        key=lambda x: CLASS_PRIORITY.get(x, 0)
    )


def get_confidence(group):

    confidence_priority = {
        "LOW": 1,
        "MEDIUM": 2,
        "HIGH": 3,
        "VERY_HIGH": 4,
    }

    values = (
        group["final_confidence"]
        .fillna("LOW")
        .astype(str)
    )

    return max(
        values,
        key=lambda x: confidence_priority.get(x, 0)
    )


def get_peak_forecast_residual(group):

    residuals = pd.to_numeric(
        group["forecast_residual"],
        errors="coerce"
    )

    residuals = residuals.dropna()

    if len(residuals) == 0:
        return np.nan

    return residuals.loc[
        residuals.abs().idxmax()
    ]


# =========================================================
# 7. NETWORK CONTEXT
# =========================================================

def classify_network_context(
    regional_fraction
):

    if pd.isna(regional_fraction):
        return "UNKNOWN"

    if regional_fraction >= 0.65:
        return "REGIONAL"

    if regional_fraction >= 0.50:
        return "MIXED"

    return "LOCALIZED"


def calculate_sensor_specificity(
    regional_fraction
):

    if pd.isna(regional_fraction):
        return np.nan

    # Higher value means the event is more localized
    # relative to the surrounding sensor network.

    return float(
        np.clip(
            1.0 - regional_fraction,
            0,
            1
        )
    )


def build_confidence_reason(
    event_type,
    network_context,
    severity,
    peak_score,
    peak_pm25
):

    reasons = []

    if severity in [
        "CRITICAL",
        "SEVERE",
        "EXTREME",
    ]:
        reasons.append(
            "high_pm25_severity"
        )

    if peak_score >= 0.70:
        reasons.append(
            "high_intelligence_score"
        )

    if network_context == "LOCALIZED":
        reasons.append(
            "sensor_specific_signal"
        )

    elif network_context == "REGIONAL":
        reasons.append(
            "multi_sensor_agreement"
        )

    elif network_context == "MIXED":
        reasons.append(
            "partial_network_agreement"
        )

    if event_type == "SENSOR_ANOMALY":
        reasons.append(
            "sensor_anomaly_signal"
        )

    elif event_type == "UNEXPECTED_RISE":
        reasons.append(
            "unexpected_rise_signal"
        )

    elif event_type == "FORECAST_SURPRISE":
        reasons.append(
            "forecast_surprise_signal"
        )

    elif event_type in [
        "REGIONAL_EVENT",
        "EXTREME_REGIONAL_EVENT",
    ]:
        reasons.append(
            "regional_event_signal"
        )

    return ";".join(reasons)


# =========================================================
# 8. EVENT AGGREGATION
# =========================================================

event_rows = []


for event_id, group in alerts.groupby(
    "event_id",
    sort=False
):

    group = group.sort_values(
        "period_from_utc"
    ).copy()

    # -----------------------------------------------------
    # Basic event values
    # -----------------------------------------------------

    start_time = (
        group["period_from_utc"].min()
    )

    end_time = (
        group["period_from_utc"].max()
    )

    peak_index = group["value"].idxmax()

    peak_row = group.loc[
        peak_index
    ]

    peak_time = (
        peak_row["period_from_utc"]
    )

    peak_pm25 = float(
        group["value"].max()
    )

    # -----------------------------------------------------
    # Event classification
    # -----------------------------------------------------

    event_type = classify_event(
        group
    )

    confidence_tier = get_confidence(
        group
    )

    # -----------------------------------------------------
    # Forecast residual
    # -----------------------------------------------------

    peak_forecast_residual = (
        get_peak_forecast_residual(
            group
        )
    )

    if pd.notna(
        peak_forecast_residual
    ):

        peak_forecast_surprise = float(
            abs(
                peak_forecast_residual
            )
        )

    else:

        peak_forecast_surprise = np.nan

    # -----------------------------------------------------
    # Duration
    # -----------------------------------------------------

    duration_hours = (
        (
            end_time
            - start_time
        ).total_seconds()
        / 3600
    ) + 1

    # -----------------------------------------------------
    # Rise / peak / recovery
    # -----------------------------------------------------

    before_peak = group[
        group["period_from_utc"]
        < peak_time
    ]

    after_peak = group[
        group["period_from_utc"]
        > peak_time
    ]

    if len(before_peak) > 0:

        pre_peak_value = float(
            before_peak[
                "value"
            ].iloc[-1]
        )

    else:

        pre_peak_value = np.nan

    if len(after_peak) > 0:

        post_peak_value = float(
            after_peak[
                "value"
            ].iloc[-1]
        )

    else:

        post_peak_value = np.nan

    # -----------------------------------------------------
    # Event phase
    # -----------------------------------------------------

    if len(before_peak) == 0:

        if len(after_peak) > 0:
            event_phase = (
                "PEAK_AND_RECOVERY"
            )
        else:
            event_phase = "PEAK"

    elif len(after_peak) == 0:

        event_phase = (
            "RISING_TO_PEAK"
        )

    else:

        event_phase = (
            "RISE_PEAK_RECOVERY"
        )

    # -----------------------------------------------------
    # Recovery fraction
    # -----------------------------------------------------

    if (
        pd.notna(pre_peak_value)
        and
        pd.notna(post_peak_value)
        and
        peak_pm25 > pre_peak_value
    ):

        recovery_fraction = (
            peak_pm25
            - post_peak_value
        ) / (
            peak_pm25
            - pre_peak_value
        )

        recovery_fraction = float(
            np.clip(
                recovery_fraction,
                0,
                1
            )
        )

    else:

        recovery_fraction = np.nan

    # -----------------------------------------------------
    # Regional metrics
    #
    # Use peak observation for network context.
    # -----------------------------------------------------

    peak_regional_fraction = (
        peak_row[
            "regional_elevated_fraction"
        ]
    )

    peak_regional_median = (
        peak_row[
            "regional_median"
        ]
    )

    if pd.notna(
        peak_regional_fraction
    ):

        peak_regional_fraction = float(
            peak_regional_fraction
        )

    else:

        peak_regional_fraction = np.nan

    if pd.notna(
        peak_regional_median
    ):

        peak_regional_median = float(
            peak_regional_median
        )

    else:

        peak_regional_median = np.nan

    # -----------------------------------------------------
    # Network context
    # -----------------------------------------------------

    network_context = (
        classify_network_context(
            peak_regional_fraction
        )
    )

    sensor_specificity = (
        calculate_sensor_specificity(
            peak_regional_fraction
        )
    )

    # -----------------------------------------------------
    # Network coverage
    #
    # regional intelligence contains sensors_reporting
    # at each timestamp.
    # -----------------------------------------------------

    if "sensors_reporting" in group.columns:

        peak_sensors_reporting = (
            peak_row[
                "sensors_reporting"
            ]
        )

        if pd.notna(
            peak_sensors_reporting
        ):

            peak_sensors_reporting = int(
                peak_sensors_reporting
            )

            network_coverage = float(
                np.clip(
                    peak_sensors_reporting
                    / TOTAL_SENSORS,
                    0,
                    1
                )
            )

        else:

            peak_sensors_reporting = np.nan
            network_coverage = np.nan

    else:

        peak_sensors_reporting = np.nan
        network_coverage = np.nan

    # -----------------------------------------------------
    # Regional support
    #
    # Conservative definition:
    # ≥65% sensors elevated at peak.
    # -----------------------------------------------------

    regional_event_supported = bool(
        pd.notna(
            peak_regional_fraction
        )
        and
        peak_regional_fraction >= 0.65
    )

    # -----------------------------------------------------
    # Peak scores
    # -----------------------------------------------------

    peak_anomaly_score = float(
        group[
            "anomaly_score"
        ].max()
    )

    peak_final_score = float(
        group[
            "final_intelligence_score"
        ].max()
    )

    # -----------------------------------------------------
    # Confidence reason
    # -----------------------------------------------------

    confidence_reason = (
        build_confidence_reason(
            event_type,
            network_context,
            "UNKNOWN",
            peak_final_score,
            peak_pm25
        )
    )

    # -----------------------------------------------------
    # Build row
    # -----------------------------------------------------

    event_rows.append({

        "event_id": event_id,

        "sensor_id": group[
            "sensor_id"
        ].iloc[0],

        "event_type": event_type,

        "network_context": network_context,

        "regional_event_supported": (
            regional_event_supported
        ),

        "event_phase": event_phase,

        "start_time": start_time,

        "peak_time": peak_time,

        "end_time": end_time,

        "duration_hours": duration_hours,

        "peak_pm25": peak_pm25,

        "mean_pm25": float(
            group["value"].mean()
        ),

        "start_pm25": float(
            group["value"].iloc[0]
        ),

        "end_pm25": float(
            group["value"].iloc[-1]
        ),

        "pre_peak_pm25": (
            pre_peak_value
        ),

        "post_peak_pm25": (
            post_peak_value
        ),

        "recovery_fraction": (
            recovery_fraction
        ),

        "peak_expected_pm25": (
            float(
                peak_row[
                    "expected_pm25_1h"
                ]
            )
            if pd.notna(
                peak_row[
                    "expected_pm25_1h"
                ]
            )
            else np.nan
        ),

        "peak_forecast_residual": (
            float(
                peak_forecast_residual
            )
            if pd.notna(
                peak_forecast_residual
            )
            else np.nan
        ),

        "peak_forecast_surprise": (
            peak_forecast_surprise
        ),

        "peak_anomaly_score": (
            peak_anomaly_score
        ),

        "peak_final_score": (
            peak_final_score
        ),

        "peak_regional_median": (
            peak_regional_median
        ),

        "peak_regional_elevated_fraction": (
            peak_regional_fraction
        ),

        "peak_sensors_reporting": (
            peak_sensors_reporting
        ),

        "network_coverage": (
            network_coverage
        ),

        "sensor_specificity": (
            sensor_specificity
        ),

        "confidence_tier": (
            confidence_tier
        ),

        "confidence_reason": (
            confidence_reason
        ),

        "observation_count": len(
            group
        ),

        "alert_observation_count": len(
            group
        ),
    })


# =========================================================
# 9. CREATE EVENT DATAFRAME
# =========================================================

events = pd.DataFrame(
    event_rows
)


# =========================================================
# 10. EVENT SEVERITY
# =========================================================

events["severity"] = np.select(

    [
        events["peak_pm25"] >= 5000,
        events["peak_pm25"] >= 1000,
        events["peak_pm25"] >= 500,
        events["peak_pm25"] >= 300,
        events["peak_pm25"] >= 100,
    ],

    [
        "CRITICAL",
        "SEVERE",
        "EXTREME",
        "HIGH",
        "MODERATE",
    ],

    default="LOW",
)


# =========================================================
# 11. EVENT TYPE REFINEMENT
#
# High PM2.5 + weak network agreement means localized
# sensor anomaly rather than regional pollution.
# =========================================================

events.loc[
    (
        events["peak_pm25"] >= 500
    )
    &
    (
        events[
            "peak_regional_elevated_fraction"
        ] < 0.50
    )
    &
    (
        events["event_type"]
        == "ELEVATED"
    ),
    "event_type"
] = "SENSOR_ANOMALY"


# =========================================================
# 12. REGIONAL EVENT REFINEMENT
#
# Only promote to regional event when the original
# intelligence system supports it.
# =========================================================

events.loc[
    (
        events[
            "regional_event_supported"
        ]
        == True
    )
    &
    (
        events[
            "peak_final_score"
        ] >= 0.60
    ),
    "event_type"
] = "REGIONAL_EVENT"


# =========================================================
# 13. EVENT DESCRIPTION
# =========================================================

events["event_description"] = (

    events[
        "event_type"
    ].astype(str)

    + " | "

    + events[
        "network_context"
    ].astype(str)

    + " | "

    + events[
        "severity"
    ].astype(str)

    + " | Peak PM2.5="

    + events[
        "peak_pm25"
    ].round(1).astype(str)

    + " µg/m³ | Score="

    + events[
        "peak_final_score"
    ].round(3).astype(str)

    + " | Phase="

    + events[
        "event_phase"
    ].astype(str)
)


# =========================================================
# 14. SEVERITY RANK
# =========================================================

severity_order = {

    "CRITICAL": 5,
    "SEVERE": 4,
    "EXTREME": 3,
    "HIGH": 2,
    "MODERATE": 1,
    "LOW": 0,
}

events["severity_rank"] = (
    events["severity"]
    .map(severity_order)
)


# =========================================================
# 15. SORT
# =========================================================

events = (
    events
    .sort_values(
        [
            "severity_rank",
            "peak_final_score",
            "peak_pm25",
        ],
        ascending=False,
    )
    .drop(
        columns=[
            "severity_rank"
        ]
    )
    .reset_index(drop=True)
)


# =========================================================
# 16. SAVE
# =========================================================

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

events.to_csv(
    OUTPUT,
    index=False
)


# =========================================================
# 17. SUMMARY
# =========================================================

print("\n" + "=" * 70)
print("EVENT ENGINE V3 COMPLETE")
print("=" * 70)

print(
    f"Alert observations: {len(alerts):,}"
)

print(
    f"Events generated:   {len(events):,}"
)

print(
    f"Unique sensors:      "
    f"{events['sensor_id'].nunique():,}"
)


print("\nEvent severity:")

print(
    events[
        "severity"
    ]
    .value_counts()
    .to_string()
)


print("\nEvent types:")

print(
    events[
        "event_type"
    ]
    .value_counts()
    .to_string()
)


print("\nNetwork context:")

print(
    events[
        "network_context"
    ]
    .value_counts()
    .to_string()
)


print("\nRegional event support:")

print(
    events[
        "regional_event_supported"
    ]
    .value_counts()
    .to_string()
)


print("\nEvent phases:")

print(
    events[
        "event_phase"
    ]
    .value_counts()
    .to_string()
)


print("\nTop events:")

top = events[
    [
        "event_id",
        "sensor_id",
        "event_type",
        "network_context",
        "event_phase",
        "severity",
        "start_time",
        "peak_time",
        "end_time",
        "peak_pm25",
        "peak_expected_pm25",
        "peak_forecast_residual",
        "peak_forecast_surprise",
        "peak_regional_elevated_fraction",
        "network_coverage",
        "sensor_specificity",
        "peak_final_score",
        "confidence_tier",
    ]
].head(20)


print(
    top.to_string(
        index=False
    )
)


print("\nOutput:")
print(OUTPUT)

print("=" * 70)