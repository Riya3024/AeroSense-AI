import json
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor


# =========================================================
# AEROSENSE — FORECAST RESIDUAL INTELLIGENCE
# =========================================================

FEATURES = Path(
    "data/processed/openaq_pm25_features.csv"
)

MODEL_PATH = Path(
    "models/xgboost_pm25.json"
)

FEATURE_PATH = Path(
    "models/xgboost_features.json"
)

OUTPUT = Path(
    "data/processed/openaq_forecast_residuals.csv"
)


DROP_COLUMNS = [
    "target_1h",
    "period_from_utc",
    "period_to_utc",
    "pollutant",
    "parameter",
    "unit",
]


print("=" * 70)
print("AEROSENSE — FORECAST RESIDUAL INTELLIGENCE")
print("=" * 70)


# ---------------------------------------------------------
# 1. Load model feature metadata
# ---------------------------------------------------------

print("\nLoading model feature metadata...")

with open(FEATURE_PATH, "r", encoding="utf-8") as f:
    feature_columns = json.load(f)

print(f"Model features: {len(feature_columns)}")


# ---------------------------------------------------------
# 2. Load XGBoost model
# ---------------------------------------------------------

print("\nLoading XGBoost model...")

model = XGBRegressor()
model.load_model(MODEL_PATH)

print("Model loaded successfully.")


# ---------------------------------------------------------
# 3. Load feature dataset
# ---------------------------------------------------------

print("\nLoading feature dataset...")

df = pd.read_csv(
    FEATURES,
    parse_dates=["period_from_utc"]
)

print(f"Rows:    {len(df):,}")
print(f"Sensors: {df['sensor_id'].nunique():,}")


# ---------------------------------------------------------
# 4. Build feature matrix
# ---------------------------------------------------------

X = df.drop(
    columns=DROP_COLUMNS,
    errors="ignore"
)

X = X.select_dtypes(
    include=[np.number]
)

# Ensure exact feature ordering used by XGBoost.
X = X[feature_columns]


# ---------------------------------------------------------
# 5. Generate predictions
# ---------------------------------------------------------

print("\nGenerating XGBoost forecasts...")

df["expected_pm25_1h"] = model.predict(X)


# ---------------------------------------------------------
# 6. Calculate forecast residual
# ---------------------------------------------------------

df["forecast_residual"] = (
    df["target_1h"] -
    df["expected_pm25_1h"]
)

df["forecast_abs_residual"] = (
    df["forecast_residual"].abs()
)


# ---------------------------------------------------------
# 7. Forecast surprise score
# ---------------------------------------------------------

# Residuals above ~100 µg/m³ are considered highly
# surprising relative to the model expectation.

df["forecast_surprise_score"] = (
    df["forecast_abs_residual"] / 100.0
).clip(0, 1)


# ---------------------------------------------------------
# 8. Direction of surprise
# ---------------------------------------------------------

df["forecast_surprise_direction"] = np.select(
    [
        df["forecast_residual"] >= 100,
        df["forecast_residual"] <= -100,
    ],
    [
        "UNEXPECTED_RISE",
        "UNEXPECTED_DROP",
    ],
    default="EXPECTED_RANGE",
)


# ---------------------------------------------------------
# 9. Save
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
# 10. Summary
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("FORECAST RESIDUAL COMPLETE")
print("=" * 70)

print(
    f"Rows:                    {len(df):,}"
)

print(
    f"Mean absolute residual:  "
    f"{df['forecast_abs_residual'].mean():.4f}"
)

print(
    f"Median absolute residual:"
    f" {df['forecast_abs_residual'].median():.4f}"
)

print(
    f"Unexpected rises:        "
    f"{(df['forecast_residual'] >= 100).sum():,}"
)

print(
    f"Unexpected drops:        "
    f"{(df['forecast_residual'] <= -100).sum():,}"
)


print("\nTop forecast surprises:")

top = (
    df.sort_values(
        "forecast_abs_residual",
        ascending=False
    )
    [
        [
            "period_from_utc",
            "sensor_id",
            "target_1h",
            "expected_pm25_1h",
            "forecast_residual",
            "forecast_abs_residual",
            "forecast_surprise_score",
            "forecast_surprise_direction",
        ]
    ]
    .head(20)
)

print(
    top.to_string(index=False)
)


print("\nOutput:")
print(OUTPUT)

print("=" * 70)