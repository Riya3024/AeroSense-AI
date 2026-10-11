
"""One-hour-ahead PM2.5 predictions using the saved AeroSense XGBoost model."""
from __future__ import annotations

import json
import math
from functools import lru_cache

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from .research_data import FEATURES_FILE, MODEL_FILE


@lru_cache(maxsize=1)
def _load_predictor():
    """Load the model and feature list once per running process."""
    if not MODEL_FILE.is_file():
        raise FileNotFoundError("The PM2.5 model file is unavailable")

    if not FEATURES_FILE.is_file():
        raise FileNotFoundError("The PM2.5 feature metadata is unavailable")

    with FEATURES_FILE.open("r", encoding="utf-8") as file:
        features = json.load(file)

    if (
        not isinstance(features, list)
        or not features
        or not all(isinstance(name, str) and name for name in features)
        or len(features) != len(set(features))
    ):
        raise RuntimeError("The PM2.5 feature metadata is invalid")

    model = XGBRegressor()
    model.load_model(str(MODEL_FILE))
    return model, features


def predict_pm25(payload: object) -> dict:
    """Predict PM2.5 one hour ahead from a complete model feature vector."""
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object")

    values = payload.get("features")
    if not isinstance(values, dict):
        raise ValueError("Provide a 'features' object containing model inputs")

    model, feature_names = _load_predictor()

    missing = [name for name in feature_names if name not in values]
    extra = [name for name in values if name not in feature_names]

    if missing:
        raise ValueError(f"Missing required features: {', '.join(missing)}")
    if extra:
        raise ValueError(f"Unexpected features: {', '.join(extra)}")

    cleaned = {}
    for name in feature_names:
        value = values[name]

        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Feature '{name}' must be a number")

        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"Feature '{name}' must be finite")

        cleaned[name] = number

    frame = pd.DataFrame(
        [[cleaned[name] for name in feature_names]],
        columns=feature_names,
    )

    prediction = float(np.asarray(model.predict(frame)).reshape(-1)[0])
    if not math.isfinite(prediction):
        raise RuntimeError("The model returned a non-finite prediction")

    return {
        "status": "success",
        "forecast_horizon_hours": 1,
        "predicted_pm25": max(0.0, prediction),
        "unit": "µg/m³",
        "model": "AeroSense XGBoost",
        "input_type": "engineered_features",
        "live_sensor_data_used": False,
    }



def predict_latest_historical() -> dict:
    """Predict one hour ahead from the latest complete historical feature row."""
    from pathlib import Path

    data_file = (
        Path(__file__).resolve().parent.parent
        / "data"
        / "processed"
        / "openaq_pm25_features.csv"
    )

    if not data_file.is_file():
        raise FileNotFoundError("Historical feature dataset is unavailable")

    _, feature_names = _load_predictor()
    frame = pd.read_csv(data_file)

    missing_columns = [
        name for name in feature_names if name not in frame.columns
    ]
    if missing_columns:
        raise RuntimeError(
            "Historical dataset is missing model features: "
            + ", ".join(missing_columns)
        )

    complete = frame.dropna(subset=feature_names)
    if complete.empty:
        raise RuntimeError("No complete historical feature rows are available")

    # Use the newest timestamp when the dataset provides one.
    timestamp_column = "period_from_utc"
    if timestamp_column in complete.columns:
        complete = complete.sort_values(timestamp_column)

    latest = complete.iloc[-1]
    features = {
        name: float(latest[name])
        for name in feature_names
    }

    result = predict_pm25({"features": features})
    result["input_type"] = "latest_historical_engineered_features"
    result["live_sensor_data_used"] = False
    result["source"] = str(data_file.name)
    result["source_timestamp"] = (
        str(latest[timestamp_column])
        if timestamp_column in complete.columns
        else None
    )
    return result