"""Shared fixtures. All data here is SYNTHETIC and exists only to exercise code paths in unit tests."""
import pandas as pd
import pytest

from data_pipeline.config import Settings

START = pd.Timestamp("2025-11-01 00:00", tz="UTC")


@pytest.fixture
def settings(tmp_path):
    return Settings(
        openaq_api_key="test-key-123",
        root=tmp_path,
        weather_variables=("temperature_2m", "relative_humidity_2m", "wind_speed_10m", "wind_direction_10m", "precipitation"),
        backoff_s=0.0,
        min_request_interval_s=0.0,
        page_limit=2,
        max_retries=3,
    )


def hourly_index(n, start=START):
    return pd.date_range(start, periods=n, freq="1h", name="timestamp_utc")


def make_merged(n=48, stations=(1,), base=None):
    """Regular-grid merged frame. pm25 = base + hour index (per station), temperature = hour index."""
    frames = []
    for i, sid in enumerate(stations):
        b = 100.0 * (i + 1) if base is None else base
        idx = hourly_index(n)
        frames.append(pd.DataFrame({
            "station_id": sid,
            "timestamp_utc": idx,
            "pm25": [b + k for k in range(n)],
            "latitude": 28.6 + 0.01 * i,
            "longitude": 77.2,
            "temperature_2m": [float(k) for k in range(n)],
            "relative_humidity_2m": [50.0 + (k % 10) for k in range(n)],
            "wind_speed_10m": [5.0] * n,
            "wind_direction_10m": [90.0] * n,
            "precipitation": [0.0] * n,
            "wind_u_10m": [-5.0] * n,
            "wind_v_10m": [0.0] * n,
        }))
    return pd.concat(frames, ignore_index=True)


def raw_openaq_row(to_utc, value, station_id=1, unit="\u00b5g/m\u00b3", pct=100.0, from_utc=None):
    """A flattened raw row shaped like fetch_openaq.RAW_COLUMNS."""
    return {
        "station_id": station_id, "sensor_id": 10, "parameter": "pm25", "unit": unit, "value": value,
        "period_from_utc": from_utc, "period_to_utc": to_utc, "percent_complete": pct, "has_flags": False,
    }
