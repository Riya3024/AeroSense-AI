"""Wiring test for merge_datasets.run + validate_data.run on SYNTHETIC raw files in a temp dir.
It proves the CLI plumbing and file formats work; it says nothing about real OpenAQ/Open-Meteo data."""
import json

import numpy as np
import pandas as pd

from data_pipeline import merge_datasets, validate_data
from data_pipeline.config import TARGET_COL


def _write_synthetic_raw(s, n=72):
    s.raw_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1)
    rows = []
    for sid in (1, 2):
        for h in range(1, n + 1):
            if sid == 2 and h == 30:
                continue  # a missing hour
            to = pd.Timestamp("2025-11-01", tz="UTC") + pd.Timedelta(hours=h)
            rows.append({"station_id": sid, "sensor_id": 10 + sid, "parameter": "pm25", "unit": "\u00b5g/m\u00b3",
                         "value": 80 + 20 * sid + 10 * np.sin(h / 5) + rng.normal(0, 2),
                         "period_from_utc": (to - pd.Timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                         "period_to_utc": to.strftime("%Y-%m-%dT%H:%M:%SZ"), "percent_complete": 100.0, "has_flags": False})
    rows.append(rows[5].copy())  # exact duplicate row
    pd.DataFrame(rows).to_csv(s.raw_dir / "openaq_pm25_hourly_raw.csv", index=False)
    pd.DataFrame({"station_id": [1, 2], "station_name": ["a", "b"], "latitude": [28.61, 28.70], "longitude": [77.20, 77.10]}).to_csv(s.raw_dir / "stations.csv", index=False)
    pd.DataFrame({"station_id": [1, 2], "weather_point_id": ["P1", "P2"], "requested_latitude": [28.61, 28.70], "requested_longitude": [77.20, 77.10]}).to_csv(s.raw_dir / "station_weather_points.csv", index=False)
    wx = []
    for pid in ("P1", "P2"):
        for h in range(0, n + 3):
            t = pd.Timestamp("2025-11-01", tz="UTC") + pd.Timedelta(hours=h)
            wx.append({"weather_point_id": pid, "timestamp_utc": t.strftime("%Y-%m-%dT%H:%M:%SZ"), "temperature_2m": 20 + h % 12,
                       "relative_humidity_2m": 60.0, "wind_speed_10m": 8.0, "wind_direction_10m": 270.0, "precipitation": 0.0,
                       "weather_grid_latitude": 28.5, "weather_grid_longitude": 77.25, "weather_grid_elevation": 215.0})
    pd.DataFrame(wx).to_csv(s.raw_dir / "weather_hourly_raw.csv", index=False)


def test_merge_then_validate_end_to_end(settings):
    _write_synthetic_raw(settings)
    assert merge_datasets.run(settings) == 0
    out = pd.read_csv(settings.processed_dir / "supervised_pm25_t_plus_1h.csv", parse_dates=["timestamp_utc"])
    # Hand-derived row count. Per station: 72 hours; lag_24h removes the first 24 and the missing
    # target removes the last -> hours 25..71 = 47 rows. Station 2 also lacks hour 30, which removes
    # hour 29 (target missing), 30 (value missing), 31 (lag_1h missing) and 54 (lag_24h missing) -> 43.
    assert len(out) == 47 + 43 and out[TARGET_COL].notna().all()
    assert out.groupby("station_id").size().to_dict() == {1: 47, 2: 43}
    stats = json.loads((settings.processed_dir / "merge_stats.json").read_text())
    assert stats["openaq"]["duplicate_rows_removed"] == 1
    assert stats["openaq"]["missing_hours_on_grid"] == 1
    assert validate_data.run(settings) == 0
    report = json.loads((settings.processed_dir / "data_quality_report.json").read_text())
    assert report["errors"] == [] and report["row_counts"]["supervised"] == len(out)
    assert (settings.processed_dir / "data_quality_report.md").exists()


def test_validate_fails_when_processed_file_is_tampered_with_leakage(settings):
    _write_synthetic_raw(settings)
    assert merge_datasets.run(settings) == 0
    p = settings.processed_dir / "supervised_pm25_t_plus_1h.csv"
    df = pd.read_csv(p)
    df["pm25_lag_1h"] = df[TARGET_COL]  # inject leakage into the saved file
    df.to_csv(p, index=False)
    assert validate_data.run(settings) == 1
