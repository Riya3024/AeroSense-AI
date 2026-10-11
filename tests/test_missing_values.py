import numpy as np
import pandas as pd

from data_pipeline.config import TARGET_COL
from data_pipeline.merge_datasets import build_supervised, clean_openaq, clean_weather, ffill_short_gaps, regularize_hourly
from tests.conftest import make_merged, raw_openaq_row


def test_null_negative_and_implausible_pm25_become_missing_not_zero(settings):
    raw = pd.DataFrame([
        raw_openaq_row("2025-11-01T01:00:00Z", None),
        raw_openaq_row("2025-11-01T02:00:00Z", -5.0),
        raw_openaq_row("2025-11-01T03:00:00Z", 99999.0),
        raw_openaq_row("2025-11-01T04:00:00Z", 55.0),
    ])
    clean, stats = clean_openaq(raw, settings)
    assert clean["pm25"].tolist() == [55.0]
    assert stats["null_values"] == 1 and stats["negative_values_set_missing"] == 1 and stats["implausible_values_set_missing"] == 1


def test_missing_hours_reappear_as_nan_and_are_never_interpolated():
    df = pd.DataFrame({"station_id": 1, "timestamp_utc": pd.to_datetime(["2025-11-01 00:00", "2025-11-01 03:00"], utc=True),
                       "pm25": [10.0, 40.0]})
    grid = regularize_hourly(df)
    assert len(grid) == 4
    assert grid["pm25"].isna().tolist() == [False, True, True, False]  # 20 and 30 are NOT invented


def test_ffill_fills_short_gap_only_causally():
    s = pd.Series([1.0, np.nan, np.nan, 4.0])
    assert ffill_short_gaps(s, 3).tolist() == [1.0, 1.0, 1.0, 4.0]  # uses the PAST value, not a blend with 4.0


def test_ffill_leaves_long_gaps_and_leading_nan_untouched():
    s = pd.Series([np.nan, 1.0] + [np.nan] * 5 + [9.0])
    out = ffill_short_gaps(s, 3)
    assert out.isna().tolist() == [True, False] + [True] * 5 + [False]


def test_weather_ffill_only_applies_to_allowlisted_variables(settings):
    idx = pd.date_range("2025-11-01", periods=4, freq="1h", tz="UTC")
    raw = pd.DataFrame({"weather_point_id": "P", "timestamp_utc": idx,
                        "temperature_2m": [10.0, np.nan, 12.0, 13.0],
                        "precipitation": [1.0, np.nan, 0.0, 0.0],
                        "weather_grid_latitude": 28.6, "weather_grid_longitude": 77.2, "weather_grid_elevation": 200.0})
    out, stats = clean_weather(raw, settings)
    assert out["temperature_2m"].tolist() == [10.0, 10.0, 12.0, 13.0]
    assert np.isnan(out["precipitation"].iloc[1])  # sums are never filled
    assert stats["gap_filled_values_by_variable"]["temperature_2m"] == 1


def test_weather_hour_missing_from_response_is_reinserted_as_nan(settings):
    idx = pd.to_datetime(["2025-11-01 00:00", "2025-11-01 02:00"], utc=True)
    raw = pd.DataFrame({"weather_point_id": "P", "timestamp_utc": idx, "precipitation": [0.0, 0.0],
                        "weather_grid_latitude": 28.6, "weather_grid_longitude": 77.2, "weather_grid_elevation": 200.0})
    out, _ = clean_weather(raw, settings)
    assert len(out) == 3 and np.isnan(out["precipitation"].iloc[1])


def test_rows_with_missing_target_or_core_features_are_dropped(settings):
    merged = make_merged(n=60)
    merged.loc[40, "pm25"] = np.nan  # hour 40 missing
    ds, _ = build_supervised(merged, settings)
    assert ds[TARGET_COL].notna().all()
    ts = set(ds["timestamp_utc"])
    assert merged["timestamp_utc"].iloc[39] not in ts  # its target would be hour 40 (missing)
    assert merged["timestamp_utc"].iloc[40] not in ts  # its own value is missing
    assert merged["timestamp_utc"].iloc[41] not in ts  # lag_1h would be hour 40 (missing)
