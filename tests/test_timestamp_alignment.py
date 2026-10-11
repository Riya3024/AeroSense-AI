import pandas as pd
import pytest

from data_pipeline.fetch_openaq import parse_hour_record
from data_pipeline.fetch_weather import parse_weather_response
from data_pipeline.http_utils import FetchError
from data_pipeline.merge_datasets import clean_openaq, merge_pollution_weather, regularize_hourly
from tests.conftest import raw_openaq_row


def test_openaq_uses_period_end_in_utc_not_local_time(settings):
    # Shape follows the documented /sensors/{id}/hours schema (synthetic values).
    rec = {
        "value": 42.0,
        "parameter": {"id": 2, "name": "pm25", "units": "\u00b5g/m\u00b3"},
        "period": {
            "datetimeFrom": {"utc": "2025-11-01T00:00:00Z", "local": "2025-11-01T05:30:00+05:30"},
            "datetimeTo": {"utc": "2025-11-01T01:00:00Z", "local": "2025-11-01T06:30:00+05:30"},
        },
        "coverage": {"percentComplete": 100.0}, "flagInfo": {"hasFlags": False},
    }
    row = parse_hour_record(rec, station_id=1, sensor_id=10)
    clean, _ = clean_openaq(pd.DataFrame([row]), settings)
    assert clean["timestamp_utc"].iloc[0] == pd.Timestamp("2025-11-01 01:00", tz="UTC")


def test_falls_back_to_period_start_plus_one_hour(settings):
    raw = pd.DataFrame([raw_openaq_row(None, 10.0, from_utc="2025-11-01T00:00:00Z")])
    clean, _ = clean_openaq(raw, settings)
    assert clean["timestamp_utc"].iloc[0] == pd.Timestamp("2025-11-01 01:00", tz="UTC")


def test_rows_not_on_the_hour_are_dropped_and_counted(settings):
    raw = pd.DataFrame([raw_openaq_row("2025-11-01T01:00:00Z", 10.0), raw_openaq_row("2025-11-01T02:30:00Z", 11.0)])
    clean, stats = clean_openaq(raw, settings)
    assert len(clean) == 1 and stats["dropped_not_on_the_hour"] == 1


def test_unexpected_units_are_dropped_not_converted(settings):
    raw = pd.DataFrame([raw_openaq_row("2025-11-01T01:00:00Z", 10.0, unit="ppm"),
                        raw_openaq_row("2025-11-01T02:00:00Z", 11.0, unit="ug/m3")])
    clean, stats = clean_openaq(raw, settings)
    assert clean["pm25"].tolist() == [11.0] and stats["dropped_unexpected_unit"] == 1


def _payload(offset=0):
    return {"latitude": 28.6, "longitude": 77.2, "elevation": 215.0, "utc_offset_seconds": offset,
            "hourly": {"time": ["2025-11-01T00:00", "2025-11-01T01:00"], "temperature_2m": [20.0, None]},
            "hourly_units": {"temperature_2m": "\u00b0C"}}


def test_weather_timestamps_are_parsed_as_utc_and_nulls_become_nan():
    df, units = parse_weather_response(_payload(), "28.60_77.20", ("temperature_2m",))
    assert str(df["timestamp_utc"].dt.tz) == "UTC"
    assert df["timestamp_utc"].iloc[1] == pd.Timestamp("2025-11-01 01:00", tz="UTC")
    assert pd.isna(df["temperature_2m"].iloc[1]) and units["temperature_2m"] == "\u00b0C"


def test_weather_non_utc_response_is_rejected():
    with pytest.raises(FetchError):
        parse_weather_response(_payload(offset=19800), "p", ("temperature_2m",))


def test_merge_matches_same_hour_only_and_uses_station_point(settings):
    poll = regularize_hourly(pd.DataFrame({
        "station_id": [1, 1, 1], "timestamp_utc": pd.date_range("2025-11-01 10:00", periods=3, freq="1h", tz="UTC"),
        "pm25": [1.0, 2.0, 3.0]}))
    weather = pd.DataFrame({
        "weather_point_id": "P", "timestamp_utc": pd.date_range("2025-11-01 09:00", periods=5, freq="1h", tz="UTC"),
        "temperature_2m": [9.0, 10.0, 11.0, 12.0, 13.0],
        "weather_grid_latitude": 28.6, "weather_grid_longitude": 77.2})
    stations = pd.DataFrame({"station_id": [1], "station_name": ["s"], "latitude": [28.6], "longitude": [77.2]})
    points = pd.DataFrame({"station_id": [1], "weather_point_id": ["P"]})
    m = merge_pollution_weather(poll, weather, stations, points)
    assert m["temperature_2m"].tolist() == [10.0, 11.0, 12.0]  # hour 10 gets weather of hour 10
    assert m["weather_grid_distance_km"].max() < 0.01
