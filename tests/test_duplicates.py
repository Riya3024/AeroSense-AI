import pandas as pd

from data_pipeline.fetch_openaq import date_chunks
from data_pipeline.merge_datasets import clean_openaq, clean_weather, merge_pollution_weather, regularize_hourly, remove_duplicates
from datetime import date
from tests.conftest import raw_openaq_row


def test_duplicate_keeps_highest_coverage():
    df = pd.DataFrame({"station_id": [1, 1], "timestamp_utc": [pd.Timestamp("2025-11-01", tz="UTC")] * 2,
                       "pm25": [10.0, 99.0], "percent_complete": [100.0, 50.0]})
    out, removed = remove_duplicates(df, ["station_id", "timestamp_utc"])
    assert removed == 1 and out["pm25"].tolist() == [10.0]


def test_duplicate_tie_keeps_last_row():
    df = pd.DataFrame({"station_id": [1, 1], "timestamp_utc": [pd.Timestamp("2025-11-01", tz="UTC")] * 2,
                       "pm25": [10.0, 20.0], "percent_complete": [100.0, 100.0]})
    out, _ = remove_duplicates(df, ["station_id", "timestamp_utc"])
    assert out["pm25"].tolist() == [20.0]


def test_same_timestamp_at_different_stations_is_not_a_duplicate(settings):
    raw = pd.DataFrame([raw_openaq_row("2025-11-01T01:00:00Z", 10.0, station_id=1),
                        raw_openaq_row("2025-11-01T01:00:00Z", 20.0, station_id=2)])
    clean, stats = clean_openaq(raw, settings)
    assert len(clean) == 2 and stats["duplicate_rows_removed"] == 0


def test_chunk_boundary_double_count_is_removed(settings):
    # Adjacent chunks can both return the boundary hour; it must collapse to one row.
    raw = pd.DataFrame([raw_openaq_row("2025-12-01T00:00:00Z", 10.0), raw_openaq_row("2025-12-01T00:00:00Z", 10.0),
                        raw_openaq_row("2025-12-01T01:00:00Z", 11.0)])
    clean, stats = clean_openaq(raw, settings)
    assert len(clean) == 2 and stats["duplicate_rows_removed"] == 1


def test_date_chunks_are_contiguous_and_cover_range():
    chunks = list(date_chunks(date(2025, 11, 1), date(2026, 1, 31), 30))
    assert chunks[0][0] == "2025-11-01T00:00:00Z" and chunks[-1][1] == "2026-02-01T00:00:00Z"
    assert all(a[1] == b[0] for a, b in zip(chunks, chunks[1:]))


def test_weather_duplicates_removed_so_merge_is_many_to_one(settings):
    idx = pd.date_range("2025-11-01", periods=3, freq="1h", tz="UTC")
    raw = pd.DataFrame({"weather_point_id": "P", "timestamp_utc": list(idx) + [idx[1]],
                        "temperature_2m": [1.0, 2.0, 3.0, 2.0], "weather_grid_latitude": 28.6,
                        "weather_grid_longitude": 77.2, "weather_grid_elevation": 200.0})
    weather, stats = clean_weather(raw, settings)
    assert stats["duplicate_rows_removed"] == 1 and len(weather) == 3
    poll = regularize_hourly(pd.DataFrame({"station_id": [1, 1], "timestamp_utc": idx[:2], "pm25": [1.0, 2.0]}))
    stations = pd.DataFrame({"station_id": [1], "station_name": ["s"], "latitude": [28.6], "longitude": [77.2]})
    points = pd.DataFrame({"station_id": [1], "weather_point_id": ["P"]})
    merged = merge_pollution_weather(poll, weather, stations, points)  # validate="many_to_one" would raise on dupes
    assert len(merged) == 2
