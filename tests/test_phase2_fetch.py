"""Phase 2 fetch layer: multi-pollutant discovery/fetch, request estimate, hourly budget, end-date clamp, malformed payloads.
HTTP is a MOCK (scripted fake session, synthetic payloads in the documented response shapes). No network."""
import json
from dataclasses import replace
from datetime import date

import pandas as pd
import pytest

from data_pipeline import fetch_openaq, fetch_weather
from data_pipeline.fetch_openaq import OpenAQClient, estimate_requests, parse_hour_record, parse_location, sensors_for_row
from data_pipeline.http_utils import FetchError
from data_pipeline.merge_datasets import clean_openaq
from tests.test_fetch_clients import FakeResponse, FakeSession

UG = "\u00b5g/m\u00b3"


def _sensor(sid, name, display, units=UG):
    return {"id": sid, "name": name, "parameter": {"id": sid, "name": name, "units": units, "displayName": display}}


LOCATION = {"id": 7, "name": "Mock Station", "timezone": "Asia/Kolkata", "isMobile": False, "isMonitor": True,
            "coordinates": {"latitude": 28.6, "longitude": 77.2}, "provider": {"name": "P"},
            "sensors": [_sensor(71, "pm25", "PM2.5"), _sensor(70, "pm10", "PM10"), _sensor(72, "no2", "NO\u2082"), _sensor(74, "pm10", "PM10"),
                        _sensor(75, "o3", "O\u2083")]}


def _hours(name, units, n=2):
    return {"results": [{"value": 10.0 + i, "parameter": {"id": 1, "name": name, "units": units},
                         "period": {"datetimeFrom": {"utc": f"2025-11-01T{i:02d}:00:00Z"}, "datetimeTo": {"utc": f"2025-11-01T{i + 1:02d}:00:00Z"}},
                         "coverage": {"percentComplete": 100.0}, "flagInfo": {"hasFlags": False}} for i in range(n)]}


def _router(fail_sensor=None):
    names = {71: "pm25", 70: "pm10", 72: "no2"}

    def route(url, params):
        if url.endswith("/locations"):
            return FakeResponse(payload={"results": [LOCATION]})
        for sid, name in names.items():
            if f"/sensors/{sid}/hours" in url:
                if sid == fail_sensor:
                    return FakeResponse(500, text="boom")
                return FakeResponse(payload=_hours(name, UG) if params["page"] == 1 else {"results": []})
        return FakeResponse(404, text="unexpected " + url)
    return route


def _s(settings, **kw):
    base = dict(page_limit=5, start_date=date(2025, 11, 1), end_date=date(2025, 11, 2), pollutants=("pm25", "pm10", "no2"))
    return replace(settings, **{**base, **kw})


def test_discovery_maps_each_requested_pollutant_to_its_first_sensor_and_omits_the_rest():
    row = parse_location(LOCATION, ("pm25",), ("pm25", "pm10", "no2", "co", "o3"))
    mapping = json.loads(row["pollutant_sensors"])
    assert mapping == {"pm25": 71, "pm10": 70, "no2": 72, "o3": 75}  # first pm10 sensor wins; co is absent, not invented
    only_pm = json.loads(parse_location(LOCATION, ("pm25",), ("pm25",))["pollutant_sensors"])
    assert only_pm == {"pm25": 71}


def test_sensors_for_row_restricts_to_requested_and_falls_back_for_legacy_station_files():
    row = {"pollutant_sensors": json.dumps({"pm25": 1, "pm10": 2, "no2": 3})}
    assert sensors_for_row(row, ("pm25", "no2")) == {"pm25": 1, "no2": 3}
    assert sensors_for_row({"pollutant_sensors": float("nan"), "pm25_sensor_id": 9.0}, ("pm25", "pm10")) == {"pm25": 9}
    assert sensors_for_row({"pm25_sensor_id": float("nan")}, ("pm25",)) == {}


def test_run_fetches_every_requested_pollutant_and_tags_rows_with_it(settings):
    s = _s(settings)
    client = OpenAQClient(s, session=FakeSession(_router()), sleep=lambda x: None)
    assert fetch_openaq.run(s, client=client) == 0
    raw = pd.read_csv(s.raw_dir / "openaq_pm25_hourly_raw.csv")
    assert raw.groupby("pollutant").size().to_dict() == {"no2": 2, "pm10": 2, "pm25": 2}
    manifest = json.loads((s.raw_dir / "openaq_manifest.json").read_text())
    assert manifest["partial"] is False and manifest["sensors_by_pollutant"] == {"pm25": 1, "pm10": 1, "no2": 1}
    assert manifest["requests_made_this_run"] == client.requests_made > 3
    assert manifest["raw_rows_by_pollutant"] == {"no2": 2, "pm10": 2, "pm25": 2}


def test_one_failing_sensor_is_recorded_as_partial_and_strict_mode_exits_3(settings):
    s = _s(settings)
    client = OpenAQClient(s, session=FakeSession(_router(fail_sensor=72)), sleep=lambda x: None)
    assert fetch_openaq.run(s, client=client) == 0  # default: keep what we have, record the problem
    manifest = json.loads((s.raw_dir / "openaq_manifest.json").read_text())
    assert manifest["partial"] is True and "no2: FAILED" in manifest["station_status"]["7"]
    assert set(pd.read_csv(s.raw_dir / "openaq_pm25_hourly_raw.csv")["pollutant"]) == {"pm25", "pm10"}
    client2 = OpenAQClient(s, session=FakeSession(_router(fail_sensor=72)), sleep=lambda x: None)
    assert fetch_openaq.run(s, strict=True, client=client2) == 3  # finished sensors come from cache; the failed one is retried


def test_estimate_only_prints_the_plan_and_downloads_no_hours(settings):
    s = _s(settings, page_limit=1000)
    session = FakeSession(_router())
    client = OpenAQClient(s, session=session, sleep=lambda x: None)
    assert fetch_openaq.run(s, estimate_only=True, client=client) == 0
    assert not any("/hours" in url for url, _ in session.calls)


def test_estimate_requests_is_derived_from_stations_sensors_and_chunks(settings):
    s = _s(settings, page_limit=1000, chunk_days=30, start_date=date(2025, 1, 1), end_date=date(2025, 3, 31))  # 90 days -> 3 chunks
    stations = pd.DataFrame([{"station_id": 1, "pollutant_sensors": json.dumps({"pm25": 1, "pm10": 2, "no2": 3}), "datetime_first_utc": None, "datetime_last_utc": None},
                             {"station_id": 2, "pollutant_sensors": json.dumps({"pm25": 4}), "datetime_first_utc": None, "datetime_last_utc": None}])
    est = estimate_requests(stations, s)
    assert est["stations"] == 2 and est["sensors"] == 4 and est["requests_per_sensor"] == 3 and est["estimated_requests"] == 12


def test_hourly_request_budget_pauses_instead_of_exceeding_the_limit(settings):
    now, sleeps = [0.0], []
    s = replace(settings, max_requests_per_hour=3, min_request_interval_s=0.0)
    client = OpenAQClient(s, session=FakeSession([]), sleep=lambda secs: (sleeps.append(secs), now.__setitem__(0, now[0] + secs)), clock=lambda: now[0])
    for _ in range(3):
        client._throttle()
    assert sleeps == []  # within budget
    client._throttle()   # 4th request inside the same hour
    assert sleeps == [3600.0] and client.requests_made == 4


def test_future_end_date_is_clamped_for_weather_and_past_dates_are_untouched(settings):
    s = replace(settings, end_date=date(2030, 1, 1))
    assert fetch_weather.effective_end_date(s, today=date(2026, 10, 8)) == date(2026, 10, 8)
    assert fetch_weather.build_params(28.6, 77.2, replace(s, end_date=date(2025, 5, 1)))["end_date"] == "2025-05-01"
    assert fetch_weather.build_params(28.6, 77.2, s)["end_date"] <= date.today().isoformat()


def test_malformed_openaq_hour_record_does_not_crash_and_is_counted_when_cleaned(settings):
    rec = parse_hour_record({"value": 12.0, "parameter": {"name": "pm25", "units": UG}}, 1, 11)  # no period at all
    assert rec["period_to_utc"] is None and rec["pollutant"] == "pm25"
    cleaned, stats = clean_openaq(pd.DataFrame([rec]), settings, "pm25")
    assert cleaned.empty and stats["dropped_no_timestamp"] == 1


def test_weather_payload_validation_rejects_bad_shapes_and_non_utc():
    vars_ = ("temperature_2m",)
    good = {"utc_offset_seconds": 0, "hourly": {"time": ["2025-11-01T00:00"], "temperature_2m": [20.0]}}
    df, _ = fetch_weather.parse_weather_response(good, "P1", vars_)
    assert len(df) == 1 and str(df["timestamp_utc"].dt.tz) == "UTC"
    for bad in ({"error": True, "reason": "nope"}, {"utc_offset_seconds": 19800, "hourly": good["hourly"]}, {"utc_offset_seconds": 0},
                {"utc_offset_seconds": 0, "hourly": {"time": ["2025-11-01T00:00"]}}):
        with pytest.raises(FetchError):
            fetch_weather.parse_weather_response(bad, "P1", vars_)
    nulls = {"utc_offset_seconds": 0, "hourly": {"time": ["2025-11-01T00:00"], "temperature_2m": [None]}}
    assert fetch_weather.parse_weather_response(nulls, "P1", vars_)[0]["temperature_2m"].isna().all()  # null stays NaN
