"""HTTP behaviour tests using a scripted fake session (a MOCK: no network, payloads are synthetic
and merely follow the documented response shapes)."""
import json
import logging

import pandas as pd
import pytest
import requests

from data_pipeline import fetch_openaq, fetch_weather
from data_pipeline.config import _RedactingFilter
from data_pipeline.fetch_openaq import OpenAQClient, is_pm25_sensor, parse_location
from data_pipeline.http_utils import AuthError, FetchError, request_json


class FakeResponse:
    def __init__(self, status=200, payload=None, headers=None, text=""):
        self.status_code, self._payload, self.headers, self.text = status, payload, headers or {}, text

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


class FakeSession:
    def __init__(self, script):
        self.script, self.calls, self.headers = script, [], {}

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, dict(params or {})))
        item = self.script(url, params) if callable(self.script) else self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _call(session, sleeps, max_retries=3):
    return request_json(session, "http://x", {}, timeout=1, max_retries=max_retries, backoff_s=0.0, sleep=sleeps.append)


def test_missing_api_key_gives_actionable_error(settings):
    from dataclasses import replace
    with pytest.raises(AuthError) as exc:
        OpenAQClient(replace(settings, openaq_api_key=None))
    assert "explore.openaq.org" in str(exc.value)


def test_pagination_follows_pages_until_short_page(settings):
    session = FakeSession([FakeResponse(payload={"results": [{"i": 1}, {"i": 2}]}),
                           FakeResponse(payload={"results": [{"i": 3}, {"i": 4}]}),
                           FakeResponse(payload={"results": [{"i": 5}]})])
    client = OpenAQClient(settings, session=session, sleep=lambda s: None)
    got = list(client.paginate("/sensors/1/hours", {"datetime_from": "a"}))
    assert [r["i"] for r in got] == [1, 2, 3, 4, 5]
    assert [c[1]["page"] for c in session.calls] == [1, 2, 3]
    assert session.headers["X-API-Key"] == "test-key-123"  # sent as header, never in the URL/params
    assert all("test-key-123" not in json.dumps(c) for c in session.calls)


def test_429_waits_for_reset_header_then_succeeds():
    sleeps = []
    session = FakeSession([FakeResponse(429, headers={"x-ratelimit-reset": "17"}), FakeResponse(payload={"ok": 1})])
    assert _call(session, sleeps) == {"ok": 1}
    assert 17.0 in sleeps


def test_timeout_is_retried():
    sleeps = []
    session = FakeSession([requests.Timeout("slow"), FakeResponse(payload={"ok": 1})])
    assert _call(session, sleeps) == {"ok": 1} and len(session.calls) == 2


def test_auth_error_is_not_retried():
    session = FakeSession([FakeResponse(401, text="invalid key")])
    with pytest.raises(AuthError):
        _call(session, [])
    assert len(session.calls) == 1


def test_422_is_not_retried_and_surfaces_server_message():
    session = FakeSession([FakeResponse(422, text='{"detail":"bad param"}')])
    with pytest.raises(FetchError) as exc:
        _call(session, [])
    assert "bad param" in str(exc.value) and len(session.calls) == 1


def test_persistent_5xx_gives_up_with_last_error():
    session = FakeSession([FakeResponse(503, text="down")] * 3)
    with pytest.raises(FetchError) as exc:
        _call(session, [], max_retries=3)
    assert "503" in str(exc.value) and len(session.calls) == 3


def test_log_filter_redacts_secret():
    rec = logging.LogRecord("t", logging.INFO, "", 0, "key is %s", ("test-key-123",), None)
    assert _RedactingFilter(["test-key-123"]).filter(rec) and "test-key-123" not in rec.getMessage()


def test_parse_location_keeps_fixed_pm25_stations_only():
    base = {"id": 7, "name": "S", "timezone": "Asia/Kolkata", "isMobile": False, "isMonitor": True,
            "coordinates": {"latitude": 28.6, "longitude": 77.2}, "provider": {"name": "P"},
            "sensors": [{"id": 70, "name": "pm10", "parameter": {"id": 1, "name": "pm10", "units": "\u00b5g/m\u00b3", "displayName": "PM10"}},
                        {"id": 71, "name": "pm25", "parameter": {"id": 2, "name": "pm25", "units": "\u00b5g/m\u00b3", "displayName": "PM2.5"}}]}
    assert parse_location(base)["pm25_sensor_id"] == 71
    assert parse_location({**base, "isMobile": True}) is None
    assert parse_location({**base, "coordinates": {"latitude": None, "longitude": None}}) is None
    assert parse_location({**base, "sensors": [base["sensors"][0]]}) is None
    assert is_pm25_sensor({"parameter": {"name": "x", "displayName": "PM2.5"}}, ("pm25",))


def _hours_payload(n):
    return {"results": [{"value": 10.0 + i, "parameter": {"id": 2, "name": "pm25", "units": "\u00b5g/m\u00b3"},
                         "period": {"datetimeFrom": {"utc": f"2025-11-01T{i:02d}:00:00Z"}, "datetimeTo": {"utc": f"2025-11-01T{i+1:02d}:00:00Z"}},
                         "coverage": {"percentComplete": 100.0}, "flagInfo": {"hasFlags": False}} for i in range(n)]}


def _router(url, params):
    if url.endswith("/locations"):
        return FakeResponse(payload={"results": [{"id": 7, "name": "S", "timezone": "Asia/Kolkata", "isMobile": False, "isMonitor": True,
                                                  "coordinates": {"latitude": 28.6, "longitude": 77.2}, "provider": {"name": "P"},
                                                  "sensors": [{"id": 71, "name": "pm25", "parameter": {"id": 2, "name": "pm25", "units": "\u00b5g/m\u00b3"}}]}]})
    if "/sensors/71/hours" in url:
        return FakeResponse(payload=_hours_payload(1) if params["page"] == 1 else {"results": []})
    return FakeResponse(404, text="unexpected url")


def test_openaq_run_writes_raw_files_and_manifest(settings):
    from dataclasses import replace
    s = replace(settings, page_limit=5, start_date=pd.Timestamp("2025-11-01").date(), end_date=pd.Timestamp("2025-11-02").date())
    client = OpenAQClient(s, session=FakeSession(_router), sleep=lambda x: None)
    assert fetch_openaq.run(s, client=client) == 0
    raw = pd.read_csv(s.raw_dir / "openaq_pm25_hourly_raw.csv")
    assert len(raw) == 1 and raw["station_id"].iloc[0] == 7
    assert json.loads((s.raw_dir / "openaq_manifest.json").read_text())["partial"] is False
    assert "test-key-123" not in (s.raw_dir / "openaq_manifest.json").read_text()


def test_openaq_run_without_key_fails_cleanly_and_writes_no_data(settings):
    from dataclasses import replace
    s = replace(settings, openaq_api_key=None)
    assert fetch_openaq.run(s) == 2
    assert not (s.raw_dir / "openaq_pm25_hourly_raw.csv").exists()


def test_openaq_discovery_failure_falls_back_to_cached_station_list_only(settings):
    from dataclasses import replace
    s = replace(settings, max_retries=1)
    s.raw_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{c: None for c in fetch_openaq.STATION_COLUMNS} | {"station_id": 7, "pm25_sensor_id": 71, "latitude": 28.6, "longitude": 77.2}]).to_csv(s.raw_dir / "stations.csv", index=False)
    client = OpenAQClient(s, session=FakeSession(lambda u, p: FakeResponse(500, text="boom")), sleep=lambda x: None)
    assert fetch_openaq.run(s, client=client) == 1  # stations came from cache, but no measurements could be fetched
    assert not (s.raw_dir / "openaq_pm25_hourly_raw.csv").exists()  # and nothing was invented


def test_weather_run_success_and_http_400_reason_is_reported(settings, caplog):
    from dataclasses import replace
    s = replace(settings, start_date=pd.Timestamp("2025-11-01").date(), end_date=pd.Timestamp("2025-11-01").date())
    s.raw_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"station_id": [7], "latitude": [28.6123], "longitude": [77.2089]}).to_csv(s.raw_dir / "stations.csv", index=False)
    times = [f"2025-11-01T{h:02d}:00" for h in range(24)]
    payload = {"latitude": 28.6, "longitude": 77.2, "elevation": 215.0, "utc_offset_seconds": 0,
               "hourly": {"time": times, **{v: [1.0] * 24 for v in s.weather_variables}},
               "hourly_units": {v: "u" for v in s.weather_variables}}
    ok = FakeSession([FakeResponse(payload=payload)])
    assert fetch_weather.run(s, session=ok, sleep=lambda x: None) == 0
    assert len(pd.read_csv(s.raw_dir / "weather_hourly_raw.csv")) == 24
    assert ok.calls[0][1]["timezone"] == "GMT" and ok.calls[0][1]["latitude"] == 28.61

    for f in (s.raw_dir / "weather_points").glob("*.csv"):
        f.unlink()
    (s.raw_dir / "weather_hourly_raw.csv").unlink()
    bad = FakeSession([FakeResponse(400, text='{"error":true,"reason":"invalid hourly variable"}')])
    with caplog.at_level(logging.ERROR, logger="aerosense"):
        assert fetch_weather.run(s, session=bad, sleep=lambda x: None) == 1
    assert "invalid hourly variable" in caplog.text
