"""Phase 2 configuration + preflight validation. Environment is patched per test; nothing real is read or written."""
import os
import re
from contextlib import contextmanager
from dataclasses import replace
from datetime import date
from pathlib import Path
from unittest import mock

import pytest

from backend.settings import AppSettings
from data_pipeline import preflight
from data_pipeline.config import Settings

PHASE2_KEYS = ["POLLUTANTS", "POLLUTANT_UNITS", "LOCATION_NAME", "LOCATION_LATITUDE", "LOCATION_LONGITUDE", "WEATHER_MODE",
               "WEATHER_LATITUDE", "WEATHER_LONGITUDE", "OPENAQ_MAX_REQUESTS_PER_HOUR", "START_DATE", "END_DATE", "OPENAQ_API_KEY",
               "DELHI_BBOX", "AEROSENSE_S3_BUCKET", "AEROSENSE_S3_PREFIX", "AWS_REGION", "AWS_DEFAULT_REGION"]


@contextmanager
def env(**overrides):
    """Clean slate for the keys under test (a developer's real .env must not leak into assertions)."""
    with mock.patch.dict(os.environ, {}, clear=False):
        for k in PHASE2_KEYS:
            os.environ.pop(k, None)
        os.environ.update(overrides)
        yield


def test_defaults_match_the_phase2_research_window_and_delhi():
    with env():
        s = Settings.from_env()
    assert (s.start_date, s.end_date) == (date(2025, 1, 1), date(2026, 9, 30))
    assert s.pollutants == ("pm25", "pm10", "no2", "o3", "co", "so2")
    assert (s.location_name, s.location_slug, s.weather_mode) == ("Delhi", "delhi", "station")
    assert 28.0 < s.location_latitude < 29.0 and 76.5 < s.location_longitude < 77.8
    assert s.run_tag == "2025-01-01_2026-09-30"


def test_date_range_location_and_pollutants_come_from_environment_not_code():
    with env(START_DATE="2024-02-01", END_DATE="2024-03-15", POLLUTANTS="PM25, pm10", LOCATION_NAME="Gurugram NCR",
             LOCATION_LATITUDE="28.4595", LOCATION_LONGITUDE="77.0266"):
        s = Settings.from_env()
    assert (s.start_date, s.end_date) == (date(2024, 2, 1), date(2024, 3, 15))
    assert s.pollutants == ("pm25", "pm10") and s.location_slug == "gurugram_ncr" and s.location_latitude == 28.4595


@pytest.mark.parametrize("bad", [
    {"POLLUTANTS": "pm25,benzene"},
    {"POLLUTANTS": "pm10,no2"},          # pm25 is the target source and is mandatory
    {"WEATHER_MODE": "grid"},
    {"LOCATION_LATITUDE": "123"},
    {"POLLUTANT_UNITS": "co"},           # not name=unit
    {"POLLUTANT_UNITS": "xx=mg/m3"},     # unknown pollutant
    {"START_DATE": "2026-01-02", "END_DATE": "2026-01-01"},
    {"OPENAQ_MAX_REQUESTS_PER_HOUR": "0"},
])
def test_invalid_configuration_is_rejected_with_a_value_error(bad):
    with env(**bad):
        with pytest.raises(ValueError):
            Settings.from_env()


def test_location_weather_mode_uses_the_representative_point_unless_overridden():
    with env(WEATHER_MODE="location"):
        s = Settings.from_env()
    assert (s.weather_latitude, s.weather_longitude) == (s.location_latitude, s.location_longitude)
    with env(WEATHER_MODE="location", WEATHER_LATITUDE="28.5", WEATHER_LONGITUDE="77.1"):
        assert Settings.from_env().weather_latitude == 28.5


def test_unit_overrides_are_parsed_per_pollutant():
    with env(POLLUTANT_UNITS="co=mg/m3, no2=ppb"):
        s = Settings.from_env()
    assert s.expected_unit("co") == "mg/m3" and s.expected_unit("no2") == "ppb" and s.expected_unit("pm25") == "ug/m3"


def test_public_config_snapshot_never_contains_the_api_key():
    s = Settings(openaq_api_key="super-secret-key-999")
    snap = s.public_dict()
    assert snap["openaq_api_key_present"] is True
    assert "super-secret-key-999" not in repr(snap) and "openaq_api_key" not in snap


def test_no_date_range_is_hard_coded_outside_the_config_defaults():
    """The pipeline must take its window from configuration. Only config.py may contain date literals (docstrings are ignored)."""
    import ast
    date_re = re.compile(r"\b20[12]\d-\d\d-\d\d\b")
    offenders = []
    for f in Path(__file__).resolve().parents[1].joinpath("data_pipeline").glob("*.py"):
        if f.name == "config.py":
            continue
        tree = ast.parse(f.read_text(encoding="utf-8"))
        docstrings = {id(n.body[0].value) for n in ast.walk(tree)
                      if isinstance(n, (ast.Module, ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef))
                      and n.body and isinstance(n.body[0], ast.Expr) and isinstance(getattr(n.body[0], "value", None), ast.Constant)}
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings and date_re.search(node.value):
                offenders.append(f"{f.name}:{node.lineno}: {node.value!r}")
    assert offenders == []


# --------------------------------------------------------------------------- preflight
S = Settings(openaq_api_key="k", start_date=date(2025, 1, 1), end_date=date(2026, 9, 30))
APP_OK = AppSettings.from_env({"AEROSENSE_S3_BUCKET": "my-bucket"})
ALL_MODULES = {"boto3": True, "pyarrow": True}
TODAY = date(2026, 10, 8)


def levels(results):
    return [lvl for lvl, _ in results]


def test_preflight_passes_when_everything_is_configured():
    res = preflight.evaluate(S, APP_OK, today=TODAY, available=ALL_MODULES)
    assert "FAIL" not in levels(res) and "WARN" not in levels(res)


def test_preflight_fails_without_openaq_key_or_bucket_but_only_when_those_steps_run():
    res = preflight.evaluate(replace(S, openaq_api_key=None), AppSettings.from_env({}), today=TODAY, available=ALL_MODULES)
    assert levels(res).count("FAIL") == 2
    res = preflight.evaluate(replace(S, openaq_api_key=None), AppSettings.from_env({}), need_fetch=False, need_upload=False, today=TODAY, available=ALL_MODULES)
    assert "FAIL" not in levels(res)


def test_preflight_warns_about_future_end_date_missing_pyarrow_and_wrong_region():
    s = replace(S, end_date=date(2026, 12, 31))
    app = AppSettings.from_env({"AEROSENSE_S3_BUCKET": "b", "AWS_REGION": "us-east-1"})
    msgs = " | ".join(m for lvl, m in preflight.evaluate(s, app, today=TODAY, available={"boto3": True, "pyarrow": False}) if lvl == "WARN")
    assert "future" in msgs and "pyarrow" in msgs and "ap-south-1" in msgs


def test_preflight_fails_when_boto3_missing_for_upload_and_for_inverted_bbox():
    res = preflight.evaluate(replace(S, bbox=(77.5, 28.4, 77.0, 28.9)), APP_OK, today=TODAY, available={"boto3": False, "pyarrow": True})
    fails = " | ".join(m for lvl, m in res if lvl == "FAIL")
    assert "boto3" in fails and "DELHI_BBOX" in fails


def test_preflight_output_never_contains_the_api_key():
    s = replace(S, openaq_api_key="SECRET-VALUE-123")
    assert "SECRET-VALUE-123" not in " ".join(m for _, m in preflight.evaluate(s, APP_OK, today=TODAY, available=ALL_MODULES))


def test_every_setup_logging_call_passes_a_name_first():
    """Regression: setup_logging(name, settings). Calling it with settings only crashes the CLI at start-up."""
    import ast
    bad = []
    for f in Path(__file__).resolve().parents[1].joinpath("data_pipeline").glob("*.py"):
        for node in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "setup_logging":
                first = node.args[0] if node.args else None
                if not (isinstance(first, ast.Constant) and isinstance(first.value, str)):
                    bad.append(f"{f.name}:{node.lineno}")
    assert bad == []
