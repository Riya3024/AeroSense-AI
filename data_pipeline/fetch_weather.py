"""Fetch hourly historical weather from the Open-Meteo Historical Weather API.

Endpoint (checked against open-meteo.com/en/docs/historical-weather-api):
  GET https://archive-api.open-meteo.com/v1/archive
  params: latitude, longitude, start_date, end_date (yyyy-mm-dd), hourly (comma list),
          timezone (we pin GMT so timestamps are UTC), plus explicit unit parameters.
No API key is needed for non-commercial use. ERA5 data lags real time by ~5 days.

Usage (PowerShell):  python -m data_pipeline.fetch_weather [--refresh]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from .config import Settings, setup_logging
from .http_utils import FetchError, request_json

log = logging.getLogger("aerosense.weather")

GRID_COLUMNS = ["weather_grid_latitude", "weather_grid_longitude", "weather_grid_elevation"]


def weather_point_id(lat: float, lon: float) -> str:
    """Stable ID for a weather request point; rounding to 0.01 deg (~1 km) dedupes near-identical stations."""
    return f"{lat:.2f}_{lon:.2f}"


def resolve_points(stations: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Map each station to the weather point used for it.

    Returns columns: station_id, weather_point_id, requested_latitude, requested_longitude.
    """
    out = stations[["station_id"]].copy()
    if settings.weather_latitude is not None and settings.weather_longitude is not None:
        out["requested_latitude"] = settings.weather_latitude
        out["requested_longitude"] = settings.weather_longitude
    else:
        out["requested_latitude"] = stations["latitude"].round(2)
        out["requested_longitude"] = stations["longitude"].round(2)
    out["weather_point_id"] = [weather_point_id(a, b) for a, b in zip(out["requested_latitude"], out["requested_longitude"])]
    return out[["station_id", "weather_point_id", "requested_latitude", "requested_longitude"]]


def effective_end_date(settings: Settings, today: Optional[date] = None) -> date:
    """END_DATE clamped to today (UTC): the archive API cannot serve future dates. Recent days may still be null."""
    today = today or datetime.now(timezone.utc).date()
    return min(settings.end_date, today)


def build_params(lat: float, lon: float, settings: Settings) -> dict:
    return {
        "latitude": lat,
        "longitude": lon,
        "start_date": settings.start_date.isoformat(),
        "end_date": effective_end_date(settings).isoformat(),
        "hourly": ",".join(settings.weather_variables),
        "timezone": "GMT",  # timestamps come back as UTC
        "temperature_unit": "celsius",
        "wind_speed_unit": "kmh",
        "precipitation_unit": "mm",
    }


def parse_weather_response(payload: dict, point_id: str, variables: tuple) -> tuple:
    """Turn an Open-Meteo JSON payload into (DataFrame, units dict). Raises FetchError on schema problems."""
    if payload.get("error"):
        raise FetchError(f"Open-Meteo error: {payload.get('reason')}")
    if payload.get("utc_offset_seconds", 0) != 0:
        raise FetchError(f"Expected UTC (GMT) timestamps but got utc_offset_seconds={payload.get('utc_offset_seconds')}")
    hourly = payload.get("hourly")
    if not hourly or "time" not in hourly:
        raise FetchError("Open-Meteo response has no hourly.time array")
    missing = [v for v in variables if v not in hourly]
    if missing:
        raise FetchError(f"Requested variables missing from response: {missing}")
    df = pd.DataFrame({"timestamp_utc": pd.to_datetime(hourly["time"], utc=True)})
    for v in variables:
        df[v] = pd.to_numeric(pd.Series(hourly[v]), errors="coerce")  # JSON null -> NaN
    df.insert(0, "weather_point_id", point_id)
    df["weather_grid_latitude"] = payload.get("latitude")
    df["weather_grid_longitude"] = payload.get("longitude")
    df["weather_grid_elevation"] = payload.get("elevation")
    return df, payload.get("hourly_units", {})


def fetch_point(session, settings: Settings, point_id: str, lat: float, lon: float, sleep=time.sleep) -> tuple:
    payload = request_json(
        session, settings.weather_url, build_params(lat, lon, settings),
        timeout=settings.timeout_s, max_retries=settings.max_retries, backoff_s=settings.backoff_s,
        sleep=sleep, before_request=lambda: sleep(0.2),
    )
    return parse_weather_response(payload, point_id, settings.weather_variables)


def _write_atomic(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def run(settings: Settings, *, refresh: bool = False, strict: bool = False, session=None, sleep=time.sleep) -> int:
    """Return 0 on success, 1 if no weather could be retrieved."""
    import requests

    raw_dir = settings.raw_dir
    stations_path = raw_dir / "stations.csv"
    if not stations_path.exists():
        log.error("%s not found. Run `python -m data_pipeline.fetch_openaq` first.", stations_path)
        return 1
    stations = pd.read_csv(stations_path)
    mapping = resolve_points(stations, settings)
    mapping.to_csv(raw_dir / "station_weather_points.csv", index=False)
    points = mapping.drop_duplicates("weather_point_id")
    log.info("%d stations -> %d unique weather points", len(stations), len(points))

    if effective_end_date(settings) < settings.end_date:
        log.warning("END_DATE %s is in the future; requesting weather only up to %s", settings.end_date, effective_end_date(settings))
    latest_reliable = datetime.now(timezone.utc).date() - timedelta(days=5)
    if settings.end_date > latest_reliable:
        log.warning("END_DATE %s is within ~5 days of today; ERA5 may not be complete yet (docs: ~5 day delay).", settings.end_date)

    cache_dir = raw_dir / "weather_points"
    cache_dir.mkdir(parents=True, exist_ok=True)
    session = session or requests.Session()
    units, status = {}, {}
    for p in points.to_dict("records"):
        pid = p["weather_point_id"]
        cache = cache_dir / f"point_{pid}_{settings.start_date}_{settings.end_date}.csv"
        if cache.exists() and not refresh:
            status[pid] = "cached"
            continue
        try:
            log.info("Fetching weather for point %s", pid)
            df, u = fetch_point(session, settings, pid, p["requested_latitude"], p["requested_longitude"], sleep=sleep)
            _write_atomic(df, cache)
            units.update(u)
            status[pid] = f"fetched ({len(df)} rows)"
        except FetchError as exc:
            status[pid] = f"FAILED: {exc}"
            log.error("Weather point %s failed: %s", pid, exc)

    frames = [pd.read_csv(c) for c in (cache_dir / f"point_{pid}_{settings.start_date}_{settings.end_date}.csv" for pid in points["weather_point_id"]) if c.exists()]
    weather = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    manifest = {
        "run_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "endpoint": settings.weather_url, "start_date": str(settings.start_date), "end_date": str(settings.end_date),
        "effective_end_date": str(effective_end_date(settings)), "weather_mode": settings.weather_mode,
        "variables": list(settings.weather_variables), "units": units, "timezone": "GMT (UTC)",
        "points_requested": int(len(points)), "raw_rows": int(len(weather)),
        "partial": any(v.startswith("FAILED") for v in status.values()), "point_status": status,
    }
    (raw_dir / "weather_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if weather.empty:
        log.error("No weather rows retrieved. See %s. Nothing was fabricated.", raw_dir / "weather_manifest.json")
        return 1
    _write_atomic(weather, raw_dir / "weather_hourly_raw.csv")
    log.info("Saved %d weather rows for %d points", len(weather), weather["weather_point_id"].nunique())
    if manifest["partial"] and strict:
        log.error("--strict: exiting with code 3 because some weather points failed. Re-run to retry only those points.")
        return 3
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--strict", action="store_true", help="exit with code 3 if any weather point failed (partial data)")
    ap.add_argument("--start", type=date.fromisoformat)
    ap.add_argument("--end", type=date.fromisoformat)
    args = ap.parse_args(argv)
    settings = Settings.from_env()
    overrides = {k: v for k, v in (("start_date", args.start), ("end_date", args.end)) if v}
    settings = replace(settings, **overrides)
    setup_logging("fetch_weather", settings)
    return run(settings, refresh=args.refresh, strict=args.strict)


if __name__ == "__main__":
    sys.exit(main())
