"""Fetch Delhi PM2.5 hourly measurements from the OpenAQ v3 API.

Endpoints used (checked against docs.openaq.org):
  GET /v3/locations            discovery, filtered by ``bbox``
  GET /v3/locations/{id}       when explicit station IDs are configured
  GET /v3/sensors/{id}/hours   hourly mean values (``datetime_from``/``datetime_to``)

Auth: ``X-API-Key`` header, key read from the OPENAQ_API_KEY environment variable.

Usage (PowerShell):
  python -m data_pipeline.fetch_openaq --list-stations
  python -m data_pipeline.fetch_openaq [--refresh] [--start 2025-11-01 --end 2026-01-31]
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
import time
from collections import deque
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator, Optional

import pandas as pd

from .config import SUPPORTED_POLLUTANTS, Settings, setup_logging
from .http_utils import AuthError, FetchError, request_json

log = logging.getLogger("aerosense.openaq")

MISSING_KEY_MSG = (
    "OPENAQ_API_KEY is not set. OpenAQ v3 requires an API key sent in the X-API-Key header. "
    "Get a free key: register at https://explore.openaq.org/register, then copy it from "
    "https://explore.openaq.org/account. PowerShell: $env:OPENAQ_API_KEY = '<your key>' "
    "or put it in .env (git-ignored)."
)

STATION_COLUMNS = [
    "station_id", "station_name", "locality", "latitude", "longitude", "timezone",
    "provider", "is_monitor", "pm25_sensor_id", "n_pm25_sensors", "pm25_units",
    "datetime_first_utc", "datetime_last_utc", "pollutant_sensors",
]
RAW_COLUMNS = [
    "station_id", "sensor_id", "pollutant", "parameter", "unit", "value",
    "period_from_utc", "period_to_utc", "percent_complete", "has_flags",
]


# --------------------------------------------------------------------------- #
# Pure parsing helpers (unit-tested; no network)
# --------------------------------------------------------------------------- #
def _utc(obj) -> Optional[str]:
    return obj.get("utc") if isinstance(obj, dict) else None


def _norm_param(text) -> str:
    return re.sub(r"[^a-z0-9]", "", str(text or "").lower())


def is_pm25_sensor(sensor: dict, names: tuple) -> bool:
    wanted = {_norm_param(n) for n in names}
    param = sensor.get("parameter") or {}
    return _norm_param(param.get("name")) in wanted or _norm_param(param.get("displayName")) in wanted


def parse_location(loc: dict, pm25_names: tuple = ("pm25",), pollutants: tuple = ("pm25",)) -> Optional[dict]:
    """Flatten one /locations result into a station row, or None if unusable.

    A station is usable only if it has a fixed position and a PM2.5 sensor (PM2.5 defines the target).
    ``pollutant_sensors`` additionally maps every other requested pollutant the station really has to its sensor id;
    pollutants a station does not report are simply absent (never invented).
    """
    coords = loc.get("coordinates") or {}
    lat, lon = coords.get("latitude"), coords.get("longitude")
    if lat is None or lon is None:
        log.warning("Location %s has no coordinates; skipped", loc.get("id"))
        return None
    if loc.get("isMobile"):
        log.info("Location %s is mobile; skipped (fixed coordinates required)", loc.get("id"))
        return None
    sensors = [s for s in (loc.get("sensors") or []) if is_pm25_sensor(s, pm25_names)]
    if not sensors:
        return None
    if len(sensors) > 1:
        log.info("Location %s has %d PM2.5 sensors; using the first (id=%s)", loc.get("id"), len(sensors), sensors[0].get("id"))
    sensor = sensors[0]
    by_pollutant = {"pm25": sensor["id"]}
    for name in pollutants:
        if name == "pm25":
            continue
        found = [s for s in (loc.get("sensors") or []) if is_pm25_sensor(s, (name,))]
        if found:
            by_pollutant[name] = found[0]["id"]
    return {
        "station_id": loc["id"],
        "station_name": loc.get("name"),
        "locality": loc.get("locality"),
        "latitude": float(lat),
        "longitude": float(lon),
        "timezone": loc.get("timezone"),
        "provider": (loc.get("provider") or {}).get("name"),
        "is_monitor": loc.get("isMonitor"),
        "pm25_sensor_id": sensor["id"],
        "n_pm25_sensors": len(sensors),
        "pm25_units": (sensor.get("parameter") or {}).get("units"),
        "datetime_first_utc": _utc(loc.get("datetimeFirst")),
        "datetime_last_utc": _utc(loc.get("datetimeLast")),
        "pollutant_sensors": json.dumps(by_pollutant, sort_keys=True),
    }


def parse_hour_record(rec: dict, station_id: int, sensor_id: int, pollutant: str = "pm25") -> dict:
    """Flatten one /sensors/{id}/hours result. Timestamps stay as the API's UTC strings."""
    period = rec.get("period") or {}
    coverage = rec.get("coverage") or {}
    param = rec.get("parameter") or {}
    return {
        "station_id": station_id,
        "sensor_id": sensor_id,
        "pollutant": pollutant,
        "parameter": param.get("name"),
        "unit": param.get("units"),
        "value": rec.get("value"),
        "period_from_utc": _utc(period.get("datetimeFrom")),
        "period_to_utc": _utc(period.get("datetimeTo")),
        "percent_complete": coverage.get("percentComplete"),
        "has_flags": (rec.get("flagInfo") or {}).get("hasFlags"),
    }


def date_chunks(start: date, end: date, chunk_days: int) -> Iterator[tuple]:
    """Yield (datetime_from, datetime_to) UTC ISO strings covering [start, end] inclusive.

    Chunk edges touch, so a boundary hour can appear twice; merge_datasets removes it.
    """
    cur = start
    while cur <= end:
        chunk_end = min(cur + timedelta(days=chunk_days), end + timedelta(days=1))
        yield f"{cur.isoformat()}T00:00:00Z", f"{chunk_end.isoformat()}T00:00:00Z"
        cur += timedelta(days=chunk_days)


def sensors_for_row(row: dict, pollutants: tuple) -> dict:
    """{pollutant: sensor_id} for one stations.csv row, restricted to the requested pollutants.

    Falls back to ``pm25_sensor_id`` for station files written before multi-pollutant support.
    """
    raw = row.get("pollutant_sensors")
    mapping = {}
    if isinstance(raw, str) and raw.strip():
        mapping = {k: int(v) for k, v in json.loads(raw).items()}
    elif row.get("pm25_sensor_id") is not None and not (isinstance(row.get("pm25_sensor_id"), float) and math.isnan(row["pm25_sensor_id"])):
        mapping = {"pm25": int(row["pm25_sensor_id"])}
    return {p: mapping[p] for p in pollutants if p in mapping}


def estimate_requests(stations: pd.DataFrame, settings: Settings) -> dict:
    """Lower-bound request count and duration for a full fetch (before caching), so the user can judge it first."""
    chunks = len(list(date_chunks(settings.start_date, settings.end_date, settings.chunk_days)))
    pages = max(1, math.ceil(settings.chunk_days * 24 / settings.page_limit))
    per_sensor = chunks * pages
    rows = [r for r in stations.to_dict("records") if _overlaps(r, settings)]
    sensors = sum(len(sensors_for_row(r, settings.pollutants)) for r in rows)
    total = sensors * per_sensor
    minutes_rate = total * settings.min_request_interval_s / 60
    minutes_budget = total / max(settings.max_requests_per_hour, 1) * 60
    return {
        "stations": len(rows), "sensors": sensors, "requests_per_sensor": per_sensor, "estimated_requests": total,
        "estimated_minutes": round(max(minutes_rate, minutes_budget), 1),
        "hourly_request_budget": settings.max_requests_per_hour,
    }


# --------------------------------------------------------------------------- #
# Client
# --------------------------------------------------------------------------- #
class OpenAQClient:
    def __init__(self, settings: Settings, session=None, sleep=time.sleep, clock=time.monotonic):
        if not settings.openaq_api_key:
            raise AuthError(MISSING_KEY_MSG)
        import requests

        self.settings = settings
        self.session = session or requests.Session()
        self.session.headers.update({"X-API-Key": settings.openaq_api_key, "Accept": "application/json"})
        self._sleep, self._clock = sleep, clock
        self._last_request = float("-inf")
        self._recent = deque()  # monotonic timestamps of requests in the last hour
        self.requests_made = 0

    def _throttle(self) -> None:
        """Enforce the per-request spacing AND a rolling one-hour request budget (OpenAQ free tier: 2000/hour)."""
        wait = self._last_request + self.settings.min_request_interval_s - self._clock()
        if wait > 0:
            self._sleep(wait)
        now = self._clock()
        while self._recent and now - self._recent[0] >= 3600.0:
            self._recent.popleft()
        if len(self._recent) >= self.settings.max_requests_per_hour:
            pause = self._recent[0] + 3600.0 - now
            if pause > 0:
                log.warning("Hourly request budget (%d) reached; pausing %.0fs to stay within OpenAQ limits",
                            self.settings.max_requests_per_hour, pause)
                self._sleep(pause)
            now = self._clock()
            while self._recent and now - self._recent[0] >= 3600.0:
                self._recent.popleft()
        self._recent.append(now)
        self.requests_made += 1
        self._last_request = now

    def get(self, path: str, params: Optional[dict] = None) -> dict:
        s = self.settings
        return request_json(
            self.session, f"{s.openaq_base_url}{path}", params,
            timeout=s.timeout_s, max_retries=s.max_retries, backoff_s=s.backoff_s,
            sleep=self._sleep, before_request=self._throttle,
        )

    def paginate(self, path: str, params: Optional[dict] = None) -> Iterator[dict]:
        """Yield results across pages. Stops on a short page (``meta.found`` may be a string like '>1000')."""
        limit = self.settings.page_limit
        for page in range(1, self.settings.max_pages + 1):
            payload = self.get(path, {**(params or {}), "limit": limit, "page": page})
            results = payload.get("results") or []
            yield from results
            if len(results) < limit:
                return
        log.warning("max_pages=%d reached for %s; results may be truncated", self.settings.max_pages, path)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def discover_stations(client: OpenAQClient, settings: Settings) -> pd.DataFrame:
    rows = []
    if settings.location_ids:
        for lid in settings.location_ids:
            payload = client.get(f"/locations/{lid}")
            rows += [parse_location(loc, settings.pm25_parameter_names, settings.pollutants) for loc in payload.get("results") or []]
    else:
        bbox = ",".join(str(x) for x in settings.bbox)
        log.info("Discovering locations with bbox=%s", bbox)
        rows = [parse_location(loc, settings.pm25_parameter_names, settings.pollutants) for loc in client.paginate("/locations", {"bbox": bbox})]
    rows = [r for r in rows if r]
    df = pd.DataFrame(rows, columns=STATION_COLUMNS)
    return df.drop_duplicates("station_id").sort_values("station_id").reset_index(drop=True)


def _overlaps(row, settings: Settings) -> bool:
    """False only when the station's known range provably misses the requested window."""
    start = pd.Timestamp(settings.start_date, tz="UTC")
    end = pd.Timestamp(settings.end_date, tz="UTC") + pd.Timedelta(days=1)
    first = pd.to_datetime(row.get("datetime_first_utc"), utc=True, errors="coerce")
    last = pd.to_datetime(row.get("datetime_last_utc"), utc=True, errors="coerce")
    return not ((pd.notna(last) and last < start) or (pd.notna(first) and first > end))


def fetch_sensor_hours(client: OpenAQClient, station_id: int, sensor_id: int, settings: Settings, pollutant: str = "pm25") -> pd.DataFrame:
    records = []
    for dt_from, dt_to in date_chunks(settings.start_date, settings.end_date, settings.chunk_days):
        for rec in client.paginate(f"/sensors/{sensor_id}/hours", {"datetime_from": dt_from, "datetime_to": dt_to}):
            records.append(parse_hour_record(rec, station_id, sensor_id, pollutant))
    return pd.DataFrame(records, columns=RAW_COLUMNS)


def _write_atomic(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def run(settings: Settings, *, list_only: bool = False, refresh: bool = False, estimate_only: bool = False,
        strict: bool = False, client: Optional[OpenAQClient] = None) -> int:
    """Return 0 on success (even if some stations failed: see manifest), 1 on no data, 2 on auth/config error."""
    raw_dir = settings.raw_dir
    cache_dir = raw_dir / "openaq_sensors"
    cache_dir.mkdir(parents=True, exist_ok=True)
    stations_path = raw_dir / "stations.csv"

    try:
        client = client or OpenAQClient(settings)
        stations = discover_stations(client, settings)
    except AuthError as exc:
        log.error("Authentication/config problem: %s", exc)
        return 2
    except FetchError as exc:
        log.error("Station discovery failed: %s", exc)
        if stations_path.exists():
            log.warning("FALLBACK: using previously saved %s (no new data was invented)", stations_path)
            stations = pd.read_csv(stations_path)
        else:
            log.error("No cached station list available. Fix the error above and rerun.")
            return 1

    if stations.empty:
        log.error("No fixed PM2.5 stations found (bbox=%s, ids=%s).", settings.bbox, settings.location_ids)
        return 1
    _write_atomic(stations, stations_path)
    log.info("Found %d PM2.5 stations -> %s", len(stations), stations_path)
    if list_only:
        print(stations[["station_id", "station_name", "locality", "latitude", "longitude", "pollutant_sensors"]].to_string(index=False))
        return 0

    est = estimate_requests(stations, settings)
    log.info("Plan: %d stations, %d sensors (%s), >= %d requests, roughly %.0f min at the configured rate/budget",
             est["stations"], est["sensors"], ",".join(settings.pollutants), est["estimated_requests"], est["estimated_minutes"])
    if estimate_only:
        print(json.dumps(est, indent=2))
        return 0

    status, sensor_counts = {}, {p: 0 for p in settings.pollutants}
    for row in stations.to_dict("records"):
        sid = int(row["station_id"])
        if not _overlaps(row, settings):
            status[sid] = "skipped: station data range does not overlap requested dates"
            log.info("Station %s: %s", sid, status[sid])
            continue
        outcomes = []
        for pollutant, sensor in sensors_for_row(row, settings.pollutants).items():
            cache = cache_dir / f"sensor_{sensor}_{settings.start_date}_{settings.end_date}.csv"
            if cache.exists() and not refresh:
                outcomes.append(f"{pollutant}: cached")
                sensor_counts[pollutant] += 1
                log.info("Station %s %s: using cache %s", sid, pollutant, cache.name)
                continue
            try:
                log.info("Station %s (%s): fetching %s sensor %s", sid, row.get("station_name"), pollutant, sensor)
                df = fetch_sensor_hours(client, sid, sensor, settings, pollutant)
                _write_atomic(df, cache)
                outcomes.append(f"{pollutant}: fetched ({len(df)} rows)")
                sensor_counts[pollutant] += 1
            except AuthError as exc:
                log.error("Authentication failed mid-run: %s", exc)
                return 2
            except FetchError as exc:
                outcomes.append(f"{pollutant}: FAILED: {exc}")
                log.error("Station %s %s failed: %s", sid, pollutant, exc)
        status[sid] = "; ".join(outcomes) if outcomes else "no requested pollutant sensors"

    frames = []
    for row in stations.to_dict("records"):
        for pollutant, sensor in sensors_for_row(row, settings.pollutants).items():
            cache = cache_dir / f"sensor_{sensor}_{settings.start_date}_{settings.end_date}.csv"
            if cache.exists():
                frame = pd.read_csv(cache)
                if "pollutant" not in frame.columns:
                    frame.insert(2, "pollutant", pollutant)
                frames.append(frame)
    frames = [f for f in frames if not f.empty]
    raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=RAW_COLUMNS)

    failed = {k: v for k, v in status.items() if "FAILED" in v}
    manifest = {
        "run_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "start_date": str(settings.start_date), "end_date": str(settings.end_date),
        "endpoint": f"{settings.openaq_base_url}/sensors/{{id}}/hours",
        "pollutants_requested": list(settings.pollutants), "sensors_by_pollutant": sensor_counts,
        "request_plan": est, "requests_made_this_run": client.requests_made,
        "stations_discovered": int(len(stations)), "raw_rows": int(len(raw)),
        "raw_rows_by_pollutant": {str(k): int(v) for k, v in raw["pollutant"].value_counts().items()} if len(raw) else {},
        "partial": bool(failed), "station_status": {str(k): v for k, v in status.items()},
    }
    (raw_dir / "openaq_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if raw.empty:
        log.error("No PM2.5 rows retrieved. See station_status in %s. Nothing was fabricated.", raw_dir / "openaq_manifest.json")
        return 1
    _write_atomic(raw, raw_dir / "openaq_pm25_hourly_raw.csv")
    log.info("Saved %d raw rows from %d stations", len(raw), raw["station_id"].nunique())
    if failed:
        log.warning("%d station(s) failed; rerun to retry them (finished stations are cached): %s", len(failed), list(failed))
        if strict:
            log.error("--strict: exiting with code 3 because the fetch is PARTIAL. Re-run to retry only the failed sensors.")
            return 3
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list-stations", action="store_true", help="discover stations, save stations.csv, then exit")
    ap.add_argument("--refresh", action="store_true", help="ignore cached per-sensor files")
    ap.add_argument("--strict", action="store_true", help="exit with code 3 if any sensor failed (partial data)")
    ap.add_argument("--estimate", action="store_true", help="discover stations, print the request/time estimate, then exit")
    ap.add_argument("--pollutants", help="comma separated subset, e.g. pm25,pm10 (must include pm25)")
    ap.add_argument("--start", type=date.fromisoformat)
    ap.add_argument("--end", type=date.fromisoformat)
    ap.add_argument("--location-ids", help="comma separated OpenAQ location IDs (overrides bbox)")
    args = ap.parse_args(argv)
    settings = Settings.from_env()
    overrides = {}
    if args.start:
        overrides["start_date"] = args.start
    if args.end:
        overrides["end_date"] = args.end
    if args.location_ids:
        overrides["location_ids"] = tuple(int(x) for x in args.location_ids.split(","))
    if args.pollutants:
        pols = tuple(p.strip().lower() for p in args.pollutants.split(",") if p.strip())
        if "pm25" not in pols or any(p not in SUPPORTED_POLLUTANTS for p in pols):
            ap.error(f"--pollutants must include pm25 and only use {list(SUPPORTED_POLLUTANTS)}")
        overrides["pollutants"] = pols
    settings = replace(settings, **overrides)
    setup_logging("fetch_openaq", settings)
    return run(settings, list_only=args.list_stations, refresh=args.refresh, estimate_only=args.estimate, strict=args.strict)


if __name__ == "__main__":
    sys.exit(main())
