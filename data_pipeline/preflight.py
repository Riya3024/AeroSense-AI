"""Step 1 of scripts/run_phase2.ps1: validate configuration BEFORE any slow or paid-for work starts.

Prints PASS / WARN / FAIL lines and exits 1 if anything would make the run fail. Never prints secret values; for the
OpenAQ key it only says whether one is set.
"""
from __future__ import annotations

import argparse
import importlib.util
import logging
import math
import sys
from datetime import date, datetime, timezone
from typing import Optional

from backend.settings import DEFAULT_REGION, AppSettings

from .config import Settings, setup_logging
from .fetch_openaq import date_chunks

log = logging.getLogger("aerosense.preflight")
ERA5_LAG_DAYS = 5


def module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def evaluate(settings: Settings, app: AppSettings, *, need_fetch: bool = True, need_upload: bool = True,
             today: Optional[date] = None, available: Optional[dict] = None) -> list:
    """Return [(level, message)] with level in PASS / WARN / FAIL. Pure: no network, no file writes."""
    today = today or datetime.now(timezone.utc).date()
    available = available if available is not None else {m: module_available(m) for m in ("boto3", "pyarrow")}
    out = []

    def add(level, msg):
        out.append((level, msg))

    add("PASS", f"Configuration parsed: location '{settings.location_name}' ({settings.location_latitude}, {settings.location_longitude}), "
                f"weather mode '{settings.weather_mode}'")
    add("PASS", f"Date range {settings.start_date} to {settings.end_date} ({(settings.end_date - settings.start_date).days + 1} days, UTC)")
    if settings.end_date > today:
        add("WARN", f"END_DATE {settings.end_date} is in the future; weather will be requested only up to {today} and the actual range will be shorter.")
    elif (today - settings.end_date).days < ERA5_LAG_DAYS:
        add("WARN", f"END_DATE is within {ERA5_LAG_DAYS} days of today; the weather reanalysis lags ~{ERA5_LAG_DAYS} days, so the last days may be missing (kept as NaN).")
    if (settings.end_date - settings.start_date).days < 30:
        add("WARN", "Fewer than 30 days requested; that is too little for a useful training/validation/test split.")

    min_lon, min_lat, max_lon, max_lat = settings.bbox
    if not (min_lon < max_lon and min_lat < max_lat):
        add("FAIL", f"DELHI_BBOX must be min_lon,min_lat,max_lon,max_lat with min < max; got {settings.bbox}")
    elif not (min_lon <= settings.location_longitude <= max_lon and min_lat <= settings.location_latitude <= max_lat):
        add("WARN", "LOCATION_LATITUDE/LONGITUDE lies outside DELHI_BBOX; stations come from the bbox, the location is only reported "
                    "(and used as the weather point when WEATHER_MODE=location).")

    add("PASS", f"Pollutants requested: {', '.join(settings.pollutants)}")
    chunks = len(list(date_chunks(settings.start_date, settings.end_date, settings.chunk_days)))
    pages = max(1, math.ceil(settings.chunk_days * 24 / settings.page_limit))
    per_sensor = chunks * pages
    add("PASS", f"OpenAQ cost: about {per_sensor} requests per sensor (one per {settings.chunk_days}-day chunk). Total = sensors x {per_sensor}; "
                f"the client throttles to {settings.max_requests_per_hour} requests/hour. Use `fetch_openaq --estimate` after discovery for the exact figure, "
                "or POLLUTANTS=pm25 for a fast first run.")

    if need_fetch:
        if settings.openaq_api_key:
            add("PASS", "OPENAQ_API_KEY is set (value hidden)")
        else:
            add("FAIL", "OPENAQ_API_KEY is not set. Create a free key at https://explore.openaq.org/register and put it in .env")
    if not available.get("pyarrow"):
        add("WARN", "pyarrow is not installed: the Parquet copy of the dataset will be skipped (CSV is still produced). pip install pyarrow")

    if need_upload:
        if not app.s3_bucket:
            add("FAIL", "AEROSENSE_S3_BUCKET is not set; the S3 upload step cannot run (use -SkipUpload to process locally only).")
        else:
            add("PASS", f"S3 target: s3://{app.s3_bucket}/{app.s3_root}/ in {app.aws_region}")
        if not app.uses_target_region:
            add("WARN", f"AWS_REGION is {app.aws_region}; the project targets {DEFAULT_REGION}.")
        if not available.get("boto3"):
            add("FAIL", "boto3 is not installed (needed for S3). Run: pip install -r requirements.txt")
    return out


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--skip-fetch", action="store_true", help="do not require the OpenAQ key")
    ap.add_argument("--skip-upload", action="store_true", help="do not require S3 settings or boto3")
    args = ap.parse_args(argv)
    try:
        settings, app = Settings.from_env(), AppSettings.from_env()
    except ValueError as exc:
        print(f"FAIL  Invalid configuration: {exc}")
        return 1
    setup_logging("preflight", settings)
    results = evaluate(settings, app, need_fetch=not args.skip_fetch, need_upload=not args.skip_upload)
    for level, msg in results:
        print(f"{level:<5} {msg}")
    failures = sum(1 for level, _ in results if level == "FAIL")
    print(f"\nPreflight: {failures} failure(s), {sum(1 for l, _ in results if l == 'WARN')} warning(s).")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
