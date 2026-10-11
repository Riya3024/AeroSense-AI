"""Phase 2 export: the ML-ready hourly dataset under its public schema, verified as written.

Input  (made by merge_datasets):  data/processed/supervised_pm25_t_plus_1h.csv + feature_columns.json
Output (this module):
    data/processed/<slug>_hourly_pm25_dataset.csv        (always)
    data/processed/<slug>_hourly_pm25_dataset.parquet    (if a parquet engine such as pyarrow is installed)
    data/processed/<slug>_hourly_pm25_dataset_manifest.json   schema, features, target, checks, checksums, config
    data/samples/<slug>_hourly_pm25_sample.csv           first week of one station (small enough to inspect/commit)

Nothing is re-computed here: columns are RENAMED to the public schema (e.g. pm25_t -> pm25) and then the exported file
is read back and checked again with the existing leakage guard, so what ships is what was verified.

Time conventions
  * ``timestamp`` / ``target_timestamp`` are UTC, on the hour, written as ISO-8601 with a trailing Z.
  * ``hour``, ``day_of_week``, ``day_of_year``, ``month`` describe LOCAL time (Asia/Kolkata, IST = UTC+05:30),
    because diurnal/weekly patterns follow local human activity. They are derived from ``timestamp`` only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import subprocess
import sys
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from .config import TARGET_COL, TARGET_TS_COL, Settings, setup_logging
from .merge_datasets import ONE_HOUR, _fmt_ts, chronological_split
from .validate_data import LeakageError, assert_no_target_leakage, check_supervised

log = logging.getLogger("aerosense.final_dataset")

INTERNAL_TO_PUBLIC = {
    "timestamp_utc": "timestamp",
    "target_timestamp_utc": "target_timestamp",
    "pm25_t": "pm25",
    "dayofweek": "day_of_week",
    "dayofyear": "day_of_year",
    "temperature_2m": "temperature",
    "relative_humidity_2m": "relative_humidity",
    "wind_speed_10m": "wind_speed",
    "wind_direction_10m": "wind_direction",
    "wind_u_10m": "wind_u",
    "wind_v_10m": "wind_v",
}
PUBLIC_TO_INTERNAL = {v: k for k, v in INTERNAL_TO_PUBLIC.items()}
TARGET_PUBLIC = TARGET_COL  # unchanged: target_pm25_t_plus_1h
LEADING_COLUMNS = ["station_id", "timestamp", "target_timestamp", "hour", "day_of_week", "day_of_year", "month",
                   "pm25", "pm10", "no2", "o3", "co", "so2",
                   "temperature", "relative_humidity", "wind_speed", "wind_direction", "precipitation", "wind_u", "wind_v"]
METADATA_COLUMNS = ["station_id", "timestamp", "target_timestamp", "weather_imputed_count"]
# Minimum schema required by the Phase 2 specification (pollutants other than pm25 are optional: "if available").
REQUIRED_PUBLIC = ["timestamp", "hour", "day_of_week", "day_of_year", "month", "pm25", "temperature", "relative_humidity",
                   "wind_speed", "wind_direction", "precipitation", "wind_u", "wind_v", TARGET_PUBLIC]

UNITS = {
    "pm25": "ug/m3", "pm10": "ug/m3", "no2": "ug/m3 (see config)", "o3": "ug/m3 (see config)", "co": "ug/m3 (see config)",
    "so2": "ug/m3 (see config)", "temperature": "degC", "relative_humidity": "%", "wind_speed": "km/h",
    "wind_direction": "deg (direction the wind blows FROM)", "precipitation": "mm (hourly sum)", "wind_u": "km/h (east+)",
    "wind_v": "km/h (north+)", "dew_point_2m": "degC", "surface_pressure": "hPa", "wind_gusts_10m": "km/h",
    "cloud_cover": "%", "shortwave_radiation": "W/m2", "hour": "IST hour 0-23", "day_of_week": "IST, Monday=0",
    "day_of_year": "IST 1-366", "month": "IST 1-12", "latitude": "deg", "longitude": "deg",
    TARGET_PUBLIC: "ug/m3 (PM2.5 at timestamp + 1 hour)",
}


def to_public(sup: pd.DataFrame, internal_features: list) -> tuple:
    """Rename to the public schema and order columns. Returns (dataframe, public_feature_list)."""
    df = sup.rename(columns=INTERNAL_TO_PUBLIC)
    features = [INTERNAL_TO_PUBLIC.get(c, c) for c in internal_features]
    others = [c for c in features if c not in LEADING_COLUMNS]
    ordered = [c for c in LEADING_COLUMNS if c in df.columns] + others
    tail = [c for c in ("weather_imputed_count", TARGET_PUBLIC) if c in df.columns]
    ordered = list(dict.fromkeys(ordered + tail))
    return df[ordered], features


def to_internal(public: pd.DataFrame) -> pd.DataFrame:
    return public.rename(columns=PUBLIC_TO_INTERNAL)


def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _check(name: str, ok: Optional[bool], detail: str = "") -> dict:
    status = "skipped" if ok is None else ("pass" if ok else "fail")
    return {"name": name, "status": status, "detail": detail}


def verify_public(public: pd.DataFrame, features: list, settings: Settings) -> list:
    """Independent checks on the dataset exactly as shipped (read back from disk). Returns a list of check dicts."""
    checks = []
    missing = [c for c in REQUIRED_PUBLIC if c not in public.columns]
    checks.append(_check("required_columns_present", not missing, f"missing: {missing}" if missing else f"{len(REQUIRED_PUBLIC)} required columns present"))
    bad_feats = [c for c in features if c in METADATA_COLUMNS or c == TARGET_PUBLIC or c not in public.columns]
    checks.append(_check("features_exclude_target_and_metadata", not bad_feats, f"offending: {bad_feats}" if bad_feats else f"{len(features)} feature columns"))
    if missing:
        return checks

    ts = public["timestamp"]
    aligned = bool(str(ts.dt.tz) == "UTC" and (ts.dt.minute == 0).all() and (ts.dt.second == 0).all())
    checks.append(_check("timestamps_are_utc_and_on_the_hour", aligned))
    checks.append(_check("unique_station_timestamp", not public.duplicated(["station_id", "timestamp"]).any()))
    checks.append(_check("sorted_chronologically_within_station", public.sort_values(["station_id", "timestamp"]).equals(public)))
    step_ok = bool(((public["target_timestamp"] - public["timestamp"]) == ONE_HOUR).all())
    checks.append(_check("target_timestamp_is_exactly_t_plus_1h", step_ok))
    required_cols = [TARGET_PUBLIC, "pm25", "pm25_lag_1h", "pm25_lag_24h"]
    nan_required = {c: int(public[c].isna().sum()) for c in required_cols if c in public.columns}
    checks.append(_check("no_missing_in_target_or_core_pm25_features", not any(nan_required.values()), json.dumps(nan_required)))

    # Target value: equals PM2.5 observed at t+1h wherever that later row is also in the dataset.
    nxt = public[["station_id", "timestamp", "pm25"]].copy()
    nxt["timestamp"] = nxt["timestamp"] - ONE_HOUR
    j = public[["station_id", "timestamp", TARGET_PUBLIC]].merge(nxt, on=["station_id", "timestamp"], how="left")
    verifiable = j["pm25"].notna()
    mismatch = int(((j.loc[verifiable, TARGET_PUBLIC] - j.loc[verifiable, "pm25"]).abs() > 1e-9).sum())
    checks.append(_check("target_equals_pm25_one_hour_later", mismatch == 0,
                         f"verified {int(verifiable.sum())} rows against the later row; {int((~verifiable).sum())} rows "
                         f"cannot be cross-checked inside the file (the later hour was excluded); mismatches={mismatch}"))

    internal = to_internal(public)
    internal_features = [PUBLIC_TO_INTERNAL.get(c, c) for c in features]
    try:
        assert_no_target_leakage(internal, internal_features)
        checks.append(_check("leakage_guard_on_exported_file", True,
                             "target not in features; no forward-looking names; target != any feature; lag columns equal values "
                             "observed k hours earlier; target equals pm25 one hour later"))
    except LeakageError as exc:
        checks.append(_check("leakage_guard_on_exported_file", False, str(exc)))
    errors = check_supervised(internal, internal_features)
    checks.append(_check("existing_supervised_validation_on_exported_file", not errors, "; ".join(errors)))

    if internal["timestamp_utc"].nunique() >= 10:
        train, val, test = chronological_split(internal)
        ok = (len(train) > 0 and len(val) > 0 and len(test) > 0
              and train[TARGET_TS_COL].max() < val["timestamp_utc"].min()
              and val[TARGET_TS_COL].max() < test["timestamp_utc"].min())
        checks.append(_check("chronological_split_has_no_window_overlap", ok, f"train={len(train)} val={len(val)} test={len(test)}"))
    else:
        checks.append(_check("chronological_split_has_no_window_overlap", None, "fewer than 10 distinct timestamps"))
    return checks


def column_schema(public: pd.DataFrame, features: list) -> list:
    out = []
    for c in public.columns:
        role = "target" if c == TARGET_PUBLIC else "metadata" if c in METADATA_COLUMNS else "feature" if c in features else "other"
        unit = UNITS.get(c) or UNITS.get(c.rsplit("_lag_", 1)[0], "")
        out.append({"name": c, "dtype": str(public[c].dtype), "role": role, "unit": unit})
    return out


def write_parquet(public: pd.DataFrame, path) -> dict:
    try:
        public.to_parquet(path, index=False)
        return {"written": True, "path": path.name}
    except ImportError as exc:  # no pyarrow / fastparquet installed
        return {"written": False, "reason": "no parquet engine installed (pip install pyarrow to enable)"}
    except Exception as exc:  # noqa: BLE001 - report, do not crash the pipeline for an optional format
        return {"written": False, "reason": f"{type(exc).__name__}: {exc}"}


def _git_commit(root) -> Optional[str]:
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, timeout=5)
        return res.stdout.strip() or None if res.returncode == 0 else None
    except Exception:  # noqa: BLE001
        return None


def dataset_paths(settings: Settings) -> dict:
    stem = f"{settings.location_slug}_hourly_pm25_dataset"
    return {"csv": settings.processed_dir / f"{stem}.csv", "parquet": settings.processed_dir / f"{stem}.parquet",
            "manifest": settings.processed_dir / f"{stem}_manifest.json",
            "sample": settings.samples_dir / f"{settings.location_slug}_hourly_pm25_sample.csv"}


def run(settings: Settings) -> int:
    p = settings.processed_dir
    src, feat_file = p / "supervised_pm25_t_plus_1h.csv", p / "feature_columns.json"
    if not src.exists() or not feat_file.exists():
        log.error("Missing %s or %s. Run merge_datasets first.", src.name, feat_file.name)
        return 1
    sup = pd.read_csv(src, parse_dates=["timestamp_utc", TARGET_TS_COL])
    internal_features = json.loads(feat_file.read_text(encoding="utf-8"))["features"]
    public, features = to_public(sup, internal_features)
    paths = dataset_paths(settings)
    settings.samples_dir.mkdir(parents=True, exist_ok=True)

    _fmt_ts(public).to_csv(paths["csv"], index=False)
    parquet = write_parquet(public, paths["parquet"])

    # Verify what was actually written, not the in-memory frame.
    shipped = pd.read_csv(paths["csv"], parse_dates=["timestamp", "target_timestamp"])
    checks = verify_public(shipped, features, settings)
    if parquet["written"]:
        n_parquet = len(pd.read_parquet(paths["parquet"]))
        checks.append(_check("parquet_matches_csv_row_count", n_parquet == len(shipped), f"csv={len(shipped)} parquet={n_parquet}"))

    biggest = shipped["station_id"].value_counts().idxmax()
    _fmt_ts(shipped[shipped["station_id"] == biggest].head(168)).to_csv(paths["sample"], index=False)

    inputs = {}
    for name in ("openaq_pm25_hourly_raw.csv", "weather_hourly_raw.csv", "stations.csv", "station_weather_points.csv"):
        f = settings.raw_dir / name
        if f.exists():
            inputs[name] = {"sha256": _sha256(f), "bytes": f.stat().st_size}
    manifest = {
        "dataset": paths["csv"].name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "purpose": "Training data for Phase 3 PM2.5 prediction (one row = one station-hour; target = PM2.5 at t+1h)",
        "location": {"name": settings.location_name, "latitude": settings.location_latitude,
                     "longitude": settings.location_longitude, "station_bbox": list(settings.bbox)},
        "requested_range": {"start": str(settings.start_date), "end": str(settings.end_date)},
        "actual_range_utc": {"first_timestamp": shipped["timestamp"].min().isoformat(), "last_timestamp": shipped["timestamp"].max().isoformat(),
                             "last_target_timestamp": shipped["target_timestamp"].max().isoformat()},
        "rows": int(len(shipped)), "stations": int(shipped["station_id"].nunique()), "columns": int(shipped.shape[1]),
        "target": {"column": TARGET_PUBLIC, "definition": "PM2.5 (ug/m3) at the same station exactly one hour after `timestamp`; "
                   "built with a per-station time shift of the hourly grid, so a missing hour yields a missing (dropped) target, never a neighbour's value"},
        "feature_columns": features, "metadata_columns": [c for c in METADATA_COLUMNS if c in shipped.columns],
        "schema": column_schema(shipped, features),
        "calendar_features_timezone": "Asia/Kolkata (IST); timestamps are UTC",
        "checks": checks, "parquet": parquet,
        "files": {k: str(v.relative_to(settings.root)) for k, v in paths.items() if v.exists()},
        "sha256_csv": _sha256(paths["csv"]),
        "input_files": inputs, "config": settings.public_dict(),
        "environment": {"python": platform.python_version(), "pandas": pd.__version__, "git_commit": _git_commit(settings.root)},
    }
    paths["manifest"].write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    failed = [c for c in checks if c["status"] == "fail"]
    for c in checks:
        (log.error if c["status"] == "fail" else log.info)("check %-55s %s %s", c["name"], c["status"].upper(), c["detail"][:140])
    if failed:
        log.error("Final dataset verification FAILED (%d checks). Dataset written but must not be used.", len(failed))
        return 1
    log.info("Final dataset: %s (%d rows, %d columns)%s", paths["csv"], len(shipped), shipped.shape[1],
             "" if parquet["written"] else f"; parquet skipped: {parquet['reason']}")
    return 0


def main(argv: Optional[list] = None) -> int:
    argparse.ArgumentParser(description=__doc__.split("\n")[0]).parse_args(argv)
    settings = Settings.from_env()
    setup_logging("final_dataset", settings)
    return run(settings)


if __name__ == "__main__":
    sys.exit(main())
