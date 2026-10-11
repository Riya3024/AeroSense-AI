"""Data-quality report and target-leakage guard for the processed datasets.

Usage (PowerShell):  python -m data_pipeline.validate_data
Writes data/processed/data_quality_report.json and .md; exits 1 if any check fails.
"""
from __future__ import annotations

import json
import logging
import re
import sys
from typing import Optional

import numpy as np
import pandas as pd

from .config import TARGET_COL, TARGET_TS_COL, Settings, setup_logging

log = logging.getLogger("aerosense.validate")
ONE_HOUR = pd.Timedelta(hours=1)
FORBIDDEN_NAME = re.compile(r"target|future|lead|t_plus|next", re.IGNORECASE)
LAG_NAME = re.compile(r"^(pm25|pm10|no2|o3|co|so2)_lag_(\d+)h$")
CURRENT_COLUMN = {"pm25": "pm25_t"}  # all other pollutants: the current value is the column named after them


class LeakageError(AssertionError):
    """Raised when a feature could contain information from the target (t+1h)."""


def assert_no_target_leakage(df: pd.DataFrame, feature_cols: list, target_col: str = TARGET_COL,
                             target_ts_col: str = TARGET_TS_COL) -> None:
    """Structural + empirical leakage checks. Raises LeakageError on the first problem."""
    if target_col in feature_cols or target_ts_col in feature_cols:
        raise LeakageError("Target (or target timestamp) is listed among the feature columns")
    bad_names = [c for c in feature_cols if FORBIDDEN_NAME.search(c)]
    if bad_names:
        raise LeakageError(f"Feature names look forward-looking: {bad_names}")
    if not ((df[target_ts_col] - df["timestamp_utc"]) == ONE_HOUR).all():
        raise LeakageError("target_timestamp_utc must equal timestamp_utc + 1h on every row")

    target = df[target_col]
    for c in feature_cols:
        col = pd.to_numeric(df[c], errors="coerce")
        both = col.notna() & target.notna()
        if both.sum() > 2 and col[both].nunique() > 1:
            if np.allclose(col[both], target[both]):
                raise LeakageError(f"Feature '{c}' is identical to the target")
            if abs(np.corrcoef(col[both], target[both])[0, 1]) > 0.9999:
                raise LeakageError(f"Feature '{c}' is perfectly correlated with the target")

    # Semantics of lag columns: lag_k at t must equal pm25_t at t-k (a past value, never a future one).
    for c in feature_cols:
        m = LAG_NAME.match(c)
        if not m:
            continue
        pollutant, k = m.group(1), int(m.group(2))
        base = CURRENT_COLUMN.get(pollutant, pollutant)
        if base not in df.columns:
            raise LeakageError(f"Lag column '{c}' present but its base column '{base}' is missing, so it cannot be verified")
        past = df[["station_id", "timestamp_utc", base]].copy()
        past["timestamp_utc"] = past["timestamp_utc"] + k * ONE_HOUR  # value observed at t-k, re-labelled to t
        j = df[["station_id", "timestamp_utc", c]].merge(past, on=["station_id", "timestamp_utc"], how="inner")
        j = j.dropna(subset=[c, base])
        if len(j) and not np.allclose(j[c], j[base]):
            raise LeakageError(f"Lag column '{c}' does not match {pollutant} observed {k}h earlier")
    # target semantics: target at t must equal pm25_t at t+1h
    nxt = df[["station_id", "timestamp_utc", "pm25_t"]].copy()
    nxt["timestamp_utc"] = nxt["timestamp_utc"] - ONE_HOUR
    j = df[["station_id", "timestamp_utc", target_col]].merge(nxt, on=["station_id", "timestamp_utc"], how="inner")
    if len(j) and not np.allclose(j[target_col], j["pm25_t"]):
        raise LeakageError("Target does not equal pm25 observed one hour later")


def missing_summary(df: pd.DataFrame) -> dict:
    n = max(len(df), 1)
    return {c: {"missing": int(df[c].isna().sum()), "missing_pct": round(100 * df[c].isna().sum() / n, 2)} for c in df.columns}


def station_coverage(grid: pd.DataFrame) -> list:
    out = []
    for sid, g in grid.groupby("station_id"):
        observed = int(g["pm25"].notna().sum())
        out.append({
            "station_id": int(sid), "grid_hours": int(len(g)), "observed_hours": observed,
            "coverage_pct": round(100 * observed / max(len(g), 1), 2),
            "first_utc": g["timestamp_utc"].min().isoformat(), "last_utc": g["timestamp_utc"].max().isoformat(),
        })
    return out


def check_supervised(df: pd.DataFrame, feature_cols: list) -> list:
    """Return a list of human-readable error strings (empty list = pass)."""
    errors = []
    for col in ["station_id", "timestamp_utc", TARGET_TS_COL, TARGET_COL, *feature_cols]:
        if col not in df.columns:
            errors.append(f"missing column: {col}")
    if errors:
        return errors
    if df.duplicated(["station_id", "timestamp_utc"]).any():
        errors.append("duplicate (station_id, timestamp_utc) rows")
    if df[TARGET_COL].isna().any():
        errors.append("target contains NaN")
    if (df[TARGET_COL] < 0).any():
        errors.append("target contains negative PM2.5")
    if not df.sort_values(["station_id", "timestamp_utc"]).equals(df):
        errors.append("rows are not sorted by station_id, timestamp_utc")
    try:
        assert_no_target_leakage(df, feature_cols)
    except LeakageError as exc:
        errors.append(f"LEAKAGE: {exc}")
    return errors


def render_markdown(report: dict) -> str:
    lines = ["# AeroSense AI - data quality report", "", f"Status: **{'PASS' if not report['errors'] else 'FAIL'}**", ""]
    if report["errors"]:
        lines += ["## Errors", *[f"- {e}" for e in report["errors"]], ""]
    lines += ["## Row counts", "", "| dataset | rows |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in report["row_counts"].items()]
    lines += ["", "## Stations (hourly grid coverage)", "", "| station_id | grid_hours | observed | coverage % | first (UTC) | last (UTC) |", "|---|---|---|---|---|---|"]
    lines += [f"| {s['station_id']} | {s['grid_hours']} | {s['observed_hours']} | {s['coverage_pct']} | {s['first_utc']} | {s['last_utc']} |" for s in report["stations"]]
    lines += ["", "## Missing values - supervised dataset", "", "| column | missing | % |", "|---|---|---|"]
    lines += [f"| {c} | {v['missing']} | {v['missing_pct']} |" for c, v in report["missing_supervised"].items()]
    lines += ["", "## Cleaning statistics", "", "```json", json.dumps(report["cleaning_stats"], indent=2, default=str), "```", ""]
    return "\n".join(lines)


def build_report(settings: Settings) -> dict:
    p = settings.processed_dir
    merged = pd.read_csv(p / "pollution_weather_hourly.csv", parse_dates=["timestamp_utc"])
    sup = pd.read_csv(p / "supervised_pm25_t_plus_1h.csv", parse_dates=["timestamp_utc", TARGET_TS_COL])
    features = json.loads((p / "feature_columns.json").read_text(encoding="utf-8"))["features"]
    stats_path = p / "merge_stats.json"
    row_counts = {"merged_hourly_grid": len(merged), "supervised": len(sup)}
    raw_path = settings.raw_dir / "openaq_pm25_hourly_raw.csv"
    if raw_path.exists():
        row_counts["openaq_raw"] = int(sum(1 for _ in open(raw_path, encoding="utf-8")) - 1)
    errors = check_supervised(sup, features)
    return {
        "row_counts": row_counts,
        "stations": station_coverage(merged),
        "missing_merged": missing_summary(merged),
        "missing_supervised": missing_summary(sup),
        "time_range_utc": [sup["timestamp_utc"].min().isoformat(), sup["timestamp_utc"].max().isoformat()],
        "cleaning_stats": json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.exists() else {},
        "errors": errors,
    }


def run(settings: Settings) -> int:
    p = settings.processed_dir
    needed = ["pollution_weather_hourly.csv", "supervised_pm25_t_plus_1h.csv", "feature_columns.json"]
    missing = [n for n in needed if not (p / n).exists()]
    if missing:
        log.error("Missing processed files %s. Run merge_datasets first.", missing)
        return 1
    report = build_report(settings)
    (p / "data_quality_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    (p / "data_quality_report.md").write_text(render_markdown(report), encoding="utf-8")
    log.info("Row counts: %s", report["row_counts"])
    if report["errors"]:
        for e in report["errors"]:
            log.error("VALIDATION FAILED: %s", e)
        return 1
    log.info("All validation checks passed. Report: %s", p / "data_quality_report.md")
    return 0


def main(argv: Optional[list] = None) -> int:
    settings = Settings.from_env()
    setup_logging("validate_data", settings)
    return run(settings)


if __name__ == "__main__":
    sys.exit(main())
