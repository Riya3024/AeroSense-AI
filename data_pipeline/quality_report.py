"""Phase 2 data-quality report: reports/data_quality_report.json (machine) and .md (human).

Every number is read from artefacts of a real run - the fetch manifests, merge_stats.json, the hourly grid and the
exported dataset with its verification checks. Nothing is typed in or estimated. If an input artefact is missing, the
section says so (and the report fails) instead of guessing.

Reuses validate_data.missing_summary / station_coverage; it does not re-implement them.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

from .config import TARGET_COL, Settings, setup_logging
from .final_dataset import dataset_paths
from .validate_data import missing_summary, station_coverage

log = logging.getLogger("aerosense.quality_report")
REPORT_VERSION = 1

IMPUTATION_POLICY = (
    "Pollutants are NEVER imputed: a missing PM2.5/PM10/NO2/O3/CO/SO2 hour stays missing (NaN) and any training row that "
    "needs it is excluded. Only the weather variables listed in `weather_forward_fill_variables` can be filled, and only when "
    "the missing run is at most `weather_max_gap_hours` hours long: the value is carried FORWARD from the last observed hour "
    "(never backward, so no future information is used). Longer missing runs stay entirely missing - they are not partially "
    "filled. Each forward-filled value is flagged in the weather grid, and `weather_imputed_count` in the dataset counts them per row."
)


def _load_json(path) -> Optional[dict]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _longest_run(mask: np.ndarray) -> tuple:
    """(number_of_runs, longest_run) of True values in a boolean array."""
    runs, longest, cur = 0, 0, 0
    for flag in mask:
        if flag:
            cur += 1
            if cur == 1:
                runs += 1
            longest = max(longest, cur)
        else:
            cur = 0
    return runs, longest


def timestamp_statistics(grid: pd.DataFrame) -> dict:
    """Gap structure of the hourly PM2.5 grid per station (computed, not assumed)."""
    per_station = []
    for sid, g in grid.sort_values("timestamp_utc").groupby("station_id"):
        n_gaps, longest = _longest_run(g["pm25"].isna().to_numpy())
        steps = g["timestamp_utc"].diff().dropna()
        per_station.append({"station_id": int(sid), "grid_hours": int(len(g)), "observed_pm25_hours": int(g["pm25"].notna().sum()),
                            "missing_pm25_hours": int(g["pm25"].isna().sum()), "n_missing_runs": n_gaps,
                            "longest_missing_run_hours": longest, "irregular_steps_on_grid": int((steps != pd.Timedelta(hours=1)).sum())})
    ts = grid["timestamp_utc"]
    return {"timezone": "UTC", "grid_frequency": "1h", "grid_first_utc": ts.min().isoformat(), "grid_last_utc": ts.max().isoformat(),
            "all_on_the_hour": bool(((ts.dt.minute == 0) & (ts.dt.second == 0)).all()), "per_station": per_station}


def pollutant_row_accounting(stats: dict) -> dict:
    """Where every raw row of one pollutant went. `unaccounted` must be 0; if not, the report warns."""
    removed = {
        "no_usable_timestamp": stats.get("dropped_no_timestamp", 0),
        "not_on_the_hour": stats.get("dropped_not_on_the_hour", 0),
        "unexpected_unit": stats.get("dropped_unexpected_unit", 0),
        "null_value": stats.get("null_values", 0),
        "negative_value": stats.get("negative_values_set_missing", 0),
        "implausible_value": stats.get("implausible_values_set_missing", 0),
        "exact_duplicate_or_conflicting_duplicate": stats.get("duplicate_rows_removed", 0),
    }
    kept = stats.get("clean_rows", 0)
    return {"raw_rows": stats.get("raw_rows", 0), "removed": removed, "kept": kept,
            "unaccounted": stats.get("raw_rows", 0) - kept - sum(removed.values())}


def build_report(settings: Settings, now: Optional[datetime] = None) -> dict:
    p, raw = settings.processed_dir, settings.raw_dir
    paths = dataset_paths(settings)
    errors, warnings = [], []

    aq_manifest = _load_json(raw / "openaq_manifest.json")
    wx_manifest = _load_json(raw / "weather_manifest.json")
    merge_stats = _load_json(p / "merge_stats.json")
    ds_manifest = _load_json(paths["manifest"])
    for name, obj in (("openaq_manifest.json", aq_manifest), ("weather_manifest.json", wx_manifest),
                      ("merge_stats.json", merge_stats), (paths["manifest"].name, ds_manifest)):
        if obj is None:
            errors.append(f"missing input artefact: {name} (run the earlier pipeline steps first)")
    if errors or not paths["csv"].exists() or not (p / "pollution_weather_hourly.csv").exists():
        if not paths["csv"].exists():
            errors.append(f"missing dataset: {paths['csv'].name}")
        return {"report_version": REPORT_VERSION, "status": "FAIL", "errors": errors, "warnings": warnings,
                "generated_at_utc": (now or datetime.now(timezone.utc)).isoformat(timespec="seconds")}

    df = pd.read_csv(paths["csv"], parse_dates=["timestamp", "target_timestamp"])
    grid = pd.read_csv(p / "pollution_weather_hourly.csv", parse_dates=["timestamp_utc"])
    wx_raw_path = raw / "weather_hourly_raw.csv"
    wx_raw = pd.read_csv(wx_raw_path, parse_dates=["timestamp_utc"]) if wx_raw_path.exists() else pd.DataFrame()

    aq_stats, other_stats, wx_stats = merge_stats["openaq"], merge_stats.get("openaq_other_pollutants", {}), merge_stats["weather"]
    sup_stats = merge_stats.get("supervised", {})

    # ---- date ranges: requested vs actually received
    actual = {"openaq_pm25_first_utc": aq_stats.get("first_timestamp_utc"), "openaq_pm25_last_utc": aq_stats.get("last_timestamp_utc"),
              "dataset_first_timestamp": df["timestamp"].min().isoformat(), "dataset_last_timestamp": df["timestamp"].max().isoformat(),
              "dataset_last_target_timestamp": df["target_timestamp"].max().isoformat()}
    if len(wx_raw):
        first_var = settings.weather_variables[0]
        valid = wx_raw[wx_raw[first_var].notna()] if first_var in wx_raw.columns else wx_raw
        actual["weather_first_utc"] = wx_raw["timestamp_utc"].min().isoformat()
        actual["weather_last_utc_with_data"] = valid["timestamp_utc"].max().isoformat() if len(valid) else None
    req_start = pd.Timestamp(settings.start_date, tz="UTC")
    req_end = pd.Timestamp(settings.end_date, tz="UTC") + pd.Timedelta(days=1)
    if aq_stats.get("first_timestamp_utc") and pd.Timestamp(aq_stats["first_timestamp_utc"]) > req_start + pd.Timedelta(days=1):
        warnings.append(f"OpenAQ PM2.5 starts at {aq_stats['first_timestamp_utc']}, later than the requested start {settings.start_date}: "
                        "the requested period is not fully covered.")
    if aq_stats.get("last_timestamp_utc") and pd.Timestamp(aq_stats["last_timestamp_utc"]) < req_end - pd.Timedelta(days=2):
        warnings.append(f"OpenAQ PM2.5 ends at {aq_stats['last_timestamp_utc']}, earlier than the requested end {settings.end_date}.")
    if wx_manifest.get("effective_end_date") != wx_manifest.get("end_date"):
        warnings.append(f"END_DATE {wx_manifest['end_date']} was in the future; weather requested only up to {wx_manifest['effective_end_date']}.")

    # ---- fetch completeness
    if aq_manifest.get("partial"):
        failed = {k: v for k, v in aq_manifest.get("station_status", {}).items() if "FAILED" in v}
        warnings.append(f"OpenAQ fetch was PARTIAL: {len(failed)} station(s) had failed sensors: {failed}")
    if wx_manifest.get("partial"):
        warnings.append("Open-Meteo fetch was PARTIAL for some weather points: see weather_manifest.json point_status.")

    # ---- pollutants
    pollutants = {"pm25": {"available": True, **pollutant_row_accounting(aq_stats), "expected_unit": aq_stats.get("expected_unit")}}
    for name, s in other_stats.items():
        pollutants[name] = {"available": bool(s.get("available")), **pollutant_row_accounting(s), "expected_unit": s.get("expected_unit"),
                            "dropped_unit_counts": s.get("dropped_unit_counts", {})}
        if not s.get("available"):
            why = (f"all {s.get('dropped_unexpected_unit', 0)} rows were in units {s.get('dropped_unit_counts')} but {s.get('expected_unit')} "
                   "was expected (set POLLUTANT_UNITS if that unit is right; nothing was converted)") if s.get("dropped_unexpected_unit") else "no rows retrieved"
            warnings.append(f"Pollutant {name} is absent from the dataset: {why}.")
    for name, acct in pollutants.items():
        if acct.get("unaccounted"):
            warnings.append(f"Row accounting for {name} does not add up (unaccounted={acct['unaccounted']}); investigate before trusting counts.")
    requested_missing = [p_ for p_ in settings.pollutants if p_ not in pollutants]
    if requested_missing:
        warnings.append(f"Requested pollutants never reached cleaning: {requested_missing}")

    # ---- missingness
    n_final = len(df)
    per_column = {c: {**v, "observed": n_final - v["missing"]} for c, v in missing_summary(df).items()}
    imputed_rows = int((df["weather_imputed_count"] > 0).sum()) if "weather_imputed_count" in df.columns else None
    weather_grid = {v: {"forward_filled_values": int(wx_stats.get("gap_filled_values_by_variable", {}).get(v, 0)),
                        "still_missing_after_cleaning": int(wx_stats.get("missing_after_cleaning", {}).get(v, 0)),
                        "weather_grid_rows": int(wx_stats.get("rows_after_regularisation", 0))}
                    for v in settings.weather_variables}

    # ---- row flow / removal
    grid_rows, final_rows = int(len(grid)), int(n_final)
    rows_flow = [
        {"stage": "OpenAQ raw rows, all pollutants", "rows": int(aq_manifest.get("raw_rows", 0))},
        {"stage": "PM2.5 rows after cleaning", "rows": int(aq_stats.get("clean_rows", 0))},
        {"stage": "Hourly PM2.5 grid per station (missing hours kept as NaN)", "rows": int(aq_stats.get("grid_rows_incl_missing_hours", 0))},
        {"stage": "Grid joined with weather", "rows": int(merge_stats.get("merged_grid_rows", grid_rows))},
        {"stage": "Final ML dataset (target, pm25 and 1h/24h lags all present)", "rows": final_rows},
    ]
    removed_supervised = {"rows_before": sup_stats.get("rows_before_required_dropna"), "rows_after": sup_stats.get("rows_after_required_dropna"),
                          "rows_removed": sup_stats.get("rows_removed"),
                          "rows_missing_each_required_column_overlapping": sup_stats.get("rows_missing_required_column"),
                          "explanation": "Rows are removed when the target (PM2.5 at t+1h), current PM2.5, or the 1h/24h PM2.5 lags are missing. "
                                         "The first 24 hours of each station and the final hour necessarily lack a lag/target."}

    checks = ds_manifest["checks"]
    failed_checks = [c["name"] for c in checks if c["status"] == "fail"]
    errors += [f"validation check failed: {n}" for n in failed_checks]
    leak_names = ("target_timestamp_is_exactly_t_plus_1h", "target_equals_pm25_one_hour_later", "leakage_guard_on_exported_file",
                  "chronological_split_has_no_window_overlap")

    report = {
        "report_version": REPORT_VERSION,
        "status": "PASS" if not errors else "FAIL",
        "generated_at_utc": (now or datetime.now(timezone.utc)).isoformat(timespec="seconds"),
        "data_collection": {"openaq_fetched_at_utc": aq_manifest.get("run_at_utc"), "weather_fetched_at_utc": wx_manifest.get("run_at_utc"),
                            "dataset_built_at_utc": ds_manifest.get("created_at_utc")},
        "source_apis": [
            {"name": "OpenAQ v3", "endpoint": aq_manifest.get("endpoint"), "pollutants_requested": aq_manifest.get("pollutants_requested"),
             "stations_discovered": aq_manifest.get("stations_discovered"), "sensors_by_pollutant": aq_manifest.get("sensors_by_pollutant"),
             "requests_made_last_run": aq_manifest.get("requests_made_this_run"), "partial": aq_manifest.get("partial")},
            {"name": "Open-Meteo Historical Weather API (reanalysis)", "endpoint": wx_manifest.get("endpoint"), "variables": wx_manifest.get("variables"),
             "units": wx_manifest.get("units"), "timezone": wx_manifest.get("timezone"), "weather_mode": wx_manifest.get("weather_mode"),
             "points_requested": wx_manifest.get("points_requested"), "partial": wx_manifest.get("partial")},
        ],
        "location": ds_manifest["location"],
        "date_range": {"requested": {"start": str(settings.start_date), "end": str(settings.end_date)}, "actual": actual},
        "record_counts": {"openaq_raw_rows": aq_manifest.get("raw_rows"), "openaq_raw_rows_by_pollutant": aq_manifest.get("raw_rows_by_pollutant"),
                          "weather_raw_rows": wx_manifest.get("raw_rows"), "hourly_grid_rows": grid_rows, "final_dataset_rows": final_rows,
                          "final_dataset_stations": ds_manifest["stations"], "final_dataset_columns": ds_manifest["columns"]},
        "rows_flow": rows_flow,
        "rows_removed": {"per_pollutant_cleaning": {k: v["removed"] for k, v in pollutants.items()}, "final_dataset_filter": removed_supervised},
        "rows_retained": {"final_dataset_rows": final_rows, "pct_of_hourly_grid_rows": round(100 * final_rows / max(grid_rows, 1), 2)},
        "pollutants": pollutants,
        "missing_values": {"per_column_final_dataset": per_column, "pm25_missing_hours_on_grid": aq_stats.get("missing_hours_on_grid"),
                           "weather_on_hourly_grid": weather_grid,
                           "final_rows_with_any_forward_filled_weather": imputed_rows},
        "imputation": {"policy": IMPUTATION_POLICY, "pollutant_values_imputed": 0,
                       "weather_forward_fill_variables": sorted(wx_stats.get("gap_filled_values_by_variable", {})),
                       "weather_max_gap_hours": settings.weather_max_gap_hours,
                       "weather_values_forward_filled": int(sum(wx_stats.get("gap_filled_values_by_variable", {}).values()))},
        "duplicates": {"openaq_duplicate_rows_removed": {k: v["removed"]["exact_duplicate_or_conflicting_duplicate"] for k, v in pollutants.items()},
                       "weather_duplicate_rows_removed": wx_stats.get("duplicate_rows_removed"),
                       "final_dataset_duplicate_station_timestamp_keys": int(df.duplicated(["station_id", "timestamp"]).sum())},
        "timestamps": timestamp_statistics(grid),
        "stations": station_coverage(grid),
        "features": {"feature_columns": ds_manifest["feature_columns"], "metadata_columns": ds_manifest["metadata_columns"]},
        "target": {**ds_manifest["target"], "availability": {"rows_with_target": int(df[TARGET_COL].notna().sum()), "rows": final_rows,
                                                              "pct_of_final_rows": round(100 * df[TARGET_COL].notna().mean(), 2)}},
        "dataset": {"file": ds_manifest["dataset"], "sha256": ds_manifest["sha256_csv"], "rows": final_rows, "schema": ds_manifest["schema"],
                    "parquet": ds_manifest["parquet"]},
        "validation": {"all_passed": not failed_checks, "checks": checks},
        "leakage": {"in_pipeline_checks": [c for c in checks if c["name"] in leak_names],
                    "note": "These checks ran on the exported file. The unit-test leakage suite (tests/test_leakage.py) is run separately "
                            "by scripts/run_phase2.ps1 / run_tests.ps1 and is not recorded in this file."},
        "config": ds_manifest["config"],
        "reproducibility": {"input_files": ds_manifest["input_files"], "environment": ds_manifest["environment"],
                            "note": "Same configuration reproduces the dataset up to source API updates (OpenAQ back-fills, ERA5 revisions)."},
        "warnings": warnings, "errors": errors,
    }
    return report


def render_markdown(r: dict) -> str:
    if "date_range" not in r:
        return "\n".join(["# AeroSense AI - Phase 2 data quality report", "", f"Status: **{r['status']}**", "", "## Errors", *[f"- {e}" for e in r["errors"]], ""])
    L = ["# AeroSense AI - Phase 2 data quality report", "", f"Status: **{r['status']}**  |  generated {r['generated_at_utc']} UTC", "",
         "Every number below was computed from the pipeline run that produced this dataset.", ""]
    dc = r["data_collection"]
    loc = r["location"]
    L += ["## Sources and collection", "", f"- Location: {loc['name']} ({loc['latitude']}, {loc['longitude']}); station bbox {loc['station_bbox']}",
          f"- OpenAQ fetched: {dc['openaq_fetched_at_utc']}  |  Open-Meteo fetched: {dc['weather_fetched_at_utc']}  |  dataset built: {dc['dataset_built_at_utc']}"]
    for s in r["source_apis"]:
        L.append(f"- **{s['name']}** - {s['endpoint']} (partial fetch: {s['partial']})")
    dr = r["date_range"]
    L += ["", "## Date range", "", f"- Requested: {dr['requested']['start']} to {dr['requested']['end']}"]
    L += [f"- Actual `{k}`: {v}" for k, v in dr["actual"].items()]
    rc = r["record_counts"]
    L += ["", "## Record counts", "", f"- OpenAQ raw rows: {rc['openaq_raw_rows']} (by pollutant: {rc['openaq_raw_rows_by_pollutant']})",
          f"- Weather raw rows: {rc['weather_raw_rows']}", f"- Hourly grid rows: {rc['hourly_grid_rows']}",
          f"- **Final dataset: {rc['final_dataset_rows']} rows, {rc['final_dataset_columns']} columns, {rc['final_dataset_stations']} stations**",
          "", "| stage | rows |", "|---|---|"] + [f"| {s['stage']} | {s['rows']} |" for s in r["rows_flow"]]
    f = r["rows_removed"]["final_dataset_filter"]
    L += ["", f"Final filter: {f['rows_before']} -> {f['rows_after']} rows ({f['rows_removed']} removed). {f['explanation']}", "",
          "## Pollutants (row accounting: where every raw row went)", "", "| pollutant | available | raw | kept | removed (reason: n) | unaccounted |", "|---|---|---|---|---|---|"]
    for k, v in r["pollutants"].items():
        rem = ", ".join(f"{a}: {b}" for a, b in v["removed"].items() if b) or "-"
        L.append(f"| {k} | {v['available']} | {v['raw_rows']} | {v['kept']} | {rem} | {v['unaccounted']} |")
    L += ["", "## Missing values (final dataset)", "", "| column | observed | missing | missing % |", "|---|---|---|---|"]
    L += [f"| {c} | {v['observed']} | {v['missing']} | {v['missing_pct']} |" for c, v in r["missing_values"]["per_column_final_dataset"].items()]
    mv = r["missing_values"]
    L += ["", f"PM2.5 hours missing on the hourly grid: {mv['pm25_missing_hours_on_grid']}. Final rows containing any forward-filled weather value: {mv['final_rows_with_any_forward_filled_weather']}.",
          "", "Weather on the hourly weather grid:", "", "| variable | forward-filled | still missing |", "|---|---|---|"]
    L += [f"| {v} | {d['forward_filled_values']} | {d['still_missing_after_cleaning']} |" for v, d in mv["weather_on_hourly_grid"].items()]
    L += ["", "## Imputation policy", "", r["imputation"]["policy"], "", f"Pollutant values imputed: {r['imputation']['pollutant_values_imputed']}. "
          f"Weather values forward-filled: {r['imputation']['weather_values_forward_filled']}.", "", "## Duplicates", "", f"```json\n{json.dumps(r['duplicates'], indent=2)}\n```",
          "", "## Timestamps (UTC, hourly)", "", "| station | grid hours | observed PM2.5 | missing | missing runs | longest missing run (h) |", "|---|---|---|---|---|---|"]
    L += [f"| {s['station_id']} | {s['grid_hours']} | {s['observed_pm25_hours']} | {s['missing_pm25_hours']} | {s['n_missing_runs']} | {s['longest_missing_run_hours']} |"
          for s in r["timestamps"]["per_station"]]
    t = r["target"]
    L += ["", "## Target", "", f"`{t['column']}`: {t['definition']}", "", f"Availability: {t['availability']['rows_with_target']} of {t['availability']['rows']} final rows ({t['availability']['pct_of_final_rows']}%).",
          "", "## Features", "", ", ".join(f"`{c}`" for c in r["features"]["feature_columns"]), "", f"Metadata (not features): {r['features']['metadata_columns']}",
          "", "## Validation and leakage checks", "", "| check | status | detail |", "|---|---|---|"]
    L += [f"| {c['name']} | {c['status'].upper()} | {c['detail'][:110]} |" for c in r["validation"]["checks"]]
    L += ["", r["leakage"]["note"], "", "## Warnings", *([f"- {w}" for w in r["warnings"]] or ["- none"]), "", "## Errors", *([f"- {e}" for e in r["errors"]] or ["- none"]), ""]
    return "\n".join(L)


def run(settings: Settings) -> int:
    report = build_report(settings)
    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    (settings.reports_dir / "data_quality_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    (settings.reports_dir / "data_quality_report.md").write_text(render_markdown(report), encoding="utf-8")
    for w in report.get("warnings", []):
        log.warning("%s", w)
    for e in report.get("errors", []):
        log.error("%s", e)
    log.info("Quality report: %s (status %s)", settings.reports_dir / "data_quality_report.md", report["status"])
    return 0 if report["status"] == "PASS" else 1


def main(argv: Optional[list] = None) -> int:
    argparse.ArgumentParser(description=__doc__.split("\n")[0]).parse_args(argv)
    settings = Settings.from_env()
    setup_logging("quality_report", settings)
    return run(settings)


if __name__ == "__main__":
    sys.exit(main())
