"""Quality report: every figure must trace back to the run artefacts. SYNTHETIC inputs (tests/synth.py)."""
import json
from dataclasses import replace
from datetime import date, datetime, timezone

import pandas as pd
import pytest

from data_pipeline import final_dataset, merge_datasets, quality_report, validate_data
from tests.synth import write_synthetic_manifests, write_synthetic_raw


@pytest.fixture
def run_env(settings):
    s = replace(settings, start_date=date(2025, 11, 1), end_date=date(2025, 11, 4))
    write_synthetic_raw(s)
    write_synthetic_manifests(s)
    assert merge_datasets.run(s) == 0 and validate_data.run(s) == 0 and final_dataset.run(s) == 0
    return s


@pytest.fixture
def report(run_env):
    return quality_report.build_report(run_env, now=datetime(2026, 1, 1, tzinfo=timezone.utc))


def test_report_contains_every_section_required_by_the_spec(report):
    required = {"source_apis", "location", "date_range", "record_counts", "rows_removed", "rows_retained", "missing_values", "duplicates",
                "timestamps", "features", "target", "validation", "leakage", "warnings", "errors", "data_collection", "dataset", "config",
                "reproducibility", "imputation", "pollutants", "rows_flow"}
    assert required <= set(report) and report["status"] == "PASS"
    assert report["generated_at_utc"] == "2026-01-01T00:00:00+00:00"
    assert set(report["date_range"]) == {"requested", "actual"}


def test_counts_in_the_report_equal_counts_recomputed_from_the_files(run_env, report):
    paths = final_dataset.dataset_paths(run_env)
    ds = pd.read_csv(paths["csv"])
    raw = pd.read_csv(run_env.raw_dir / "openaq_pm25_hourly_raw.csv")
    assert report["record_counts"]["final_dataset_rows"] == len(ds) == report["rows_flow"][-1]["rows"]
    assert report["record_counts"]["final_dataset_columns"] == ds.shape[1]
    assert report["record_counts"]["openaq_raw_rows"] == len(raw)
    assert report["dataset"]["sha256"] == json.loads(paths["manifest"].read_text())["sha256_csv"]
    for col, v in report["missing_values"]["per_column_final_dataset"].items():
        assert v["missing"] == int(ds[col].isna().sum()) and v["observed"] == len(ds) - v["missing"]
    assert report["target"]["availability"] == {"rows_with_target": len(ds), "rows": len(ds), "pct_of_final_rows": 100.0}
    assert report["rows_retained"]["final_dataset_rows"] == len(ds)


def test_every_raw_row_is_accounted_for_per_pollutant(report):
    for name, acct in report["pollutants"].items():
        assert acct["unaccounted"] == 0, name
        assert acct["raw_rows"] == acct["kept"] + sum(acct["removed"].values())
    assert report["pollutants"]["pm10"]["removed"]["exact_duplicate_or_conflicting_duplicate"] == 1  # the synthetic duplicate row is a PM10 row
    assert report["pollutants"]["pm25"]["removed"]["exact_duplicate_or_conflicting_duplicate"] == 0
    assert report["duplicates"]["final_dataset_duplicate_station_timestamp_keys"] == 0


def test_timestamp_statistics_and_gap_structure_are_computed(report):
    per = {s["station_id"]: s for s in report["timestamps"]["per_station"]}
    assert per[1]["missing_pm25_hours"] == 0 and per[2]["missing_pm25_hours"] == 1
    assert per[2]["n_missing_runs"] == 1 and per[2]["longest_missing_run_hours"] == 1
    assert report["timestamps"]["all_on_the_hour"] is True and report["timestamps"]["timezone"] == "UTC"
    assert all(s["irregular_steps_on_grid"] == 0 for s in per.values())


def test_absent_pollutant_because_of_unit_mismatch_is_reported_as_a_warning_not_hidden(report):
    assert report["pollutants"]["co"]["available"] is False and report["pollutants"]["co"]["dropped_unit_counts"] == {"ppm": 72}
    assert any("co" in w and "ppm" in w and "nothing was converted" in w for w in report["warnings"])
    assert any("o3" in w for w in report["warnings"]) and "co" not in report["features"]["feature_columns"]


def test_actual_range_narrower_than_requested_is_warned_about(run_env):
    wide = replace(run_env, start_date=date(2025, 1, 1), end_date=date(2026, 9, 30))
    r = quality_report.build_report(wide)
    assert any("starts at" in w and "not fully covered" in w for w in r["warnings"])
    assert any("ends at" in w for w in r["warnings"])
    assert r["date_range"]["requested"] == {"start": "2025-01-01", "end": "2026-09-30"}  # requested and actual both recorded


def test_failed_validation_check_or_partial_fetch_surface_in_status_and_warnings(run_env):
    paths = final_dataset.dataset_paths(run_env)
    m = json.loads(paths["manifest"].read_text())
    m["checks"][0]["status"] = "fail"
    paths["manifest"].write_text(json.dumps(m))
    aq = json.loads((run_env.raw_dir / "openaq_manifest.json").read_text())
    aq["partial"], aq["station_status"]["2"] = True, "pm25: FAILED: HTTP 500"
    (run_env.raw_dir / "openaq_manifest.json").write_text(json.dumps(aq))
    r = quality_report.build_report(run_env)
    assert r["status"] == "FAIL" and any("validation check failed" in e for e in r["errors"])
    assert any("PARTIAL" in w for w in r["warnings"])


def test_missing_artefacts_produce_a_failed_report_instead_of_invented_numbers(settings):
    r = quality_report.build_report(settings)
    assert r["status"] == "FAIL" and any("missing input artefact" in e for e in r["errors"])
    assert "record_counts" not in r
    assert "FAIL" in quality_report.render_markdown(r)
    assert quality_report.run(settings) == 1


def test_run_writes_machine_and_human_readable_reports_without_secrets(run_env):
    assert quality_report.run(run_env) == 0
    j = (run_env.reports_dir / "data_quality_report.json").read_text(encoding="utf-8")
    md = (run_env.reports_dir / "data_quality_report.md").read_text(encoding="utf-8")
    assert json.loads(j)["status"] == "PASS" and run_env.openaq_api_key not in j + md
    for heading in ("## Date range", "## Record counts", "## Missing values (final dataset)", "## Imputation policy", "## Duplicates",
                    "## Target", "## Validation and leakage checks", "## Warnings"):
        assert heading in md
    assert "target_pm25_t_plus_1h" in md and "never" in md.lower()


def test_longest_run_helper():
    import numpy as np
    assert quality_report._longest_run(np.array([False, True, True, False, True])) == (2, 2)
    assert quality_report._longest_run(np.array([False, False])) == (0, 0)
