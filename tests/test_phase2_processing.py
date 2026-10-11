"""Phase 2 processing on SYNTHETIC raw inputs (tests/synth.py): units, alignment, duplicates, missing-data policy,
target creation, leakage, schema, export verification and reproducibility. Passing says nothing about real data quality."""
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from data_pipeline import final_dataset, merge_datasets
from data_pipeline.config import TARGET_COL
from data_pipeline.merge_datasets import ONE_HOUR, attach_pollutant, build_supervised, clean_openaq, regularize_hourly
from data_pipeline.validate_data import LeakageError, assert_no_target_leakage
from tests.conftest import make_merged, raw_openaq_row
from tests.synth import T0, UG, write_synthetic_raw


@pytest.fixture
def exported(settings):
    write_synthetic_raw(settings)
    assert merge_datasets.run(settings) == 0
    assert final_dataset.run(settings) == 0
    paths = final_dataset.dataset_paths(settings)
    return settings, pd.read_csv(paths["csv"], parse_dates=["timestamp", "target_timestamp"]), paths


# ------------------------------------------------------------------ units / hourly aggregation / duplicates
def _rows(pollutant, unit, values, station=1):
    out = []
    for i, v in enumerate(values, start=1):
        r = raw_openaq_row(T0 + pd.Timedelta(hours=i), v, station_id=station, unit=unit)
        r["pollutant"] = pollutant
        out.append(r)
    return pd.DataFrame(out)


def test_wrong_unit_rows_are_dropped_counted_by_unit_and_never_converted(settings):
    raw = pd.concat([_rows("co", UG, [1.0, 2.0]), _rows("co", "ppm", [9.0, 9.0, 9.0])], ignore_index=True)
    cleaned, stats = clean_openaq(raw, settings, "co")
    assert stats["dropped_unexpected_unit"] == 3 and stats["dropped_unit_counts"] == {"ppm": 3}
    assert cleaned["co"].tolist() == [1.0, 2.0]  # the ppm values were not rescaled into the data


def test_unit_expectation_is_configurable_per_pollutant(settings):
    from dataclasses import replace
    s = replace(settings, pollutant_units={"co": "mg/m3"})
    cleaned, stats = clean_openaq(_rows("co", "mg/m\u00b3", [1.1, 1.2]), s, "co")
    assert len(cleaned) == 2 and stats["expected_unit"] == "mg/m3"


def test_pollutants_are_separated_by_name_and_off_hour_rows_are_counted(settings):
    raw = pd.concat([_rows("pm25", UG, [10.0, 11.0]), _rows("pm10", UG, [20.0, 21.0, 22.0])], ignore_index=True)
    off = raw.iloc[[0]].copy()
    off["period_to_utc"] = "2025-11-01T01:30:00Z"  # not on the hour
    raw = pd.concat([raw, off], ignore_index=True)
    c25, s25 = clean_openaq(raw, settings, "pm25")
    c10, s10 = clean_openaq(raw, settings, "pm10")
    assert len(c25) == 2 and s25["raw_rows"] == 3 and s25["dropped_not_on_the_hour"] == 1
    assert len(c10) == 3 and s10["raw_rows"] == 3 + 0 and "pm25" not in c10.columns


def test_duplicate_observations_are_removed_and_counted(settings):
    rows = _rows("pm10", UG, [20.0, 21.0])
    cleaned, stats = clean_openaq(pd.concat([rows, rows.iloc[[0]]], ignore_index=True), settings, "pm10")
    assert stats["duplicate_rows_removed"] == 1 and len(cleaned) == 2


def test_extra_pollutant_is_joined_on_the_pm25_grid_without_extending_it(settings):
    pm25, _ = clean_openaq(_rows("pm25", UG, [10.0, 11.0, 12.0]), settings, "pm25")
    grid = regularize_hourly(pm25)
    pm10, _ = clean_openaq(_rows("pm10", UG, [20.0, 21.0, 22.0, 23.0, 24.0]), settings, "pm10")
    merged, extra = attach_pollutant(grid, pm10, "pm10")
    assert len(merged) == len(grid) == 3 and extra["rows_outside_pm25_grid_ignored"] == 2
    assert merged["pm10"].tolist() == [20.0, 21.0, 22.0]


# ------------------------------------------------------------------ missing-data policy
def test_pm25_gaps_are_never_filled(exported):
    settings, ds, _ = exported
    grid = pd.read_csv(settings.processed_dir / "pollution_weather_hourly.csv", parse_dates=["timestamp_utc"])
    gap = grid[(grid["station_id"] == 2) & (grid["timestamp_utc"] == T0 + pd.Timedelta(hours=30))]
    assert len(gap) == 1 and gap["pm25"].isna().all()
    stats = json.loads((settings.processed_dir / "merge_stats.json").read_text())
    assert stats["openaq"]["missing_hours_on_grid"] == 1
    # the hours whose lag/target would have needed the missing value are excluded, not patched
    s2 = ds[ds["station_id"] == 2]["timestamp"]
    assert (T0 + pd.Timedelta(hours=30)) not in set(s2) and (T0 + pd.Timedelta(hours=29)) not in set(s2)


def test_short_weather_gap_is_forward_filled_from_the_past_and_flagged_long_gap_stays_missing(exported):
    settings, ds, _ = exported
    grid = pd.read_csv(settings.processed_dir / "pollution_weather_hourly.csv", parse_dates=["timestamp_utc"])
    s1 = grid[grid["station_id"] == 1].set_index("timestamp_utc")
    at = lambda h: s1.loc[T0 + pd.Timedelta(hours=h)]
    # synthetic temperature at hour h is 20 + h % 12; hours 30,31 were missing in the source
    assert at(29)["temperature_2m"] == 20 + 29 % 12
    assert at(30)["temperature_2m"] == at(31)["temperature_2m"] == at(29)["temperature_2m"]  # carried from the PAST (25), not from hour 32 (28)
    assert at(30)["temperature_2m_imputed"] == at(31)["temperature_2m_imputed"] == 1 and at(29)["temperature_2m_imputed"] == 0
    # the 6-hour gap (hours 40-45) is longer than weather_max_gap_hours=3: the WHOLE gap stays missing, nothing is partially filled
    assert all(pd.isna(at(h)["temperature_2m"]) and at(h)["temperature_2m_imputed"] == 0 for h in range(40, 46))
    assert at(39)["temperature_2m_imputed"] == 0 and at(46)["temperature_2m_imputed"] == 0
    assert ds["weather_imputed_count"].gt(0).sum() > 0 and set(ds.columns) >= {"weather_imputed_count"}


def test_missing_data_statistics_come_from_the_run(exported):
    settings, ds, _ = exported
    stats = json.loads((settings.processed_dir / "merge_stats.json").read_text())
    sup = stats["supervised"]
    assert sup["rows_after_required_dropna"] == len(ds) and sup["rows_removed"] == sup["rows_before_required_dropna"] - len(ds)
    assert stats["weather"]["gap_filled_values_by_variable"]["temperature_2m"] == 2  # only the 2-hour gap; the 6-hour gap is left missing
    assert stats["weather"]["missing_after_cleaning"]["temperature_2m"] == 6


# ------------------------------------------------------------------ target, ordering, leakage
def test_target_is_exactly_pm25_one_hour_later_checked_against_the_raw_source(exported):
    """Independent of the pipeline code: rebuild pm25(t) and pm25(t+1h) straight from the raw synthetic file."""
    settings, ds, _ = exported
    raw = pd.read_csv(settings.raw_dir / "openaq_pm25_hourly_raw.csv", parse_dates=["period_to_utc"])
    raw = raw[(raw["pollutant"] == "pm25")].drop_duplicates(["station_id", "period_to_utc"])
    lookup = {(r.station_id, r.period_to_utc): r.value for r in raw.itertuples()}
    assert len(ds) > 50
    for r in ds.itertuples():
        assert r.timestamp + ONE_HOUR == r.target_timestamp
        assert np.isclose(lookup[(r.station_id, r.timestamp)], r.pm25)
        assert np.isclose(lookup[(r.station_id, r.target_timestamp)], getattr(r, TARGET_COL))


def test_final_dataset_is_chronological_unique_and_hourly(exported):
    _, ds, _ = exported
    assert ds.sort_values(["station_id", "timestamp"]).equals(ds)
    assert not ds.duplicated(["station_id", "timestamp"]).any()
    assert ((ds["timestamp"].dt.minute == 0) & (ds["timestamp"].dt.second == 0)).all() and str(ds["timestamp"].dt.tz) == "UTC"


def test_features_at_or_before_t0_are_unchanged_when_future_pm10_and_no2_are_rewritten(settings):
    rng = np.random.default_rng(3)
    merged = make_merged(n=96, stations=(1, 2))
    merged["pm25"] += rng.normal(0, 3, len(merged))
    merged["pm10"], merged["no2"] = rng.normal(150, 5, len(merged)), rng.normal(40, 2, len(merged))
    ds, feats = build_supervised(merged, settings)
    assert {"pm10", "pm10_lag_1h", "no2", "no2_lag_1h"} <= set(feats)
    t0 = merged["timestamp_utc"].iloc[60]
    tampered = merged.copy()
    future = tampered["timestamp_utc"] > t0
    tampered.loc[future, ["pm10", "no2"]] = tampered.loc[future, ["pm10", "no2"]] * 9 + 500
    ds2, _ = build_supervised(tampered, settings)
    key = ["station_id", "timestamp_utc"]
    pd.testing.assert_frame_equal(ds[ds["timestamp_utc"] <= t0].set_index(key)[feats], ds2[ds2["timestamp_utc"] <= t0].set_index(key)[feats])


def test_leakage_guard_catches_a_forward_looking_other_pollutant_lag(settings):
    merged = make_merged(n=96, stations=(1,))
    merged["pm25"] += np.random.default_rng(0).normal(0, 3, len(merged))
    merged["pm10"] = np.random.default_rng(1).normal(150, 5, len(merged))
    ds, feats = build_supervised(merged, settings)
    assert_no_target_leakage(ds, feats)
    bad = ds.copy()
    bad["pm10_lag_1h"] = bad["pm10"].shift(-1)  # a "lag" that actually looks forward
    with pytest.raises(LeakageError):
        assert_no_target_leakage(bad.dropna(subset=["pm10_lag_1h"]), feats)


# ------------------------------------------------------------------ schema / export / reproducibility
def test_public_schema_has_the_required_columns_and_units(exported):
    settings, ds, paths = exported
    for col in final_dataset.REQUIRED_PUBLIC:
        assert col in ds.columns, col
    assert {"pm10", "no2"} <= set(ds.columns) and not ({"co", "o3", "so2"} & set(ds.columns))  # absent pollutants are omitted, not NaN-filled
    manifest = json.loads(paths["manifest"].read_text())
    roles = {c["name"]: c["role"] for c in manifest["schema"]}
    assert roles[TARGET_COL] == "target" and roles["timestamp"] == "metadata" and roles["pm25"] == "feature"
    units = {c["name"]: c["unit"] for c in manifest["schema"]}
    assert units["temperature"] == "degC" and units["wind_speed"] == "km/h" and "ug/m3" in units["pm25"]
    assert TARGET_COL not in manifest["feature_columns"] and "target_timestamp" not in manifest["feature_columns"]


def test_exported_file_passes_all_checks_and_manifest_records_provenance(exported):
    settings, ds, paths = exported
    manifest = json.loads(paths["manifest"].read_text())
    assert [c for c in manifest["checks"] if c["status"] == "fail"] == []
    assert manifest["rows"] == len(ds) and manifest["sha256_csv"] == hashlib.sha256(paths["csv"].read_bytes()).hexdigest()
    assert set(manifest["input_files"]) >= {"openaq_pm25_hourly_raw.csv", "weather_hourly_raw.csv"}
    assert settings.openaq_api_key not in json.dumps(manifest) and manifest["config"]["openaq_api_key_present"] is True
    assert manifest["parquet"]["written"] is True or "parquet engine" in manifest["parquet"]["reason"]
    sample = pd.read_csv(paths["sample"])
    assert 0 < len(sample) <= 168 and sample["station_id"].nunique() == 1


def test_tampering_with_the_exported_target_or_a_lag_is_detected(exported):
    settings, ds, _ = exported
    feats = json.loads(final_dataset.dataset_paths(settings)["manifest"].read_text())["feature_columns"]
    bad_target = ds.copy()
    bad_target.loc[bad_target.index[10], TARGET_COL] += 5.0
    res = {c["name"]: c["status"] for c in final_dataset.verify_public(bad_target, feats, settings)}
    assert res["target_equals_pm25_one_hour_later"] == "fail" and res["leakage_guard_on_exported_file"] == "fail"
    bad_lag = ds.copy()
    bad_lag["pm25_lag_1h"] = bad_lag[TARGET_COL]
    assert {c["name"]: c["status"] for c in final_dataset.verify_public(bad_lag, feats, settings)}["leakage_guard_on_exported_file"] == "fail"


def test_same_inputs_reproduce_a_byte_identical_dataset(settings, tmp_path):
    from dataclasses import replace
    digests = []
    for i in range(2):
        s = replace(settings, root=tmp_path / f"run{i}")
        write_synthetic_raw(s)
        assert merge_datasets.run(s) == 0 and final_dataset.run(s) == 0
        digests.append(hashlib.sha256(final_dataset.dataset_paths(s)["csv"].read_bytes()).hexdigest())
    assert digests[0] == digests[1]


def test_export_refuses_when_processed_inputs_are_missing(settings):
    assert final_dataset.run(settings) == 1
