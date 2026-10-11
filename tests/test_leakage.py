import numpy as np
import pandas as pd
import pytest

from data_pipeline.config import TARGET_COL, TARGET_TS_COL
from data_pipeline.merge_datasets import build_supervised, chronological_split
from data_pipeline.validate_data import LeakageError, assert_no_target_leakage, check_supervised
from tests.conftest import make_merged


@pytest.fixture
def built(settings):
    rng = np.random.default_rng(0)
    merged = make_merged(n=96, stations=(1, 2))
    merged["pm25"] = merged["pm25"] + rng.normal(0, 3, len(merged))  # break exact linearity
    ds, feats = build_supervised(merged, settings)
    return merged, ds, feats


def test_guard_passes_on_correctly_built_dataset(built):
    _, ds, feats = built
    assert_no_target_leakage(ds, feats)
    assert check_supervised(ds, feats) == []


def test_guard_rejects_target_in_features(built):
    _, ds, feats = built
    with pytest.raises(LeakageError):
        assert_no_target_leakage(ds, feats + [TARGET_COL])


def test_guard_rejects_forward_looking_feature_names(built):
    _, ds, feats = built
    ds = ds.assign(pm25_next_hour=ds[TARGET_COL])
    with pytest.raises(LeakageError):
        assert_no_target_leakage(ds, feats + ["pm25_next_hour"])


def test_guard_rejects_feature_that_copies_the_target(built):
    _, ds, feats = built
    ds = ds.assign(innocent_name=ds[TARGET_COL] * 1.0)
    with pytest.raises(LeakageError):
        assert_no_target_leakage(ds, feats + ["innocent_name"])


def test_guard_rejects_lag_column_holding_future_values(built):
    _, ds, feats = built
    bad = ds.copy()
    bad["pm25_lag_1h"] = bad[TARGET_COL]  # a "lag" that is secretly t+1h
    with pytest.raises(LeakageError):
        assert_no_target_leakage(bad, feats)


def test_features_are_unchanged_when_future_pm25_and_weather_are_altered(settings, built):
    """Strongest check: rewrite everything after t0; every feature row at or before t0 must be identical."""
    merged, ds, feats = built
    t0 = merged["timestamp_utc"].iloc[60]
    tampered = merged.copy()
    future = tampered["timestamp_utc"] > t0
    tampered.loc[future, "pm25"] = tampered.loc[future, "pm25"] * 7 + 1000
    for col in ("temperature_2m", "relative_humidity_2m", "wind_speed_10m", "precipitation"):
        tampered.loc[future, col] = tampered.loc[future, col] * 3 + 50
    ds2, feats2 = build_supervised(tampered, settings)
    assert feats == feats2
    a = ds[ds["timestamp_utc"] <= t0].set_index(["station_id", "timestamp_utc"])[feats]
    b = ds2[ds2["timestamp_utc"] <= t0].set_index(["station_id", "timestamp_utc"])[feats]
    pd.testing.assert_frame_equal(a, b)
    # sanity: the tampering really did change targets, i.e. the test is able to detect leakage
    assert not ds.set_index(["station_id", "timestamp_utc"])[TARGET_COL].equals(ds2.set_index(["station_id", "timestamp_utc"])[TARGET_COL])


def test_chronological_split_has_no_overlap_and_no_target_crossing(built):
    _, ds, _ = built
    train, val, test = chronological_split(ds, 0.6, 0.2)
    assert len(train) and len(val) and len(test)
    assert train["timestamp_utc"].max() < val["timestamp_utc"].min()
    assert val["timestamp_utc"].max() < test["timestamp_utc"].min()
    assert train[TARGET_TS_COL].max() < val["timestamp_utc"].min()   # purge: no train target inside validation window
    assert val[TARGET_TS_COL].max() < test["timestamp_utc"].min()
    assert len(set(map(tuple, train[["station_id", "timestamp_utc"]].values)) & set(map(tuple, test[["station_id", "timestamp_utc"]].values))) == 0


def test_split_rejects_bad_fractions(built):
    _, ds, _ = built
    with pytest.raises(ValueError):
        chronological_split(ds, 0.9, 0.2)
