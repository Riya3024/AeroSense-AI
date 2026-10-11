import pandas as pd
import pytest

from data_pipeline.config import TARGET_COL, TARGET_TS_COL
from data_pipeline.merge_datasets import assert_regular_grid, build_supervised
from tests.conftest import make_merged


def test_target_is_pm25_one_hour_later(settings):
    merged = make_merged(n=48)
    ds, _ = build_supervised(merged, settings)
    lookup = merged.set_index("timestamp_utc")["pm25"]
    expected = lookup.reindex(ds["timestamp_utc"] + pd.Timedelta(hours=1)).to_numpy()
    assert (ds[TARGET_COL].to_numpy() == expected).all()
    assert ((ds[TARGET_TS_COL] - ds["timestamp_utc"]) == pd.Timedelta(hours=1)).all()


def test_last_hour_of_each_station_has_no_target_and_is_dropped(settings):
    merged = make_merged(n=48)
    ds, _ = build_supervised(merged, settings)
    assert ds["timestamp_utc"].max() == merged["timestamp_utc"].max() - pd.Timedelta(hours=1)


def test_target_never_bleeds_across_stations(settings):
    merged = make_merged(n=48, stations=(1, 2))  # station 1 pm25 ~100+, station 2 ~200+
    ds, _ = build_supervised(merged, settings)
    for sid, g in ds.groupby("station_id"):
        base = 100.0 * sid
        assert g[TARGET_COL].between(base, base + 48).all()


def test_missing_hour_gives_nan_target_not_next_available_value(settings):
    merged = make_merged(n=60)
    merged.loc[40, "pm25"] = float("nan")
    ds, _ = build_supervised(merged, settings)
    row39 = merged["timestamp_utc"].iloc[39]
    assert row39 not in set(ds["timestamp_utc"])  # would have wrongly taken hour 41's value if rows were just shifted


def test_irregular_grid_is_rejected_instead_of_silently_mis_shifting(settings):
    merged = make_merged(n=48).drop(index=20).reset_index(drop=True)  # hour 20 row removed entirely
    with pytest.raises(ValueError):
        assert_regular_grid(merged)
    with pytest.raises(ValueError):
        build_supervised(merged, settings)


def test_feature_list_excludes_target_and_keys(settings):
    _, feats = build_supervised(make_merged(n=48), settings)
    assert TARGET_COL not in feats and TARGET_TS_COL not in feats
    assert "station_id" not in feats and "timestamp_utc" not in feats
    assert "pm25_t" in feats and "pm25_lag_1h" in feats and "temperature_2m" in feats
