"""Clean, align and merge PM2.5 + weather; build the leakage-safe supervised dataset.

POLICIES (also documented in README.md)
  Timezone   All timestamps are UTC and labelled HOUR-ENDING: a row stamped T covers (T-1h, T].
             OpenAQ rows use period.datetimeTo.utc; Open-Meteo is requested with timezone=GMT and its
             "preceding hour" variables use the same convention. IST (Asia/Kolkata) is used only to
             derive hour-of-day / weekday features.
  Units      PM2.5 in ug/m3 (rows with any other unit are dropped, never converted silently).
             Weather: degC, %, hPa, mm, km/h, degrees, W/m2 (units pinned in the request).
  Resampling None. OpenAQ /hours and Open-Meteo are already hourly.
  Missing    PM2.5: never interpolated or filled. Missing hours become explicit NaN rows on a regular
             hourly grid, so shifts are exact. Rows lacking the target or core features are dropped.
             Weather: forward-fill only (past values), only for instantaneous variables, only for gaps
             <= WEATHER_MAX_GAP_HOURS. Linear interpolation is deliberately NOT used because it would
             use values from after t.
  Duplicates One row per (station, hour); keep the highest coverage, ties -> last.
  Target     PM2.5 at t+1h, same station. Features use data at or before t only.

Usage (PowerShell):  python -m data_pipeline.merge_datasets
"""
from __future__ import annotations

import json
import logging
import sys
from typing import Optional

import numpy as np
import pandas as pd

from .config import (
    EXPECTED_PM25_UNIT, HORIZON_HOURS, LAG_HOURS, LOCAL_TIMEZONE, TARGET_COL, TARGET_TS_COL,
    Settings, setup_logging,
)

log = logging.getLogger("aerosense.merge")
ONE_HOUR = pd.Timedelta(hours=1)
KEY_COLS = ["station_id", "timestamp_utc"]


# --------------------------------------------------------------------------- #
# Cleaning
# --------------------------------------------------------------------------- #
def _norm_unit(unit) -> str:
    if unit is None or (isinstance(unit, float) and np.isnan(unit)):
        return ""
    return str(unit).lower().replace("\u00b5", "u").replace("\u03bc", "u").replace("\u00b3", "3").replace(" ", "")


def remove_duplicates(df: pd.DataFrame, keys: list, order_col: str = "percent_complete") -> tuple:
    """Keep one row per key: highest ``order_col`` (coverage), ties -> last occurrence. Returns (df, n_removed)."""
    n0 = len(df)
    d = df.copy()
    if order_col in d.columns:
        d["_order"] = pd.to_numeric(d[order_col], errors="coerce").fillna(-1)
        d = d.sort_values("_order", kind="stable").drop(columns="_order")
    d = d.drop_duplicates(subset=keys, keep="last").sort_values(keys).reset_index(drop=True)
    return d, n0 - len(d)


def _norm_name(text) -> str:
    return "".join(ch for ch in str(text or "").lower() if ch.isalnum())


def _select_pollutant(raw: pd.DataFrame, pollutant: str) -> pd.DataFrame:
    """Rows of one pollutant. Uses our own ``pollutant`` column, else the API ``parameter`` name; legacy PM2.5-only files pass through."""
    for col in ("pollutant", "parameter"):
        if col in raw.columns:
            return raw[raw[col].map(_norm_name) == pollutant]
    return raw if pollutant == "pm25" else raw.iloc[0:0]


def clean_openaq(raw: pd.DataFrame, settings: Settings, pollutant: str = "pm25") -> tuple:
    """Raw OpenAQ rows of ONE pollutant -> (station_id, sensor_id, timestamp_utc, <pollutant>, percent_complete) + stats.

    Rows whose unit differs from the expected unit are dropped and counted per unit (never converted silently).
    """
    df = _select_pollutant(raw, pollutant).copy()
    stats = {"pollutant": pollutant, "raw_rows": int(len(df))}
    df["period_to_utc"] = pd.to_datetime(df["period_to_utc"], utc=True, errors="coerce")
    df["period_from_utc"] = pd.to_datetime(df["period_from_utc"], utc=True, errors="coerce")
    df["timestamp_utc"] = df["period_to_utc"].fillna(df["period_from_utc"] + ONE_HOUR)  # hour-ending label

    no_ts = df["timestamp_utc"].isna()
    stats["dropped_no_timestamp"] = int(no_ts.sum())
    df = df[~no_ts]

    off_hour = (df["timestamp_utc"].dt.minute != 0) | (df["timestamp_utc"].dt.second != 0) | (df["timestamp_utc"].dt.microsecond != 0)
    stats["dropped_not_on_the_hour"] = int(off_hour.sum())
    df = df[~off_hour]

    expected_unit = _norm_unit(settings.expected_unit(pollutant))
    unit_ok = df["unit"].map(_norm_unit) == expected_unit
    stats["expected_unit"] = expected_unit
    stats["dropped_unexpected_unit"] = int((~unit_ok).sum())
    stats["dropped_unit_counts"] = {str(k): int(v) for k, v in df.loc[~unit_ok, "unit"].map(_norm_unit).value_counts().items()}
    df = df[unit_ok]

    df = df.assign(**{pollutant: pd.to_numeric(df["value"], errors="coerce")})
    stats["null_values"] = int(df[pollutant].isna().sum())
    neg = df[pollutant] < 0
    stats["negative_values_set_missing"] = int(neg.sum())
    df.loc[neg, pollutant] = np.nan
    too_big = df[pollutant] > settings.max_plausible(pollutant)
    stats["implausible_values_set_missing"] = int(too_big.sum())
    df.loc[too_big, pollutant] = np.nan
    if "has_flags" in df.columns:
        stats["rows_with_provider_flags_kept"] = int(df["has_flags"].astype(str).str.lower().eq("true").sum())

    df = df[df[pollutant].notna()]  # missing measurements: re-created explicitly by regularize_hourly
    df, n_dup = remove_duplicates(df, KEY_COLS)
    stats["duplicate_rows_removed"] = n_dup
    stats["clean_rows"] = int(len(df))
    stats["first_timestamp_utc"] = df["timestamp_utc"].min().isoformat() if len(df) else None
    stats["last_timestamp_utc"] = df["timestamp_utc"].max().isoformat() if len(df) else None
    return df[["station_id", "sensor_id", "timestamp_utc", pollutant, "percent_complete"]], stats


def regularize_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Reindex each station to a complete hourly grid between its first and last observation (NaN = missing)."""
    frames = []
    for sid, g in df.groupby("station_id"):
        idx = pd.date_range(g["timestamp_utc"].min(), g["timestamp_utc"].max(), freq="1h", name="timestamp_utc")
        s = g.set_index("timestamp_utc")["pm25"].reindex(idx)
        frames.append(pd.DataFrame({"station_id": sid, "timestamp_utc": idx, "pm25": s.to_numpy()}))
    if not frames:
        return pd.DataFrame(columns=["station_id", "timestamp_utc", "pm25"])
    return pd.concat(frames, ignore_index=True)


def attach_pollutant(poll: pd.DataFrame, cleaned: pd.DataFrame, name: str) -> tuple:
    """Left-join one more pollutant onto the PM2.5 hourly grid (same station, same UTC hour). Never extends the grid."""
    keep = cleaned[["station_id", "timestamp_utc", name]]
    on_grid = keep.merge(poll[KEY_COLS], on=KEY_COLS, how="inner")
    merged = poll.merge(keep, on=KEY_COLS, how="left", validate="one_to_one")
    return merged, {"rows_outside_pm25_grid_ignored": int(len(keep) - len(on_grid)),
                    "grid_hours_observed": int(merged[name].notna().sum())}


def ffill_short_gaps(s: pd.Series, max_gap: int) -> pd.Series:
    """Causal forward-fill that only fills NaN runs of length <= max_gap (longer runs stay entirely NaN)."""
    isna = s.isna()
    if not isna.any():
        return s
    run_len = isna.groupby((~isna).cumsum()).transform("sum")
    filled = s.ffill()
    return filled.mask(isna & (run_len > max_gap))


def add_wind_components(df: pd.DataFrame) -> pd.DataFrame:
    """u (eastward) / v (northward) wind from speed and meteorological 'from' direction, at the same hour."""
    if {"wind_speed_10m", "wind_direction_10m"} <= set(df.columns):
        rad = np.deg2rad(df["wind_direction_10m"])
        df["wind_u_10m"] = -df["wind_speed_10m"] * np.sin(rad)
        df["wind_v_10m"] = -df["wind_speed_10m"] * np.cos(rad)
    return df


def clean_weather(raw: pd.DataFrame, settings: Settings) -> tuple:
    stats = {"raw_rows": int(len(raw))}
    df = raw.copy()
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    variables = [v for v in settings.weather_variables if v in df.columns]
    for v in variables:
        df[v] = pd.to_numeric(df[v], errors="coerce")
    grid_cols = [c for c in ("weather_grid_latitude", "weather_grid_longitude", "weather_grid_elevation") if c in df.columns]
    static = df.groupby("weather_point_id")[grid_cols].first().reset_index()

    df, n_dup = remove_duplicates(df, ["weather_point_id", "timestamp_utc"], order_col="__none__")
    stats["duplicate_rows_removed"] = n_dup

    frames, filled_counts = [], {v: 0 for v in settings.weather_ffill_variables if v in variables}
    for pid, g in df.groupby("weather_point_id"):
        idx = pd.date_range(g["timestamp_utc"].min(), g["timestamp_utc"].max(), freq="1h", name="timestamp_utc")
        g = g.set_index("timestamp_utc")[variables].reindex(idx)
        for v in filled_counts:
            was_missing = g[v].isna()
            before = int(was_missing.sum())
            g[v] = ffill_short_gaps(g[v], settings.weather_max_gap_hours)
            filled_counts[v] += before - int(g[v].isna().sum())
            g[f"{v}_imputed"] = (was_missing & g[v].notna()).astype(int)  # 1 = value was forward-filled, not observed
        g = g.reset_index()
        g.insert(0, "weather_point_id", pid)
        frames.append(g)
    out = pd.concat(frames, ignore_index=True).merge(static, on="weather_point_id", how="left")
    out = add_wind_components(out)
    stats["rows_after_regularisation"] = int(len(out))
    stats["gap_filled_values_by_variable"] = filled_counts
    stats["missing_after_cleaning"] = {v: int(out[v].isna().sum()) for v in variables}
    return out, stats


# --------------------------------------------------------------------------- #
# Merge
# --------------------------------------------------------------------------- #
def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.deg2rad, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0088 * 2 * np.arcsin(np.sqrt(a))


def merge_pollution_weather(poll: pd.DataFrame, weather: pd.DataFrame, stations: pd.DataFrame, station_points: pd.DataFrame) -> pd.DataFrame:
    """Left-join weather onto pollution by (weather point of the station's coordinates, UTC hour)."""
    m = poll.merge(stations[["station_id", "station_name", "latitude", "longitude"]], on="station_id", how="left")
    m = m.merge(station_points[["station_id", "weather_point_id"]], on="station_id", how="left")
    m = m.merge(weather, on=["weather_point_id", "timestamp_utc"], how="left", validate="many_to_one")
    if "weather_grid_latitude" in m.columns:
        m["weather_grid_distance_km"] = haversine_km(m["latitude"], m["longitude"], m["weather_grid_latitude"], m["weather_grid_longitude"])
    return m.sort_values(KEY_COLS).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Supervised dataset (target + leakage-safe features)
# --------------------------------------------------------------------------- #
def assert_regular_grid(df: pd.DataFrame) -> None:
    if df.duplicated(KEY_COLS).any():
        raise ValueError("Duplicate (station_id, timestamp_utc) rows; remove duplicates first")
    gaps = df.sort_values(KEY_COLS).groupby("station_id")["timestamp_utc"].diff().dropna()
    if (gaps != ONE_HOUR).any():
        raise ValueError("Each station must be on a regular 1-hour grid (use regularize_hourly)")


def weather_feature_columns(merged: pd.DataFrame, settings: Settings) -> list:
    cols = [v for v in settings.weather_variables if v in merged.columns]
    return cols + [c for c in ("wind_u_10m", "wind_v_10m") if c in merged.columns]


def build_supervised(merged: pd.DataFrame, settings: Settings, stats: Optional[dict] = None) -> tuple:
    """Return (dataset, feature_columns).

    LEAKAGE RULES: every feature at row t is a function of data at timestamps <= t
    (current value, positive shifts, trailing rolling windows, weather at t, calendar of t).
    The only use of t+1h data is the target column, computed with a negative shift on the
    regular grid and never reused as an input.
    """
    df = merged.sort_values(KEY_COLS).reset_index(drop=True)
    assert_regular_grid(df)
    g = df.groupby("station_id", sort=False)["pm25"]

    feats = {"pm25_t": df["pm25"]}
    for k in LAG_HOURS:
        feats[f"pm25_lag_{k}h"] = g.shift(k)
    feats["pm25_roll_mean_3h"] = g.transform(lambda s: s.rolling(3, min_periods=2).mean())
    feats["pm25_roll_mean_24h"] = g.transform(lambda s: s.rolling(24, min_periods=18).mean())
    feats["pm25_diff_1h"] = df["pm25"] - feats["pm25_lag_1h"]
    # Other pollutants: only what stations really measured (NaN elsewhere), current value and 1h lag (both <= t).
    for name in settings.pollutants:
        if name != "pm25" and name in df.columns:
            feats[name] = df[name]
            feats[f"{name}_lag_1h"] = df.groupby("station_id", sort=False)[name].shift(1)

    local = df["timestamp_utc"].dt.tz_convert(LOCAL_TIMEZONE)
    feats["hour_sin"] = np.sin(2 * np.pi * local.dt.hour / 24)
    feats["hour_cos"] = np.cos(2 * np.pi * local.dt.hour / 24)
    feats["dayofweek"] = local.dt.dayofweek
    feats["is_weekend"] = (local.dt.dayofweek >= 5).astype(int)
    feats["month"] = local.dt.month
    feats["hour"] = local.dt.hour
    feats["dayofyear"] = local.dt.dayofyear
    feats["latitude"] = df["latitude"]
    feats["longitude"] = df["longitude"]
    for col in weather_feature_columns(df, settings):
        feats[col] = df[col]

    feature_df = pd.DataFrame(feats)
    parts = [df[KEY_COLS], (df["timestamp_utc"] + HORIZON_HOURS * ONE_HOUR).rename(TARGET_TS_COL), feature_df]
    flag_cols = [c for c in df.columns if c.endswith("_imputed")]
    if flag_cols:  # metadata (not a feature): how many weather values at t were forward-filled rather than observed
        parts.append(df[flag_cols].fillna(0).sum(axis=1).astype(int).rename("weather_imputed_count"))
    parts.append(g.shift(-HORIZON_HOURS).rename(TARGET_COL))
    out = pd.concat(parts, axis=1)
    required = [TARGET_COL, "pm25_t", "pm25_lag_1h", "pm25_lag_24h"]
    if stats is not None:
        stats["rows_before_required_dropna"] = int(len(out))
        stats["required_columns"] = required
        stats["rows_missing_required_column"] = {c: int(out[c].isna().sum()) for c in required}  # overlapping counts
    out = out.dropna(subset=required).reset_index(drop=True)
    if stats is not None:
        stats["rows_after_required_dropna"] = int(len(out))
        stats["rows_removed"] = stats["rows_before_required_dropna"] - stats["rows_after_required_dropna"]
    return out, list(feature_df.columns)


def chronological_split(df: pd.DataFrame, train_frac: float = 0.70, val_frac: float = 0.15,
                        time_col: str = "timestamp_utc", target_ts_col: str = TARGET_TS_COL) -> tuple:
    """Split by calendar time (never randomly), with a purge so no train target falls in a later window."""
    if not (0 < train_frac < 1 and 0 < val_frac < 1 and train_frac + val_frac < 1):
        raise ValueError("fractions must satisfy 0 < train, val and train + val < 1")
    ts = df[time_col].drop_duplicates().sort_values().reset_index(drop=True)
    n = len(ts)
    val_start, test_start = ts.iloc[int(n * train_frac)], ts.iloc[int(n * (train_frac + val_frac))]
    train = df[(df[time_col] < val_start) & (df[target_ts_col] < val_start)]
    val = df[(df[time_col] >= val_start) & (df[target_ts_col] < test_start)]
    test = df[df[time_col] >= test_start]
    return train, val, test


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _fmt_ts(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if isinstance(out[c].dtype, pd.DatetimeTZDtype):
            out[c] = out[c].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return out


def run(settings: Settings) -> int:
    raw_dir, out_dir = settings.raw_dir, settings.processed_dir
    needed = [raw_dir / n for n in ("openaq_pm25_hourly_raw.csv", "weather_hourly_raw.csv", "stations.csv", "station_weather_points.csv")]
    missing = [p.name for p in needed if not p.exists()]
    if missing:
        log.error("Missing raw inputs %s. Run fetch_openaq and fetch_weather first.", missing)
        return 1
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_aq, raw_wx, stations, points = (pd.read_csv(p) for p in needed)

    clean_aq, aq_stats = clean_openaq(raw_aq, settings, "pm25")
    if clean_aq.empty:
        log.error("No usable PM2.5 rows after cleaning: %s", aq_stats)
        return 1
    poll = regularize_hourly(clean_aq)
    aq_stats["grid_rows_incl_missing_hours"] = int(len(poll))
    aq_stats["missing_hours_on_grid"] = int(poll["pm25"].isna().sum())
    other_stats = {}
    for name in settings.pollutants:
        if name == "pm25":
            continue
        cleaned, s = clean_openaq(raw_aq, settings, name)
        if cleaned.empty:
            s["available"] = False
            log.warning("Pollutant %s: no usable rows (raw=%d, wrong-unit dropped=%d %s). Column omitted - nothing invented.",
                        name, s["raw_rows"], s["dropped_unexpected_unit"], s["dropped_unit_counts"])
        else:
            poll, extra = attach_pollutant(poll, cleaned, name)
            s.update(extra, available=True)
        other_stats[name] = s
    weather, wx_stats = clean_weather(raw_wx, settings)

    merged = merge_pollution_weather(poll, weather, stations, points)
    wvars = [v for v in settings.weather_variables if v in merged.columns]
    aq_stats["rows_without_weather_match"] = int(merged[wvars].isna().all(axis=1).sum()) if wvars else None

    sup_stats = {}
    dataset, feature_cols = build_supervised(merged, settings, sup_stats)
    log.info("merged=%d rows, supervised=%d rows, features=%d", len(merged), len(dataset), len(feature_cols))
    if dataset.empty:
        log.error("Supervised dataset is empty (too few consecutive hours?).")
        return 1

    _fmt_ts(merged).to_csv(out_dir / "pollution_weather_hourly.csv", index=False)
    _fmt_ts(dataset).to_csv(out_dir / "supervised_pm25_t_plus_1h.csv", index=False)
    (out_dir / "feature_columns.json").write_text(json.dumps({"features": feature_cols, "target": TARGET_COL}, indent=2), encoding="utf-8")
    (out_dir / "merge_stats.json").write_text(json.dumps({"openaq": aq_stats, "openaq_other_pollutants": other_stats, "weather": wx_stats,
                                                                       "merged_grid_rows": int(len(merged)), "supervised": sup_stats}, indent=2, default=str), encoding="utf-8")
    log.info("Wrote processed datasets to %s", out_dir)
    return 0


def main(argv: Optional[list] = None) -> int:
    settings = Settings.from_env()
    setup_logging("merge_datasets", settings)
    return run(settings)


if __name__ == "__main__":
    sys.exit(main())
