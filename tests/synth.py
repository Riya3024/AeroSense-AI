"""SYNTHETIC raw inputs for Phase 2 unit tests only. Nothing here is, or may be presented as, real data.

Layout mirrors what fetch_openaq / fetch_weather write, so merge_datasets.run can be exercised end to end:
  * pm25 and pm10 in ug/m3 for two stations (station 2 has one missing hour, one exact duplicate row exists)
  * no2 only at station 1; co reported in a DIFFERENT unit (must be dropped+counted); o3 and so2 absent
  * weather with a short 2h gap (forward-fillable) and a long 6h gap (must stay missing) at point P1
"""
import numpy as np
import pandas as pd

T0 = pd.Timestamp("2025-11-01", tz="UTC")
UG = "\u00b5g/m\u00b3"


def _stamp(ts):
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ")


def write_synthetic_raw(s, n=72, seed=1):
    s.raw_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows = []

    def add(sid, sensor, pollutant, unit, h, value):
        to = T0 + pd.Timedelta(hours=h)
        rows.append({"station_id": sid, "sensor_id": sensor, "pollutant": pollutant, "parameter": pollutant, "unit": unit,
                     "value": value, "period_from_utc": _stamp(to - pd.Timedelta(hours=1)), "period_to_utc": _stamp(to),
                     "percent_complete": 100.0, "has_flags": False})

    for sid in (1, 2):
        for h in range(1, n + 1):
            if sid == 2 and h == 30:
                continue
            add(sid, 10 + sid, "pm25", UG, h, 80 + 20 * sid + 10 * np.sin(h / 5) + rng.normal(0, 2))
            add(sid, 20 + sid, "pm10", UG, h, 150 + 10 * sid + rng.normal(0, 3))
    for h in range(1, n + 1):
        add(1, 31, "no2", UG, h, 40 + rng.normal(0, 1))
        add(1, 41, "co", "ppm", h, 1.2)  # wrong unit on purpose
    rows.append(rows[5].copy())  # exact duplicate
    pd.DataFrame(rows).to_csv(s.raw_dir / "openaq_pm25_hourly_raw.csv", index=False)
    pd.DataFrame({"station_id": [1, 2], "station_name": ["Synthetic A", "Synthetic B"],
                  "latitude": [28.61, 28.70], "longitude": [77.20, 77.10]}).to_csv(s.raw_dir / "stations.csv", index=False)
    pd.DataFrame({"station_id": [1, 2], "weather_point_id": ["P1", "P2"], "requested_latitude": [28.61, 28.70],
                  "requested_longitude": [77.20, 77.10]}).to_csv(s.raw_dir / "station_weather_points.csv", index=False)
    wx = []
    for pid in ("P1", "P2"):
        for h in range(0, n + 3):
            t = T0 + pd.Timedelta(hours=h)
            temp = 20.0 + h % 12
            if pid == "P1" and h in (30, 31):  # short gap: eligible for forward fill
                temp = np.nan
            if pid == "P1" and 40 <= h <= 45:  # long gap: must NOT be filled
                temp = np.nan
            wx.append({"weather_point_id": pid, "timestamp_utc": _stamp(t), "temperature_2m": temp, "relative_humidity_2m": 60.0,
                       "wind_speed_10m": 8.0, "wind_direction_10m": 270.0, "precipitation": 0.0,
                       "weather_grid_latitude": 28.5, "weather_grid_longitude": 77.25, "weather_grid_elevation": 215.0})
    pd.DataFrame(wx).to_csv(s.raw_dir / "weather_hourly_raw.csv", index=False)


def write_synthetic_manifests(s):
    """Manifests shaped like the real fetchers' output (SYNTHETIC values, for report-plumbing tests only)."""
    import json
    raw = pd.read_csv(s.raw_dir / "openaq_pm25_hourly_raw.csv")
    wx = pd.read_csv(s.raw_dir / "weather_hourly_raw.csv")
    (s.raw_dir / "openaq_manifest.json").write_text(json.dumps({
        "run_at_utc": "2025-11-05T00:00:00+00:00", "endpoint": "synthetic://openaq/sensors/{id}/hours",
        "pollutants_requested": list(s.pollutants), "sensors_by_pollutant": {"pm25": 2, "pm10": 2, "no2": 1, "co": 1},
        "requests_made_this_run": 0, "stations_discovered": 2, "raw_rows": int(len(raw)),
        "raw_rows_by_pollutant": {str(k): int(v) for k, v in raw["pollutant"].value_counts().items()},
        "partial": False, "station_status": {"1": "ok", "2": "ok"}}), encoding="utf-8")
    (s.raw_dir / "weather_manifest.json").write_text(json.dumps({
        "run_at_utc": "2025-11-05T00:00:00+00:00", "endpoint": "synthetic://open-meteo/archive", "start_date": str(s.start_date),
        "end_date": str(s.end_date), "effective_end_date": str(s.end_date), "weather_mode": s.weather_mode,
        "variables": list(s.weather_variables), "units": {}, "timezone": "GMT (UTC)", "points_requested": 2,
        "raw_rows": int(len(wx)), "partial": False, "point_status": {}}), encoding="utf-8")
