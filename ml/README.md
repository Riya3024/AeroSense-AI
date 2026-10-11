# ml/ - model code (Phase 3, not started)

No model has been trained and nothing in this folder is executable yet. This file records the
contract Phase 2 must respect so the work builds on the existing, already-tested data pipeline.

**Input:** `data/processed/delhi_hourly_pm25_dataset.csv` (or the copy in S3 under `aerosense/processed/delhi_hourly/<START>_<END>/`) and the
`feature_columns` / `schema` in `delhi_hourly_pm25_dataset_manifest.json`, produced by `.\scripts\run_phase2.ps1`.
Read `reports/data_quality_report.md` first: it states the real date range, missingness and which pollutants exist.

**Target:** `target_pm25_t_plus_1h` (PM2.5 one hour ahead, same station).

**Rules that must not be broken**
* Split with `data_pipeline.merge_datasets.chronological_split` (calendar time with a purge; it expects the internal column names, so map them back with `data_pipeline.final_dataset.to_internal`). Never split randomly.
* Use only the columns listed as features in the manifest (never `station_id`, `timestamp`, `target_timestamp`, `weather_imputed_count`). Adding forecast-based weather features requires
  Open-Meteo's Historical *Forecast* API, not the reanalysis archive (see the main README, "Target and leakage").
* Always report a naive persistence baseline (predict `pm25_t`) next to any model, on the same test window.
* Report only metrics that were actually computed on real data.
