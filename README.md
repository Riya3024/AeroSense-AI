# AeroSense AI

Environmental intelligence for the **Air track** of the Environmental Hacks / Bharat Builds Tour.
Given a *from* location, *to* location, *departure time* and *travel mode*, AeroSense is meant to predict
near-term PM2.5, compare route alternatives, estimate exposure per route, recommend the lower-exposure
option and explain why.

## Status - read this first

| Area | State |
|---|---|
| Phase 2 data pipeline (OpenAQ + Open-Meteo -> hourly ML dataset -> quality report -> S3), multi-pollutant | **Implemented and unit-tested on synthetic data. NOT yet executed against the live APIs or a real S3 bucket - run `.\scripts\run_phase2.ps1` on your machine; no real-data numbers exist in this repository.** |
| Backend skeleton (`backend/`: env-driven settings, Lambda handler, local dev server) | Built. Exposes `GET /health` only. |
| AWS preparation (env vars, IAM policy templates, PowerShell scripts) | Prepared. **Nothing is deployed and no AWS resource has been created.** |
| PM2.5 model | **Not built. No metrics exist.** |
| Route engine / exposure estimation | **Not built.** |
| Frontend | **Not built.** |

Details and open risks: [`docs/architecture.md`](docs/architecture.md).

## Architecture (summary)

```
OpenAQ v3 --+                                 +--> S3 raw/ , processed/ , models/
            +--> data_pipeline --> datasets --+--> SageMaker training (Phase 3)
Open-Meteo -+      (local today)                                  |
                                                                  v
Amplify (frontend) --> API Gateway --> Lambda (backend/handler.py) --> model --> recommendation
                                           +--> CloudWatch
```

Region: `ap-south-1`. DynamoDB is deliberately **not** part of the MVP. Full explanation, service roles, security
model and known gaps are in `docs/architecture.md`.

## Repository layout

```
data_pipeline/   fetch_openaq, fetch_weather, merge_datasets, validate_data, config, http_utils
backend/         settings.py, handler.py (Lambda entry point), local_server.py
ml/              Phase 3 contract (README only, no code yet)
aws/policies/    IAM policy templates (placeholders; not applied)
aws/scripts/     check-aws.ps1 (read-only), create-s3-bucket.ps1 (supports -WhatIf)
scripts/         run_pipeline.ps1, setup_dev.ps1, run_tests.ps1, run_backend_local.ps1
tests/           pipeline tests + backend / repository-config tests
docs/            architecture.md
data/ logs/      generated, git-ignored (placeholders are kept)
```

## Setup (Windows PowerShell)

```powershell
cd C:\path\to\aerosense-ai
.\scripts\setup_dev.ps1            # creates .venv, installs requirements.txt, copies .env.example to .env
.\.venv\Scripts\Activate.ps1       # if blocked: Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
notepad .env                       # add OPENAQ_API_KEY (free key: https://explore.openaq.org/register)
```

## Environment variables

Copy `.env.example` to `.env` (git-ignored). **Never commit keys and never put AWS access keys in any project
file** - use `aws configure` / `aws sso login` locally and IAM roles in AWS.

| Variable | Used by | Purpose |
|---|---|---|
| `OPENAQ_API_KEY` | data pipeline | **Required** for ingestion. Sent as `X-API-Key`; redacted from logs. |
| `DELHI_BBOX`, `OPENAQ_LOCATION_IDS`, `START_DATE`, `END_DATE`, ... | data pipeline | Station selection, date range, request limits, cleaning (see `.env.example`). |
| `AEROSENSE_ENV` | backend | `local` (default), `dev` or `prod`. |
| `AWS_REGION` | backend / scripts | Defaults to `ap-south-1`. |
| `AEROSENSE_S3_BUCKET` | backend / scripts | Project bucket (prefixes `raw/`, `processed/`, `models/`). |
| `SAGEMAKER_ROLE_ARN` | Phase 3 | SageMaker execution role. |
| `SAGEMAKER_ENDPOINT_NAME` | later | Inference endpoint, if one is used. |
| `AEROSENSE_S3_PREFIX` | pipeline / backend | Root folder in the bucket (default `aerosense`). |
| `POLLUTANTS`, `POLLUTANT_UNITS`, `LOCATION_*`, `WEATHER_MODE`, `OPENAQ_MAX_REQUESTS_PER_HOUR` | Phase 2 | See "Phase 2" below. |
| `DYNAMODB_TABLE_NAME` | optional | Only if DynamoDB is adopted. |
| `CLOUDWATCH_LOG_RETENTION_DAYS` | backend | Log retention (default 14). |
| `API_ALLOWED_ORIGIN` | backend | CORS origin; set to the Amplify domain after the frontend exists. |

## Local development

```powershell
.\scripts\run_pipeline.ps1           # full data pipeline (needs OPENAQ_API_KEY and internet)
.\scripts\run_backend_local.ps1      # API on http://127.0.0.1:8000
Invoke-RestMethod http://127.0.0.1:8000/health
```

`/health` reports whether local PM2.5 model artifacts are present; this alone does not guarantee a successful prediction. `route_comparison` remains false until implemented. The current forecast uses historical engineered features, not live sensor readings. Cloud connectivity must not be claimed until AWS connectivity is configured and verified.

## Testing

```powershell
.\scripts\run_tests.ps1              # or: python -m pytest -q
```

The suite covers the data pipeline (alignment, duplicates, missing values, target creation, leakage, HTTP
behaviour), the backend settings and handler (including one real localhost request), and repository hygiene
(no committed credentials, `.gitignore` coverage, `.env.example` completeness, IAM templates pinned to
`ap-south-1`, ASCII-only PowerShell). All fixtures are synthetic; passing tests say nothing about live data quality.

## AWS deployment roadmap (nothing below has been done yet)

1. **Check access (read-only):** `.\aws\scripts\check-aws.ps1`
2. **Create the bucket:** preview with `.\aws\scripts\create-s3-bucket.ps1 -BucketName <unique-name> -WhatIf`, then run
   it without `-WhatIf` and set `AEROSENSE_S3_BUCKET`. It blocks public access, enables encryption and versioning.
3. **Create IAM roles** from `aws/policies/*.json` after replacing `<ACCOUNT_ID>` and `<BUCKET>` and reviewing them.
4. **Phase 2 (implemented, awaiting your first real run):** `.\scripts\run_phase2.ps1` -> real data, hourly dataset, quality report, S3.
5. **Phase 3:** train a persistence baseline and the first PM2.5 model on the Phase 2 dataset (SageMaker or local); report only real metrics.
6. **Later phases:** live data path + route engine + Lambda/API Gateway; frontend on Amplify; CloudWatch alarms and a cost review;
   DynamoDB only if a concrete need appears.

---

## Phase 2 - real Delhi/NCR data -> hourly ML dataset -> S3

**Purpose:** produce the real, validated, leakage-safe hourly dataset that **Phase 3 will use to train the PM2.5 prediction
model**. Phase 2 does not train anything, build routes, or score exposure.

**Honest status:** the code and its tests are done (synthetic data only). The first real run has to happen on a machine with
internet access, an OpenAQ key and AWS credentials. Until you have run it, this repository contains **no real measurements,
no real data-quality statistics and no S3 objects**. The generated `reports\data_quality_report.md` is the only source of
numbers about the data.

### 1. Data sources

| Source | Used for | Notes |
|---|---|---|
| **OpenAQ v3** (`/locations`, `/sensors/{id}/hours`) | PM2.5 (required), PM10, NO2, O3, CO, SO2 hourly averages | Needs a free API key. OpenAQ aggregates data from providers (e.g. government monitors); check and cite each provider's terms when you publish. |
| **Open-Meteo Historical Weather API** | temperature, relative humidity, wind speed/direction, precipitation (+ dew point, pressure, gusts, cloud cover, shortwave radiation) | Reanalysis-based archive; roughly 5 days behind real time. Units: degC, %, km/h, degrees, mm, UTC. |

Pollutants a station does not measure are simply absent for that station (NaN) - nothing is invented. A pollutant whose
unit differs from the expected one is dropped and **counted by unit in the report**, never silently converted.

### 2. Delhi location configuration

`.env` (defaults shown). Stations are discovered inside `DELHI_BBOX`; `LOCATION_*` is the documented representative point
(New Delhi city centre, approximate) used for reporting and, if `WEATHER_MODE=location`, as the single weather point.

```
DELHI_BBOX=76.84,28.40,77.35,28.88      # min_lon,min_lat,max_lon,max_lat - widen it for more of NCR
LOCATION_NAME=Delhi
LOCATION_LATITUDE=28.6139
LOCATION_LONGITUDE=77.2090
WEATHER_MODE=station                    # station = one weather point per station; location = one point for all
OPENAQ_LOCATION_IDS=                    # optional: pin exact OpenAQ location IDs instead of the bbox
```

### 3. Date-range configuration

```
START_DATE=2025-01-01
END_DATE=2026-09-30
```

Nothing in the code is tied to these dates (a test enforces it); override per run with `--start/--end` on the fetchers. If a source
cannot supply the whole range (stations that started later, weather lagging ~5 days, future `END_DATE`), the pipeline keeps
what exists, marks the rest as missing, and the report records **requested vs. actual** ranges plus a warning.

### 4. Pipeline execution

```
OpenAQ -> raw -> clean/validate -> hourly grid ----+
                                                   +-> align -> missing-data policy -> features -> target -> dataset -> report -> S3
Open-Meteo -> raw -> clean/validate -> hourly -----+
```

| Step | Module | Output |
|---|---|---|
| 1 Validate environment | `data_pipeline.preflight` | PASS/WARN/FAIL list (no secrets printed) |
| 2 Fetch | `fetch_openaq`, `fetch_weather` | `data\raw\*` + manifests (cached per sensor; resumable) |
| 3 Process | `merge_datasets` | `data\processed\pollution_weather_hourly.csv`, `supervised_pm25_t_plus_1h.csv`, `merge_stats.json` |
| 4 Validate | `validate_data` | structural + leakage checks |
| 5 Export | `final_dataset` | `data\processed\delhi_hourly_pm25_dataset.csv` (+ `.parquet` if `pyarrow`), `*_manifest.json`, `data\samples\*` |
| 6 Report | `quality_report` | `reports\data_quality_report.json` / `.md` |
| 7 Test | pytest | `reports\pytest_last_run.txt` |
| 8 Upload | `upload_s3` | S3 objects + `reports\s3_upload_manifest.json` |

**Request budget.** OpenAQ's free tier allows 2000 requests/hour. A full 21-month, six-pollutant run needs on the order of
`sensors x 22` requests and **can take hours**; the client throttles itself to `OPENAQ_MAX_REQUESTS_PER_HOUR` (default 1800) and
resumes from cache if interrupted. Check the cost first with `.\scripts\run_phase2.ps1 -EstimateOnly`, or start with
`-Pollutants pm25` for a fast first run.

### 5. Dataset schema (`delhi_hourly_pm25_dataset.csv`)

One row = one station-hour. `*_manifest.json` lists every column with dtype, role and unit.

| Group | Columns |
|---|---|
| Keys / metadata (not features) | `station_id`, `timestamp` (UTC, hour-ending), `target_timestamp` (= `timestamp` + 1 h), `weather_imputed_count` |
| Calendar (local IST, derived from `timestamp`) | `hour`, `day_of_week` (Mon=0), `day_of_year`, `month`, `hour_sin`, `hour_cos`, `is_weekend` |
| Air quality at t (ug/m3) | `pm25`, and `pm10`, `no2`, `o3`, `co`, `so2` **only if** real data survived cleaning |
| Weather at t | `temperature`, `relative_humidity`, `wind_speed`, `wind_direction`, `precipitation`, `wind_u`, `wind_v`, plus the extra variables configured in `WEATHER_VARIABLES` |
| History (all <= t) | `pm25_lag_{1,2,3,6,12,24}h`, `pm25_roll_mean_{3,24}h`, `pm25_diff_1h`, `<pollutant>_lag_1h` |
| Station | `latitude`, `longitude` |
| **Target** | `target_pm25_t_plus_1h` |

`wind_u/wind_v` come from the existing wind-vector code (meteorological "from" direction: u = -speed*sin(dir), v = -speed*cos(dir)).
`timestamp` is **hour-ending**: a row stamped T summarises (T-1h, T]; Open-Meteo precipitation uses the same convention.

### 6. Target definition

`target_pm25_t_plus_1h` = the PM2.5 value (ug/m3) **at the same station exactly one hour after `timestamp`**, created with a
per-station shift on the complete hourly grid. A missing hour produces a missing target and the row is dropped - a neighbouring
value is never substituted.

### 7. Missing-data policy

* **Pollutants are never imputed.** A missing hour stays NaN; rows needing it (target, `pm25`, 1 h / 24 h lags) are removed and counted.
* **Weather** variables that vary smoothly (temperature, humidity, dew point, pressure) are carried **forward** from the last
  observed hour only when the whole gap is at most `WEATHER_MAX_GAP_HOURS` (default 3). Longer gaps stay **entirely** missing.
  Every filled value is flagged (`*_imputed` on the weather grid; `weather_imputed_count` per dataset row).
* `pm25_roll_mean_3h` / `_24h` average the **observed** values in a trailing window and are missing unless at least 2 of 3 / 18 of 24 hours were observed (no values are created).
* Values below 0 or above a generous plausibility bound become missing (counted), duplicates are removed (counted), off-the-hour
  and un-timestamped rows are dropped (counted). **Every raw row is accounted for** in the report (`unaccounted` must be 0).
* The report distinguishes observed / forward-filled / unavailable values.

### 8. Leakage prevention

* Target is a forward shift; every feature is computed from rows at or before `t` (lags use backward shifts; rolling windows end at `t`).
* Weather is joined on the exact same `(station, hour)`; no forward-looking weather columns exist, and filling is causal.
* Name guard rejects features that look forward-looking; a numeric guard rejects any feature identical or perfectly correlated with the target.
* Lag columns are verified to equal the value observed k hours earlier; the target is verified to equal `pm25` one hour later.
* The **exported file is read back and re-verified** (`leakage_guard_on_exported_file` in the manifest and report).
* The train/validation/test split helper splits by calendar time with a purge so no training target falls in a later window.
* Unit tests rewrite all future values and assert earlier feature rows do not change (for PM2.5, weather, PM10 and NO2).

### 9. Data-quality report

`reports\data_quality_report.json` (machine) and `.md` (human). Contains: source APIs, location, requested vs. actual date range,
raw and final record counts, per-pollutant row accounting, rows removed and why, missing-value statistics, imputation counts,
duplicate statistics, timestamp/gap statistics per station, feature list, target definition and availability, all validation and
leakage check results, warnings/errors, data-collection timestamps, config snapshot (no secrets) and input checksums.
Every figure is computed from the run's artefacts; if an artefact is missing the report says so and fails.

### 10. S3 structure (region `ap-south-1`)

```
s3://<bucket>/aerosense/
    processed/delhi_hourly/<START>_<END>/delhi_hourly_pm25_dataset.csv (+ .parquet, _manifest.json)
    reports/<START>_<END>/data_quality_report.json | .md
    raw/air_quality/<START>_<END>/...      (only with -IncludeRaw)
    raw/weather/<START>_<END>/...          (only with -IncludeRaw)
```

Objects are uploaded with server-side encryption (AES256) and their size is verified after upload. Only files from
`data\processed`, `data\raw` and `reports` with extension csv/parquet/json/md are eligible; `.env`, credentials, keys and
anything secret-looking are refused.

### 11. AWS configuration

```powershell
aws configure                       # or: aws sso login   (credentials stay in your AWS CLI config, never in the project)
.\aws\scripts\check-aws.ps1         # read-only check
.\aws\scripts\create-s3-bucket.ps1 -BucketName aerosense-ai-<your-suffix> -WhatIf    # preview, then run without -WhatIf
# then in .env:   AEROSENSE_S3_BUCKET=aerosense-ai-<your-suffix>   AWS_REGION=ap-south-1
```

Needed IAM permissions for the uploader: `s3:ListBucket` on the bucket and `s3:PutObject` / `s3:GetObject` on `aerosense/*`.

### 12. PowerShell commands

```powershell
.\scripts\setup_dev.ps1                          # once: venv + requirements + .env
.\scripts\run_phase2.ps1 -EstimateOnly           # how many OpenAQ requests / how long?
.\scripts\run_phase2.ps1 -Pollutants pm25 -SkipUpload     # fast local first run
.\scripts\run_phase2.ps1                         # everything, including tests and the S3 upload
.\scripts\run_phase2.ps1 -SkipFetch -DryRunUpload          # re-process existing raw data; show the upload plan only
.\scripts\run_phase2.ps1 -IncludeRaw             # also upload raw files
```

The script is non-destructive (nothing is deleted), stops at the first failing step, runs the tests **before** uploading, and
uses `--strict` so a partial fetch stops the run (re-run to retry only the failed sensors; finished ones are cached).

### 13. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `OPENAQ_API_KEY is not set` / HTTP 401 | Create a key at explore.openaq.org/register, put it in `.env`. |
| HTTP 429 or the run pauses | Rate limit; the client waits automatically. Lower `OPENAQ_MAX_REQUESTS_PER_HOUR` or use `-Pollutants pm25`. |
| Fetch exits with code 3 | Partial fetch; re-run the same command - cached sensors are skipped. |
| A pollutant is missing from the dataset | See `warnings` in the report; usually a unit mismatch - set `POLLUTANT_UNITS` (e.g. `co=mg/m3`) after confirming the unit; nothing is converted automatically. |
| Dataset ends days before `END_DATE` | Weather reanalysis lags ~5 days; recent hours are NaN and those rows are excluded. |
| `Parquet skipped` | `pip install pyarrow`; the CSV is always produced. |
| `Unable to locate credentials` | Run `aws configure` or `aws sso login`; confirm with `aws sts get-caller-identity`. |
| `Bucket ... does not exist` / Access denied | Create it with `create-s3-bucket.ps1`; check the IAM permissions above and `AWS_REGION=ap-south-1`. |
| PowerShell blocks scripts | `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` |
| Opt-in AWS integration test | `$env:RUN_AWS_INTEGRATION="1"; python -m pytest -m integration` (read-only `head_bucket`). |

### What Phase 2 reused and changed

Reused unchanged: HTTP/retry client, pagination, caching, station discovery, hourly regularisation, duplicate handling, wind
vectors, lag/rolling features, target shift, `chronological_split`, `validate_data` checks, logging with key redaction, all 49
original tests. Changed: `config` (pollutants, location, request budget; default window), `fetch_openaq` (one sensor per
pollutant, hourly budget, estimate, `--strict`), `fetch_weather` (future-date clamp, `--strict`), `merge_datasets` (per-pollutant
cleaning, imputation flags, extra features, row accounting), `validate_data` (lag check for all pollutants), `backend.settings`
(S3 root prefix). New: `final_dataset`, `quality_report`, `upload_s3`, `preflight`, `scripts\run_phase2.ps1`.

---

# Data pipeline (existing, preserved)

Builds a reproducible, leakage-safe dataset for predicting **PM2.5 one hour ahead** at Delhi
monitoring stations (and, later, estimating exposure along routes).

```
OpenAQ v3 (PM2.5, hourly) --+
                            +--> merge_datasets --> supervised_pm25_t_plus_1h.csv --> validate_data (report + leakage guard)
Open-Meteo archive (hourly)-+
```

### Pipeline status (as written by the original author)

| Item | State |
|---|---|
| 49 unit tests (alignment, duplicates, missing values, target, leakage, HTTP behaviour) | **Passed** in the authoring sandbox (Python 3.12, pandas 3.0.2). Test data is synthetic. |
| Leakage tests can actually fail | **Verified** by mutation: a 2h target shift, a future-valued lag, and a centred rolling window each made tests fail. |
| Failure paths (no key; network unreachable) | **Run for real**: exit code 2 with instructions; exit 1 with the actual connection error; no data written. |
| **Live OpenAQ / Open-Meteo pulls** | **NOT executed.** The authoring sandbox cannot resolve `api.openaq.org` (DNS error, logged verbatim). No real Delhi data has been fetched or validated. |
| Model | **Not built.** This phase is data only. |

Treat the first live run on your machine as the real integration test (see "First-run checklist").

### API facts this code relies on (from the official docs)

* **OpenAQ v3**, base `https://api.openaq.org/v3`. Auth: `X-API-Key` header. Free key: register at
  <https://explore.openaq.org/register>, copy it from <https://explore.openaq.org/account>.
  Limits: 60 requests/min, 2,000/hour; 429 on excess; repeated abuse can get you banned.
* `GET /locations?bbox=min_lon,min_lat,max_lon,max_lat&limit&page` for discovery;
  `GET /locations/{id}`; `GET /sensors/{id}/hours?datetime_from&datetime_to&limit&page` for hourly means.
  `limit` max 1000. `meta.found` may be a string such as `">1000"`, so pagination stops on a short page.
* **Open-Meteo Historical** `https://archive-api.open-meteo.com/v1/archive` with `latitude, longitude,
  start_date, end_date, hourly, timezone`. No key for non-commercial use. ERA5 lags real time ~5 days.
  Non-commercial terms apply; attribute Open-Meteo / ERA5 in anything you publish. OpenAQ requires source attribution too.

**Doc inconsistency handled:** OpenAQ's Pagination page names the date params `datetimeFrom/To`; the API reference says
`datetime_from/datetime_to`. We use the API reference. The docs also don't say whether the filter matches period start or end,
so chunk boundaries may return a boundary hour twice; duplicates are removed in the merge.

### Policies

**Timezone.** Everything is UTC, **hour-ending**: a row stamped `T` describes `(T-1h, T]`. OpenAQ rows use
`period.datetimeTo.utc`; Open-Meteo is requested with `timezone=GMT` (response with a non-zero offset is rejected) and its
"preceding hour" variables follow the same convention. IST (`Asia/Kolkata`) is used **only** to derive hour-of-day/weekday features.

**Units.** PM2.5 in ug/m3 (any other unit is dropped, never converted silently). Weather units are pinned in the request:
degC, %, hPa, mm, km/h, degrees, W/m2.

**Resampling.** None; both sources are already hourly.

**Missing data.**
* PM2.5 is never interpolated or filled. Null/negative/implausible (> `PM25_MAX_PLAUSIBLE`) values become missing. Missing hours
  become explicit NaN rows on a regular hourly grid (so shifts are exact). Rows lacking the target, `pm25_t`, `pm25_lag_1h` or
  `pm25_lag_24h` are dropped; other NaNs are kept and reported.
* Weather: causal forward-fill only, only for instantaneous variables (`temperature_2m, relative_humidity_2m, dew_point_2m,
  surface_pressure`), only for gaps <= `WEATHER_MAX_GAP_HOURS`. Linear interpolation is deliberately avoided because it would
  use values from after `t`. Sums (precipitation, radiation) and angles are never filled.
* Provider-flagged PM2.5 rows are kept and counted in `merge_stats.json`.

**Duplicates.** One row per (station, hour); keep the highest `coverage.percentComplete`, ties -> last.

**Coordinates.** Each station is matched to the weather grid via its own coordinates (rounded to 0.01 deg, ~1 km, and de-duplicated),
or to a single point if `WEATHER_LATITUDE/LONGITUDE` are set. Open-Meteo returns the grid-cell centre actually used; the
station-to-cell distance is stored as `weather_grid_distance_km`. ERA5 cells are ~25 km, so nearby stations share weather.

**Target and leakage.** `target_pm25_t_plus_1h` = PM2.5 at `t+1h`, same station.
Every feature at `t` uses only data at or before `t` (current value, positive lags 1/2/3/6/12/24h, trailing rolling means, weather
at `t`, calendar of `t`, static coordinates). Weather at `t+1h` is deliberately excluded: at prediction time you would only have a
*forecast*, and this dataset holds reanalysis. If you add forecast features later, use Open-Meteo's Historical *Forecast* API, not archive data.
Enforced by `validate_data.assert_no_target_leakage` and tests that rewrite all future values and require identical features.

**Splitting.** Use `merge_datasets.chronological_split(df)`: split by calendar time, with a purge so no training row's target
falls inside the validation window. Never split randomly.

### Windows PowerShell commands

```powershell
# 1) Create and activate the environment (Python 3.10+; tested on 3.12)
cd C:\path\to\aerosense-ai
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
# If activation is blocked:  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# 2) Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 3) Configure environment variables (pick ONE)
Copy-Item .env.example .env ; notepad .env              # put your key after OPENAQ_API_KEY=
#   or, for this terminal session only:
$env:OPENAQ_API_KEY = "paste-your-key-here"

# 4) Run the pipeline
python -m data_pipeline.fetch_openaq --list-stations    # discover stations; writes data\raw\stations.csv
python -m data_pipeline.fetch_openaq                    # PM2.5 hourly -> data\raw
python -m data_pipeline.fetch_weather                   # weather      -> data\raw
python -m data_pipeline.merge_datasets                  # -> data\processed
python -m data_pipeline.validate_data                   # quality report + leakage guard
#   or all four with stop-on-error:
.\scripts\run_pipeline.ps1

# 5) Run the tests
python -m pytest -q
```

Useful options: `--start 2025-11-01 --end 2026-01-31`, `--location-ids 123,456`, `--refresh` (ignore cache).
Re-running resumes: finished sensors/points are cached under `data\raw\openaq_sensors` and `data\raw\weather_points`.
Budget: roughly (stations x ceil(days/30) x pages) OpenAQ requests at ~1.1 s each; stay under 2,000/hour.

### Outputs

| Path | Content |
|---|---|
| `data/raw/stations.csv`, `openaq_pm25_hourly_raw.csv`, `openaq_manifest.json` | stations, raw API rows, per-station status/errors |
| `data/raw/weather_hourly_raw.csv`, `weather_manifest.json`, `station_weather_points.csv` | raw weather, units, station-to-point map |
| `data/processed/pollution_weather_hourly.csv` | cleaned, merged hourly grid (NaN = missing) |
| `data/processed/supervised_pm25_t_plus_1h.csv`, `feature_columns.json` | model-ready rows + the exact feature list |
| `data/processed/data_quality_report.{md,json}`, `merge_stats.json` | row counts, missingness, coverage, cleaning stats, checks |
| `logs/*.log` | run logs (API key is redacted) |

### Failure behaviour (no invented data)

* No key -> exit 2 with how to get one. 401/403 -> exit 2 with the server message.
* 429 -> waits per `x-ratelimit-reset`/`Retry-After`; timeouts, 5xx, 408 -> exponential backoff; other 4xx -> real server message, no retry.
* One failing station does not abort the run; it is recorded in `openaq_manifest.json` (`partial: true`) and retried on rerun.
* Discovery failure -> reuses a previously saved `stations.csv` if present; **never** substitutes made-up measurements.
* Zero rows retrieved -> exit 1.

### First-run checklist (things I could not verify offline)

1. Open `data/raw/openaq_pm25_hourly_raw.csv`: confirm `period_to_utc` is on the hour and `unit` is ug/m3 (rows with another unit are dropped and counted).
2. Parameter name is assumed to be `pm25` (override via `pm25_parameter_names` in `config.py` if `--list-stations` finds nothing).
3. `DELHI_BBOX` is an approximate extent and can include border-city stations; pin `OPENAQ_LOCATION_IDS` for a curated set.
4. Check `data_quality_report.md` for per-station coverage; drop stations with little data before modelling.
