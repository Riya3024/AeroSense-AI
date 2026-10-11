import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw/openaq_sensors")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

frames = []

for file in RAW_DIR.glob("*.csv"):
    try:
        df = pd.read_csv(file, encoding="utf-8")

        required = {
            "station_id",
            "sensor_id",
            "pollutant",
            "parameter",
            "unit",
            "value",
            "period_from_utc",
            "period_to_utc",
            "percent_complete",
            "has_flags",
        }

        if not required.issubset(df.columns):
            print(f"SKIP invalid schema: {file.name}")
            continue

        if df.empty:
            print(f"SKIP empty: {file.name}")
            continue

        df["source_file"] = file.name
        frames.append(df)

        print(f"OK: {file.name} -> {len(df):,} rows")

    except Exception as e:
        print(f"SKIP {file.name}: {e}")

if not frames:
    raise RuntimeError("No valid OpenAQ files found.")

master = pd.concat(frames, ignore_index=True)

# Parse timestamps
master["period_from_utc"] = pd.to_datetime(
    master["period_from_utc"],
    utc=True,
    errors="coerce"
)

master["period_to_utc"] = pd.to_datetime(
    master["period_to_utc"],
    utc=True,
    errors="coerce"
)

# Numeric fields
master["value"] = pd.to_numeric(master["value"], errors="coerce")
master["percent_complete"] = pd.to_numeric(
    master["percent_complete"],
    errors="coerce"
)

# Keep PM2.5 records only
master = master[
    master["pollutant"].astype(str).str.lower().eq("pm25")
].copy()

# Remove invalid measurements
master = master.dropna(
    subset=["sensor_id", "value", "period_from_utc"]
)

# Remove duplicate observations
master = master.drop_duplicates(
    subset=["sensor_id", "period_from_utc", "value"]
)

# Sort chronologically
master = master.sort_values(
    ["sensor_id", "period_from_utc"]
).reset_index(drop=True)

# Save
output = OUT_DIR / "openaq_pm25_master.csv"
master.to_csv(output, index=False, encoding="utf-8")

print("\n========================================")
print("AEROSENSE DATASET CREATED")
print("========================================")
print(f"Rows:        {len(master):,}")
print(f"Sensors:     {master['sensor_id'].nunique():,}")
print(f"Stations:    {master['station_id'].nunique():,}")
print(f"Min date:    {master['period_from_utc'].min()}")
print(f"Max date:    {master['period_from_utc'].max()}")
print(f"PM2.5 mean:  {master['value'].mean():.2f}")
print(f"PM2.5 max:   {master['value'].max():.2f}")
print(f"Output:      {output}")
