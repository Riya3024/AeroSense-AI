"""Central configuration for the AeroSense AI data pipeline.

All tunables come from environment variables (optionally loaded from a local
``.env`` file) so runs are reproducible and secrets never live in code.
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field, fields
from datetime import date
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

# ----------------------------------------------------------------------------
# Constants that define the ML problem (not environment-tunable on purpose)
# ----------------------------------------------------------------------------
HORIZON_HOURS = 1
TARGET_COL = "target_pm25_t_plus_1h"
TARGET_TS_COL = "target_timestamp_utc"
LOCAL_TIMEZONE = "Asia/Kolkata"  # used ONLY to derive time-of-day features
LAG_HOURS = (1, 2, 3, 6, 12, 24)
EXPECTED_PM25_UNIT = "ug/m3"  # normalised spelling of micrograms per cubic metre

# --- Phase 2: pollutants, location ------------------------------------------
SUPPORTED_POLLUTANTS = ("pm25", "pm10", "no2", "o3", "co", "so2")  # OpenAQ parameter names
DEFAULT_POLLUTANT_UNIT = "ug/m3"
# Generous sensor-artefact bounds (same unit as the data) - NOT health thresholds. Values above become missing.
DEFAULT_MAX_PLAUSIBLE = {"pm10": 5000.0, "no2": 2000.0, "o3": 1000.0, "so2": 2000.0, "co": 200000.0}
# Representative point for Delhi (New Delhi city centre, approximate). Override with LOCATION_LATITUDE/LONGITUDE.
DEFAULT_LOCATION = ("Delhi", 28.6139, 77.2090)
WEATHER_MODES = ("station", "location")

DEFAULT_WEATHER_VARIABLES = (
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "surface_pressure",
    "precipitation",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
    "cloud_cover",
    "shortwave_radiation",
)
# Instantaneous, slowly varying variables that may be forward-filled over short
# gaps. Sums/means (precipitation, radiation) and circular angles are NOT filled.
WEATHER_FFILL_VARIABLES = (
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "surface_pressure",
)


def _get(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.getenv(name)
    return value.strip() if value is not None and value.strip() != "" else default


def _get_float(name: str, default: float) -> float:
    raw = _get(name)
    return float(raw) if raw is not None else default


def _get_int(name: str, default: int) -> int:
    raw = _get(name)
    return int(raw) if raw is not None else default


def _get_csv(name: str, default: tuple) -> tuple:
    raw = _get(name)
    return tuple(p.strip() for p in raw.split(",") if p.strip()) if raw else default


def _get_date(name: str, default: date) -> date:
    raw = _get(name)
    return date.fromisoformat(raw) if raw else default


def _parse_units(raw: Optional[str]) -> dict:
    """'co=mg/m3,no2=ppb' -> {'co': 'mg/m3', 'no2': 'ppb'} (names validated by the caller)."""
    out = {}
    for part in (raw or "").split(","):
        if part.strip():
            if "=" not in part:
                raise ValueError(f"POLLUTANT_UNITS entries must look like name=unit, got {part!r}")
            name, unit = part.split("=", 1)
            out[name.strip().lower()] = unit.strip()
    return out


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_") or "location"


@dataclass(frozen=True)
class Settings:
    # --- OpenAQ ---
    openaq_api_key: Optional[str] = field(default=None, repr=False)  # never printed
    openaq_base_url: str = "https://api.openaq.org/v3"
    bbox: tuple = (76.84, 28.40, 77.35, 28.88)  # min_lon, min_lat, max_lon, max_lat
    location_ids: tuple = ()
    pm25_parameter_names: tuple = ("pm25",)
    page_limit: int = 1000
    max_pages: int = 200
    chunk_days: int = 30
    min_request_interval_s: float = 1.1
    # --- shared HTTP ---
    timeout_s: float = 30.0
    max_retries: int = 5
    backoff_s: float = 2.0
    # --- dates (UTC, inclusive) ---
    start_date: date = date(2025, 1, 1)
    end_date: date = date(2026, 9, 30)
    # --- weather ---
    weather_url: str = "https://archive-api.open-meteo.com/v1/archive"
    weather_latitude: Optional[float] = None
    weather_longitude: Optional[float] = None
    weather_variables: tuple = DEFAULT_WEATHER_VARIABLES
    weather_ffill_variables: tuple = WEATHER_FFILL_VARIABLES
    weather_max_gap_hours: int = 3
    # --- cleaning ---
    pm25_max_plausible: float = 2000.0
    # --- Phase 2: pollutants / location / request budget ---
    pollutants: tuple = SUPPORTED_POLLUTANTS
    pollutant_units: dict = field(default_factory=dict)  # overrides of the expected unit per pollutant
    location_name: str = DEFAULT_LOCATION[0]
    location_latitude: float = DEFAULT_LOCATION[1]
    location_longitude: float = DEFAULT_LOCATION[2]
    weather_mode: str = "station"  # "station": one weather point per station; "location": one point (the location)
    max_requests_per_hour: int = 1800  # OpenAQ free tier allows 2000/hour; stay below it
    # --- paths ---
    root: Path = PROJECT_ROOT

    @property
    def raw_dir(self) -> Path:
        return self.root / "data" / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.root / "data" / "processed"

    @property
    def log_dir(self) -> Path:
        return self.root / "logs"

    @property
    def reports_dir(self) -> Path:
        return self.root / "reports"

    @property
    def samples_dir(self) -> Path:
        return self.root / "data" / "samples"

    @property
    def location_slug(self) -> str:
        return slugify(self.location_name)

    @property
    def run_tag(self) -> str:
        """Reproducible label for the requested window, used in output and S3 names."""
        return f"{self.start_date.isoformat()}_{self.end_date.isoformat()}"

    def expected_unit(self, pollutant: str) -> str:
        return self.pollutant_units.get(pollutant, DEFAULT_POLLUTANT_UNIT)

    def max_plausible(self, pollutant: str) -> float:
        return self.pm25_max_plausible if pollutant == "pm25" else DEFAULT_MAX_PLAUSIBLE[pollutant]

    def public_dict(self) -> dict:
        """Configuration snapshot that is safe to store in reports: never contains the API key."""
        out = {}
        for f in fields(self):
            if f.name in ("openaq_api_key", "root"):
                continue
            value = getattr(self, f.name)
            out[f.name] = value.isoformat() if isinstance(value, date) else (list(value) if isinstance(value, tuple) else value)
        out["openaq_api_key_present"] = bool(self.openaq_api_key)
        return out

    @classmethod
    def from_env(cls) -> "Settings":
        bbox = tuple(float(x) for x in _get_csv("DELHI_BBOX", cls.bbox))
        if len(bbox) != 4:
            raise ValueError("DELHI_BBOX must be min_lon,min_lat,max_lon,max_lat")
        lat, lon = _get("WEATHER_LATITUDE"), _get("WEATHER_LONGITUDE")
        if (lat is None) != (lon is None):
            raise ValueError("Set both WEATHER_LATITUDE and WEATHER_LONGITUDE, or neither")
        pollutants = tuple(dict.fromkeys(p.lower() for p in _get_csv("POLLUTANTS", SUPPORTED_POLLUTANTS)))
        unknown = [p for p in pollutants if p not in SUPPORTED_POLLUTANTS]
        if unknown:
            raise ValueError(f"POLLUTANTS contains unsupported names {unknown}; supported: {list(SUPPORTED_POLLUTANTS)}")
        if "pm25" not in pollutants:
            raise ValueError("POLLUTANTS must include pm25 (it defines the prediction target)")
        units = _parse_units(_get("POLLUTANT_UNITS"))
        bad_units = [p for p in units if p not in SUPPORTED_POLLUTANTS]
        if bad_units:
            raise ValueError(f"POLLUTANT_UNITS names unsupported pollutants {bad_units}")
        loc_lat = _get_float("LOCATION_LATITUDE", DEFAULT_LOCATION[1])
        loc_lon = _get_float("LOCATION_LONGITUDE", DEFAULT_LOCATION[2])
        if not (-90 <= loc_lat <= 90 and -180 <= loc_lon <= 180):
            raise ValueError("LOCATION_LATITUDE must be in [-90, 90] and LOCATION_LONGITUDE in [-180, 180]")
        weather_mode = (_get("WEATHER_MODE", "station") or "station").lower()
        if weather_mode not in WEATHER_MODES:
            raise ValueError(f"WEATHER_MODE must be one of {WEATHER_MODES}, got {weather_mode!r}")
        if weather_mode == "location" and not lat and not lon:
            lat, lon = str(loc_lat), str(loc_lon)  # one weather point at the representative location
        settings = cls(
            pollutants=pollutants,
            pollutant_units=units,
            location_name=_get("LOCATION_NAME", DEFAULT_LOCATION[0]),
            location_latitude=loc_lat,
            location_longitude=loc_lon,
            weather_mode=weather_mode,
            max_requests_per_hour=_get_int("OPENAQ_MAX_REQUESTS_PER_HOUR", cls.max_requests_per_hour),
            openaq_api_key=_get("OPENAQ_API_KEY"),
            openaq_base_url=_get("OPENAQ_BASE_URL", cls.openaq_base_url).rstrip("/"),
            bbox=bbox,
            location_ids=tuple(int(x) for x in _get_csv("OPENAQ_LOCATION_IDS", ())),
            page_limit=min(_get_int("OPENAQ_PAGE_LIMIT", cls.page_limit), 1000),  # API max is 1000
            chunk_days=_get_int("OPENAQ_CHUNK_DAYS", cls.chunk_days),
            min_request_interval_s=_get_float("OPENAQ_MIN_REQUEST_INTERVAL_S", cls.min_request_interval_s),
            timeout_s=_get_float("REQUEST_TIMEOUT_S", cls.timeout_s),
            max_retries=_get_int("MAX_RETRIES", cls.max_retries),
            backoff_s=_get_float("BACKOFF_S", cls.backoff_s),
            start_date=_get_date("START_DATE", cls.start_date),
            end_date=_get_date("END_DATE", cls.end_date),
            weather_url=_get("WEATHER_BASE_URL", cls.weather_url),
            weather_latitude=float(lat) if lat else None,
            weather_longitude=float(lon) if lon else None,
            weather_variables=_get_csv("WEATHER_VARIABLES", DEFAULT_WEATHER_VARIABLES),
            weather_max_gap_hours=_get_int("WEATHER_MAX_GAP_HOURS", cls.weather_max_gap_hours),
            pm25_max_plausible=_get_float("PM25_MAX_PLAUSIBLE", cls.pm25_max_plausible),
        )
        if settings.start_date > settings.end_date:
            raise ValueError("START_DATE must be <= END_DATE")
        if settings.max_requests_per_hour < 1:
            raise ValueError("OPENAQ_MAX_REQUESTS_PER_HOUR must be >= 1")
        return settings


class _RedactingFilter(logging.Filter):
    """Replace secret values in log messages so keys can never reach a log file."""

    def __init__(self, secrets):
        super().__init__()
        self._secrets = [s for s in secrets if s]

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        for secret in self._secrets:
            message = message.replace(secret, "***REDACTED***")
        record.msg, record.args = message, ()
        return True


def setup_logging(name: str, settings: Optional[Settings] = None) -> logging.Logger:
    """Console + file logging under ``logs/<name>.log`` with secret redaction."""
    settings = settings or Settings.from_env()
    settings.log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("aerosense")
    logger.setLevel(logging.INFO)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
    redactor = _RedactingFilter([settings.openaq_api_key])
    for handler in (logging.StreamHandler(), logging.FileHandler(settings.log_dir / f"{name}.log", encoding="utf-8")):
        handler.setFormatter(fmt)
        handler.addFilter(redactor)
        logger.addHandler(handler)
    return logger
