"""Access AeroSense research artifacts without loading the large intelligence CSV."""
from __future__ import annotations

import csv
import io
import json
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Optional

from .settings import AppSettings, PROJECT_ROOT

EVENTS_FILE = PROJECT_ROOT / "data" / "processed" / "aerosense_events.csv"
MODEL_FILE = PROJECT_ROOT / "models" / "xgboost_pm25.json"
FEATURES_FILE = PROJECT_ROOT / "models" / "xgboost_features.json"

# Separate research keys: do not change the legacy S3_PREFIXES mapping.
RESEARCH_KEYS = {
    "events": "processed/events/aerosense_events.csv",
    "final_intelligence": "processed/intelligence/openaq_final_intelligence.csv",
    "model": "models/xgboost/xgboost_pm25.json",
    "features": "models/xgboost/xgboost_features.json",
}


def _s3_client(settings: AppSettings):
    """Import boto3 only when cloud access is actually needed."""
    import boto3
    return boto3.client("s3", region_name=settings.aws_region)


def _s3_key(settings: AppSettings, artifact: str) -> str:
    if artifact not in RESEARCH_KEYS:
        raise ValueError("Unknown research artifact")
    return f"{settings.s3_root}/{RESEARCH_KEYS[artifact]}"


def _typed_row(row: dict) -> dict:
    """Convert CSV scalar values to JSON-friendly values."""
    result = {}
    for key, value in row.items():
        if value is None or value.strip() == "":
            result[key] = None
            continue

        lower = value.strip().lower()
        if lower in ("true", "false"):
            result[key] = lower == "true"
            continue

        try:
            number = float(value)
            result[key] = int(number) if number.is_integer() else number
            continue
        except (ValueError, OverflowError):
            pass

        result[key] = value
    return result


@lru_cache(maxsize=4)
def _read_local_events(path_string: str) -> tuple:
    """Cache the small event CSV between requests in a warm process."""
    path = Path(path_string)
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return tuple(_typed_row(row) for row in csv.DictReader(file))


def load_events(settings: AppSettings) -> list[dict]:
    """Prefer the local research artifact; fall back to S3 when necessary."""
    if EVENTS_FILE.is_file():
        return list(_read_local_events(str(EVENTS_FILE)))

    if not settings.s3_bucket:
        raise FileNotFoundError("Event dataset is not available")

    key = _s3_key(settings, "events")
    try:
        response = _s3_client(settings).get_object(
            Bucket=settings.s3_bucket, Key=key
        )
        content = response["Body"].read().decode("utf-8-sig")
        reader = csv.DictReader(io.StringIO(content))
        return [_typed_row(row) for row in reader]
    except Exception as exc:
        # Keep AWS internals, bucket names, and credentials out of API responses.
        raise RuntimeError("Event dataset could not be loaded") from exc


def intelligence_summary(settings: AppSettings) -> dict:
    events = load_events(settings)
    if not events:
        return {
            "status": "available",
            "event_count": 0,
            "regional_event_supported_count": 0,
            "network_context_counts": {},
            "event_type_counts": {},
            "severity_counts": {},
            "date_range": {"start": None, "end": None},
            "peak_pm25_max": None,
        }

    def counts(field: str) -> dict:
        return dict(sorted(Counter(
            str(row[field]) for row in events
            if row.get(field) is not None
        ).items()))

    times = [
        str(row["start_time"]) for row in events if row.get("start_time")
    ]
    peaks = [
        float(row["peak_pm25"]) for row in events
        if row.get("peak_pm25") is not None
    ]

    return {
        "status": "available",
        "event_count": len(events),
        "regional_event_supported_count": sum(
            row.get("regional_event_supported") is True for row in events
        ),
        "network_context_counts": counts("network_context"),
        "event_type_counts": counts("event_type"),
        "severity_counts": counts("severity"),
        "date_range": {
            "start": min(times) if times else None,
            "end": max(
                (str(row["end_time"]) for row in events if row.get("end_time")),
                default=None,
            ),
        },
        "peak_pm25_max": max(peaks) if peaks else None,
        "source": "aerosense_events.csv",
    }


def find_event(settings: AppSettings, event_id: str) -> Optional[dict]:
    for row in load_events(settings):
        if row.get("event_id") == event_id:
            return row
    return None


def model_status(settings: AppSettings) -> dict:
    local_model = MODEL_FILE.is_file()
    local_features = FEATURES_FILE.is_file()
    feature_count = None

    if local_features:
        try:
            with FEATURES_FILE.open("r", encoding="utf-8") as file:
                features = json.load(file)
            if isinstance(features, list):
                feature_count = len(features)
        except (OSError, ValueError):
            pass

    cloud_model = False
    cloud_features = False
    if settings.s3_bucket:
        try:
            client = _s3_client(settings)
            for artifact, variable in (
                ("model", "model"),
                ("features", "features"),
            ):
                try:
                    client.head_object(
                        Bucket=settings.s3_bucket,
                        Key=_s3_key(settings, artifact),
                    )
                    if variable == "model":
                        cloud_model = True
                    else:
                        cloud_features = True
                except Exception:
                    # Missing or inaccessible objects are reported as unavailable.
                    pass
        except Exception:
            pass

    return {
        "status": "available" if (local_model or cloud_model) else "unavailable",
        "local_model_available": local_model,
        "local_feature_metadata_available": local_features,
        "cloud_model_available": cloud_model,
        "cloud_feature_metadata_available": cloud_features,
        "feature_count": feature_count,
        "prediction_endpoint_available": local_model and local_features,
    }


def cloud_status(settings: AppSettings) -> dict:
    if not settings.s3_bucket:
        return {
            "configured": False,
            "connectivity": "not_checked",
            "region": settings.aws_region,
            "events_artifact_available": False,
        }

    connected = False
    events_available = False
    try:
        client = _s3_client(settings)
        client.head_bucket(Bucket=settings.s3_bucket)
        connected = True
        try:
            client.head_object(
                Bucket=settings.s3_bucket,
                Key=_s3_key(settings, "events"),
            )
            events_available = True
        except Exception:
            pass
    except Exception:
        pass

    return {
        "configured": True,
        "connectivity": "verified" if connected else "unavailable",
        "region": settings.aws_region,
        "events_artifact_available": events_available,
    }
