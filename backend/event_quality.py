"""Non-destructive quality checks for AeroSense event records."""
from __future__ import annotations

import math
import re
from collections import Counter
from datetime import datetime, timezone
from typing import Any

DEFAULT_OUTLIER_THRESHOLD = 1000.0
REQUIRED_FIELDS = (
    "event_id",
    "sensor_id",
    "event_type",
    "start_time",
    "peak_time",
    "end_time",
    "peak_pm25",
)
SENSOR_ID_PATTERN = re.compile(r"^[0-9]+$")


def _present(value: Any) -> bool:
    return value is not None and str(value).strip() != ""


def _timestamp(value: Any) -> datetime | None:
    if not _present(value):
        return None
    try:
        parsed = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            return None
        return parsed.astimezone(timezone.utc)
    except (TypeError, ValueError, OverflowError):
        return None


def build_event_quality_report(
    events: list[dict],
    outlier_threshold: float = DEFAULT_OUTLIER_THRESHOLD,
) -> dict:
    """Report data issues without changing, filtering, or deleting input records."""
    if not math.isfinite(outlier_threshold) or outlier_threshold < 0:
        raise ValueError("outlier_threshold must be a finite non-negative number")

    issues: list[dict] = []
    suspicious: list[dict] = []
    event_id_counts = Counter(
        str(row.get("event_id")).strip()
        for row in events
        if _present(row.get("event_id"))
    )

    def add_issue(index: int, event_id: Any, reason: str) -> None:
        issues.append({
            "row_number": index + 1,
            "event_id": event_id if _present(event_id) else None,
            "reason": reason,
        })

    for index, row in enumerate(events):
        event_id = row.get("event_id")

        for field in REQUIRED_FIELDS:
            if not _present(row.get(field)):
                add_issue(index, event_id, f"missing_required:{field}")

        if _present(event_id) and event_id_counts[str(event_id).strip()] > 1:
            add_issue(index, event_id, "duplicate_event_id")

        sensor_id = row.get("sensor_id")
        if _present(sensor_id) and not SENSOR_ID_PATTERN.fullmatch(str(sensor_id).strip()):
            add_issue(index, event_id, "invalid_sensor_id")

        parsed_times = {}
        for field in ("start_time", "peak_time", "end_time"):
            if _present(row.get(field)):
                parsed = _timestamp(row.get(field))
                if parsed is None:
                    add_issue(index, event_id, f"invalid_timestamp:{field}")
                else:
                    parsed_times[field] = parsed

        if len(parsed_times) == 3:
            if not (
                parsed_times["start_time"]
                <= parsed_times["peak_time"]
                <= parsed_times["end_time"]
            ):
                add_issue(index, event_id, "inconsistent_event_timestamps")

        raw_peak = row.get("peak_pm25")
        if _present(raw_peak):
            try:
                peak = float(raw_peak)
                if not math.isfinite(peak):
                    add_issue(index, event_id, "invalid_reading:peak_pm25_non_finite")
                elif peak < 0:
                    add_issue(index, event_id, "invalid_reading:peak_pm25_negative")
                elif peak > outlier_threshold:
                    suspicious.append({
                        "row_number": index + 1,
                        "event_id": event_id if _present(event_id) else None,
                        "field": "peak_pm25",
                        "value": peak,
                        "reason": "above_review_threshold",
                    })
            except (TypeError, ValueError, OverflowError):
                add_issue(index, event_id, "invalid_reading:peak_pm25_not_numeric")

    reason_counts = dict(sorted(Counter(item["reason"] for item in issues).items()))
    suspicious_counts = dict(sorted(Counter(
        item["reason"] for item in suspicious
    ).items()))

    rows_with_issues = len({item["row_number"] for item in issues})
    rows_suspicious = len({item["row_number"] for item in suspicious})

    return {
        "status": "issues_found" if issues else "passed",
        "source": "aerosense_events.csv",
        "records_checked": len(events),
        "records_with_quality_issues": rows_with_issues,
        "quality_issue_count": len(issues),
        "issue_counts_by_reason": reason_counts,
        "suspicious_record_count": rows_suspicious,
        "suspicious_reading_count": len(suspicious),
        "outlier_threshold_ug_m3": outlier_threshold,
        "suspicious_counts_by_reason": suspicious_counts,
        "issues": issues,
        "suspicious_readings": suspicious,
        "source_modified": False,
    }
