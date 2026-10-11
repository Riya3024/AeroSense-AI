from copy import deepcopy

from backend.event_quality import build_event_quality_report


def valid_event(**overrides):
    event = {
        "event_id": "AS-TEST-001",
        "sensor_id": "12234735",
        "event_type": "SENSOR_ANOMALY",
        "start_time": "2026-05-14T15:30:00+00:00",
        "peak_time": "2026-05-14T16:30:00+00:00",
        "end_time": "2026-05-14T17:30:00+00:00",
        "peak_pm25": 120.0,
        "post_peak_pm25": None,  # Optional field may be blank.
    }
    event.update(overrides)
    return event


def test_valid_event_passes_and_optional_blank_is_allowed():
    report = build_event_quality_report([valid_event()])
    assert report["status"] == "passed"
    assert report["quality_issue_count"] == 0
    assert report["suspicious_reading_count"] == 0


def test_missing_required_field_is_reported():
    event = valid_event()
    event["event_type"] = ""
    report = build_event_quality_report([event])
    assert "missing_required:event_type" in report["issue_counts_by_reason"]


def test_duplicate_event_ids_are_reported_for_both_records():
    report = build_event_quality_report([
        valid_event(),
        valid_event(sensor_id="14258988"),
    ])
    assert report["issue_counts_by_reason"]["duplicate_event_id"] == 2


def test_invalid_timestamp_sensor_id_and_negative_reading_are_reported():
    report = build_event_quality_report([valid_event(
        sensor_id="sensor-X",
        start_time="not-a-timestamp",
        peak_pm25=-5,
    )])
    reasons = report["issue_counts_by_reason"]
    assert "invalid_sensor_id" in reasons
    assert "invalid_timestamp:start_time" in reasons
    assert "invalid_reading:peak_pm25_negative" in reasons


def test_extreme_reading_is_suspicious_not_a_quality_failure():
    event = valid_event(peak_pm25=7080.0)
    original = deepcopy(event)
    report = build_event_quality_report([event])
    assert report["quality_issue_count"] == 0
    assert report["suspicious_reading_count"] == 1
    assert report["suspicious_readings"][0]["value"] == 7080.0
    assert event == original
    assert report["source_modified"] is False


def test_invalid_timestamp_order_is_reported():
    report = build_event_quality_report([valid_event(
        start_time="2026-05-14T18:30:00+00:00",
        peak_time="2026-05-14T16:30:00+00:00",
    )])
    assert "inconsistent_event_timestamps" in report["issue_counts_by_reason"]
