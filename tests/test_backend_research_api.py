"""Backend research API tests. These use local artifacts and never require AWS."""
import json

from backend.handler import lambda_handler
from backend.settings import AppSettings

SETTINGS = AppSettings.from_env({
    "API_ALLOWED_ORIGIN": "https://example.amplifyapp.test",
    "AEROSENSE_ENV": "local",
})


def get(path, query=None):
    return lambda_handler({
        "rawPath": path,
        "requestContext": {"http": {"method": "GET"}},
        "queryStringParameters": query or {},
    }, settings=SETTINGS)


def body(response):
    return json.loads(response["body"])


def test_summary_returns_event_counts_and_no_regional_claim():
    response = get("/api/intelligence/summary")
    assert response["statusCode"] == 200
    result = body(response)
    assert result["event_count"] == 1057
    assert result["regional_event_supported_count"] == 0
    assert result["network_context_counts"] == {"LOCALIZED": 1057}


def test_events_pagination_and_filters():
    response = get("/api/events", {"severity": "HIGH", "limit": "10", "offset": "0"})
    assert response["statusCode"] == 200
    result = body(response)
    assert result["pagination"]["total"] == 533
    assert result["pagination"]["returned"] == 10
    assert len(result["items"]) == 10
    assert all(row["severity"] == "HIGH" for row in result["items"])


def test_events_support_regional_boolean_filter():
    response = get("/api/events", {"regional_event_supported": "true"})
    assert response["statusCode"] == 200
    assert body(response)["pagination"]["total"] == 0


def test_events_reject_invalid_pagination_and_boolean_filter():
    assert get("/api/events", {"limit": "oops"})["statusCode"] == 400
    assert get("/api/events", {"limit": "201"})["statusCode"] == 400
    assert get("/api/events", {"regional_event_supported": "maybe"})["statusCode"] == 400


def test_event_detail_and_missing_event():
    found = get("/api/events/AS-20260514-12234735-36")
    assert found["statusCode"] == 200
    assert body(found)["event"]["peak_pm25"] == 7080
    assert get("/api/events/not-a-real-event")["statusCode"] == 404


def test_model_status_reports_artifacts_but_not_live_prediction():
    result = body(get("/api/model/status"))
    assert result["status"] == "available"
    assert result["local_model_available"] is True
    assert result["feature_count"] == 31
    assert result["prediction_endpoint_available"] is True


def test_cloud_status_does_not_leak_bucket_name():
    settings = AppSettings.from_env({"AEROSENSE_S3_BUCKET": "sensitive-bucket-name"})
    response = lambda_handler({
        "rawPath": "/api/cloud/status",
        "requestContext": {"http": {"method": "GET"}},
    }, settings=settings)
    assert response["statusCode"] == 200
    assert "sensitive-bucket-name" not in response["body"]


def test_existing_health_contract_is_unchanged():
    result = body(get("/health"))
    assert result["status"] == "ok"
    assert result["capabilities"] == {"pm25_prediction": True, "route_comparison": False}

