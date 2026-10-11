
"""AeroSense Lambda API handler for API Gateway v1 and v2 events."""
from __future__ import annotations

import base64
import json
from typing import Optional
from urllib.parse import parse_qs, unquote

from . import __version__
from .settings import AppSettings
from . import research_data
from .event_quality import build_event_quality_report


def _response(status: int, body: dict, settings: AppSettings) -> dict:
    return {
        "statusCode": status,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": settings.allowed_origin,
            "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type",
        },
        "body": "" if status == 204 else json.dumps(body, allow_nan=False),
    }


def _method_and_path(event: dict) -> tuple:
    http = (event.get("requestContext") or {}).get("http") or {}
    method = http.get("method") or event.get("httpMethod") or "GET"
    path = event.get("rawPath") or event.get("path") or "/"
    return method.upper(), path.split("?", 1)[0].rstrip("/") or "/"


def _query(event: dict) -> dict:
    params = event.get("queryStringParameters") or {}
    if params:
        return {str(k): str(v) for k, v in params.items() if v is not None}
    raw = event.get("rawQueryString") or ""
    return {k: values[-1] for k, values in parse_qs(raw).items() if values}


def _request_body(event: dict) -> dict:
    """Decode a JSON object from API Gateway or the local HTTP server."""
    body = event.get("body")
    if isinstance(body, dict):
        return body
    if not isinstance(body, str) or not body.strip():
        raise ValueError("A JSON request body is required")

    if event.get("isBase64Encoded"):
        try:
            body = base64.b64decode(body, validate=True).decode("utf-8")
        except (ValueError, UnicodeDecodeError):
            raise ValueError("Request body is not valid base64-encoded UTF-8") from None

    try:
        parsed = json.loads(body)
    except (json.JSONDecodeError, ValueError):
        raise ValueError("Request body must contain valid JSON") from None

    if not isinstance(parsed, dict):
        raise ValueError("Request body must be a JSON object")
    return parsed


def health_body(settings: AppSettings) -> dict:
    return {
        "status": "ok",
        "service": "aerosense-api",
        "version": __version__,
        "environment": settings.environment,
        "region": settings.aws_region,
        "capabilities": {
    "pm25_prediction": research_data.MODEL_FILE.is_file() and research_data.FEATURES_FILE.is_file(),
    "route_comparison": False,
},
    }


def _events_list(event: dict, settings: AppSettings) -> dict:
    query = _query(event)
    try:
        limit = int(query.get("limit", "50"))
        offset = int(query.get("offset", "0"))
    except ValueError:
        return {
            "statusCode": 400,
            "body": {
                "error": "invalid_pagination",
                "message": "limit and offset must be integers",
            },
        }

    if limit < 1 or limit > 200 or offset < 0:
        return {
            "statusCode": 400,
            "body": {
                "error": "invalid_pagination",
                "message": "limit must be 1–200 and offset must be non-negative",
            },
        }

    filters = {
        "event_type": "event_type",
        "severity": "severity",
        "network_context": "network_context",
        "sensor_id": "sensor_id",
        "regional_event_supported": "regional_event_supported",
    }

    try:
        rows = research_data.load_events(settings)
    except FileNotFoundError:
        return {
            "statusCode": 503,
            "body": {
                "error": "data_unavailable",
                "message": "Event data is not available",
            },
        }
    except RuntimeError:
        return {
            "statusCode": 503,
            "body": {
                "error": "data_unavailable",
                "message": "Event data could not be loaded",
            },
        }

    for param, field in filters.items():
        wanted = query.get(param)
        if wanted is None:
            continue
        if field == "regional_event_supported":
            if wanted.lower() not in ("true", "false"):
                return {
                    "statusCode": 400,
                    "body": {
                        "error": "invalid_filter",
                        "message": "regional_event_supported must be true or false",
                    },
                }
            expected = wanted.lower() == "true"
            rows = [r for r in rows if r.get(field) is expected]
        else:
            rows = [
                r for r in rows
                if str(r.get(field, "")).casefold() == wanted.casefold()
            ]

    rows.sort(key=lambda r: str(r.get("start_time") or ""), reverse=True)
    page = rows[offset:offset + limit]
    return {
        "statusCode": 200,
        "body": {
            "items": page,
            "pagination": {
                "limit": limit,
                "offset": offset,
                "total": len(rows),
                "returned": len(page),
            },
            "filters": {k: query[k] for k in filters if k in query},
            "source": "aerosense_events.csv",
        },
    }


def lambda_handler(
    event: dict,
    context: Optional[object] = None,
    settings: Optional[AppSettings] = None,
) -> dict:
    settings = settings or AppSettings.from_env()
    event = event or {}
    method, path = _method_and_path(event)

    if method == "OPTIONS":
        return _response(204, {}, settings)

    # Prediction is the only POST route currently supported.
    if method == "POST":
        if path != "/api/prediction/pm25":
            return _response(
                404,
                {"error": "not_found", "message": f"No route for {method} {path}"},
                settings,
            )

        try:
            payload = _request_body(event)
        except ValueError as exc:
            return _response(
                400,
                {"error": "invalid_request", "message": str(exc)},
                settings,
            )

        try:
            # Import lazily so existing GET endpoints remain usable if the
            # optional prediction dependency is unavailable.
            from .prediction import predict_pm25

            result = predict_pm25(payload)
            return _response(200, result, settings)
        except ValueError as exc:
            return _response(
                400,
                {"error": "invalid_features", "message": str(exc)},
                settings,
            )
        except FileNotFoundError:
            return _response(
                503,
                {
                    "error": "model_unavailable",
                    "message": "The prediction model or feature metadata is unavailable",
                },
                settings,
            )
        except Exception:
            return _response(
                503,
                {
                    "error": "prediction_unavailable",
                    "message": "The prediction could not be completed",
                },
                settings,
            )

    if method != "GET":
        return _response(
            404,
            {"error": "not_found", "message": f"No route for {method} {path}"},
            settings,
        )

    if path == "/health":
        return _response(200, health_body(settings), settings)

    try:
        if path == "/api/data-quality":
            try:
                events = research_data.load_events(settings)
            except FileNotFoundError:
                return _response(
                    503,
                    {"error": "data_unavailable", "message": "Event data is not available"},
                    settings,
                )
            except RuntimeError:
                return _response(
                    503,
                    {"error": "data_unavailable", "message": "Event data could not be loaded"},
                    settings,
                )
            return _response(200, build_event_quality_report(events), settings)

        if path == "/api/intelligence/summary":
            return _response(200, research_data.intelligence_summary(settings), settings)

        if path == "/api/events":
            result = _events_list(event, settings)
            return _response(result["statusCode"], result["body"], settings)

        if path.startswith("/api/events/"):
            event_id = unquote(path[len("/api/events/"):])
            if not event_id or "/" in event_id:
                return _response(
                    404, {"error": "not_found", "message": "Event not found"}, settings
                )
            try:
                row = research_data.find_event(settings, event_id)
            except FileNotFoundError:
                return _response(
                    503,
                    {"error": "data_unavailable", "message": "Event data is not available"},
                    settings,
                )
            except RuntimeError:
                return _response(
                    503,
                    {"error": "data_unavailable", "message": "Event data could not be loaded"},
                    settings,
                )
            if row is None:
                return _response(
                    404, {"error": "not_found", "message": "Event not found"}, settings
                )
            return _response(200, {"event": row}, settings)


        
        
        if path == "/api/prediction/latest":
            try:
                from .prediction import predict_latest_historical

                result = predict_latest_historical()
                return _response(200, result, settings)
            except FileNotFoundError as exc:
                return _response(
                    503,
                    {
                        "error": "historical_data_unavailable",
                        "message": str(exc),
                    },
                    settings,
                )
            except Exception:
                return _response(
                    503,
                    {"error": "prediction_unavailable"},
                    settings,
                )

        if path == "/api/model/status":
            return _response(
                200,
                research_data.model_status(settings),
                settings,
            )

        if path == "/api/cloud/status":
            return _response(
                200,
                research_data.cloud_status(settings),
                settings,
            )

    except Exception:
        # Do not expose stack traces, credentials, or cloud resource names.
        return _response(
            500,
            {"error": "internal_error", "message": "The request could not be completed"},
            settings,
        )

    return _response(
        404,
        {"error": "not_found", "message": f"No route for {method} {path}"},
        settings,
    )