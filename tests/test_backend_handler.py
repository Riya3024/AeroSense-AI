"""Lambda handler: routing, CORS, honesty of the health payload, and one real localhost round-trip."""
import json
import threading
import urllib.request
from http.server import HTTPServer

from backend import __version__
from backend.handler import lambda_handler
from backend.local_server import _Handler
from backend.settings import AppSettings

SETTINGS = AppSettings.from_env({"API_ALLOWED_ORIGIN": "https://example.amplifyapp.test"})


def _v2(method, path):
    return {"rawPath": path, "requestContext": {"http": {"method": method}}}


def _v1(method, path):
    return {"httpMethod": method, "path": path}


def test_health_works_for_http_api_v2_and_rest_v1_events():
    for event in (_v2("GET", "/health"), _v1("GET", "/health"), _v1("GET", "/health/")):
        res = lambda_handler(event, None, SETTINGS)
        body = json.loads(res["body"])
        assert res["statusCode"] == 200 and body["status"] == "ok" and body["version"] == __version__
        assert body["region"] == "ap-south-1"


def test_health_does_not_claim_unbuilt_capabilities():
    body = json.loads(lambda_handler(_v2("GET", "/health"), None, SETTINGS)["body"])
    assert body["capabilities"] == {"pm25_prediction": True, "route_comparison": False}


def test_health_does_not_leak_resource_names():
    s = AppSettings.from_env({"AEROSENSE_S3_BUCKET": "secret-bucket-name",
                              "SAGEMAKER_ROLE_ARN": "arn:aws:iam::123456789012:role/x",
                              "SAGEMAKER_ENDPOINT_NAME": "my-endpoint"})
    raw = lambda_handler(_v2("GET", "/health"), None, s)["body"]
    assert "secret-bucket-name" not in raw and "123456789012" not in raw and "my-endpoint" not in raw


def test_unknown_route_and_wrong_method_are_404():
    assert lambda_handler(_v2("GET", "/recommend"), None, SETTINGS)["statusCode"] == 404
    assert lambda_handler(_v2("POST", "/health"), None, SETTINGS)["statusCode"] == 404


def test_cors_origin_comes_from_settings_and_options_is_204():
    res = lambda_handler(_v2("OPTIONS", "/health"), None, SETTINGS)
    assert res["statusCode"] == 204
    assert res["headers"]["Access-Control-Allow-Origin"] == "https://example.amplifyapp.test"


def test_local_server_serves_health_over_real_http():
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}/health"
        with urllib.request.urlopen(url, timeout=5) as resp:
            assert resp.status == 200 and json.loads(resp.read())["status"] == "ok"
    finally:
        server.shutdown()
        server.server_close()
