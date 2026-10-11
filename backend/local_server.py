
"""Run the Lambda handler locally with the standard library."""
from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlsplit

from .handler import lambda_handler


class _Handler(BaseHTTPRequestHandler):
    def _serve(self):
        parsed = urlsplit(self.path)
        length = int(self.headers.get("Content-Length", "0"))

        raw_body = self.rfile.read(length) if length else b""
        try:
            body = raw_body.decode("utf-8") if raw_body else None
        except UnicodeDecodeError:
            body = None

        event = {
            "httpMethod": self.command,
            "path": parsed.path,
            "queryStringParameters": {
                key: values[-1]
                for key, values in parse_qs(parsed.query).items()
                if values
            },
            "headers": dict(self.headers.items()),
            "body": body,
            "isBase64Encoded": False,
        }

        result = lambda_handler(event)
        response_body = (
            result.get("body", "").encode("utf-8")
            if result["statusCode"] != 204
            else b""
        )

        self.send_response(result["statusCode"])
        for key, value in result["headers"].items():
            self.send_header(key, value)
        self.send_header("Content-Length", str(len(response_body)))
        self.end_headers()

        if response_body:
            self.wfile.write(response_body)

    do_GET = _serve
    do_POST = _serve
    do_OPTIONS = _serve


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    server = HTTPServer(("127.0.0.1", args.port), _Handler)
    print(f"AeroSense API running at http://127.0.0.1:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping AeroSense API.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()