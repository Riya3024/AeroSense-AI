"""Shared HTTP helper: timeouts, retries with backoff, 429 handling."""
from __future__ import annotations

import logging
import time
from typing import Callable, Optional

import requests

log = logging.getLogger("aerosense.http")


class FetchError(RuntimeError):
    """A request failed for good (after retries, or a non-retryable status)."""

    def __init__(self, message: str, status: Optional[int] = None):
        super().__init__(message)
        self.status = status


class AuthError(FetchError):
    """401/403 or a missing credential. Retrying cannot help."""


def _retry_after_seconds(resp, default: float = 60.0) -> float:
    """Seconds to wait after a 429. Uses Retry-After or x-ratelimit-reset (OpenAQ)."""
    for header in ("Retry-After", "x-ratelimit-reset"):
        raw = resp.headers.get(header)
        try:
            if raw is not None:
                return min(max(float(raw), 1.0), 120.0)
        except (TypeError, ValueError):
            continue
    return default


def request_json(
    session,
    url: str,
    params: Optional[dict] = None,
    *,
    timeout: float,
    max_retries: int,
    backoff_s: float,
    sleep: Callable[[float], None] = time.sleep,
    before_request: Optional[Callable[[], None]] = None,
) -> dict:
    """GET ``url`` and return decoded JSON, or raise FetchError/AuthError.

    Retried: timeouts, connection errors, 408, 429, 5xx, undecodable bodies.
    Not retried: 401/403 (AuthError), other 4xx (FetchError with the server body).
    Headers are never logged (they carry the API key).
    """
    last_error = "no attempt made"
    for attempt in range(1, max_retries + 1):
        if before_request:
            before_request()
        wait = backoff_s * (2 ** (attempt - 1))
        try:
            resp = session.get(url, params=params, timeout=timeout)
        except (requests.Timeout, requests.ConnectionError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            log.warning("Attempt %d/%d failed (%s); retrying in %.1fs", attempt, max_retries, last_error, wait)
            sleep(wait)
            continue

        status = resp.status_code
        if status == 200:
            try:
                return resp.json()
            except ValueError:
                last_error = "HTTP 200 but body is not valid JSON"
                log.warning("Attempt %d/%d: %s", attempt, max_retries, last_error)
                sleep(wait)
                continue

        body = (resp.text or "")[:500]
        if status in (401, 403):
            raise AuthError(f"HTTP {status} from {url}: {body}", status)
        if status == 429:
            wait = _retry_after_seconds(resp)
            last_error = f"HTTP 429 rate limited: {body}"
            log.warning("Rate limited (429); sleeping %.0fs (attempt %d/%d)", wait, attempt, max_retries)
            sleep(wait)
            continue
        if status == 408 or status >= 500:
            last_error = f"HTTP {status}: {body}"
            log.warning("Attempt %d/%d: %s; retrying in %.1fs", attempt, max_retries, last_error, wait)
            sleep(wait)
            continue
        raise FetchError(f"HTTP {status} from {url} params={params}: {body}", status)

    raise FetchError(f"Gave up on {url} after {max_retries} attempts. Last error: {last_error}")
