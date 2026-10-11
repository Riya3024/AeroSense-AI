"""Application and AWS settings for the backend, read from environment variables.

Design rules
  * No credentials here, ever. AWS access comes from the standard credential chain
    (an AWS CLI profile / SSO locally, the execution role inside Lambda/SageMaker).
    Do NOT add AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY handling to this module.
  * Every value is optional at import time so the app and the tests start without AWS.
    Use ``AppSettings.missing_for("s3")`` etc. to find out what a feature still needs.
  * Separate from ``data_pipeline.config.Settings`` on purpose: that one configures
    OpenAQ/Open-Meteo ingestion; this one configures the deployed application.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

DEFAULT_REGION = "ap-south-1"
_REGION_RE = re.compile(r"^[a-z]{2}(-[a-z]+)+-\d$")
# Object-key layout inside the project bucket (one bucket, one root prefix, then these folders).
#   aerosense/raw/air_quality/  aerosense/raw/weather/  aerosense/processed/delhi_hourly/  aerosense/reports/
DEFAULT_S3_ROOT = "aerosense"
S3_PREFIXES = {
    "raw_air_quality": "raw/air_quality/",
    "raw_weather": "raw/weather/",
    "processed": "processed/delhi_hourly/",
    "reports": "reports/",
    "models": "models/",
}
VALID_ENVIRONMENTS = ("local", "dev", "prod")

# What each AWS feature needs before it can be used. Used by missing_for().
_REQUIREMENTS = {
    "s3": ("s3_bucket",),
    "sagemaker": ("s3_bucket", "sagemaker_role_arn"),
    "sagemaker_inference": ("sagemaker_endpoint_name",),
    "dynamodb": ("dynamodb_table",),
}


def _get(env: Mapping[str, str], name: str, default: Optional[str] = None) -> Optional[str]:
    value = env.get(name)
    return value.strip() if value is not None and value.strip() != "" else default


@dataclass(frozen=True)
class AppSettings:
    environment: str = "local"
    aws_region: str = DEFAULT_REGION
    s3_bucket: Optional[str] = None
    s3_root: str = DEFAULT_S3_ROOT
    sagemaker_role_arn: Optional[str] = None
    sagemaker_endpoint_name: Optional[str] = None
    dynamodb_table: Optional[str] = None  # optional: only if caching/storage is actually needed
    log_retention_days: int = 14
    allowed_origin: str = "http://localhost:5173"  # CORS; set to the Amplify domain when deployed

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None) -> "AppSettings":
        env = os.environ if env is None else env
        settings = cls(
            environment=_get(env, "AEROSENSE_ENV", "local"),
            aws_region=_get(env, "AWS_REGION") or _get(env, "AWS_DEFAULT_REGION") or DEFAULT_REGION,
            s3_bucket=_get(env, "AEROSENSE_S3_BUCKET"),
            s3_root=(_get(env, "AEROSENSE_S3_PREFIX", DEFAULT_S3_ROOT) or DEFAULT_S3_ROOT).strip("/"),
            sagemaker_role_arn=_get(env, "SAGEMAKER_ROLE_ARN"),
            sagemaker_endpoint_name=_get(env, "SAGEMAKER_ENDPOINT_NAME"),
            dynamodb_table=_get(env, "DYNAMODB_TABLE_NAME"),
            log_retention_days=int(_get(env, "CLOUDWATCH_LOG_RETENTION_DAYS", "14")),
            allowed_origin=_get(env, "API_ALLOWED_ORIGIN", "http://localhost:5173"),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        if self.environment not in VALID_ENVIRONMENTS:
            raise ValueError(f"AEROSENSE_ENV must be one of {VALID_ENVIRONMENTS}, got {self.environment!r}")
        if not _REGION_RE.match(self.aws_region):
            raise ValueError(f"AWS_REGION does not look like an AWS region: {self.aws_region!r}")
        if not re.fullmatch(r"[A-Za-z0-9._\-/]+", self.s3_root):
            raise ValueError(f"AEROSENSE_S3_PREFIX has unsupported characters: {self.s3_root!r}")
        if self.log_retention_days <= 0:
            raise ValueError("CLOUDWATCH_LOG_RETENTION_DAYS must be a positive number of days")

    @property
    def uses_target_region(self) -> bool:
        """True when running in the project's target region (ap-south-1)."""
        return self.aws_region == DEFAULT_REGION

    def s3_key(self, kind: str, key: str = "") -> str:
        """Object key ``<root>/<folder><key>`` for kind in raw_air_quality / raw_weather / processed / reports / models."""
        if kind not in S3_PREFIXES:
            raise ValueError(f"unknown S3 prefix kind {kind!r}; expected one of {sorted(S3_PREFIXES)}")
        return f"{self.s3_root}/{S3_PREFIXES[kind]}{key}"

    def s3_uri(self, kind: str, key: str = "") -> str:
        """``s3://<bucket>/<root>/<folder><key>``."""
        if not self.s3_bucket:
            raise ValueError("AEROSENSE_S3_BUCKET is not set")
        return f"s3://{self.s3_bucket}/{self.s3_key(kind, key)}"

    def missing_for(self, feature: str) -> list:
        """Names of settings that ``feature`` still needs (empty list = ready)."""
        if feature not in _REQUIREMENTS:
            raise ValueError(f"unknown feature {feature!r}; expected one of {sorted(_REQUIREMENTS)}")
        return [name for name in _REQUIREMENTS[feature] if not getattr(self, name)]

