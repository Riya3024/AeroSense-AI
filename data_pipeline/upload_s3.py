"""Upload the Phase 2 artefacts to Amazon S3 (region ap-south-1).

Layout (bucket from AEROSENSE_S3_BUCKET, root from AEROSENSE_S3_PREFIX, default "aerosense"):

    aerosense/processed/delhi_hourly/<start>_<end>/delhi_hourly_pm25_dataset.csv (+ .parquet, + _manifest.json)
    aerosense/reports/<start>_<end>/data_quality_report.json | .md
    aerosense/raw/air_quality/<start>_<end>/...      only with --include-raw
    aerosense/raw/weather/<start>_<end>/...          only with --include-raw

Credentials: NEVER read from the project. boto3 uses the standard chain (AWS CLI profile / SSO / environment / IAM role),
i.e. whatever `aws configure` or `aws sso login` set up. Secret values are never printed.

Safety: only files from data/processed, data/raw and reports with an allow-listed extension can be uploaded, and names that
look like secrets (.env*, credentials*, *.pem, *.key, *secret*) are refused even there.
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import logging
import mimetypes
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from backend.settings import DEFAULT_REGION, AppSettings

from .config import Settings, setup_logging
from .final_dataset import dataset_paths

log = logging.getLogger("aerosense.upload_s3")

ALLOWED_SUFFIXES = {".csv", ".parquet", ".json", ".md"}
BLOCKED_PATTERNS = (".env*", "*credential*", "*.pem", "*.key", "*secret*", "*.p12", "*.pfx", "id_rsa*", "*.tfstate")

CREDENTIAL_ERRORS = {"NoCredentialsError", "PartialCredentialsError", "CredentialRetrievalError", "TokenRetrievalError",
                     "SSOTokenLoadError", "UnauthorizedSSOTokenError", "ProfileNotFound", "NoRegionError"}


class UploadError(RuntimeError):
    """A problem the user can act on; the message says how. ``code``: 2 = configuration/credentials, 1 = anything else."""

    def __init__(self, message: str, code: int = 1):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class UploadItem:
    local: Path
    key: str


def assert_safe_to_upload(path: Path, settings: Settings) -> None:
    resolved = path.resolve()
    allowed_roots = [(settings.root / "data" / "processed"), (settings.root / "data" / "raw"), settings.reports_dir]
    if not any(root.resolve() in resolved.parents for root in allowed_roots):
        raise UploadError(f"Refusing to upload {path}: it is outside data/processed, data/raw and reports.")
    if any(fnmatch.fnmatch(path.name.lower(), pat) for pat in BLOCKED_PATTERNS):
        raise UploadError(f"Refusing to upload {path.name}: the name looks like a secret/credentials file.")
    if path.suffix.lower() not in ALLOWED_SUFFIXES:
        raise UploadError(f"Refusing to upload {path.name}: extension {path.suffix!r} is not in {sorted(ALLOWED_SUFFIXES)}.")


def plan_uploads(settings: Settings, app: AppSettings, include_raw: bool = False) -> list:
    """Pure function: which local files go to which S3 keys. Raises UploadError if the required outputs are missing."""
    tag, paths = settings.run_tag, dataset_paths(settings)
    items = []
    for name in ("csv", "parquet", "manifest"):
        if paths[name].exists():
            items.append(UploadItem(paths[name], app.s3_key("processed", f"{tag}/{paths[name].name}")))
    for fname in ("data_quality_report.json", "data_quality_report.md"):
        f = settings.reports_dir / fname
        if f.exists():
            items.append(UploadItem(f, app.s3_key("reports", f"{tag}/{fname}")))
    if include_raw:
        groups = {"raw_air_quality": ("openaq_pm25_hourly_raw.csv", "openaq_manifest.json", "stations.csv"),
                  "raw_weather": ("weather_hourly_raw.csv", "weather_manifest.json", "station_weather_points.csv")}
        for kind, names in groups.items():
            for n in names:
                f = settings.raw_dir / n
                if f.exists():
                    items.append(UploadItem(f, app.s3_key(kind, f"{tag}/{n}")))
    have = {i.local for i in items}
    for required in (paths["csv"], settings.reports_dir / "data_quality_report.json"):
        if required not in have:
            raise UploadError(f"Required file missing: {required}. Run the pipeline (fetch, merge, export, report) first.")
    for item in items:
        assert_safe_to_upload(item.local, settings)
    return items


def explain_aws_error(exc: Exception, bucket: str = "", region: str = DEFAULT_REGION) -> Optional[str]:
    """Translate common boto3/botocore failures into an actionable sentence (by class name, so botocore is not imported)."""
    name = type(exc).__name__
    if name in CREDENTIAL_ERRORS:
        return ("AWS credentials were not found or could not be loaded. Run `aws configure` (or `aws sso login`) in PowerShell, "
                "then `aws sts get-caller-identity` to confirm. Credentials are never read from the project files.")
    if name == "EndpointConnectionError" or name == "ConnectTimeoutError":
        return f"Could not reach AWS S3 in {region}. Check your internet connection/VPN/proxy."
    response = getattr(exc, "response", None)
    if isinstance(response, dict):
        code = str(response.get("Error", {}).get("Code", ""))
        if code in ("404", "NoSuchBucket", "NotFound"):
            return (f"Bucket '{bucket}' does not exist or is not visible to you. Create it with "
                    f".\\aws\\scripts\\create-s3-bucket.ps1 -BucketName {bucket} (try -WhatIf first), or fix AEROSENSE_S3_BUCKET.")
        if code in ("403", "AccessDenied", "Forbidden", "AllAccessDisabled"):
            return f"Access denied for bucket '{bucket}'. Your IAM identity needs s3:ListBucket on the bucket and s3:PutObject/GetObject on the aerosense/ prefix."
        if code in ("InvalidAccessKeyId", "SignatureDoesNotMatch", "ExpiredToken", "RequestExpired", "InvalidToken"):
            return "AWS credentials are invalid or expired. Refresh them (`aws sso login` or `aws configure`) and retry."
        if code == "PermanentRedirect" or code == "301":
            return f"Bucket '{bucket}' lives in a different region than {region}. Set AWS_REGION to the bucket's region."
    return None


def _content_type(path: Path) -> str:
    return {".csv": "text/csv", ".json": "application/json", ".md": "text/markdown"}.get(path.suffix.lower()) or mimetypes.guess_type(path.name)[0] or "application/octet-stream"


def make_client(app: AppSettings):
    try:
        import boto3  # imported lazily: unit tests and local-only runs do not need it
    except ImportError as exc:
        raise UploadError("boto3 is not installed. Run:  pip install -r requirements.txt", code=2) from exc
    return boto3.client("s3", region_name=app.aws_region)


def upload(settings: Settings, app: AppSettings, *, include_raw: bool = False, dry_run: bool = False, client=None) -> int:
    """0 = uploaded (or dry-run printed), 1 = an upload failed, 2 = configuration/credential problem."""
    try:
        if not app.s3_bucket:
            raise UploadError("AEROSENSE_S3_BUCKET is not set. Put the bucket name in .env (see .env.example).", code=2)
        if not app.uses_target_region:
            log.warning("AWS_REGION is %s but the project targets %s.", app.aws_region, DEFAULT_REGION)
        items = plan_uploads(settings, app, include_raw)
        total = sum(i.local.stat().st_size for i in items)
        log.info("Upload plan: %d files, %.1f MB -> s3://%s/", len(items), total / 1e6, app.s3_bucket)
        for i in items:
            log.info("  %-60s -> s3://%s/%s", i.local.relative_to(settings.root), app.s3_bucket, i.key)
        if dry_run:
            log.info("Dry run: nothing was uploaded.")
            return 0

        client = client or make_client(app)
        client.head_bucket(Bucket=app.s3_bucket)  # fails fast with a clear message on bad credentials / bucket / permissions
        uploaded = []
        for i in items:
            extra = {"ServerSideEncryption": "AES256", "ContentType": _content_type(i.local)}
            client.upload_file(Filename=str(i.local), Bucket=app.s3_bucket, Key=i.key, ExtraArgs=extra)
            head = client.head_object(Bucket=app.s3_bucket, Key=i.key)
            if int(head["ContentLength"]) != i.local.stat().st_size:
                raise UploadError(f"Size mismatch after uploading {i.key}: local {i.local.stat().st_size} bytes, S3 {head['ContentLength']} bytes.")
            uploaded.append({"key": i.key, "uri": f"s3://{app.s3_bucket}/{i.key}", "bytes": int(head["ContentLength"]), "etag": str(head.get("ETag", "")).strip('"')})
            log.info("Uploaded and verified s3://%s/%s (%d bytes)", app.s3_bucket, i.key, uploaded[-1]["bytes"])
        settings.reports_dir.mkdir(parents=True, exist_ok=True)
        (settings.reports_dir / "s3_upload_manifest.json").write_text(json.dumps({
            "uploaded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "bucket": app.s3_bucket,
            "region": app.aws_region, "objects": uploaded}, indent=2), encoding="utf-8")
        return 0
    except UploadError as exc:
        log.error("%s", exc)
        return exc.code
    except Exception as exc:  # noqa: BLE001 - convert AWS failures into guidance; unknown ones are still reported, briefly
        hint = explain_aws_error(exc, app.s3_bucket or "", app.aws_region)
        if hint:
            log.error("%s", hint)
            return 2
        log.error("S3 upload failed: %s: %s", type(exc).__name__, exc)
        return 1


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dry-run", action="store_true", help="print what would be uploaded; touch nothing")
    ap.add_argument("--include-raw", action="store_true", help="also upload raw air-quality and weather files (can be large)")
    args = ap.parse_args(argv)
    settings, app = Settings.from_env(), AppSettings.from_env()
    setup_logging("upload_s3", settings)
    return upload(settings, app, include_raw=args.include_raw, dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
