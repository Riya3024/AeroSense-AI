"""S3 upload behaviour with a FAKE client (no AWS, no boto3, no credentials). One opt-in integration test at the end."""
import json
import os
import sys
from dataclasses import replace
from datetime import date
from unittest import mock

import pytest

from backend.settings import AppSettings
from data_pipeline import final_dataset, merge_datasets, quality_report, upload_s3, validate_data
from data_pipeline.upload_s3 import UploadError, UploadItem, assert_safe_to_upload, explain_aws_error, plan_uploads
from tests.synth import write_synthetic_manifests, write_synthetic_raw

APP = AppSettings.from_env({"AEROSENSE_S3_BUCKET": "test-bucket"})
TAG = "2025-11-01_2025-11-04"


class NoCredentialsError(Exception):  # same class name boto/botocore uses
    pass


class ClientError(Exception):
    def __init__(self, code):
        super().__init__(code)
        self.response = {"Error": {"Code": code}}


class FakeS3:
    def __init__(self, head_error=None, size_delta=0):
        self.head_error, self.size_delta, self.uploads = head_error, size_delta, {}

    def head_bucket(self, Bucket):
        if self.head_error:
            raise self.head_error
        return {}

    def upload_file(self, Filename, Bucket, Key, ExtraArgs=None):
        self.uploads[Key] = (Filename, Bucket, ExtraArgs or {})

    def head_object(self, Bucket, Key):
        return {"ContentLength": os.path.getsize(self.uploads[Key][0]) + self.size_delta, "ETag": '"abc"'}


@pytest.fixture
def ready(settings):
    s = replace(settings, start_date=date(2025, 11, 1), end_date=date(2025, 11, 4))
    write_synthetic_raw(s)
    write_synthetic_manifests(s)
    assert merge_datasets.run(s) == 0 and validate_data.run(s) == 0 and final_dataset.run(s) == 0 and quality_report.run(s) == 0
    return s


def test_upload_plan_follows_the_documented_s3_layout(ready):
    keys = {i.key for i in plan_uploads(ready, APP)}
    assert f"aerosense/processed/delhi_hourly/{TAG}/delhi_hourly_pm25_dataset.csv" in keys
    assert f"aerosense/processed/delhi_hourly/{TAG}/delhi_hourly_pm25_dataset_manifest.json" in keys
    assert f"aerosense/reports/{TAG}/data_quality_report.json" in keys and f"aerosense/reports/{TAG}/data_quality_report.md" in keys
    assert not any("/raw/" in k for k in keys)  # raw data is opt-in


def test_raw_files_are_uploaded_only_on_request_into_their_own_folders(ready):
    keys = {i.key for i in plan_uploads(ready, APP, include_raw=True)}
    assert f"aerosense/raw/air_quality/{TAG}/openaq_pm25_hourly_raw.csv" in keys
    assert f"aerosense/raw/weather/{TAG}/weather_hourly_raw.csv" in keys


def test_custom_root_prefix_is_honoured(ready):
    app = AppSettings.from_env({"AEROSENSE_S3_BUCKET": "b", "AEROSENSE_S3_PREFIX": "/team/aero/"})
    assert all(i.key.startswith("team/aero/") for i in plan_uploads(ready, app))


def test_plan_fails_clearly_when_pipeline_outputs_are_missing(settings):
    with pytest.raises(UploadError) as exc:
        plan_uploads(settings, APP)
    assert "Run the pipeline" in str(exc.value)


def test_secret_looking_or_out_of_scope_files_are_refused(ready):
    d = ready.processed_dir
    for name in (".env", "credentials.json", "my.pem", "aws_secret_notes.csv"):
        f = d / name
        f.write_text("x")
        with pytest.raises(UploadError):
            assert_safe_to_upload(f, ready)
    outside = ready.root / "notes.csv"
    outside.write_text("x")
    with pytest.raises(UploadError):
        assert_safe_to_upload(outside, ready)
    exe = d / "tool.exe"
    exe.write_text("x")
    with pytest.raises(UploadError):
        assert_safe_to_upload(exe, ready)


def test_successful_upload_uses_encryption_verifies_sizes_and_records_a_manifest(ready):
    fake = FakeS3()
    assert upload_s3.upload(ready, APP, client=fake) == 0
    assert len(fake.uploads) == len(plan_uploads(ready, APP))
    assert all(extra["ServerSideEncryption"] == "AES256" and bucket == "test-bucket" for _, bucket, extra in fake.uploads.values())
    assert fake.uploads[f"aerosense/processed/delhi_hourly/{TAG}/delhi_hourly_pm25_dataset.csv"][2]["ContentType"] == "text/csv"
    manifest = json.loads((ready.reports_dir / "s3_upload_manifest.json").read_text())
    assert manifest["bucket"] == "test-bucket" and manifest["region"] == "ap-south-1" and len(manifest["objects"]) == len(fake.uploads)
    assert ready.openaq_api_key not in json.dumps(manifest)
    assert not any(".env" in k or "credential" in k for k in fake.uploads)


def test_dry_run_uploads_nothing_and_needs_no_credentials_or_boto3(ready):
    with mock.patch.dict(sys.modules, {"boto3": None}):  # importing boto3 would raise; dry run must not try
        assert upload_s3.upload(ready, APP, dry_run=True) == 0
    assert not (ready.reports_dir / "s3_upload_manifest.json").exists()


def test_missing_bucket_setting_gives_exit_2_and_a_clear_message(ready, caplog):
    import logging
    with caplog.at_level(logging.ERROR, logger="aerosense.upload_s3"):
        assert upload_s3.upload(ready, AppSettings.from_env({}), client=FakeS3()) == 2
    assert "AEROSENSE_S3_BUCKET" in caplog.text


def test_missing_credentials_is_reported_with_guidance_not_a_traceback(ready, caplog):
    import logging
    with caplog.at_level(logging.ERROR, logger="aerosense.upload_s3"):
        assert upload_s3.upload(ready, APP, client=FakeS3(head_error=NoCredentialsError("Unable to locate credentials"))) == 2
    assert "aws configure" in caplog.text and "Traceback" not in caplog.text


@pytest.mark.parametrize("code,needle", [("404", "does not exist"), ("403", "Access denied"), ("ExpiredToken", "expired"), ("PermanentRedirect", "different region")])
def test_common_aws_errors_are_translated(code, needle):
    assert needle in explain_aws_error(ClientError(code), "test-bucket", "ap-south-1")


def test_unrecognised_errors_are_not_swallowed_into_a_fake_hint():
    assert explain_aws_error(RuntimeError("weird")) is None


def test_boto3_not_installed_is_a_configuration_error(ready, caplog):
    import logging
    with mock.patch.dict(sys.modules, {"boto3": None}), caplog.at_level(logging.ERROR, logger="aerosense.upload_s3"):
        assert upload_s3.upload(ready, APP) == 2
    assert "pip install" in caplog.text


def test_size_mismatch_after_upload_is_a_failure(ready):
    assert upload_s3.upload(ready, APP, client=FakeS3(size_delta=1)) == 1
    assert not (ready.reports_dir / "s3_upload_manifest.json").exists()  # no manifest claiming success


def test_wrong_region_only_warns(ready, caplog):
    import logging
    app = AppSettings.from_env({"AEROSENSE_S3_BUCKET": "b", "AWS_REGION": "eu-west-1"})
    with caplog.at_level(logging.WARNING, logger="aerosense.upload_s3"):
        assert upload_s3.upload(ready, app, client=FakeS3()) == 0
    assert "ap-south-1" in caplog.text


@pytest.mark.integration
@pytest.mark.skipif(os.environ.get("RUN_AWS_INTEGRATION") != "1", reason="opt-in: set RUN_AWS_INTEGRATION=1 with real AWS credentials and AEROSENSE_S3_BUCKET")
def test_integration_real_bucket_is_reachable():
    """Read-only: head_bucket against the configured bucket. Needs real credentials; skipped by default."""
    app = AppSettings.from_env()
    assert app.s3_bucket, "AEROSENSE_S3_BUCKET must be set"
    upload_s3.make_client(app).head_bucket(Bucket=app.s3_bucket)
