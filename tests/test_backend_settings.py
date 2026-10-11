"""AppSettings: env parsing, defaults, validation. Uses explicit dicts, never the real environment."""
import pytest

from backend.settings import DEFAULT_REGION, AppSettings


def test_defaults_target_ap_south_1_and_need_no_aws_setup():
    s = AppSettings.from_env({})
    assert s.aws_region == "ap-south-1" == DEFAULT_REGION
    assert s.environment == "local" and s.uses_target_region
    assert s.s3_bucket is None and s.dynamodb_table is None


def test_aws_region_beats_aws_default_region_which_beats_default():
    assert AppSettings.from_env({"AWS_DEFAULT_REGION": "us-east-1"}).aws_region == "us-east-1"
    both = {"AWS_REGION": "eu-west-1", "AWS_DEFAULT_REGION": "us-east-1"}
    s = AppSettings.from_env(both)
    assert s.aws_region == "eu-west-1" and not s.uses_target_region


def test_blank_values_count_as_unset():
    s = AppSettings.from_env({"AEROSENSE_S3_BUCKET": "   ", "SAGEMAKER_ROLE_ARN": "", "AWS_REGION": ""})
    assert s.s3_bucket is None and s.sagemaker_role_arn is None and s.aws_region == "ap-south-1"


@pytest.mark.parametrize("env", [
    {"AEROSENSE_ENV": "staging"},
    {"AWS_REGION": "not a region"},
    {"CLOUDWATCH_LOG_RETENTION_DAYS": "0"},
])
def test_invalid_values_are_rejected(env):
    with pytest.raises(ValueError):
        AppSettings.from_env(env)


def test_s3_uri_uses_known_prefixes_only():
    s = AppSettings.from_env({"AEROSENSE_S3_BUCKET": "my-bucket"})
    assert s.s3_uri("processed", "a.csv") == "s3://my-bucket/aerosense/processed/delhi_hourly/a.csv"
    assert s.s3_uri("models") == "s3://my-bucket/aerosense/models/"
    with pytest.raises(ValueError):
        s.s3_uri("secrets")
    with pytest.raises(ValueError):
        AppSettings.from_env({}).s3_uri("reports")


def test_missing_for_reports_what_each_feature_still_needs():
    s = AppSettings.from_env({})
    assert s.missing_for("s3") == ["s3_bucket"]
    assert s.missing_for("sagemaker") == ["s3_bucket", "sagemaker_role_arn"]
    ready = AppSettings.from_env({"AEROSENSE_S3_BUCKET": "b", "SAGEMAKER_ROLE_ARN": "arn:aws:iam::1:role/r"})
    assert ready.missing_for("sagemaker") == []
    with pytest.raises(ValueError):
        s.missing_for("quantum")
