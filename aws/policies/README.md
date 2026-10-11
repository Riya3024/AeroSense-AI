# IAM policy templates (NOT applied to any account)

These are **templates** with placeholders. Nothing here has been created in AWS.

| Placeholder | Replace with |
|---|---|
| `<ACCOUNT_ID>` | your 12-digit AWS account ID |
| `<BUCKET>` | the project S3 bucket name (same value as `AEROSENSE_S3_BUCKET`); objects live under the `aerosense/` prefix |

All ARNs are pinned to region `ap-south-1`. Review each statement before attaching it, and prefer
creating roles through your own process (console, CloudFormation, or the CLI) once the Phase 2 design is fixed.

| File | Purpose |
|---|---|
| `lambda-trust-policy.json` | who may assume the Lambda execution role (`lambda.amazonaws.com`) |
| `lambda-execution-policy.json` | CloudWatch Logs, read model/processed objects in S3, invoke the AeroSense SageMaker endpoint |
| `sagemaker-trust-policy.json` | who may assume the SageMaker execution role (`sagemaker.amazonaws.com`) |
| `sagemaker-execution-policy.json` | read training data / write model artifacts in S3, CloudWatch Logs + metrics, pull container images |

No DynamoDB statements are included because DynamoDB is not required for the MVP. Add them only if a
real need appears (see `docs/architecture.md`).
