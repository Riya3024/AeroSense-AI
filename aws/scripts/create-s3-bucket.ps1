<#
.SYNOPSIS
  Creates the single AeroSense S3 bucket in ap-south-1 with safe defaults.
.DESCRIPTION
  Creates the bucket, blocks all public access, enables default AES256 encryption and versioning.
  Supports -WhatIf so you can preview without creating anything.
  Prefixes used later: raw/  processed/  models/
.EXAMPLE
  .\aws\scripts\create-s3-bucket.ps1 -BucketName aerosense-ai-<your-suffix> -WhatIf
  .\aws\scripts\create-s3-bucket.ps1 -BucketName aerosense-ai-<your-suffix>
#>
[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [Parameter(Mandatory = $true)][string]$BucketName,
    [string]$Region = "ap-south-1"
)
$ErrorActionPreference = "Stop"

if ($BucketName -notmatch '^[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]$') {
    throw "Invalid bucket name '$BucketName'. Use 3-63 chars: lowercase letters, digits, dots, hyphens."
}
if (-not (Get-Command aws -ErrorAction SilentlyContinue)) { throw "AWS CLI not found. See aws\scripts\check-aws.ps1" }

function Invoke-Aws {
    param([string[]]$AwsArgs)
    & aws @AwsArgs
    if ($LASTEXITCODE -ne 0) { throw "aws $($AwsArgs -join ' ') failed with exit code $LASTEXITCODE" }
}

if ($PSCmdlet.ShouldProcess($BucketName, "Create S3 bucket in $Region")) {
    Invoke-Aws @("s3api", "create-bucket", "--bucket", $BucketName, "--region", $Region,
                 "--create-bucket-configuration", "LocationConstraint=$Region")
    Invoke-Aws @("s3api", "put-public-access-block", "--bucket", $BucketName, "--public-access-block-configuration",
                 "BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true")

    $tmp = Join-Path $env:TEMP "aerosense-sse.json"
    Set-Content -Path $tmp -Encoding ASCII -Value '{"Rules":[{"ApplyServerSideEncryptionByDefault":{"SSEAlgorithm":"AES256"}}]}'
    try {
        Invoke-Aws @("s3api", "put-bucket-encryption", "--bucket", $BucketName, "--server-side-encryption-configuration", "file://$tmp")
    } finally {
        Remove-Item $tmp -ErrorAction SilentlyContinue
    }
    Invoke-Aws @("s3api", "put-bucket-versioning", "--bucket", $BucketName, "--versioning-configuration", "Status=Enabled")

    Write-Host "Bucket '$BucketName' is ready in $Region." -ForegroundColor Green
    Write-Host "Add to your .env:  AEROSENSE_S3_BUCKET=$BucketName   (objects will go under aerosense/)"
}
