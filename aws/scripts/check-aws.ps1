<#
.SYNOPSIS
  Read-only AWS readiness check for AeroSense AI. Creates nothing.
.DESCRIPTION
  Verifies that the AWS CLI is installed, that credentials resolve (via your profile / SSO / environment),
  and that the configured region is the project target (ap-south-1).
  Usage (from the project root):  .\aws\scripts\check-aws.ps1
#>
# "Continue" on purpose: Windows PowerShell 5.1 turns native-command stderr into terminating errors under "Stop".
# Exit codes are checked explicitly below.
$ErrorActionPreference = "Continue"
$TargetRegion = "ap-south-1"

if (-not (Get-Command aws -ErrorAction SilentlyContinue)) {
    Write-Host "AWS CLI not found. Install AWS CLI v2: https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html" -ForegroundColor Red
    exit 1
}
Write-Host ("AWS CLI: " + (aws --version 2>&1)) -ForegroundColor Cyan

$region = $env:AWS_REGION
if (-not $region) { $region = $env:AWS_DEFAULT_REGION }
if (-not $region) { $region = (aws configure get region 2>$null) }
if (-not $region) {
    Write-Host "No region configured. Set one for this session:  `$env:AWS_REGION = '$TargetRegion'" -ForegroundColor Yellow
} elseif ($region -ne $TargetRegion) {
    Write-Host "Region is '$region' but the project targets '$TargetRegion'." -ForegroundColor Yellow
} else {
    Write-Host "Region OK: $region" -ForegroundColor Green
}

# Credentials are resolved by the AWS CLI itself; this script never reads or stores keys.
$identity = aws sts get-caller-identity --output json 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "Could not resolve AWS credentials. Run 'aws configure' or 'aws sso login' (never paste keys into project files)." -ForegroundColor Red
    Write-Host $identity
    exit 1
}
$id = $identity | ConvertFrom-Json
Write-Host "Authenticated as: $($id.Arn)" -ForegroundColor Green
Write-Host "Account: $($id.Account)"
Write-Host "Nothing was created or modified." -ForegroundColor Cyan
