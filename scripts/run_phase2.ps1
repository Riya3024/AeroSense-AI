<#
.SYNOPSIS
  Phase 2 end to end: real Delhi data (OpenAQ + Open-Meteo) -> hourly ML dataset -> quality report -> S3.

.DESCRIPTION
  Steps (stops at the first failure):
    1 Validate environment        2 Fetch real data          3 Process (clean, align, features, target)
    4 Validate dataset            5 Export final dataset     6 Quality report
    7 Run tests                   8 Upload to S3
  Non-destructive: nothing is deleted. Outputs under data\processed, data\samples and reports\ are overwritten by a re-run.
  Fetched sensor files are cached in data\raw\openaq_cache so an interrupted run resumes instead of starting over.
  Credentials are never read from the project: the S3 step uses your AWS CLI profile / SSO (aws configure / aws sso login).

.PARAMETER SkipFetch      Reuse the raw files already in data\raw.
.PARAMETER EstimateOnly   Discover stations and print the OpenAQ request/time estimate, then stop.
.PARAMETER Refresh        Ignore cached raw files and download again.
.PARAMETER Pollutants     Override POLLUTANTS for this run only, e.g. pm25 or pm25,pm10 (a fast first run).
.PARAMETER SkipTests      Skip the pytest step.
.PARAMETER SkipUpload     Process locally; do not touch S3.
.PARAMETER DryRunUpload   Print what would be uploaded; upload nothing.
.PARAMETER IncludeRaw     Also upload raw air-quality and weather files (can be large).

.EXAMPLE
  .\scripts\run_phase2.ps1 -EstimateOnly
  .\scripts\run_phase2.ps1 -Pollutants pm25 -SkipUpload
  .\scripts\run_phase2.ps1
#>
[CmdletBinding()]
param(
    [switch]$SkipFetch, [switch]$EstimateOnly, [switch]$Refresh, [string]$Pollutants,
    [switch]$SkipTests, [switch]$SkipUpload, [switch]$DryRunUpload, [switch]$IncludeRaw
)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
$py = if (Test-Path ".\.venv\Scripts\python.exe") { ".\.venv\Scripts\python.exe" } else { "python" }
$total = 8
$script:n = 0

function Start-Step([string]$title) {
    $script:n++
    Write-Host ""
    Write-Host ("[{0}/{1}] {2}" -f $script:n, $total, $title) -ForegroundColor Cyan
}
function Invoke-Py([string[]]$PyArgs) {
    & $py @PyArgs
    if ($LASTEXITCODE -ne 0) { throw ("'python {0}' failed with exit code {1}. See logs\ for details." -f ($PyArgs -join " "), $LASTEXITCODE) }
}

if ($Pollutants) { $env:POLLUTANTS = $Pollutants; Write-Host "POLLUTANTS overridden for this run: $Pollutants" -ForegroundColor Yellow }

Start-Step "Validate environment"
$pre = @("-m", "data_pipeline.preflight")
if ($SkipFetch) { $pre += "--skip-fetch" }
if ($SkipUpload -or $DryRunUpload -or $EstimateOnly) { $pre += "--skip-upload" }
Invoke-Py $pre

if (-not $SkipFetch) {
    Start-Step "Fetch real air-quality data (OpenAQ) - can take a long time; progress is logged and cached"
    $aq = @("-m", "data_pipeline.fetch_openaq", "--strict")
    if ($EstimateOnly) { $aq += "--estimate" }
    if ($Refresh) { $aq += "--refresh" }
    Invoke-Py $aq
    if ($EstimateOnly) {
        Write-Host "Estimate only: stopping before any download." -ForegroundColor Yellow
        return
    }
    Write-Host "Fetch real weather data (Open-Meteo)" -ForegroundColor Cyan
    $wx = @("-m", "data_pipeline.fetch_weather", "--strict")
    if ($Refresh) { $wx += "--refresh" }
    Invoke-Py $wx
} else {
    Start-Step "Fetch skipped (-SkipFetch): using existing files in data\raw"
}

Start-Step "Process: clean, align hourly, handle missing values, engineer features, create target"
Invoke-Py @("-m", "data_pipeline.merge_datasets")

Start-Step "Validate dataset (structure, duplicates, ordering, leakage guard)"
Invoke-Py @("-m", "data_pipeline.validate_data")

Start-Step "Export final ML dataset (CSV, Parquet if available) and re-verify the exported file"
Invoke-Py @("-m", "data_pipeline.final_dataset")

Start-Step "Generate data-quality report (reports\data_quality_report.json / .md)"
Invoke-Py @("-m", "data_pipeline.quality_report")

if (-not $SkipTests) {
    Start-Step "Run the test suite"
    New-Item -ItemType Directory -Force -Path "reports" | Out-Null
    & $py -m pytest -q | Tee-Object -FilePath "reports\pytest_last_run.txt"
    if ($LASTEXITCODE -ne 0) { throw "Tests failed. Nothing was uploaded. See reports\pytest_last_run.txt" }
} else {
    Start-Step "Tests skipped (-SkipTests)"
}

if (-not $SkipUpload) {
    Start-Step "Upload to S3 (ap-south-1)"
    $up = @("-m", "data_pipeline.upload_s3")
    if ($DryRunUpload) { $up += "--dry-run" }
    if ($IncludeRaw) { $up += "--include-raw" }
    Invoke-Py $up
} else {
    Start-Step "Upload skipped (-SkipUpload)"
}

Write-Host ""
Write-Host "Phase 2 workflow finished." -ForegroundColor Green
Write-Host "  Dataset : data\processed\*_hourly_pm25_dataset.csv"
Write-Host "  Report  : reports\data_quality_report.md"
