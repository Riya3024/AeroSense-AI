# Runs the full Phase 1 pipeline and stops at the first failing step.
# Usage (from the project root, venv activated):  .\scripts\run_pipeline.ps1
$ErrorActionPreference = "Stop"
$steps = @(
    "data_pipeline.fetch_openaq",
    "data_pipeline.fetch_weather",
    "data_pipeline.merge_datasets",
    "data_pipeline.validate_data"
)
foreach ($step in $steps) {
    Write-Host "=== python -m $step ===" -ForegroundColor Cyan
    python -m $step
    if ($LASTEXITCODE -ne 0) { throw "Step '$step' failed with exit code $LASTEXITCODE. See logs\ for details." }
}
Write-Host "Pipeline finished. See data\processed\data_quality_report.md" -ForegroundColor Green
