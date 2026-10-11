<#
.SYNOPSIS
  One-time local setup: virtual environment, dependencies, and a .env file.
  Usage (from anywhere):  .\scripts\setup_dev.ps1
#>
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

if (-not (Test-Path ".venv")) {
    py -3 -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw "Could not create .venv (is Python 3.10+ installed? try: py -3 --version)" }
}
$py = ".\.venv\Scripts\python.exe"
& $py -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed" }
& $py -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw "pip install failed" }

if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "Created .env from .env.example. Edit it and add OPENAQ_API_KEY (the file is git-ignored)." -ForegroundColor Yellow
}
Write-Host "Setup complete. Activate with:  .\.venv\Scripts\Activate.ps1" -ForegroundColor Green
