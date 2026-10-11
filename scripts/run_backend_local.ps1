<#
.SYNOPSIS
  Starts the local API (stdlib server wrapping the Lambda handler) on http://127.0.0.1:8000
  Usage:  .\scripts\run_backend_local.ps1 [-Port 8000]
  Check:  Invoke-RestMethod http://127.0.0.1:8000/health
#>
param([int]$Port = 8000)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
$py = if (Test-Path ".\.venv\Scripts\python.exe") { ".\.venv\Scripts\python.exe" } else { "python" }
& $py -m backend.local_server --port $Port
exit $LASTEXITCODE
