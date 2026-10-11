<#
.SYNOPSIS
  Runs the full test suite. Uses the project .venv when present.
  Usage:  .\scripts\run_tests.ps1 [extra pytest args]
#>
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
$py = if (Test-Path ".\.venv\Scripts\python.exe") { ".\.venv\Scripts\python.exe" } else { "python" }
& $py -m pytest -q @args
exit $LASTEXITCODE
