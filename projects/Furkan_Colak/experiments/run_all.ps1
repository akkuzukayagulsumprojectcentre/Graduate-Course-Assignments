# Runs every stage in order (PowerShell version of run_all.sh).
# Usage: .\experiments\run_all.ps1 [-From build] [-Python C:\path\to\python.exe]
param([string]$From = "collect", [string]$Python = "python")
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:PYTHONIOENCODING = "utf-8"

$stages = "collect", "collect_ext", "generate", "variants", "build", "analyze", "zeroshot", "detect", "evaluate", "calibrate", "compare", "robust", "errors", "leakage", "cost", "report", "verify"
$started = $false
foreach ($st in $stages) {
    if ($st -eq $From) { $started = $true }
    if (-not $started) { continue }
    $cfg = if ($st -in "collect", "collect_ext", "generate", "variants") { "configs/generation.yaml" } else { "configs/experiments.yaml" }
    Write-Output "=== $st ($(Get-Date -Format HH:mm))"
    & $Python -m src.cli $st --config $cfg
    if ($LASTEXITCODE -ne 0) { throw "stage $st failed" }
}
& $Python -m pytest -q tests
