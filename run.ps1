# One-command setup + launch for the Weekly Replenishment Planner.
#
#   Right-click -> "Run with PowerShell", or from a terminal:  .\run.ps1
#
# Creates the virtual environment, installs dependencies, makes sure the
# prepared data + trained model exist, then opens the app in your browser.
# Safe to run repeatedly — it skips any step that's already done.

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# --- 1. Python check ------------------------------------------------------- #
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "Python was not found on your PATH." -ForegroundColor Red
    Write-Host "Install Python 3.11+ from https://www.python.org/downloads/ " -NoNewline
    Write-Host "(tick 'Add python.exe to PATH' in the installer), then re-run this script."
    exit 1
}

# --- 2. Virtual environment ------------------------------------------------ #
$py = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    Write-Host "Creating virtual environment (.venv)..." -ForegroundColor Cyan
    python -m venv .venv
}

# --- 3. Dependencies ------------------------------------------------------- #
Write-Host "Checking dependencies..." -ForegroundColor Cyan
& $py -m pip install --quiet --disable-pip-version-check -r requirements.txt

# --- 4. Prepared data + trained model -------------------------------------- #
# These are committed to the repo, so normally both already exist. If they're
# missing (e.g. a fresh regen), rebuild them — this needs the raw M5 CSVs in
# data\ (see the Data section of the README).
if (-not (Test-Path "data\m5_long.parquet")) {
    Write-Host "Preparing dataset (one-time, ~20s)..." -ForegroundColor Cyan
    & $py -m src.prepare_data
}
if (-not (Test-Path "artifacts\lgb_baseline.txt")) {
    Write-Host "Training model (one-time, ~20s)..." -ForegroundColor Cyan
    & $py -m src.train
}

# --- 5. Launch ------------------------------------------------------------- #
Write-Host ""
Write-Host "Starting the app — it will open at http://localhost:8501" -ForegroundColor Green
Write-Host "Press Ctrl+C in this window to stop it." -ForegroundColor Green
Write-Host ""
& $py -m streamlit run app.py
