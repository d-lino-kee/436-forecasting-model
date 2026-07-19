@echo off
REM Double-click this file to set up and launch the app.
REM It just runs run.ps1 with the execution policy relaxed for this one process,
REM so you don't have to touch any PowerShell settings.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run.ps1"
pause
