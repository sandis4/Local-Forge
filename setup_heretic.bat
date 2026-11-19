@echo off
rem Optional: Heretic (ARA) support. Requires uv: https://docs.astral.sh/uv/
where uv >nul 2>nul
if errorlevel 1 (
    echo [setup] uv not found. Install uv or skip Heretic training.
    exit /b 1
)
cd /d "%~dp0heretic-2.1.0"
uv sync
if errorlevel 1 (
    echo [setup] Heretic setup failed.
    exit /b 1
)
echo [setup] Heretic ready.
