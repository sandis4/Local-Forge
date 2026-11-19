@echo off
rem Local Forge beta: создаёт .venv и ставит зависимости движка.
where python >nul 2>nul
if errorlevel 1 (
    echo [setup] Python not found. Install Python 3.10+ from python.org and run again.
    exit /b 1
)
python -m venv .venv
if errorlevel 1 (
    echo [setup] Failed to create .venv.
    exit /b 1
)
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r requirements.txt
if errorlevel 1 (
    echo [setup] Failed to install requirements.
    exit /b 1
)
echo [setup] Done. Start local-forge.exe.
