@echo off
setlocal

set "PYTHON_EXE=D:\Python313\python.exe"
set "SCRIPT_DIR=%~dp0"

if not exist "%PYTHON_EXE%" (
    echo [fall-event-server] Python not found: %PYTHON_EXE%
    echo Install or extract Python to D:\Python313 first.
    exit /b 1
)

if "%~1"=="" (
    "%PYTHON_EXE%" "%SCRIPT_DIR%fall_event_server.py" --host 0.0.0.0 --port 18080
) else (
    "%PYTHON_EXE%" "%SCRIPT_DIR%fall_event_server.py" %*
)
