@echo off
setlocal enabledelayedexpansion

:: Quick start - launches the TTS API server directly
set "SCRIPT_DIR=%~dp0"

:: Determine Python executable
set "PYTHON_EXE=%SCRIPT_DIR%python_embedded\python.exe"
if not exist "%PYTHON_EXE%" (
    echo ERROR: Embedded Python not found.
    echo   Run install.bat first to set up the environment.
    pause
    exit /b 1
)

:: Add portable Git to PATH if available
if exist "%SCRIPT_DIR%git_portable\cmd\git.exe" (
    set "PATH=%SCRIPT_DIR%git_portable\cmd;%PATH%"
)

:: Add FFmpeg to PATH if available
for /d %%D in ("%SCRIPT_DIR%ffmpeg\ffmpeg-*") do (
    if exist "%%D\bin\ffmpeg.exe" (
        set "PATH=%%D\bin;%PATH%"
    )
)
if exist "%SCRIPT_DIR%ffmpeg\bin\ffmpeg.exe" (
    set "PATH=%SCRIPT_DIR%ffmpeg\bin;%PATH%"
)

:: Add Rubberband to PATH if available
for /d %%D in ("%SCRIPT_DIR%rubberband\rubberband-*") do (
    if exist "%%D\rubberband.exe" (
        set "PATH=%%D;%PATH%"
    )
)
if exist "%SCRIPT_DIR%rubberband\rubberband.exe" (
    set "PATH=%SCRIPT_DIR%rubberband;%PATH%"
)

:: Add eSpeak NG to PATH if available
if exist "%SCRIPT_DIR%espeak_ng\espeak-ng.exe" (
    set "PATH=%SCRIPT_DIR%espeak_ng;%PATH%"
    set "ESPEAK_DATA_PATH=%SCRIPT_DIR%espeak_ng\espeak-ng-data"
)
for /d %%D in ("%SCRIPT_DIR%espeak_ng\eSpeak*") do (
    if exist "%%D\espeak-ng.exe" (
        set "PATH=%%D;%PATH%"
        set "ESPEAK_DATA_PATH=%%D\espeak-ng-data"
    )
)

cd /d "%SCRIPT_DIR%"
"%PYTHON_EXE%" "%SCRIPT_DIR%tts_api_server.py" %*
if errorlevel 1 (
    echo.
    echo ERROR: API server exited with an error.
    pause
)
