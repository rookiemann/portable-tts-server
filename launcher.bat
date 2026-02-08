@echo off
setlocal enabledelayedexpansion

:: TTS Module Launcher
:: Usage: launcher.bat [command] [options]

set "SCRIPT_DIR=%~dp0"

:: Determine Python executable (embedded first, then system)
set "PYTHON_EXE=%SCRIPT_DIR%python_embedded\python.exe"
if not exist "%PYTHON_EXE%" (
    where python >nul 2>&1
    if %errorlevel% equ 0 (
        for /f "delims=" %%i in ('where python') do set "PYTHON_EXE=%%i"
    )
)

:: Add portable Git to PATH if available
if exist "%SCRIPT_DIR%git_portable\cmd\git.exe" (
    set "PATH=%SCRIPT_DIR%git_portable\cmd;%PATH%"
)

:: Add FFmpeg to PATH if available (check various locations)
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

:: Set working directory to script location (required for module imports)
cd /d "%SCRIPT_DIR%"

echo.
echo ========================================
echo   TTS Module Launcher
echo ========================================
echo.

:: Check for python
if not exist "%PYTHON_EXE%" (
    echo ERROR: Python not found.
    echo.
    echo   Run install.bat first to set up the environment.
    echo.
    pause
    exit /b 1
)

:: Parse command
set "COMMAND=%~1"
if "%COMMAND%"=="" set "COMMAND=gui"

if /i "%COMMAND%"=="help" goto :show_help
if /i "%COMMAND%"=="setup" goto :run_setup
if /i "%COMMAND%"=="gui" goto :run_gui
if /i "%COMMAND%"=="api" goto :run_api
if /i "%COMMAND%"=="server" goto :run_api
if /i "%COMMAND%"=="download" goto :run_download

echo Unknown command: %COMMAND%
goto :show_help

:show_help
echo.
echo Usage: launcher.bat [command] [options]
echo.
echo Commands:
echo   gui             Launch the TTS Manager GUI (default)
echo   api, server     Start the TTS API server directly
echo   download        Download models (use --model NAME or --all)
echo   setup           Run install.bat for full environment setup
echo   help            Show this help
echo.
echo API Server options (for 'api' command):
echo   --port PORT     Server port (default: 8100)
echo   --host HOST     Server host (default: 127.0.0.1)
echo.
echo Download options (for 'download' command):
echo   --model NAME    Download specific model
echo   --all           Download all models
echo   --list          List available models
echo.
echo Examples:
echo   launcher.bat                     Launch GUI
echo   launcher.bat api                 Start API server
echo   launcher.bat api --port 8200     Start API on port 8200
echo   launcher.bat download --all      Download all models
echo.
pause
exit /b 0

:run_setup
echo Running full environment setup...
call "%SCRIPT_DIR%install.bat"
pause
exit /b 0

:run_gui
echo Launching TTS Manager GUI...
echo.
echo Python: %PYTHON_EXE%
echo.

"%PYTHON_EXE%" "%SCRIPT_DIR%tts_manager.py"

if errorlevel 1 (
    echo.
    echo ERROR: The TTS Manager exited with an error.
    echo.
    pause
)
exit /b 0

:run_api
echo Starting TTS API Server...

:: Collect remaining arguments
set "ARGS="
:collect_api_args
shift
if "%~1"=="" goto :start_api
set "ARGS=%ARGS% %~1"
goto :collect_api_args

:start_api
"%PYTHON_EXE%" "%SCRIPT_DIR%tts_api_server.py" %ARGS%
if errorlevel 1 (
    echo.
    echo ERROR: API server exited with an error.
    echo.
    pause
)
exit /b 0

:run_download
echo Running model download...

:: Collect remaining arguments
set "ARGS="
:collect_dl_args
shift
if "%~1"=="" goto :start_download
set "ARGS=%ARGS% %~1"
goto :collect_dl_args

:start_download
"%PYTHON_EXE%" "%SCRIPT_DIR%download_all_models.py" %ARGS%
if errorlevel 1 (
    echo.
    echo ERROR: Download failed.
    echo.
)
pause
endlocal
exit /b 0
