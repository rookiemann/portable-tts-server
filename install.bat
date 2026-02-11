@echo off
setlocal enabledelayedexpansion

echo.
echo ============================================
echo   TTS Module - Autonomous Installer
echo ============================================
echo.
echo   This script sets up everything from scratch:
echo   - Embedded Python 3.10 (no system Python needed)
echo   - Portable Git (no system Git needed)
echo   - Portable FFmpeg (for audio processing)
echo   - Rubberband (for pitch-preserving tempo adjustment)
echo   - eSpeak NG (required for Kokoro TTS)
echo   - All Python dependencies
echo   - Then launches the TTS Manager GUI
echo.

set "SCRIPT_DIR=%~dp0"
set "PYTHON_DIR=%SCRIPT_DIR%python_embedded"
set "PYTHON_EXE=%PYTHON_DIR%\python.exe"
set "GIT_DIR=%SCRIPT_DIR%git_portable"
set "GIT_EXE=%GIT_DIR%\cmd\git.exe"
set "FFMPEG_DIR=%SCRIPT_DIR%ffmpeg"
set "FFMPEG_EXE=%FFMPEG_DIR%\bin\ffmpeg.exe"

:: Python 3.10 for best TTS compatibility (many TTS libs don't support 3.12 yet)
set "PYTHON_VERSION=3.10.11"
set "PYTHON_URL=https://www.python.org/ftp/python/%PYTHON_VERSION%/python-%PYTHON_VERSION%-embed-amd64.zip"
set "PYTHON_ZIP=%SCRIPT_DIR%python_embedded.zip"

set "GIT_VERSION=2.47.1"
set "GIT_URL=https://github.com/git-for-windows/git/releases/download/v%GIT_VERSION%.windows.1/MinGit-%GIT_VERSION%-64-bit.zip"
set "GIT_ZIP=%SCRIPT_DIR%git_portable.zip"

set "FFMPEG_URL=https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
set "FFMPEG_ZIP=%SCRIPT_DIR%ffmpeg_portable.zip"

set "RUBBERBAND_DIR=%SCRIPT_DIR%rubberband"
set "RUBBERBAND_VERSION=4.0.0"
set "RUBBERBAND_URL=https://breakfastquay.com/files/releases/rubberband-%RUBBERBAND_VERSION%-gpl-executable-windows.zip"
set "RUBBERBAND_ZIP=%SCRIPT_DIR%rubberband_portable.zip"

:: ============================================
:: Step 1: Download Embedded Python
:: ============================================
if exist "%PYTHON_EXE%" (
    echo [OK] Embedded Python already installed.
    goto :check_pip
)

echo [1/9] Downloading Python %PYTHON_VERSION% embedded...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
    "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12;" ^
    "$ProgressPreference = 'SilentlyContinue';" ^
    "Invoke-WebRequest -Uri '%PYTHON_URL%' -OutFile '%PYTHON_ZIP%'"

if not exist "%PYTHON_ZIP%" (
    echo.
    echo ERROR: Failed to download Python.
    echo   - Check your internet connection
    echo   - URL: %PYTHON_URL%
    echo.
    pause
    exit /b 1
)

echo [2/9] Extracting Python...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
    "Expand-Archive -Path '%PYTHON_ZIP%' -DestinationPath '%PYTHON_DIR%' -Force"

if not exist "%PYTHON_EXE%" (
    echo ERROR: Python extraction failed.
    pause
    exit /b 1
)

del "%PYTHON_ZIP%" 2>nul

:: ============================================
:: Step 2: Configure ._pth for site-packages
:: ============================================
echo [2/9] Configuring Python for package installation...

:: Create Lib\site-packages directory
if not exist "%PYTHON_DIR%\Lib\site-packages" (
    mkdir "%PYTHON_DIR%\Lib\site-packages"
)

:: Rewrite the ._pth file to enable import site and site-packages
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
    "$pthFiles = Get-ChildItem '%PYTHON_DIR%\python*._pth';" ^
    "if ($pthFiles.Count -gt 0) {" ^
    "  $pth = $pthFiles[0];" ^
    "  $zipName = (Get-ChildItem '%PYTHON_DIR%\python*.zip' | Select-Object -First 1).Name;" ^
    "  if (-not $zipName) { $zipName = 'python310.zip' };" ^
    "  $content = @($zipName, '.', 'Lib', 'Lib\site-packages', '', 'import site');" ^
    "  $content | Set-Content -Path $pth.FullName -Encoding ASCII;" ^
    "  Write-Host '   Configured:' $pth.Name" ^
    "} else {" ^
    "  Write-Host 'WARNING: No ._pth file found'" ^
    "}"

:: ============================================
:: Step 3: Bootstrap pip (via get-pip.py)
:: ============================================
:check_pip
"%PYTHON_EXE%" -m pip --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [3/9] Downloading get-pip.py...
    powershell -NoProfile -ExecutionPolicy Bypass -Command ^
        "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12;" ^
        "$ProgressPreference = 'SilentlyContinue';" ^
        "Invoke-WebRequest -Uri 'https://bootstrap.pypa.io/get-pip.py' -OutFile '%PYTHON_DIR%\get-pip.py'"

    if not exist "%PYTHON_DIR%\get-pip.py" (
        echo ERROR: Failed to download get-pip.py.
        pause
        exit /b 1
    )

    echo [3/9] Installing pip...
    "%PYTHON_EXE%" "%PYTHON_DIR%\get-pip.py"
    if errorlevel 1 (
        echo ERROR: Failed to install pip.
        pause
        exit /b 1
    )

    del "%PYTHON_DIR%\get-pip.py" 2>nul
    "%PYTHON_EXE%" -m pip install --upgrade pip 2>nul
) else (
    echo [OK] pip already available.
)

:: ============================================
:: Step 4: Set up tkinter (needed for GUI)
:: ============================================
"%PYTHON_EXE%" -c "import _tkinter" >nul 2>&1
if %errorlevel% neq 0 (
    echo [4/9] Setting up tkinter for GUI...

    set "TCLTK_MSI_URL=https://www.python.org/ftp/python/%PYTHON_VERSION%/amd64/tcltk.msi"
    set "TCLTK_MSI=%SCRIPT_DIR%_tcltk.msi"
    set "TCLTK_DIR=%SCRIPT_DIR%_tcltk_extract"

    echo   Downloading tcltk.msi (~3.4 MB)...
    powershell -NoProfile -ExecutionPolicy Bypass -Command ^
        "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12;" ^
        "$ProgressPreference = 'SilentlyContinue';" ^
        "Invoke-WebRequest -Uri '%TCLTK_MSI_URL%' -OutFile '%TCLTK_MSI%'"

    if not exist "%TCLTK_MSI%" (
        echo WARNING: Failed to download tcltk.msi. GUI may not work.
        goto :tkinter_done
    )

    echo   Extracting tkinter components...
    :: Administrative install extracts files without system registration
    :: Must use Start-Process -Wait because msiexec returns immediately otherwise
    if exist "!TCLTK_DIR!" rmdir /S /Q "!TCLTK_DIR!" 2>nul
    powershell -NoProfile -Command "Start-Process -FilePath 'msiexec.exe' -ArgumentList '/a','!TCLTK_MSI!','/qn','TARGETDIR=!TCLTK_DIR!' -Wait -NoNewWindow"

    :: Copy DLLs next to python.exe
    if exist "!TCLTK_DIR!\DLLs\_tkinter.pyd" (
        copy /Y "!TCLTK_DIR!\DLLs\_tkinter.pyd" "%PYTHON_DIR%\" >nul 2>&1
        copy /Y "!TCLTK_DIR!\DLLs\tcl86t.dll" "%PYTHON_DIR%\" >nul 2>&1
        copy /Y "!TCLTK_DIR!\DLLs\tk86t.dll" "%PYTHON_DIR%\" >nul 2>&1
        if exist "!TCLTK_DIR!\DLLs\zlib1.dll" (
            copy /Y "!TCLTK_DIR!\DLLs\zlib1.dll" "%PYTHON_DIR%\" >nul 2>&1
        )
    )

    :: Copy Lib/tkinter/ Python package
    if exist "!TCLTK_DIR!\Lib\tkinter" (
        if exist "%PYTHON_DIR%\Lib\tkinter" rmdir /S /Q "%PYTHON_DIR%\Lib\tkinter" 2>nul
        xcopy /E /I /Y "!TCLTK_DIR!\Lib\tkinter" "%PYTHON_DIR%\Lib\tkinter" >nul 2>&1
    )

    :: Copy tcl/ library (tcl8.6, tk8.6)
    if exist "!TCLTK_DIR!\tcl" (
        if exist "%PYTHON_DIR%\tcl" rmdir /S /Q "%PYTHON_DIR%\tcl" 2>nul
        xcopy /E /I /Y "!TCLTK_DIR!\tcl" "%PYTHON_DIR%\tcl" >nul 2>&1
    )

    :: Cleanup
    rmdir /S /Q "!TCLTK_DIR!" 2>nul
    del "!TCLTK_MSI!" 2>nul

    :: Verify
    "%PYTHON_EXE%" -c "import _tkinter" >nul 2>&1
    if errorlevel 1 (
        echo WARNING: Failed to set up tkinter. GUI may not work.
        echo   You can still use the API server directly.
    ) else (
        echo [OK] tkinter setup complete.
    )
) else (
    echo [OK] tkinter already available.
)
:tkinter_done

:: ============================================
:: Step 5: Download Portable Git
:: ============================================
if exist "%GIT_EXE%" (
    echo [OK] Portable Git already installed.
    goto :check_ffmpeg
)

echo [5/9] Downloading portable Git %GIT_VERSION%...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
    "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12;" ^
    "$ProgressPreference = 'SilentlyContinue';" ^
    "Invoke-WebRequest -Uri '%GIT_URL%' -OutFile '%GIT_ZIP%'"

if not exist "%GIT_ZIP%" (
    echo WARNING: Failed to download Git. Some features may not work.
    echo   Git clone operations require Git to be available.
    goto :check_ffmpeg
)

echo [5/9] Extracting portable Git...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
    "Expand-Archive -Path '%GIT_ZIP%' -DestinationPath '%GIT_DIR%' -Force"

del "%GIT_ZIP%" 2>nul

:: ============================================
:: Step 6: Download Portable FFmpeg
:: ============================================
:check_ffmpeg
:: Check for existing FFmpeg in various locations
set "FFMPEG_FOUND=0"
if exist "%FFMPEG_DIR%\bin\ffmpeg.exe" set "FFMPEG_FOUND=1"
if exist "%FFMPEG_DIR%\ffmpeg.exe" set "FFMPEG_FOUND=1"

:: Check for extracted builds with version in name
for /d %%D in ("%FFMPEG_DIR%\ffmpeg-*") do (
    if exist "%%D\bin\ffmpeg.exe" set "FFMPEG_FOUND=1"
)

if "%FFMPEG_FOUND%"=="1" (
    echo [OK] FFmpeg already installed.
    goto :check_rubberband
)

echo [6/9] Downloading portable FFmpeg...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
    "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12;" ^
    "$ProgressPreference = 'SilentlyContinue';" ^
    "Invoke-WebRequest -Uri '%FFMPEG_URL%' -OutFile '%FFMPEG_ZIP%'"

if not exist "%FFMPEG_ZIP%" (
    echo WARNING: Failed to download FFmpeg. Audio features may not work.
    goto :check_rubberband
)

echo [6/9] Extracting portable FFmpeg...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
    "Expand-Archive -Path '%FFMPEG_ZIP%' -DestinationPath '%FFMPEG_DIR%' -Force"

del "%FFMPEG_ZIP%" 2>nul

:: ============================================
:: Step 6b: Download Rubberband (for tempo adjustment)
:: ============================================
:check_rubberband
set "RUBBERBAND_FOUND=0"
if exist "%RUBBERBAND_DIR%\rubberband.exe" set "RUBBERBAND_FOUND=1"
if exist "%RUBBERBAND_DIR%\rubberband-r3.exe" set "RUBBERBAND_FOUND=1"
:: Check for files inside a versioned subdirectory
for /d %%D in ("%RUBBERBAND_DIR%\rubberband-*") do (
    if exist "%%D\rubberband.exe" set "RUBBERBAND_FOUND=1"
    if exist "%%D\rubberband-r3.exe" set "RUBBERBAND_FOUND=1"
)

if "%RUBBERBAND_FOUND%"=="1" (
    echo [OK] Rubberband already installed.
    goto :check_espeak
)

echo [6b/9] Downloading Rubberband %RUBBERBAND_VERSION% ^(tempo adjustment^)...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
    "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12;" ^
    "$ProgressPreference = 'SilentlyContinue';" ^
    "Invoke-WebRequest -Uri '%RUBBERBAND_URL%' -OutFile '%RUBBERBAND_ZIP%'"

if not exist "%RUBBERBAND_ZIP%" (
    echo WARNING: Failed to download Rubberband. Tempo adjustment will be unavailable.
    goto :check_espeak
)

echo [6b/9] Extracting Rubberband...
if not exist "%RUBBERBAND_DIR%" mkdir "%RUBBERBAND_DIR%"
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
    "Expand-Archive -Path '%RUBBERBAND_ZIP%' -DestinationPath '%RUBBERBAND_DIR%' -Force"

del "%RUBBERBAND_ZIP%" 2>nul

:: ============================================
:: Step 6c: Download eSpeak NG (required for Kokoro)
:: ============================================
:check_espeak
set "ESPEAK_DIR=%SCRIPT_DIR%espeak_ng"
set "ESPEAK_MSI=%SCRIPT_DIR%espeak_ng.msi"

if exist "%ESPEAK_DIR%\espeak-ng.exe" (
    echo [OK] eSpeak NG already installed.
    goto :setup_path
)

:: Check for espeak-ng inside PFiles subdirectory (MSI extract layout)
for /d %%D in ("%ESPEAK_DIR%\eSpeak*") do (
    if exist "%%D\espeak-ng.exe" (
        echo [OK] eSpeak NG already installed.
        goto :setup_path
    )
)

echo [6c/9] Downloading eSpeak NG 1.52 ^(required for Kokoro TTS^)...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
    "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12;" ^
    "$ProgressPreference = 'SilentlyContinue';" ^
    "Invoke-WebRequest -Uri 'https://github.com/espeak-ng/espeak-ng/releases/download/1.52.0/espeak-ng.msi' -OutFile '%ESPEAK_MSI%'"

if not exist "%ESPEAK_MSI%" (
    echo WARNING: Failed to download eSpeak NG. Kokoro TTS will not work.
    goto :setup_path
)

echo [6c/9] Extracting eSpeak NG...
if not exist "%ESPEAK_DIR%" mkdir "%ESPEAK_DIR%"
start /wait msiexec /a "%ESPEAK_MSI%" /qn TARGETDIR="%ESPEAK_DIR%"

del "%ESPEAK_MSI%" 2>nul

:: ============================================
:: Step 7: Set up PATH and install requirements
:: ============================================
:setup_path
:: Add portable Git to PATH for this session
if exist "%GIT_EXE%" (
    set "PATH=%GIT_DIR%\cmd;%PATH%"
    echo [OK] Portable Git added to PATH.
)

:: Find and add FFmpeg to PATH
for /d %%D in ("%FFMPEG_DIR%\ffmpeg-*") do (
    if exist "%%D\bin\ffmpeg.exe" (
        set "PATH=%%D\bin;%PATH%"
        echo [OK] FFmpeg added to PATH from %%D\bin
    )
)
if exist "%FFMPEG_DIR%\bin\ffmpeg.exe" (
    set "PATH=%FFMPEG_DIR%\bin;%PATH%"
    echo [OK] FFmpeg added to PATH.
)

:: Add Rubberband to PATH if available
for /d %%D in ("%RUBBERBAND_DIR%\rubberband-*") do (
    if exist "%%D\rubberband.exe" (
        set "PATH=%%D;%PATH%"
        echo [OK] Rubberband added to PATH from %%D
    )
)
if exist "%RUBBERBAND_DIR%\rubberband.exe" (
    set "PATH=%RUBBERBAND_DIR%;%PATH%"
    echo [OK] Rubberband added to PATH.
)

:: Add eSpeak NG to PATH if available
set "ESPEAK_DIR=%SCRIPT_DIR%espeak_ng"
if exist "%ESPEAK_DIR%\espeak-ng.exe" (
    set "PATH=%ESPEAK_DIR%;%PATH%"
    set "ESPEAK_DATA_PATH=%ESPEAK_DIR%\espeak-ng-data"
    echo [OK] eSpeak NG added to PATH.
)
for /d %%D in ("%ESPEAK_DIR%\eSpeak*") do (
    if exist "%%D\espeak-ng.exe" (
        set "PATH=%%D;%PATH%"
        set "ESPEAK_DATA_PATH=%%D\espeak-ng-data"
        echo [OK] eSpeak NG added to PATH from %%D
    )
)

echo [7/9] Installing requirements...
"%PYTHON_EXE%" -m pip install -r "%SCRIPT_DIR%requirements.txt" --quiet 2>nul
if errorlevel 1 (
    echo WARNING: Some requirements failed to install. Retrying without quiet...
    "%PYTHON_EXE%" -m pip install -r "%SCRIPT_DIR%requirements.txt"
)

echo [8/9] Installing Whisper (without torch - uses GPU torch from model environments)...
"%PYTHON_EXE%" -m pip install openai-whisper --no-deps --quiet 2>nul

:: ============================================
:: Step 9: Launch the TTS Manager GUI
:: ============================================
echo [9/9] Launching TTS Manager...
echo.
echo ============================================
echo.

:: Set working directory to script location (required for module imports)
cd /d "%SCRIPT_DIR%"

"%PYTHON_EXE%" "%SCRIPT_DIR%tts_manager.py" %*

if errorlevel 1 (
    echo.
    echo The TTS Manager exited with an error.
    echo.
    pause
)

endlocal
