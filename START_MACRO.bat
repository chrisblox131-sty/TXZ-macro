@echo off
setlocal
title Roblox Motorcycle Mayhem Autopilot
cd /d "%~dp0"

:: Check if Python is installed
where python >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python was not found in your system PATH!
    echo Please install Python from https://www.python.org/
    echo and ensure "Add Python to PATH" is checked during installation.
    echo.
    pause
    exit /b 1
)

:menu
cls
echo ====================================================================
echo             ROBLOX MOTORCYCLE MAYHEM - AUTOPILOT MACRO
echo ====================================================================
echo.
echo   [1] Start Autopilot (Live + Debug Overlay)
echo   [2] Start Autopilot (First-Person / FPV Mode)
echo   [3] Tuning / Calibration Mode (Trackbars / Sliders)
echo   [4] Safe Test Run (Dry Run - Vision Only, No Key Presses)
echo   [5] Install / Check Dependencies (OpenCV, MSS, PyDirectInput)
echo   [0] Exit
echo.
echo ====================================================================
echo Hotkeys while running:
echo   ESC / Q = Emergency Killswitch
echo   F8      = Pause / Resume
echo   F9      = Save Screenshot Snapshot
echo ====================================================================
echo.

choice /c 123450 /m "Select an option [1-5, 0 to exit]:"

if errorlevel 6 exit /b 0
if errorlevel 5 goto install_deps
if errorlevel 4 goto run_dry
if errorlevel 3 goto run_tune
if errorlevel 2 goto run_fpv
if errorlevel 1 goto run_live

:run_live
cls
echo Starting Autopilot in Normal Mode with Debug Window...
echo Switch to your Roblox window now!
echo.
python mayhem_autopilot.py --debug
goto finished

:run_fpv
cls
echo Starting Autopilot in First-Person View (FPV) Mode...
echo Switch to your Roblox window now!
echo.
python mayhem_autopilot.py --debug --fpv
goto finished

:run_tune
cls
echo Starting Tuning Mode with Trackbars...
echo Adjust sliders to calibrate road and car detection.
echo Config will automatically save to tuned_config.json on exit.
echo.
python mayhem_autopilot.py --debug --tune
goto finished

:run_dry
cls
echo Starting DRY RUN (Test Mode)...
echo The macro will detect lanes and cars, but WILL NOT press any keys.
echo.
python mayhem_autopilot.py --debug --dry-run
goto finished

:install_deps
cls
echo Installing/updating requirements from requirements.txt...
echo.
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
echo.
echo Done!
pause
goto menu

:finished
echo.
echo ====================================================================
echo Autopilot session ended.
echo ====================================================================
pause
goto menu
