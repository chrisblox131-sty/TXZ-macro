@echo off
title Roblox Motorcycle Mayhem Autopilot - Quick Launch
cd /d "%~dp0"

:: Check if Python is installed
where python >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python was not found in your system PATH!
    echo Please install Python and check "Add Python to PATH".
    pause
    exit /b 1
)

echo ====================================================================
echo        ROBLOX MOTORCYCLE MAYHEM AUTOPILOT - QUICK LAUNCH
echo ====================================================================
echo Hotkeys:
echo   ESC / Q = Emergency Killswitch
echo   F8      = Pause / Resume
echo   F9      = Save Snapshot
echo ====================================================================
echo.
python mayhem_autopilot.py --debug

if %ERRORLEVEL% neq 0 (
    echo.
    echo [NOTE] Macro stopped with code %ERRORLEVEL%.
)
pause
