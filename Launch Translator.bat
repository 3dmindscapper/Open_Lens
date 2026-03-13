@echo off
cd /d "%~dp0Open_Lens_Core"
py translator_ui.py
if %errorlevel% neq 0 (
    echo.
    echo Something went wrong - see message above.
    pause
)
