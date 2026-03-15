@echo off
cd /d "%~dp0Open_Lens_Core"

echo.
echo  Starting Open Lens Web Translator...
echo  The server URL will appear below.
echo  Open that URL on any device on your network.
echo.

py web_app.py

if %errorlevel% neq 0 (
    echo.
    echo  Something went wrong - see message above.
    echo  Make sure Flask is installed:  py -m pip install flask
    pause
)
