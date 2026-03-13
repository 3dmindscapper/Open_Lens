@echo off
cd /d "%~dp0layout-parser"
"C:\Users\migne\AppData\Local\Python\pythoncore-3.14-64\python.exe" translator_ui.py
if %errorlevel% neq 0 (
    echo.
    echo Something went wrong - see message above.
    pause
)
