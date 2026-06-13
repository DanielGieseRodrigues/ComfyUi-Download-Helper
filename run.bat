@echo off
cd /d %~dp0

if not exist .venv (
    echo Creating virtual environment...
    python -m venv .venv
)

call .venv\Scripts\activate.bat

echo Stopping old instances on port 5000...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":5000 " ^| findstr LISTENING') do taskkill /F /PID %%a >nul 2>&1

echo Installing dependencies...
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -r requirements.txt

echo Opening browser...
start "" http://127.0.0.1:5000

python app.py
pause
