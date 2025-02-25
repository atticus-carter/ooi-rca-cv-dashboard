@echo off
echo Starting OOI RCA CV Dashboard...

REM Create and activate virtual environment if it doesn't exist
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Install requirements if needed
if not exist "venv\Lib\site-packages\streamlit" (
    echo Installing requirements...
    pip install -r requirements.txt
)

REM Run the Streamlit app
echo Running Streamlit app...
echo 🐈
streamlit run main.py

REM Keep the window open if there's an error
if %ERRORLEVEL% NEQ 0 (
    echo Error running the application
    pause
)
