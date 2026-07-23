@echo off
REM Start the Nemetron LangChain proxy
cd /d "%~dp0"
echo Starting Nemetron proxy on http://localhost:8000 ...
python server.py
pause
