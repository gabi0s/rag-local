@echo off
setlocal

start "RAG Backend" cmd /k "python scripts\server.py"
start "RAG UI" cmd /k "cd /d %~dp0UI && python -m http.server 5173"
echo UI: http://localhost:5173
echo API: http://localhost:8000
