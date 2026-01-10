@echo off
setlocal

rem Ensure no stale Ollama instance blocks the port.
taskkill /IM ollama.exe /F >nul 2>&1
start "Ollama (Vulkan)" cmd /k "set OLLAMA_VULKAN=1 && ollama serve"
start "RAG Backend" cmd /k "python scripts\server.py"
start "RAG UI" cmd /k "cd /d %~dp0UI && python -m http.server 5173"
echo UI: http://localhost:5173
echo API: http://localhost:8000
