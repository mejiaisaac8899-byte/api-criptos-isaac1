@echo off
title UNEFA BIOPAGO - Iniciar Servidor y Tunel
color 0A
chcp 65001 > nul
cls

echo ======================================================================
echo             UNEFA BIOPAGO - SISTEMA BIOMETRICO VASCULAR
echo ======================================================================
echo.
echo [1/3] Iniciando Servidor Bancario FastAPI (Puerto 8000)...
start "UNEFA BIOPAGO - Backend API" /min cmd /c "cd /d C:\Users\Isaac\OneDrive\Desktop\COSAS\ISACS\api-criptos-isaac1 && python -m uvicorn main:app --host 0.0.0.0 --port 8000"

timeout /t 2 /nobreak > nul

echo [2/3] Iniciando Tunel ngrok permanente (qualify-denim-armed.ngrok-free.dev)...
start "UNEFA BIOPAGO - Tunel ngrok" /min cmd /c "C:\Users\Isaac\AppData\Local\Microsoft\WinGet\Packages\Ngrok.Ngrok_Microsoft.Winget.Source_8wekyb3d8bbwe\ngrok.exe http --domain=qualify-denim-armed.ngrok-free.dev 8000"

timeout /t 2 /nobreak > nul

echo [3/3] Abriendo Interfaz Bancaria en el Navegador...
start http://localhost:8000/

echo.
echo ======================================================================
echo                       ¡TODO EN LINEA Y LISTO!
echo ======================================================================
echo.
echo  * Panel Web POS Bancario:  http://localhost:8000/
echo  * Endpoint ESP32 (Mundial): https://qualify-denim-armed.ngrok-free.dev/upload
echo.
echo  Puedes minimizar esta ventana. Para apagar todo, simplemente cierra
echo  las ventanas minimizadas o presiona una tecla aqui.
echo ======================================================================
echo.
pause
taskkill /f /im ngrok.exe > nul 2>&1
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8000 ^| findstr LISTENING') do taskkill /f /pid %%a > nul 2>&1
echo Sistema detenido.
