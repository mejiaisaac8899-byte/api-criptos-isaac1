@echo off
title UNEFA BIOPAGO - Sistema Bancario Biometrico
color 0A
chcp 65001 > nul
cls

echo ======================================================================
echo             UNEFA BIOPAGO - SISTEMA BIOMETRICO VASCULAR
echo ======================================================================
echo.

echo [1/4] Verificando y liberando el Puerto 8000...
for /f "tokens=5" %%a in ('netstat -aon 2^>nul ^| findstr :8000 ^| findstr LISTENING') do (
    echo   - Cerrando proceso anterior (PID: %%a)...
    taskkill /F /PID %%a >nul 2>&1
)
timeout /t 1 /nobreak > nul

echo.
echo [2/4] Iniciando Servidor Bancario FastAPI (Puerto 8000)...
start "UNEFA BIOPAGO - Backend" cmd /k "title Backend FastAPI && cd /d C:\Users\Isaac\OneDrive\Desktop\COSAS\ISACS\api-criptos-isaac1 && python -m uvicorn main:app --host 0.0.0.0 --port 8000"

timeout /t 3 /nobreak > nul

echo.
echo [3/4] Iniciando Tunel de Alta Velocidad (Cloudflare)...
start "UNEFA BIOPAGO - Tunel Cloudflare" cmd /k "title Tunel Cloudflare && cloudflared tunnel --url http://127.0.0.1:8000"

timeout /t 2 /nobreak > nul

echo.
echo [4/4] Abriendo Interfaz Bancaria en el Navegador...
start http://localhost:8000/

echo.
echo ======================================================================
echo                       ¡TODO EN LINEA Y LISTO!
echo ======================================================================
echo.
echo  * Panel Web POS Bancario (PC Local):  http://localhost:8000/
echo  * Panel Web ESP32 (Ajustes y LEDs):   http://localhost:8000/esp32-cam
echo.
echo  ==================================================================
echo  CONEXION DESDE EL TELEFONO O EL ESP32:
echo  1. Mira la ventana llamada "Tunel Cloudflare"
echo  2. Ahi veras tu enlace publico (https://....trycloudflare.com)
echo  3. Abre ese enlace en tu telefono para entrar desde cualquier parte
echo  ==================================================================
echo.
echo  Las ventanas del Backend y del Tunel deben permanecer abiertas.
echo  Puedes minimizar esta ventana.
echo ======================================================================
echo.
pause
