@echo off
title UNEFA BIOPAGO - Servidor y Red
color 0A
chcp 65001 > nul
cls

echo ======================================================================
echo             UNEFA BIOPAGO - SISTEMA BIOMETRICO VASCULAR
echo ======================================================================
echo.
echo [1/3] Iniciando Servidor Bancario FastAPI (Puerto 8000)...
start "UNEFA BIOPAGO - Backend" cmd /k "title Backend FastAPI && cd /d C:\Users\Isaac\OneDrive\Desktop\COSAS\ISACS\api-criptos-isaac1 && python -m uvicorn main:app --host 0.0.0.0 --port 8000"

timeout /t 3 /nobreak > nul

echo [2/3] Iniciando Tunel Fijo Mundial (unefa-biopago-isaac.loca.lt)...
start "UNEFA BIOPAGO - Tunel Mundial" cmd /k "title Tunel Mundial && npx -y localtunnel --port 8000 --subdomain unefa-biopago-isaac"

timeout /t 3 /nobreak > nul

echo [3/3] Abriendo Interfaz Bancaria en el Navegador...
start http://localhost:8000/

echo.
echo ======================================================================
echo                       ¡TODO EN LINEA Y LISTO!
echo ======================================================================
echo.
echo  * Panel Web POS Bancario:  http://localhost:8000/
echo  * Endpoint ESP32 (Mundial): https://unefa-biopago-isaac.loca.lt/upload
echo.
echo  Las ventanas del Backend y del Tunel estan abiertas y funcionando.
echo  Puedes minimizar esta ventana.
echo ======================================================================
echo.
pause
