@echo off
title Lanzador de Seismic Facies 3D Net GUI
echo ========================================================
echo   Seismic Facies 3D Net - Interfaz Grafica de Prediccion
echo ========================================================
echo Iniciando aplicacion...
python gui_app.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Error al iniciar la aplicacion. Verifique los requerimientos.
    pause
)
