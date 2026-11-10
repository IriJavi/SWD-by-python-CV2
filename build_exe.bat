@echo off
rem Genera dist\MSPe-QA.exe (un solo archivo, sin consola). Ejecutar desde la raiz del proyecto: build_exe.bat
cd /d "%~dp0"
set PY=.venv\Scripts\python.exe
if not exist "%PY%" set PY=python

"%PY%" -m pip install -r requirements.txt
"%PY%" -m PyInstaller --noconfirm --clean --onefile --windowed --name MSPe-QA ^
    --exclude-module matplotlib --exclude-module pandas --exclude-module PyQt5 --exclude-module PyQt6 ^
    gui_contour.py

echo Listo: %~dp0dist\MSPe-QA.exe
