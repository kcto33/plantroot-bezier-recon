@echo off
rem Build desktop app, output in dist\PlantRootRecon\
setlocal
cd /d "%~dp0"
chcp 65001 >nul

if not exist .venv-desktop (
    echo [1/3] Creating venv .venv-desktop ...
    python -m venv .venv-desktop || goto :err
)

call .venv-desktop\Scripts\activate.bat

echo [2/3] Installing dependencies ...
python -m pip install -r requirements-desktop.txt || goto :err

echo [3/3] PyInstaller build ...
pyinstaller --noconfirm --clean plantroot.spec || goto :err

echo.
echo Build done: dist\PlantRootRecon\PlantRootRecon.exe
goto :eof

:err
echo.
echo Build FAILED, see errors above.
exit /b 1
