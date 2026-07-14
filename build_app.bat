@echo off
REM Build a standalone Elastic-Strain Analyzer for Windows -> dist\StrainAnalyzer.exe
REM Run from inside the project folder:  build_app.bat
setlocal

echo ==^> Creating an isolated build environment (.buildenv)...
python -m venv .buildenv
call .buildenv\Scripts\activate.bat

echo ==^> Installing dependencies + PyInstaller...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt pyinstaller

echo ==^> Building...
pyinstaller --noconfirm --clean --windowed --onefile ^
    --name "StrainAnalyzer" ^
    --collect-submodules skimage ^
    --collect-submodules scipy ^
    --collect-data skimage ^
    --collect-data matplotlib ^
    app.py

call deactivate
echo.
echo ==^> Done. Your app is the single file:  dist\StrainAnalyzer.exe
endlocal
