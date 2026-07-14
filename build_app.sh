#!/usr/bin/env bash
# Build a standalone, double-clickable Elastic-Strain Analyzer.
#   macOS  -> dist/StrainAnalyzer.app
#   Linux  -> dist/StrainAnalyzer/StrainAnalyzer
# Run from inside the project folder:  bash build_app.sh
set -euo pipefail

PY="${PYTHON:-python3}"

echo "==> Creating an isolated build environment (.buildenv)…"
"$PY" -m venv .buildenv
# shellcheck disable=SC1091
source .buildenv/bin/activate

echo "==> Installing dependencies + PyInstaller…"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt pyinstaller

echo "==> Building…"
pyinstaller --noconfirm --clean --windowed \
    --name "StrainAnalyzer" \
    --collect-submodules skimage \
    --collect-submodules scipy \
    --collect-data skimage \
    --collect-data matplotlib \
    app.py

deactivate
echo
echo "==> Done. Your app is in:  dist/StrainAnalyzer*"
echo "    macOS:  open dist/StrainAnalyzer.app"
echo "    Linux:  ./dist/StrainAnalyzer/StrainAnalyzer"
