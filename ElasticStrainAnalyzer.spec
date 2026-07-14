# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the Elastic-Strain Analyzer.

Build a standalone app (no Python required on the target machine):

    macOS / Linux:
        .venv/bin/pyinstaller ElasticStrainAnalyzer.spec
    Windows:
        .venv\\Scripts\\pyinstaller ElasticStrainAnalyzer.spec

Output:
    dist/ElasticStrainAnalyzer            (one-file executable)
    dist/ElasticStrainAnalyzer.app        (macOS app bundle)
"""
from PyInstaller.utils.hooks import collect_submodules, collect_data_files

hiddenimports = (
    collect_submodules("skimage")
    + collect_submodules("scipy")
    + ["PIL._tkinter_finder"]
)
datas = collect_data_files("skimage") + collect_data_files("matplotlib")

a = Analysis(
    ["app.py"],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["PyQt5", "PySide2", "PySide6", "PyQt6", "pytest"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="ElasticStrainAnalyzer",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    runtime_tmpdir=None,
    console=False,          # GUI app: no terminal window
    disable_windowed_traceback=False,
    argv_emulation=True,    # lets macOS file-open events reach the app
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

app = BUNDLE(
    exe,
    name="ElasticStrainAnalyzer.app",
    icon=None,
    bundle_identifier="com.raghavlab.elasticstrain",
    info_plist={
        "CFBundleName": "Elastic-Strain Analyzer",
        "CFBundleDisplayName": "Elastic-Strain Analyzer",
        "CFBundleShortVersionString": "1.0.0",
        "NSHighResolutionCapable": True,
    },
)
