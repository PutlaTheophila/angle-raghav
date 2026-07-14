# Elastic-Strain Analyzer

Measure the elastic bending strain of a crystal from a three-point bending
video, using the Euler–Bernoulli geometry of Figure S17:

```
R = ((L/2)^2 + h_max^2) / (2 * h_max)        # bend radius
epsilon(%) = t / (2 * R) * 100               # bending strain
```

where **L** is the chord between the two support contacts, **h_max** the
deflection at the probe (third) contact, **t** the crystal thickness, and
**R** the fitted bend radius.

---

## Option A — Run the standalone app (no Python needed)

After building (see *Packaging* below), distribute the file in `dist/`:

- **macOS:** double-click `ElasticStrainAnalyzer.app` (or run
  `./dist/ElasticStrainAnalyzer`).
- **Windows:** run `dist\ElasticStrainAnalyzer.exe`.

First launch can take a few seconds while libraries unpack.

> macOS Gatekeeper: the app is unsigned, so the first time you may need to
> right-click → **Open**, or run
> `xattr -dr com.apple.quarantine ElasticStrainAnalyzer.app`.

---

## Option B — Run from source

Requires Python 3.10+.

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py                      # GUI
python app.py path/to/clip.mp4     # open a specific clip
python app.py clip.mp4 --batch     # headless -> out/measurements.csv + annotated.mp4
```

---

## Using the GUI

1. **Open Video** – load an `.mp4/.mov/.avi/...` (or a single image).
   The scale bar is auto-detected when present.
2. **Analyze** – every frame is processed and cached (progress shown). When it
   finishes, the chip reports how many frames were tracked.
3. **Scrub / Play** – inspect the per-frame fit instantly: green chord (L),
   orange sagitta (h_max), red bend circle, and the detected tweezer contacts.
4. **Export Results…** – choose a folder; a `<video>_results/` directory is
   created next to it containing the measurements CSV, a print-ready strain
   graph (`_strain_graph.png` at 200 dpi **and** `.pdf`), the annotated video,
   and `_calibration.json`. The folder opens automatically when done.
   After a fracture is detected, frames are flagged `fractured` and excluded
   from the strain series.

**Calibration & ROI**
- Click **Set scale**, enter the bar length in mm, then click its two ends.
- Drag on the preview to limit analysis to a region of interest; right-click
  clears it. (Changing scale/ROI requires a re-analyze.)

**Shortcuts:** `Space` play/pause · `←` / `→` step frames.

---

## Packaging (build the standalone app)

PyInstaller builds for the OS it runs on, so build on each target OS.

**Windows (`StrainAnalyzer.exe`)** — two options:

1. On any Windows PC with Python installed: copy this folder over and run
   `build_app.bat`. The single-file `dist\StrainAnalyzer.exe` is produced.
2. Without a Windows machine: push this repo to GitHub and run the
   **Build Windows executable** workflow (`.github/workflows/build-windows.yml`)
   from the Actions tab — download `StrainAnalyzer-windows` from the run's
   artifacts.

**macOS / Linux:** `bash build_app.sh` (or
`pyinstaller --noconfirm ElasticStrainAnalyzer.spec`). Results land in `dist/`.

---

## Files

| File | Purpose |
|------|---------|
| `app.py` | GUI application (open → analyze → inspect → export) |
| `tracker.py` | Segmentation, three-point geometry, strain math, overlays |
| `make_debug_frames.py` | Dump annotated QA frames into `debug/` |
| `requirements.txt` | Runtime dependencies |
| `ElasticStrainAnalyzer.spec` | PyInstaller build recipe |

## Output columns (`*_measurements.csv`)

`frame, t_s, ok, status, R_mm, D_mm, L_mm, h_max_mm, theta_deg, t_mm,
strain_pct, fit_resid_px, n_contacts, arc_samples, skel_samples`

`status` is one of `tracked / straight / fractured / no-crystal`. Strain uses
the video-median (undeformed) thickness, as in the paper.
