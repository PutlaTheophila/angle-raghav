"""Dump annotated test frames into debug/ for visual QA of the circle fit
and strain measurement.

Usage
-----
    .venv/bin/python make_debug_frames.py                 # 16 frames from test.mp4
    .venv/bin/python make_debug_frames.py clip.mp4 24     # 24 frames from clip.mp4

Each saved PNG shows the fitted (red dashed) circle, the chord A-B, the kept
arc samples, and an info panel with R / D / angle / thickness / strain so you
can eyeball whether the geometry is sensible frame by frame.  A contact-sheet
strip (debug/_strip.png) stitches the tracked frames together, and the chosen
frames + measurements are echoed to the console.
"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np

from tracker import TrackParams, analyze_frame, detect_scale_bar, draw_overlay

DEBUG_DIR = "debug"


def _autocalibrate(cap: cv2.VideoCapture, n: int) -> float:
    for _ in range(min(n, 30)):
        ok, frame = cap.read()
        if not ok:
            break
        s = detect_scale_bar(frame)
        if s:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return s
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return 1.0 / 487.0


def main(path: str = "test.mp4", n_frames: int = 16) -> int:
    os.makedirs(DEBUG_DIR, exist_ok=True)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"cannot open {path}")
        return 1
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    mm_per_px = _autocalibrate(cap, total)
    print(f"{path}: {total} frames @ {fps:.1f} fps  scale={mm_per_px*1000:.3f} um/px")

    params = TrackParams()
    idxs = sorted(set(int(round(i)) for i in
                      np.linspace(0, max(total - 1, 0), n_frames)))
    thumbs: list[np.ndarray] = []
    for fi in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok:
            continue
        res = analyze_frame(frame, fi, mm_per_px, params)
        annotated = draw_overlay(frame, res, show_panel=True,
                                 t_seconds=fi / fps)
        out_path = os.path.join(DEBUG_DIR, f"fit_{fi:04d}.png")
        cv2.imwrite(out_path, annotated)
        tag = (f"strain={res.strain_pct:.3f}% R={res.R_mm:.3f}mm"
               if res.ok else f"no-needle ({res.needle_px}px)")
        print(f"  frame {fi:4d}  {tag}  -> {out_path}")

        thumb = cv2.resize(annotated, (480, int(480 * frame.shape[0] / frame.shape[1])))
        thumbs.append(thumb)

    if thumbs:
        h = min(t.shape[0] for t in thumbs)
        row = np.hstack([t[:h] for t in thumbs])
        strip_path = os.path.join(DEBUG_DIR, "_strip.png")
        cv2.imwrite(strip_path, row)
        print(f"contact sheet -> {strip_path}")
    return 0


if __name__ == "__main__":
    p = sys.argv[1] if len(sys.argv) > 1 else "test.mp4"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    raise SystemExit(main(p, n))
