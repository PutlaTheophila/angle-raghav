"""Elastic-strain analyser - professional GUI.

Workflow
--------
    1. Open a video (or image).
    2. Click "Analyze" - every frame is processed and cached (progress shown).
    3. Scrub / play to inspect the per-frame circle fit and strain.
    4. Export a CSV + annotated MP4.

    source .venv/bin/activate
    python app.py                       # opens GUI on test.mp4 (if present)
    python app.py path/to/clip.mp4      # open a specific clip
    python app.py clip.mp4 --batch      # headless -> out/annotated.mp4 + CSV
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import threading
import tkinter as tk
from dataclasses import asdict
from tkinter import filedialog, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from tracker import (
    FrameResult,
    TrackParams,
    analyze_frame,
    detect_scale_bar,
    draw_overlay,
)

try:                                   # optional: embedded strain plot
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    _HAVE_MPL = True
except Exception:                      # pragma: no cover
    _HAVE_MPL = False

DEFAULT_MEDIA = "test.mp4"
OUT_DIR = "out"

# ----------------------------------------------------------------- palette
APP_BG = "#eef1f6"
HEADER_BG = "#1e293b"
HEADER_FG = "#f8fafc"
CARD_BG = "#ffffff"
ACCENT = "#2563eb"
ACCENT_ACTIVE = "#1d4ed8"
TEXT = "#1e293b"
MUTED = "#64748b"
CANVAS_BG = "#0b0e14"
SUCCESS = "#16a34a"
DANGER = "#dc2626"
BORDER = "#d8dee9"
FONT = "Helvetica Neue"


def _upscale_result(res: FrameResult, frame_shape: tuple, scale: float
                    ) -> FrameResult:
    """Lift a FrameResult computed on a downscaled frame back to full
    resolution (px geometry only; mm-fields already correct)."""
    if scale >= 1.0 or not res.ok:
        if res.mask is not None and res.mask.shape[:2] != frame_shape[:2]:
            res.mask = cv2.resize(res.mask, (frame_shape[1], frame_shape[0]),
                                  interpolation=cv2.INTER_NEAREST)
        return res
    H, W = frame_shape[:2]
    if res.mask is not None and res.mask.shape[:2] != (H, W):
        res.mask = cv2.resize(res.mask, (W, H), interpolation=cv2.INTER_NEAREST)
    inv = 1.0 / scale
    for a in ("cx", "cy", "R_px", "L_px", "h_max_px", "thickness_px",
              "fit_resid_px"):
        setattr(res, a, getattr(res, a) * inv)
    if res.arc_pts is not None:
        res.arc_pts = (res.arc_pts.astype(np.float64) * inv).astype(np.int32)
    if res.skeleton_pts is not None:
        res.skeleton_pts = (res.skeleton_pts.astype(np.float64) * inv
                            ).astype(np.int32)
    for a in ("chord_p1", "chord_p2", "apex", "apex_foot"):
        p = getattr(res, a)
        setattr(res, a, (int(p[0] * inv), int(p[1] * inv)))
    return res


def _autocalibrate(cap: cv2.VideoCapture, n: int) -> float | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for _ in range(min(n, 30)):
        ok, frame = cap.read()
        if not ok:
            break
        s = detect_scale_bar(frame)
        if s:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return s
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return None


# ----------------------------------------------------------------- media
class MediaSource:
    """Uniform interface over (video, single image)."""

    def __init__(self, path: str):
        self.path = path
        self.is_image = path.lower().endswith(
            (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))
        if self.is_image:
            img = cv2.imread(path)
            if img is None:
                raise SystemExit(f"cannot open {path}")
            self._img = img
            self.frame_count = 1
            self.fps = 1.0
            self.height, self.width = img.shape[:2]
        else:
            self._cap = cv2.VideoCapture(path)
            if not self._cap.isOpened():
                raise SystemExit(f"cannot open {path}")
            self.frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
            self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def read(self, idx: int) -> np.ndarray | None:
        if self.is_image:
            return self._img.copy()
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = self._cap.read()
        return frame if ok else None

    def autocalibrate(self) -> float | None:
        if self.is_image:
            return detect_scale_bar(self._img)
        return _autocalibrate(self._cap, self.frame_count)

    def reopen(self) -> "MediaSource":
        return MediaSource(self.path)


# ----------------------------------------------------------------- batch mode
def run_batch(path: str) -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    src = MediaSource(path)
    mm_per_px = src.autocalibrate() or (1.0 / 487.0)
    print(f"{src.width}x{src.height} @{src.fps:.1f}fps  "
          f"{src.frame_count} frames  scale={mm_per_px:.6f} mm/px")
    writer = None
    csv_f = open(os.path.join(OUT_DIR, "measurements.csv"), "w", newline="")
    cw = csv.writer(csv_f)
    cw.writerow(["frame", "t_s", "ok", "R_mm", "D_mm", "L_mm", "h_max_mm",
                 "theta_deg", "t_mm", "strain_pct", "fit_resid_px",
                 "arc_samples", "skel_samples"])
    params = TrackParams()
    for fi in range(src.frame_count):
        frame = src.read(fi)
        if frame is None:
            break
        if writer is None and not src.is_image:
            writer = cv2.VideoWriter(
                os.path.join(OUT_DIR, "annotated.mp4"),
                cv2.VideoWriter_fourcc(*"mp4v"), src.fps,
                (src.width, src.height))
        res = analyze_frame(frame, fi, mm_per_px, params)
        out = draw_overlay(frame, res, t_seconds=fi / src.fps)
        if src.is_image:
            cv2.imwrite(os.path.join(OUT_DIR, "annotated.png"), out)
        else:
            writer.write(out)
        cw.writerow([fi, fi / src.fps, int(res.ok),
                     res.R_mm, res.D_mm, res.L_mm, res.h_max_mm,
                     res.theta_deg, res.thickness_mm, res.strain_pct,
                     res.fit_resid_px, res.n_arc_samples, res.n_skel_samples])
        if fi % 20 == 0:
            print(f"  {fi}/{src.frame_count}")
    if writer is not None:
        writer.release()
    csv_f.close()
    json.dump({"mm_per_px": mm_per_px, "media": path,
               "width": src.width, "height": src.height, "fps": src.fps,
               "params": asdict(params)},
              open(os.path.join(OUT_DIR, "calibration.json"), "w"), indent=2)
    print("wrote", OUT_DIR + ("/annotated.mp4" if not src.is_image
                              else "/annotated.png"))
    return 0


# ----------------------------------------------------------------- GUI
class TrackerApp(tk.Tk):
    def __init__(self, path: str | None):
        super().__init__()
        self.title("Elastic-Strain Analyzer")
        self.geometry("1500x940")
        self.minsize(1180, 720)
        self.configure(bg=APP_BG)

        self.src: MediaSource | None = None
        self.mm_per_px = 1.0 / 487.0
        self.params = TrackParams()
        self.frame_cache: dict[int, np.ndarray] = {}
        self.result_cache: dict[int, FrameResult] = {}
        self.analyzed = False
        self.analyzing = False
        self.strain_series: list[tuple[int, float]] = []

        self.show_mask = tk.BooleanVar(value=True)
        self.show_circle = tk.BooleanVar(value=True)
        self.show_chord = tk.BooleanVar(value=True)
        self.show_arc_pts = tk.BooleanVar(value=True)
        self.show_panel = tk.BooleanVar(value=False)
        self.playing = False

        self.roi: tuple[int, int, int, int] | None = None
        self.scale_pts: list[tuple[int, int]] = []
        self.scale_mm = tk.DoubleVar(value=1.0)
        self._drag_start: tuple[int, int] | None = None
        self._drag_mode = "roi"
        self._pending_refresh: str | None = None
        self._analysis_cache: tuple | None = None
        self._preview_scale = 1.0
        self._fast_mode = False

        self.frame_var = tk.IntVar(value=0)
        self.frame_entry_var = tk.StringVar(value="0")

        self._init_style()
        self._build_ui()
        self.bind("<space>", lambda e: self.toggle_play())
        self.bind("<Left>", lambda e: self.step(-1))
        self.bind("<Right>", lambda e: self.step(1))

        if path and os.path.exists(path):
            self._load(path)
        else:
            self._set_status("Open a video to begin.", MUTED)

    # ---------------------------------------------------------- styling
    def _init_style(self) -> None:
        st = ttk.Style(self)
        try:
            st.theme_use("clam")
        except tk.TclError:
            pass
        st.configure(".", background=APP_BG, foreground=TEXT,
                     font=(FONT, 11))
        st.configure("App.TFrame", background=APP_BG)
        st.configure("Header.TFrame", background=HEADER_BG)
        st.configure("Card.TFrame", background=CARD_BG)
        st.configure("Canvas.TFrame", background=CANVAS_BG)

        st.configure("HeaderTitle.TLabel", background=HEADER_BG,
                     foreground=HEADER_FG, font=(FONT, 17, "bold"))
        st.configure("HeaderSub.TLabel", background=HEADER_BG,
                     foreground="#94a3b8", font=(FONT, 10))
        st.configure("CardTitle.TLabel", background=CARD_BG, foreground=MUTED,
                     font=(FONT, 10, "bold"))
        st.configure("Card.TLabel", background=CARD_BG, foreground=TEXT,
                     font=(FONT, 11))
        st.configure("CardMuted.TLabel", background=CARD_BG, foreground=MUTED,
                     font=(FONT, 10))
        st.configure("Metric.TLabel", background=CARD_BG, foreground=TEXT,
                     font=(FONT, 13, "bold"))
        st.configure("MetricBig.TLabel", background=CARD_BG, foreground=ACCENT,
                     font=(FONT, 20, "bold"))
        st.configure("Card.TCheckbutton", background=CARD_BG, foreground=TEXT,
                     font=(FONT, 10))
        st.map("Card.TCheckbutton", background=[("active", CARD_BG)])

        st.configure("TButton", font=(FONT, 11), padding=(12, 7),
                     background="#e2e8f0", foreground=TEXT, borderwidth=0)
        st.map("TButton", background=[("active", "#cbd5e1")])
        st.configure("Primary.TButton", font=(FONT, 11, "bold"),
                     padding=(14, 8), background=ACCENT, foreground="#ffffff",
                     borderwidth=0)
        st.map("Primary.TButton",
               background=[("active", ACCENT_ACTIVE), ("disabled", "#9db7f0")])
        st.configure("Ghost.TButton", font=(FONT, 11), padding=(10, 6),
                     background=HEADER_BG, foreground=HEADER_FG, borderwidth=0)
        st.map("Ghost.TButton", background=[("active", "#334155")])

        st.configure("Strain.Horizontal.TProgressbar", troughcolor="#e2e8f0",
                     background=ACCENT, borderwidth=0, thickness=10)
        st.configure("TScale", background=CARD_BG)
        st.configure("Trans.TScale", background=APP_BG)

    # ---------------------------------------------------------- UI shell
    def _build_ui(self) -> None:
        # ---- header bar -------------------------------------------------
        header = ttk.Frame(self, style="Header.TFrame")
        header.pack(side="top", fill="x")
        htext = ttk.Frame(header, style="Header.TFrame")
        htext.pack(side="left", padx=18, pady=12)
        ttk.Label(htext, text="Elastic-Strain Analyzer",
                  style="HeaderTitle.TLabel").pack(anchor="w")
        self.header_sub = ttk.Label(
            htext, text="Three-point crystal bending  ·  Euler-Bernoulli fit",
            style="HeaderSub.TLabel")
        self.header_sub.pack(anchor="w")

        hbtn = ttk.Frame(header, style="Header.TFrame")
        hbtn.pack(side="right", padx=16, pady=12)
        self.open_btn = ttk.Button(hbtn, text="Open Video",
                                   style="Ghost.TButton", command=self._on_open)
        self.open_btn.pack(side="left", padx=4)
        self.analyze_btn = ttk.Button(hbtn, text="Analyze",
                                      style="Primary.TButton",
                                      command=self._on_analyze)
        self.analyze_btn.pack(side="left", padx=4)
        self.export_btn = ttk.Button(hbtn, text="Export CSV + Video",
                                     style="Ghost.TButton",
                                     command=self._on_process)
        self.export_btn.pack(side="left", padx=4)

        # ---- body -------------------------------------------------------
        body = ttk.Frame(self, style="App.TFrame")
        body.pack(fill="both", expand=True, padx=14, pady=12)

        left = ttk.Frame(body, style="App.TFrame")
        left.pack(side="left", fill="both", expand=True)

        canvas_wrap = ttk.Frame(left, style="Canvas.TFrame")
        canvas_wrap.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(canvas_wrap, background=CANVAS_BG,
                                highlightthickness=0, bd=0)
        self.canvas.pack(fill="both", expand=True, padx=2, pady=2)
        self.canvas.bind("<Configure>", lambda e: self._refresh())
        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.canvas.bind("<ButtonPress-3>", self._on_right_click)

        self._build_transport(left)
        self._build_sidebar(body)
        self._show_placeholder()

    def _build_transport(self, parent) -> None:
        bar = ttk.Frame(parent, style="App.TFrame")
        bar.pack(fill="x", pady=(10, 0))
        self.play_btn = ttk.Button(bar, text="▶  Play", width=9,
                                   command=self.toggle_play)
        self.play_btn.pack(side="left")
        ttk.Button(bar, text="◀", width=3,
                   command=lambda: self.step(-1)).pack(side="left", padx=(6, 2))
        ttk.Button(bar, text="▶", width=3,
                   command=lambda: self.step(1)).pack(side="left", padx=2)
        self.slider = ttk.Scale(bar, from_=0, to=1, orient="horizontal",
                                variable=self.frame_var,
                                command=self._on_slider)
        self.slider.pack(side="left", fill="x", expand=True, padx=10)
        self.slider.bind("<ButtonPress-1>", lambda e: self._set_fast(True))
        self.slider.bind("<ButtonRelease-1>", lambda e: self._set_fast(False))
        fe = ttk.Entry(bar, width=6, textvariable=self.frame_entry_var,
                       justify="right")
        fe.pack(side="left", padx=(2, 0))
        fe.bind("<Return>", self._on_frame_entry)
        self.frame_lbl = ttk.Label(bar, text="/ 0", style="App.TFrame",
                                   width=8)
        self.frame_lbl.configure(background=APP_BG, foreground=MUTED)
        self.frame_lbl.pack(side="left", padx=(4, 0))

        prog = ttk.Frame(parent, style="App.TFrame")
        prog.pack(fill="x", pady=(8, 0))
        self.progress = ttk.Progressbar(prog, mode="determinate", maximum=1,
                                        style="Strain.Horizontal.TProgressbar")
        self.progress.pack(side="left", fill="x", expand=True)
        self.status_dot = tk.Label(prog, text="●", bg=APP_BG, fg=MUTED,
                                   font=(FONT, 12))
        self.status_dot.pack(side="left", padx=(10, 4))
        self.status = tk.Label(prog, text="No media.", bg=APP_BG, fg=MUTED,
                               font=(FONT, 10), anchor="w")
        self.status.pack(side="left", fill="x")

    # ---------------------------------------------------------- sidebar
    def _card(self, parent, title: str) -> ttk.Frame:
        outer = tk.Frame(parent, bg=BORDER)
        outer.pack(fill="x", pady=(0, 12))
        card = ttk.Frame(outer, style="Card.TFrame")
        card.pack(fill="both", expand=True, padx=1, pady=1)
        ttk.Label(card, text=title.upper(), style="CardTitle.TLabel").pack(
            anchor="w", padx=14, pady=(12, 6))
        return card

    def _build_sidebar(self, body) -> None:
        side = ttk.Frame(body, style="App.TFrame", width=360)
        side.pack(side="right", fill="y", padx=(14, 0))
        side.pack_propagate(False)

        # ---- measurements (hero) ---------------------------------------
        mc = self._card(side, "Measurement")
        hero = ttk.Frame(mc, style="Card.TFrame")
        hero.pack(fill="x", padx=14)
        self.strain_big = ttk.Label(hero, text="—", style="MetricBig.TLabel")
        self.strain_big.pack(side="left")
        ttk.Label(hero, text="  strain", style="CardMuted.TLabel").pack(
            side="left", pady=(8, 0))
        self.status_chip = tk.Label(mc, text="no media", bg="#e2e8f0",
                                    fg=MUTED, font=(FONT, 9, "bold"),
                                    padx=8, pady=2)
        self.status_chip.pack(anchor="w", padx=14, pady=(2, 8))

        grid = ttk.Frame(mc, style="Card.TFrame")
        grid.pack(fill="x", padx=14, pady=(0, 12))
        self.metric_lbls: dict[str, ttk.Label] = {}
        rows = [("R_mm", "Radius R"), ("D_mm", "Diameter D"),
                ("L_mm", "Chord L"), ("h_mm", "Sagitta h_max"),
                ("theta", "Subtended angle"), ("t_mm", "Thickness t"),
                ("resid", "Fit RMS"), ("t", "Time")]
        for i, (key, label) in enumerate(rows):
            r = i // 2
            c = (i % 2) * 2
            ttk.Label(grid, text=label, style="CardMuted.TLabel").grid(
                row=r, column=c, sticky="w", pady=3, padx=(0, 6))
            v = ttk.Label(grid, text="—", style="Metric.TLabel")
            v.grid(row=r, column=c + 1, sticky="w", pady=3, padx=(0, 14))
            self.metric_lbls[key] = v

        # ---- strain plot -----------------------------------------------
        if _HAVE_MPL:
            pc = self._card(side, "Strain vs Frame")
            self.fig = Figure(figsize=(3.2, 1.7), dpi=100)
            self.fig.patch.set_facecolor(CARD_BG)
            self.ax = self.fig.add_subplot(111)
            self._style_plot()
            self.plot_canvas = FigureCanvasTkAgg(self.fig, master=pc)
            self.plot_canvas.get_tk_widget().pack(fill="x", padx=8,
                                                  pady=(0, 10))

        # ---- calibration ------------------------------------------------
        cal = self._card(side, "Calibration & ROI")
        self.scale_lbl = ttk.Label(cal, text="scale: —", style="Card.TLabel")
        self.scale_lbl.pack(anchor="w", padx=14)
        srow = ttk.Frame(cal, style="Card.TFrame")
        srow.pack(fill="x", padx=14, pady=6)
        ttk.Label(srow, text="bar (mm)", style="CardMuted.TLabel").pack(
            side="left")
        ttk.Entry(srow, textvariable=self.scale_mm, width=6).pack(
            side="left", padx=6)
        ttk.Button(srow, text="Set scale",
                   command=self._set_scale_mode).pack(side="left", padx=2)
        ttk.Button(srow, text="Clear",
                   command=self._clear_scale).pack(side="left")
        self.roi_lbl = ttk.Label(cal, text="ROI: whole frame  (drag to set)",
                                 style="CardMuted.TLabel")
        self.roi_lbl.pack(anchor="w", padx=14, pady=(2, 10))

        # ---- overlay toggles -------------------------------------------
        ov = self._card(side, "Overlay")
        row = ttk.Frame(ov, style="Card.TFrame")
        row.pack(fill="x", padx=12, pady=(0, 10))
        for txt, var in [("Mask", self.show_mask),
                         ("Circle", self.show_circle),
                         ("Chord/h", self.show_chord),
                         ("Samples", self.show_arc_pts),
                         ("Panel", self.show_panel)]:
            ttk.Checkbutton(row, text=txt, variable=var, style="Card.TCheckbutton",
                            command=self._refresh).pack(side="left", padx=4)

    def _style_plot(self) -> None:
        ax = self.ax
        ax.clear()
        ax.set_facecolor("#f8fafc")
        for s in ax.spines.values():
            s.set_color(BORDER)
        ax.tick_params(colors=MUTED, labelsize=7)
        ax.set_xlabel("frame", color=MUTED, fontsize=8)
        ax.set_ylabel("strain %", color=MUTED, fontsize=8)
        ax.grid(True, color="#eef1f6", linewidth=0.8)
        self.fig.subplots_adjust(left=0.16, right=0.97, top=0.92, bottom=0.26)

    def _update_plot(self, cur_idx: int | None = None) -> None:
        if not _HAVE_MPL:
            return
        self._style_plot()
        if self.strain_series:
            xs = [f for f, _ in self.strain_series]
            ys = [s for _, s in self.strain_series]
            self.ax.plot(xs, ys, color=ACCENT, linewidth=1.4)
            self.ax.scatter(xs, ys, s=6, color=ACCENT, zorder=3)
        if cur_idx is not None:
            self.ax.axvline(cur_idx, color=DANGER, linewidth=1.0, alpha=0.8)
        self.plot_canvas.draw_idle()

    # ---------------------------------------------------------- status
    def _set_status(self, text: str, color: str = MUTED) -> None:
        self.status.configure(text=text, fg=color)
        self.status_dot.configure(fg=color)

    def _set_chip(self, text: str, fg: str, bg: str) -> None:
        self.status_chip.configure(text=text, fg=fg, bg=bg)

    def _show_placeholder(self) -> None:
        self.canvas.delete("all")
        self.canvas.update_idletasks()
        cw = max(self.canvas.winfo_width(), 400)
        ch = max(self.canvas.winfo_height(), 300)
        self.canvas.create_text(cw // 2, ch // 2,
                                text="Open a video to begin",
                                fill="#475569", font=(FONT, 16))

    # ---------------------------------------------------------- load
    def _load(self, path: str) -> None:
        try:
            src = MediaSource(path)
        except SystemExit as e:
            self._set_status(str(e), DANGER)
            return
        self.src = src
        self.frame_cache.clear()
        self.result_cache.clear()
        self.analyzed = False
        self.strain_series = []
        self.roi = None
        self.scale_pts = []
        self._analysis_cache = None
        self.mm_per_px = src.autocalibrate() or (1.0 / 487.0)
        self.title(f"Elastic-Strain Analyzer — {os.path.basename(path)}")
        self.header_sub.configure(
            text=f"{os.path.basename(path)}   ·   {src.width}×{src.height}"
                 f"   ·   {src.fps:.0f} fps   ·   {src.frame_count} frames")
        self.scale_lbl.configure(text=f"scale: {self.mm_per_px*1000:.3f} µm/px")
        self.roi_lbl.configure(text="ROI: whole frame  (drag to set)")
        self.slider.configure(to=max(src.frame_count - 1, 0))
        self.progress.configure(maximum=max(src.frame_count, 1), value=0)
        self.frame_var.set(0)
        self.frame_entry_var.set("0")
        if _HAVE_MPL:
            self._update_plot(None)
        self._set_chip("not analyzed", MUTED, "#e2e8f0")
        self._set_status("Ready. Click Analyze to process all frames.", ACCENT)
        self._show_frame(0)

    # ---------------------------------------------------------- analyze
    def _current_params(self) -> TrackParams:
        from dataclasses import replace
        return replace(self.params, roi=self.roi)

    def _analyze_one(self, idx: int, frame: np.ndarray) -> FrameResult:
        return analyze_frame(frame, idx, self.mm_per_px, self._current_params())

    def _on_analyze(self) -> None:
        if self.src is None or self.analyzing:
            return
        self.analyzing = True
        self.analyze_btn.configure(state="disabled", text="Analyzing…")
        self.export_btn.configure(state="disabled")
        self.result_cache.clear()
        self.strain_series = []
        self.progress.configure(value=0)
        self._set_chip("analyzing…", "#92400e", "#fef3c7")
        self._set_status("Analyzing all frames…", ACCENT)
        threading.Thread(target=self._run_analysis, daemon=True).start()

    def _run_analysis(self) -> None:
        src = self.src.reopen()
        params = self._current_params()
        n = src.frame_count
        for fi in range(n):
            frame = src.read(fi)
            if frame is None:
                break
            res = analyze_frame(frame, fi, self.mm_per_px, params)
            self.result_cache[fi] = res
            if fi % 4 == 0 or fi == n - 1:
                self.after(0, self._analysis_progress, fi, n)
        self.after(0, self._analysis_done)

    def _analysis_progress(self, fi: int, n: int) -> None:
        self.progress.configure(value=fi + 1)
        pct = int(100 * (fi + 1) / max(n, 1))
        self._set_status(f"Analyzing… {pct}%  ({fi + 1}/{n})", ACCENT)
        self._set_chip(f"analyzing {pct}%", "#92400e", "#fef3c7")

    def _analysis_done(self) -> None:
        self.analyzing = False
        self.analyzed = True
        self.analyze_btn.configure(state="normal", text="Re-analyze")
        self.export_btn.configure(state="normal")
        self.strain_series = [(fi, r.strain_pct)
                              for fi, r in sorted(self.result_cache.items())
                              if r.ok]
        ok = len(self.strain_series)
        tot = len(self.result_cache)
        self._set_chip(f"analyzed · {ok}/{tot} tracked", SUCCESS, "#dcfce7")
        self._set_status(f"Analysis complete — {ok}/{tot} frames tracked. "
                         f"Scrub or play to inspect fits.", SUCCESS)
        if _HAVE_MPL:
            self._update_plot(int(float(self.frame_var.get())))
        self._show_frame(int(float(self.frame_var.get())))

    # ---------------------------------------------------------- caching
    def _read_frame(self, idx: int) -> np.ndarray | None:
        if idx in self.frame_cache:
            return self.frame_cache[idx]
        if self.src is None:
            return None
        frame = self.src.read(idx)
        if frame is None:
            return None
        if len(self.frame_cache) > 40:
            self.frame_cache.pop(next(iter(self.frame_cache)))
        self.frame_cache[idx] = frame
        return frame

    def _result_for(self, idx: int, frame: np.ndarray) -> FrameResult | None:
        if self._fast_mode:
            return None
        if idx in self.result_cache:
            return self.result_cache[idx]
        # on-demand single-frame analysis (before a full Analyze run)
        key = (idx, self.roi, round(self.mm_per_px, 9))
        if self._analysis_cache and self._analysis_cache[0] == key:
            return self._analysis_cache[1]
        res = self._analyze_one(idx, frame)
        self._analysis_cache = (key, res)
        return res

    # ---------------------------------------------------------- display
    def _show_frame(self, idx: int) -> None:
        if self.src is None:
            return
        idx = max(0, min(self.src.frame_count - 1, idx))
        frame = self._read_frame(idx)
        if frame is None:
            return
        res = self._result_for(idx, frame)
        if res is None:
            annotated = frame.copy()
            cv2.putText(annotated, "scrubbing…", (24, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2,
                        cv2.LINE_AA)
        else:
            annotated = draw_overlay(
                frame, res,
                show_mask=self.show_mask.get(),
                show_circle=self.show_circle.get(),
                show_chord=self.show_chord.get(),
                show_arc_pts=self.show_arc_pts.get(),
                show_panel=self.show_panel.get(),
                t_seconds=idx / self.src.fps)

        if self.roi is not None:
            rx, ry, rw, rh = self.roi
            cv2.rectangle(annotated, (rx, ry), (rx + rw, ry + rh),
                          (255, 0, 0), max(2, frame.shape[0] // 600))
        if len(self.scale_pts) == 2:
            p1, p2 = self.scale_pts
            cv2.line(annotated, p1, p2, (255, 0, 255),
                     max(2, frame.shape[0] // 800))
            cv2.circle(annotated, p1, 8, (255, 0, 255), -1)
            cv2.circle(annotated, p2, 8, (255, 0, 255), -1)

        self._last_annotated = annotated
        self._blit()
        self.frame_lbl.configure(text=f"/ {self.src.frame_count - 1}")
        if self.frame_entry_var.get() != str(idx):
            self.frame_entry_var.set(str(idx))
        self._update_metrics(res, idx)
        if _HAVE_MPL and self.strain_series:
            self._update_plot(idx)

    def _blit(self) -> None:
        if not hasattr(self, "_last_annotated"):
            return
        img = self._last_annotated
        cw = max(self.canvas.winfo_width(), 400)
        ch = max(self.canvas.winfo_height(), 300)
        s = min(cw / img.shape[1], ch / img.shape[0])
        nw, nh = max(1, int(img.shape[1] * s)), max(1, int(img.shape[0] * s))
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        self._tk_img = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.canvas.delete("all")
        self.canvas.create_image((cw - nw) // 2, (ch - nh) // 2,
                                 image=self._tk_img, anchor="nw")
        self._display_scale = s
        self._display_offset = ((cw - nw) // 2, (ch - nh) // 2)

    def _update_metrics(self, res: FrameResult | None, idx: int) -> None:
        m = self.metric_lbls
        m["t"].configure(text=f"{idx / self.src.fps:.3f} s"
                         if not self.src.is_image else "—")
        if res is None:
            self.strain_big.configure(text="—")
            for k in ("R_mm", "D_mm", "L_mm", "h_mm", "theta", "t_mm", "resid"):
                m[k].configure(text="—")
            self._set_chip("scrubbing…", MUTED, "#e2e8f0")
            return
        if res.ok:
            self.strain_big.configure(text=f"{res.strain_pct:.2f}%")
            m["R_mm"].configure(text=f"{res.R_mm:.3f} mm")
            m["D_mm"].configure(text=f"{res.D_mm:.3f} mm")
            m["L_mm"].configure(text=f"{res.L_mm:.3f} mm")
            m["h_mm"].configure(text=f"{res.h_max_mm:.3f} mm")
            m["theta"].configure(text=f"{res.theta_deg:.1f}°")
            m["t_mm"].configure(text=f"{res.thickness_mm * 1000:.1f} µm")
            m["resid"].configure(text=f"{res.fit_resid_px:.1f} px")
            self._set_chip("tracked", SUCCESS, "#dcfce7")
        else:
            self.strain_big.configure(text="—")
            for k in ("R_mm", "D_mm", "L_mm", "h_mm", "theta", "t_mm", "resid"):
                m[k].configure(text="—")
            self._set_chip("no fit", DANGER, "#fee2e2")

    # ---------------------------------------------------------- transport
    def _on_params_changed(self) -> None:
        self._schedule_refresh()

    def _on_slider(self, _evt=None) -> None:
        idx = int(float(self.frame_var.get()))
        self.frame_entry_var.set(str(idx))
        self._schedule_refresh()

    def _on_frame_entry(self, _evt=None) -> None:
        try:
            idx = int(self.frame_entry_var.get())
        except ValueError:
            return
        if self.src is None:
            return
        idx = max(0, min(self.src.frame_count - 1, idx))
        self.frame_var.set(idx)
        self._do_refresh()

    def _refresh(self) -> None:
        self._schedule_refresh()

    def _schedule_refresh(self, delay: int = 30) -> None:
        if self._pending_refresh is not None:
            self.after_cancel(self._pending_refresh)
        self._pending_refresh = self.after(delay, self._do_refresh)

    def _do_refresh(self) -> None:
        self._pending_refresh = None
        if self.src is not None:
            self._show_frame(int(float(self.frame_var.get())))
        else:
            self._show_placeholder()

    def _set_fast(self, on: bool) -> None:
        # Only use fast (raw) scrubbing before analysis; cached lookups are
        # instant afterwards.
        self._fast_mode = on and not self.analyzed
        if not on:
            self._schedule_refresh(delay=40)

    def step(self, delta: int) -> None:
        if self.src is None:
            return
        new_idx = max(0, min(self.src.frame_count - 1,
                             int(float(self.frame_var.get())) + delta))
        self.frame_var.set(new_idx)
        self._show_frame(new_idx)

    def toggle_play(self) -> None:
        if self.src is None or self.src.is_image:
            return
        self.playing = not self.playing
        self.play_btn.configure(text="❚❚  Pause" if self.playing else "▶  Play")
        if self.playing:
            self.after(int(1000 / max(self.src.fps, 1)), self._play_tick)

    def _play_tick(self) -> None:
        if not self.playing or self.src is None:
            return
        cur = int(float(self.frame_var.get()))
        nxt = cur + 1
        if nxt >= self.src.frame_count:
            self.playing = False
            self.play_btn.configure(text="▶  Play")
            return
        self.frame_var.set(nxt)
        self._show_frame(nxt)
        # Cached playback can run near real-time; raw analysis is slower.
        delay = 33 if self.analyzed else 120
        self.after(delay, self._play_tick)

    # ---------------------------------------------------------- canvas xforms
    def _canvas_to_image(self, x: int, y: int) -> tuple[int, int] | None:
        if not hasattr(self, "_display_scale") or self.src is None:
            return None
        ox, oy = self._display_offset
        s = self._display_scale
        ix, iy = int((x - ox) / s), int((y - oy) / s)
        if ix < 0 or iy < 0 or ix >= self.src.width or iy >= self.src.height:
            return None
        return ix, iy

    def _set_scale_mode(self) -> None:
        self._drag_mode = "scale"
        self.scale_pts = []
        self._set_status("Click the two endpoints of the scale bar.", ACCENT)

    def _clear_scale(self) -> None:
        self.scale_pts = []
        self._set_status("Scale cleared.", MUTED)
        self._refresh()

    def _clear_roi(self) -> None:
        self.roi = None
        self.roi_lbl.configure(text="ROI: whole frame  (drag to set)")
        self._analysis_cache = None
        self._refresh()

    def _on_press(self, evt) -> None:
        pt = self._canvas_to_image(evt.x, evt.y)
        if pt is None:
            return
        if self._drag_mode == "scale":
            self.scale_pts.append(pt)
            if len(self.scale_pts) == 2:
                p1, p2 = self.scale_pts
                dist_px = float(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))
                bar_mm = max(self.scale_mm.get(), 1e-6)
                self.mm_per_px = bar_mm / max(dist_px, 1e-6)
                self.scale_lbl.configure(
                    text=f"scale: {self.mm_per_px*1000:.3f} µm/px")
                self._set_status(f"Scale set: {dist_px:.0f} px = {bar_mm} mm. "
                                 f"Re-analyze to apply.", SUCCESS)
                self._drag_mode = "roi"
                self.result_cache.clear()
                self.analyzed = False
                self._analysis_cache = None
            self._refresh()
            return
        self._drag_start = pt

    def _on_drag(self, evt) -> None:
        if self._drag_mode != "roi" or self._drag_start is None:
            return
        pt = self._canvas_to_image(evt.x, evt.y)
        if pt is None:
            return
        x0, y0 = self._drag_start
        x1, y1 = pt
        rx, ry = min(x0, x1), min(y0, y1)
        rw, rh = abs(x1 - x0), abs(y1 - y0)
        self.roi = (rx, ry, rw, rh) if rw > 5 and rh > 5 else None
        self.roi_lbl.configure(
            text=f"ROI: {rx},{ry}  {rw}×{rh}" if self.roi
            else "ROI: whole frame  (drag to set)")
        self._analysis_cache = None
        self._refresh()

    def _on_release(self, _evt) -> None:
        self._drag_start = None
        if self.roi is not None and self.analyzed:
            self.analyzed = False
            self.result_cache.clear()
            self._set_status("ROI changed — re-analyze to apply.", ACCENT)

    def _on_right_click(self, _evt) -> None:
        self._clear_roi()

    # ---------------------------------------------------------- file
    def _on_open(self) -> None:
        path = filedialog.askopenfilename(
            title="Open video or image",
            filetypes=[("Media", "*.mp4 *.mov *.avi *.mkv *.png *.jpg *.jpeg "
                                  "*.bmp *.tif *.tiff"),
                       ("All files", "*.*")])
        if path:
            self._load(path)

    # ---------------------------------------------------------- export
    def _on_process(self) -> None:
        if self.src is None:
            return
        self.export_btn.configure(state="disabled")
        self._set_status("Exporting CSV + annotated video…", ACCENT)
        self.progress.configure(value=0)
        threading.Thread(target=self._run_export, daemon=True).start()

    def _run_export(self) -> None:
        try:
            os.makedirs(OUT_DIR, exist_ok=True)
            src = self.src.reopen()
            writer = None
            params = self._current_params()
            csv_path = os.path.join(OUT_DIR, "measurements.csv")
            with open(csv_path, "w", newline="") as csv_f:
                cw = csv.writer(csv_f)
                cw.writerow(["frame", "t_s", "ok", "R_mm", "D_mm", "L_mm",
                             "h_max_mm", "theta_deg", "t_mm", "strain_pct",
                             "fit_resid_px", "arc_samples", "skel_samples"])
                for fi in range(src.frame_count):
                    frame = src.read(fi)
                    if frame is None:
                        break
                    res = (self.result_cache.get(fi)
                           if self.analyzed else None)
                    if res is None:
                        res = analyze_frame(frame, fi, self.mm_per_px, params)
                    out = draw_overlay(frame, res, t_seconds=fi / src.fps)
                    if src.is_image:
                        cv2.imwrite(os.path.join(OUT_DIR, "annotated.png"), out)
                    else:
                        if writer is None:
                            writer = cv2.VideoWriter(
                                os.path.join(OUT_DIR, "annotated.mp4"),
                                cv2.VideoWriter_fourcc(*"mp4v"), src.fps,
                                (src.width, src.height))
                        writer.write(out)
                    cw.writerow([fi, fi / src.fps, int(res.ok),
                                 res.R_mm, res.D_mm, res.L_mm, res.h_max_mm,
                                 res.theta_deg, res.thickness_mm,
                                 res.strain_pct, res.fit_resid_px,
                                 res.n_arc_samples, res.n_skel_samples])
                    self.after(0, lambda v=fi: self.progress.configure(value=v))
            if writer is not None:
                writer.release()
            json.dump({"mm_per_px": self.mm_per_px, "media": self.src.path,
                       "width": self.src.width, "height": self.src.height,
                       "fps": self.src.fps,
                       "params": asdict(self._current_params())},
                      open(os.path.join(OUT_DIR, "calibration.json"), "w"),
                      indent=2)
            self.after(0, lambda: self._set_status(
                f"Exported → {OUT_DIR}/  (measurements.csv, annotated.mp4)",
                SUCCESS))
        except Exception as e:
            self.after(0, lambda: self._set_status(f"Export error: {e}", DANGER))
        finally:
            self.after(0, lambda: self.export_btn.configure(state="normal"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("media", nargs="?", default=DEFAULT_MEDIA,
                    help="Video file (.mp4/.mov/...) or still image.")
    ap.add_argument("--batch", action="store_true",
                    help="Process all frames headless and exit.")
    args = ap.parse_args()
    if args.batch:
        if not os.path.exists(args.media):
            print("media not found:", args.media)
            return 1
        return run_batch(args.media)
    start = args.media if os.path.exists(args.media) else None
    app = TrackerApp(start)
    app.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
