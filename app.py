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
    compute_strain,
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
# Professional dark theme (slate + blue accent), consistent across widgets.
APP_VERSION = "1.1.0"
APP_BG = "#0f172a"          # window background        (slate-900)
PANEL_BG = "#111c33"        # sidebar background
HEADER_BG = "#0b1120"       # header / status bar
HEADER_FG = "#f1f5f9"
CARD_BG = "#1e293b"         # cards                    (slate-800)
CARD_BG2 = "#243349"        # hovered / inset fields
ACCENT = "#3b82f6"          # primary blue
ACCENT_ACTIVE = "#2563eb"
TEXT = "#e2e8f0"
MUTED = "#8fa3bd"
CANVAS_BG = "#020617"
SUCCESS = "#4ade80"
SUCCESS_BG = "#052e16"
DANGER = "#f87171"
DANGER_BG = "#450a0a"
INFO = "#93c5fd"
INFO_BG = "#172554"
WARN = "#fbbf24"
WARN_BG = "#3b2703"
BORDER = "#334155"
FONT = "Helvetica Neue"
MONO = "Menlo"


def _autocalibrate(cap: cv2.VideoCapture, n: int) -> float | None:
    """Scan frames sampled across the whole clip for the scale bar (it may
    only be stamped on part of the video)."""
    found = None
    for idx in sorted(set(int(i) for i in np.linspace(0, max(n - 1, 0), 24))):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        s = detect_scale_bar(frame)
        if s:
            found = s
            break
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return found


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
            self._pos = 0            # decoder position (next frame index)

    def read(self, idx: int) -> np.ndarray | None:
        """Read frame idx; sequential access decodes without seeking (both
        faster and frame-exact)."""
        if self.is_image:
            return self._img.copy()
        if idx != self._pos:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = self._cap.read()
        self._pos = idx + 1 if ok else -1
        return frame if ok else None

    def autocalibrate(self) -> float | None:
        if self.is_image:
            return detect_scale_bar(self._img)
        return _autocalibrate(self._cap, self.frame_count)

    def reopen(self) -> "MediaSource":
        return MediaSource(self.path)


# ----------------------------------------------------------------- shared
def lock_thickness(results: dict[int, FrameResult], mm_per_px: float) -> float:
    """The paper's strain uses the thickness of the UNDEFORMED crystal - a
    per-video constant.  Lock every frame to the median measured thickness
    and recompute the strain, removing per-frame measurement noise.
    Returns the locked thickness (mm), or 0.0 if nothing was measurable."""
    ts = [r.thickness_mm for r in results.values()
          if r.ok and r.thickness_mm > 0]
    if not ts:
        return 0.0
    t_mm = float(np.median(ts))
    for r in results.values():
        if not r.ok:
            continue
        r.thickness_mm = t_mm
        r.thickness_px = t_mm / mm_per_px if mm_per_px > 0 else 0.0
        if r.R_mm > 0:
            r.strain_pct = compute_strain(t_mm, r.R_mm)
    return t_mm


def export_graph(results: dict[int, FrameResult], fps: float,
                 png_path: str, title: str = "",
                 thickness_mm: float = 0.0, mm_per_px: float = 0.0) -> None:
    """Publication-style measurement graph -> PNG (200 dpi) + PDF.
    Top panel: strain vs time (peak annotated, fracture marked).
    Bottom panel: bend radius R and subtended angle vs time."""
    from matplotlib.figure import Figure

    rows = [results[k] for k in sorted(results)]
    t = np.array([r.frame_idx / fps for r in rows])
    strain = np.array([r.strain_pct if r.ok else np.nan for r in rows])
    R = np.array([r.R_mm if (r.ok and r.R_mm > 0) else np.nan for r in rows])
    theta = np.array([r.theta_deg if r.ok else np.nan for r in rows])
    frac = [r.frame_idx / fps for r in rows if r.status == "fractured"]

    fig = Figure(figsize=(9, 6.4), dpi=200)
    fig.patch.set_facecolor("white")
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212, sharex=ax1)

    for ax in (ax1, ax2):
        ax.set_facecolor("white")
        ax.grid(True, color="#e5e7eb", linewidth=0.7)
        for s in ax.spines.values():
            s.set_color("#94a3b8")
        ax.tick_params(colors="#334155", labelsize=8)

    ax1.plot(t, strain, color="#2563eb", linewidth=1.6)
    ax1.scatter(t, strain, s=8, color="#2563eb", zorder=3)
    ax1.set_ylabel("elastic strain ε (%)", fontsize=9, color="#0f172a")
    if np.isfinite(strain).any():
        pk = int(np.nanargmax(strain))
        ax1.annotate(f"peak {strain[pk]:.2f}%  (t = {t[pk]:.2f} s)",
                     xy=(t[pk], strain[pk]),
                     xytext=(t[pk], strain[pk] + np.nanmax(strain) * 0.08),
                     fontsize=8, color="#0f172a", ha="center",
                     arrowprops=dict(arrowstyle="-", color="#64748b", lw=0.8))
        ax1.set_ylim(0, np.nanmax(strain) * 1.25)
    if frac:
        for ax in (ax1, ax2):
            ax.axvspan(frac[0], t[-1], color="#fee2e2", zorder=0)
            ax.axvline(frac[0], color="#dc2626", linewidth=1.2,
                       linestyle="--")
        ax1.text(frac[0], ax1.get_ylim()[1] * 0.97, " fracture",
                 color="#dc2626", fontsize=8, va="top")

    ax2.plot(t, R, color="#059669", linewidth=1.6, label="bend radius R (mm)")
    ax2.set_ylabel("bend radius R (mm)", fontsize=9, color="#059669")
    ax2.set_xlabel("time (s)", fontsize=9, color="#0f172a")
    if np.isfinite(R).any():
        ax2.set_ylim(0, min(np.nanmax(R), np.nanmin(R) * 12) * 1.1)
    ax2b = ax2.twinx()
    ax2b.plot(t, theta, color="#d97706", linewidth=1.3, linestyle=":")
    ax2b.set_ylabel("subtended angle θ (°)", fontsize=9, color="#d97706")
    ax2b.tick_params(colors="#d97706", labelsize=8)
    for s in ax2b.spines.values():
        s.set_color("#94a3b8")

    sub = []
    if thickness_mm > 0:
        sub.append(f"t = {thickness_mm*1000:.1f} µm")
    if mm_per_px > 0:
        sub.append(f"scale {mm_per_px*1000:.3f} µm/px")
    n_ok = sum(1 for r in rows if r.ok)
    sub.append(f"{n_ok}/{len(rows)} frames tracked")
    fig.suptitle(title or "Elastic-strain measurement", fontsize=12,
                 color="#0f172a", y=0.98)
    ax1.set_title("   ·   ".join(sub), fontsize=8, color="#64748b")
    fig.subplots_adjust(left=0.09, right=0.91, top=0.90, bottom=0.09,
                        hspace=0.22)
    fig.savefig(png_path, dpi=200)
    fig.savefig(os.path.splitext(png_path)[0] + ".pdf")


def reveal_folder(path: str) -> None:
    """Open a folder in Finder / Explorer / the file manager."""
    try:
        if sys.platform == "darwin":
            import subprocess
            subprocess.Popen(["open", path])
        elif os.name == "nt":
            os.startfile(path)                       # type: ignore[attr-defined]
        else:
            import subprocess
            subprocess.Popen(["xdg-open", path])
    except Exception:
        pass


CSV_HEADER = ["frame", "t_s", "ok", "status", "R_mm", "D_mm", "L_mm",
              "h_max_mm", "theta_deg", "t_mm", "strain_pct", "fit_resid_px",
              "n_contacts", "arc_samples", "skel_samples"]


def _csv_row(fi: int, fps: float, res: FrameResult) -> list:
    return [fi, fi / fps, int(res.ok), res.status,
            res.R_mm, res.D_mm, res.L_mm, res.h_max_mm,
            res.theta_deg, res.thickness_mm, res.strain_pct,
            res.fit_resid_px, res.n_contacts,
            res.n_arc_samples, res.n_skel_samples]


# ----------------------------------------------------------------- batch mode
def run_batch(path: str) -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    src = MediaSource(path)
    mm_per_px = src.autocalibrate() or (1.0 / 487.0)
    print(f"{src.width}x{src.height} @{src.fps:.1f}fps  "
          f"{src.frame_count} frames  scale={mm_per_px:.6f} mm/px")
    params = TrackParams()

    # pass 1: analyze every frame (sequential decode), thickness locked after
    results: dict[int, FrameResult] = {}
    prev = None
    for fi in range(src.frame_count):
        frame = src.read(fi)
        if frame is None:
            break
        res = analyze_frame(frame, fi, mm_per_px, params, prev=prev)
        prev = res
        res.pack_mask()
        results[fi] = res
        if fi % 20 == 0:
            print(f"  analyze {fi}/{src.frame_count}")
    t_mm = lock_thickness(results, mm_per_px)
    print(f"locked thickness: {t_mm*1000:.1f} um")

    # pass 2: annotated video + CSV
    writer = None
    with open(os.path.join(OUT_DIR, "measurements.csv"), "w",
              newline="") as csv_f:
        cw = csv.writer(csv_f)
        cw.writerow(CSV_HEADER)
        for fi in sorted(results):
            frame = src.read(fi)
            if frame is None:
                break
            res = results[fi]
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
            cw.writerow(_csv_row(fi, src.fps, res))
            if fi % 20 == 0:
                print(f"  export {fi}/{src.frame_count}")
    if writer is not None:
        writer.release()
    try:
        export_graph(results, src.fps,
                     os.path.join(OUT_DIR, "strain_graph.png"),
                     title=f"Elastic-strain measurement — "
                           f"{os.path.basename(path)}",
                     thickness_mm=t_mm, mm_per_px=mm_per_px)
        print("wrote", OUT_DIR + "/strain_graph.png (+ .pdf)")
    except Exception as e:
        print("graph export skipped:", e)
    json.dump({"mm_per_px": mm_per_px, "media": path,
               "width": src.width, "height": src.height, "fps": src.fps,
               "thickness_mm": t_mm, "params": asdict(params)},
              open(os.path.join(OUT_DIR, "calibration.json"), "w"), indent=2)
    n_ok = sum(1 for r in results.values() if r.ok)
    print(f"tracked {n_ok}/{len(results)} frames")
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
                     font=(FONT, 11), bordercolor=BORDER,
                     lightcolor=BORDER, darkcolor=BORDER,
                     troughcolor=CARD_BG2, fieldbackground=CARD_BG2,
                     insertcolor=TEXT, selectbackground=ACCENT,
                     selectforeground="#ffffff")
        st.configure("App.TFrame", background=APP_BG)
        st.configure("Panel.TFrame", background=PANEL_BG)
        st.configure("Header.TFrame", background=HEADER_BG)
        st.configure("Card.TFrame", background=CARD_BG)
        st.configure("Canvas.TFrame", background=CANVAS_BG)

        st.configure("HeaderTitle.TLabel", background=HEADER_BG,
                     foreground=HEADER_FG, font=(FONT, 16, "bold"))
        st.configure("HeaderSub.TLabel", background=HEADER_BG,
                     foreground=MUTED, font=(FONT, 10))
        st.configure("CardTitle.TLabel", background=CARD_BG, foreground=MUTED,
                     font=(FONT, 9, "bold"))
        st.configure("Card.TLabel", background=CARD_BG, foreground=TEXT,
                     font=(FONT, 11))
        st.configure("CardMuted.TLabel", background=CARD_BG, foreground=MUTED,
                     font=(FONT, 10))
        st.configure("Metric.TLabel", background=CARD_BG, foreground=TEXT,
                     font=(MONO, 12, "bold"))
        st.configure("MetricBig.TLabel", background=CARD_BG, foreground=ACCENT,
                     font=(FONT, 26, "bold"))
        st.configure("Card.TCheckbutton", background=CARD_BG, foreground=TEXT,
                     font=(FONT, 10))
        st.map("Card.TCheckbutton",
               background=[("active", CARD_BG)],
               indicatorcolor=[("selected", ACCENT), ("!selected", CARD_BG2)])

        st.configure("TButton", font=(FONT, 11), padding=(12, 7),
                     background=CARD_BG, foreground=TEXT, borderwidth=0,
                     focuscolor=CARD_BG)
        st.map("TButton", background=[("active", CARD_BG2),
                                      ("disabled", "#16203a")],
               foreground=[("disabled", "#475569")])
        st.configure("Primary.TButton", font=(FONT, 11, "bold"),
                     padding=(16, 8), background=ACCENT, foreground="#ffffff",
                     borderwidth=0, focuscolor=ACCENT)
        st.map("Primary.TButton",
               background=[("active", ACCENT_ACTIVE), ("disabled", "#1e3a5f")],
               foreground=[("disabled", "#64748b")])
        st.configure("Ghost.TButton", font=(FONT, 11), padding=(12, 7),
                     background="#1a2440", foreground=HEADER_FG,
                     borderwidth=0, focuscolor="#1a2440")
        st.map("Ghost.TButton", background=[("active", "#243356")])
        st.configure("Icon.TButton", font=(FONT, 12), padding=(8, 5),
                     background=CARD_BG, foreground=TEXT, borderwidth=0,
                     focuscolor=CARD_BG)
        st.map("Icon.TButton", background=[("active", CARD_BG2)])

        st.configure("Strain.Horizontal.TProgressbar", troughcolor=CARD_BG,
                     background=ACCENT, borderwidth=0, thickness=8)
        st.configure("Horizontal.TScale", background=APP_BG,
                     troughcolor=CARD_BG, borderwidth=0,
                     lightcolor=APP_BG, darkcolor=APP_BG)
        st.map("Horizontal.TScale", background=[("active", APP_BG)])
        st.configure("TEntry", fieldbackground=CARD_BG2, foreground=TEXT,
                     bordercolor=BORDER, padding=4)

    # ---------------------------------------------------------- UI shell
    def _build_ui(self) -> None:
        # ---- header bar -------------------------------------------------
        header = ttk.Frame(self, style="Header.TFrame")
        header.pack(side="top", fill="x")
        logo = tk.Label(header, text="◈", bg=HEADER_BG, fg=ACCENT,
                        font=(FONT, 24, "bold"))
        logo.pack(side="left", padx=(18, 10), pady=10)
        htext = ttk.Frame(header, style="Header.TFrame")
        htext.pack(side="left", pady=10)
        trow = ttk.Frame(htext, style="Header.TFrame")
        trow.pack(anchor="w")
        ttk.Label(trow, text="Elastic-Strain Analyzer",
                  style="HeaderTitle.TLabel").pack(side="left")
        tk.Label(trow, text=f"  v{APP_VERSION}", bg=HEADER_BG, fg="#475569",
                 font=(FONT, 10)).pack(side="left", pady=(4, 0))
        self.header_sub = ttk.Label(
            htext, text="Three-point crystal bending  ·  Figure-S17 geometry",
            style="HeaderSub.TLabel")
        self.header_sub.pack(anchor="w")

        hbtn = ttk.Frame(header, style="Header.TFrame")
        hbtn.pack(side="right", padx=16, pady=12)
        self.open_btn = ttk.Button(hbtn, text="⬆  Open Video…",
                                   style="Ghost.TButton", command=self._on_open)
        self.open_btn.pack(side="left", padx=4)
        self.analyze_btn = ttk.Button(hbtn, text="▶  Analyze",
                                      style="Primary.TButton",
                                      command=self._on_analyze)
        self.analyze_btn.pack(side="left", padx=4)
        self.export_btn = ttk.Button(hbtn, text="⬇  Export Results…",
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
        ttk.Button(bar, text="⏮", width=3, style="Icon.TButton",
                   command=lambda: self.goto(0)).pack(side="left", padx=(0, 2))
        ttk.Button(bar, text="◀", width=3, style="Icon.TButton",
                   command=lambda: self.step(-1)).pack(side="left", padx=2)
        self.play_btn = ttk.Button(bar, text="▶  Play", width=9,
                                   style="Primary.TButton",
                                   command=self.toggle_play)
        self.play_btn.pack(side="left", padx=2)
        ttk.Button(bar, text="▶", width=3, style="Icon.TButton",
                   command=lambda: self.step(1)).pack(side="left", padx=2)
        ttk.Button(bar, text="⏭", width=3, style="Icon.TButton",
                   command=lambda: self.goto(10 ** 9)).pack(side="left",
                                                            padx=(2, 6))
        self.slider = ttk.Scale(bar, from_=0, to=1, orient="horizontal",
                                variable=self.frame_var,
                                command=self._on_slider)
        self.slider.pack(side="left", fill="x", expand=True, padx=10)
        self.slider.bind("<ButtonPress-1>", lambda e: self._set_fast(True))
        self.slider.bind("<ButtonRelease-1>", lambda e: self._set_fast(False))
        self.time_lbl = tk.Label(bar, text="0.00 s", bg=APP_BG, fg=MUTED,
                                 font=(MONO, 10), width=8, anchor="e")
        self.time_lbl.pack(side="left")
        fe = ttk.Entry(bar, width=6, textvariable=self.frame_entry_var,
                       justify="right")
        fe.pack(side="left", padx=(6, 0))
        fe.bind("<Return>", self._on_frame_entry)
        self.frame_lbl = tk.Label(bar, text="/ 0", bg=APP_BG, fg=MUTED,
                                  font=(MONO, 10), width=7, anchor="w")
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

    def goto(self, idx: int) -> None:
        if self.src is None:
            return
        idx = max(0, min(self.src.frame_count - 1, idx))
        self.frame_var.set(idx)
        self._show_frame(idx)

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
        self.status_chip = tk.Label(mc, text="no media", bg=CARD_BG2,
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
        ax.set_facecolor("#0b1120")
        for s in ax.spines.values():
            s.set_color(BORDER)
        ax.tick_params(colors=MUTED, labelsize=7)
        ax.set_xlabel("frame", color=MUTED, fontsize=8)
        ax.set_ylabel("strain %", color=MUTED, fontsize=8)
        ax.grid(True, color="#1c2a45", linewidth=0.8)
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
        self._set_chip("not analyzed", MUTED, CARD_BG2)
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
        self._set_chip("analyzing…", WARN, WARN_BG)
        self._set_status("Analyzing all frames…", ACCENT)
        threading.Thread(target=self._run_analysis, daemon=True).start()

    def _run_analysis(self) -> None:
        src = self.src.reopen()
        params = self._current_params()
        n = src.frame_count
        prev = None
        for fi in range(n):
            frame = src.read(fi)
            if frame is None:
                break
            res = analyze_frame(frame, fi, self.mm_per_px, params, prev=prev)
            prev = res
            res.pack_mask()          # 8x smaller in the cache
            self.result_cache[fi] = res
            if fi % 4 == 0 or fi == n - 1:
                self.after(0, self._analysis_progress, fi, n)
        lock_thickness(self.result_cache, self.mm_per_px)
        self.after(0, self._analysis_done)

    def _analysis_progress(self, fi: int, n: int) -> None:
        self.progress.configure(value=fi + 1)
        pct = int(100 * (fi + 1) / max(n, 1))
        self._set_status(f"Analyzing… {pct}%  ({fi + 1}/{n})", ACCENT)
        self._set_chip(f"analyzing {pct}%", WARN, WARN_BG)

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
        self._set_chip(f"analyzed · {ok}/{tot} tracked", SUCCESS, SUCCESS_BG)
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
        self.time_lbl.configure(text=f"{idx / self.src.fps:.2f} s")
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
            self._set_chip("scrubbing…", MUTED, CARD_BG2)
            return
        if res.ok:
            self.strain_big.configure(text=f"{res.strain_pct:.2f}%")
            m["R_mm"].configure(text=f"{res.R_mm:.3f} mm"
                                if res.R_mm > 0 else "∞ (straight)")
            m["D_mm"].configure(text=f"{res.D_mm:.3f} mm"
                                if res.R_mm > 0 else "—")
            m["L_mm"].configure(text=f"{res.L_mm:.3f} mm")
            m["h_mm"].configure(text=f"{res.h_max_mm:.3f} mm")
            m["theta"].configure(text=f"{res.theta_deg:.1f}°")
            m["t_mm"].configure(text=f"{res.thickness_mm * 1000:.1f} µm")
            m["resid"].configure(text=f"{res.fit_resid_px:.1f} px")
            if res.status == "straight":
                self._set_chip("straight (no bend)", INFO, INFO_BG)
            else:
                self._set_chip("tracked", SUCCESS, SUCCESS_BG)
        else:
            self.strain_big.configure(text="—")
            for k in ("R_mm", "D_mm", "L_mm", "h_mm", "theta", "t_mm", "resid"):
                m[k].configure(text="—")
            if res.status == "fractured":
                self._set_chip("crystal fractured", DANGER, DANGER_BG)
            else:
                self._set_chip(res.status or "no fit", DANGER, DANGER_BG)

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
        stem = os.path.splitext(os.path.basename(self.src.path))[0]
        initial = os.path.dirname(os.path.abspath(self.src.path))
        base = filedialog.askdirectory(
            title="Choose where to save the results folder",
            initialdir=initial)
        if not base:
            return
        outdir = os.path.join(base, f"{stem}_results")
        self.export_btn.configure(state="disabled")
        self._set_status("Exporting CSV + graph + annotated video…", ACCENT)
        self.progress.configure(value=0)
        threading.Thread(target=self._run_export, args=(outdir,),
                         daemon=True).start()

    def _run_export(self, outdir: str) -> None:
        try:
            os.makedirs(outdir, exist_ok=True)
            src = self.src.reopen()
            stem = os.path.splitext(os.path.basename(src.path))[0]
            writer = None
            params = self._current_params()
            results: dict[int, FrameResult] = {}
            csv_path = os.path.join(outdir, f"{stem}_measurements.csv")
            with open(csv_path, "w", newline="") as csv_f:
                cw = csv.writer(csv_f)
                cw.writerow(CSV_HEADER)
                prev = None
                for fi in range(src.frame_count):
                    frame = src.read(fi)
                    if frame is None:
                        break
                    res = (self.result_cache.get(fi)
                           if self.analyzed else None)
                    if res is None:
                        res = analyze_frame(frame, fi, self.mm_per_px, params,
                                            prev=prev)
                    prev = res
                    results[fi] = res
                    out = draw_overlay(frame, res, t_seconds=fi / src.fps)
                    if src.is_image:
                        cv2.imwrite(os.path.join(
                            outdir, f"{stem}_annotated.png"), out)
                    else:
                        if writer is None:
                            writer = cv2.VideoWriter(
                                os.path.join(outdir, f"{stem}_annotated.mp4"),
                                cv2.VideoWriter_fourcc(*"mp4v"), src.fps,
                                (src.width, src.height))
                        writer.write(out)
                    cw.writerow(_csv_row(fi, src.fps, res))
                    self.after(0, lambda v=fi: self.progress.configure(value=v))
            if writer is not None:
                writer.release()
            t_mm = next((r.thickness_mm for r in results.values()
                         if r.ok and r.thickness_mm > 0), 0.0)
            if _HAVE_MPL and results:
                export_graph(results, src.fps,
                             os.path.join(outdir, f"{stem}_strain_graph.png"),
                             title=f"Elastic-strain measurement — "
                                   f"{os.path.basename(src.path)}",
                             thickness_mm=t_mm, mm_per_px=self.mm_per_px)
            json.dump({"mm_per_px": self.mm_per_px, "media": self.src.path,
                       "width": self.src.width, "height": self.src.height,
                       "fps": self.src.fps, "thickness_mm": t_mm,
                       "params": asdict(self._current_params())},
                      open(os.path.join(outdir, f"{stem}_calibration.json"),
                           "w"), indent=2)
            self.after(0, lambda: self._set_status(
                f"Exported → {outdir}  (CSV, strain graph PNG+PDF, "
                f"annotated video)", SUCCESS))
            self.after(0, lambda: reveal_folder(outdir))
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
