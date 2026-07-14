"""Three-point crystal-bend tracker (Figure S17 geometry).

Pipeline
--------
1.  Crop the black letterbox to the content rectangle.
2.  Detect the METALWARE: large near-black wedges entering from the image
    border.  Each wedge's TIP is the blob point farthest from the border(s)
    it enters through.
3.  Segment the CRYSTAL as a *thin dark line*: a morphological black-hat
    (responds to thin dark structures, not to large dark blobs or blurred
    tweezer-edge halos) unioned with a gold-colour cue (R - B), minus the
    dilated metal mask and the scale-bar region, then filtered to long,
    thin connected components.
4.  Contacts: for every metalware tip, the nearest crystal-centreline point
    within `contact_dist_px` is a contact.  The two farthest-apart contacts
    are the SUPPORTS -> chord A-B (length L).  The remaining tip is the
    PROBE pushing from the opposite side.
5.  Sagitta h_max = the largest perpendicular deflection of the centreline
    from chord A-B, taken between the supports on the bulge side (this is
    the crystal point at the probe contact, per Figure S17).
6.  Closed-form geometry, exactly as the paper:
        R     = ((L/2)^2 + h_max^2) / (2 h_max)          [eq. 2]
        theta = 2 asin(L / (2 R))                         (subtended angle)
        eps   = t / (2 R) * 100                           [eq. 3]
    with the crystal thickness t = 2 x median distance-transform value
    along the skeleton.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np


# ----------------------------------------------------------------- parameters
@dataclass
class TrackParams:
    """Tunable knobs for analyze_frame.  Defaults calibrated for microscope
    footage like test.mp4 (2908x2160, gold crystal, dark tweezers)."""

    # -- crystal segmentation (thin dark line) -------------------------------
    blackhat_kernel_px: int = 31    # slightly wider than the crystal line
    blackhat_thresh: int = 26       # min black-hat response for a crystal px
    yellow_rb_min: int = 15         # min (R - B) for the gold-colour cue
    crystal_v_lo: int = 40          # darker than this is metal, never crystal
    crystal_v_hi: int = 215         # brighter than this is background
    min_component_px: int = 350     # smaller connected components are specks
    min_extent_px: float = 80.0     # min bbox diagonal of a crystal piece
    max_thickness_px: float = 30.0  # area / bbox-diagonal above this = blob
    min_chain_len: int = 60         # skeleton samples for a significant chain

    # -- metalware (tweezers / probe) -----------------------------------------
    metal_gray_max: int = 90        # pixels darker than this are metalware
    min_metal_area: int = 25000     # min blob area (px) for a real tweezer
    metal_margin_px: int = 5        # metal dilation excluded from the crystal

    # -- contacts & geometry ---------------------------------------------------
    contact_dist_px: float = 90.0   # max tip-to-centreline gap that = touching
    max_rms_frac: float = 0.12      # fit RMS above this fraction of R means
                                    # the shape is no single arc (fracture)
    straight_frac: float = 0.004    # h/L below this = effectively straight
    min_chord_px: float = 200.0     # reject implausibly short chords
    max_strain_pct: float = 25.0    # reject non-physical strains
    min_arc_samples: int = 25       # min centreline samples to measure at all

    # Restrict analysis to this ROI (x, y, w, h) in full-frame pixel coords.
    roi: Optional[tuple] = None


# ----------------------------------------------------------------- letterbox
def content_rect(gray: np.ndarray) -> tuple[int, int, int, int]:
    """(x, y, w, h) of the non-letterbox content (black bars stripped)."""
    H, W = gray.shape[:2]
    g = cv2.resize(gray, (max(W // 8, 1), max(H // 8, 1)),
                   interpolation=cv2.INTER_AREA)
    rows = np.where(g.mean(axis=1) > 12)[0]
    cols = np.where(g.mean(axis=0) > 12)[0]
    if rows.size == 0 or cols.size == 0:
        return 0, 0, W, H
    m = 1                                    # one downsampled px inward
    y0 = min(int((rows[0] + m) * 8), H - 1)
    y1 = max(int((rows[-1] + 1 - m) * 8), y0 + 1)
    x0 = min(int((cols[0] + m) * 8), W - 1)
    x1 = max(int((cols[-1] + 1 - m) * 8), x0 + 1)
    return x0, y0, min(x1, W) - x0, min(y1, H) - y0


# ----------------------------------------------------------------- metalware
def detect_metalware(gray: np.ndarray, params: TrackParams
                     ) -> tuple[np.ndarray, list[np.ndarray]]:
    """(metal_mask, tips) - the dark tweezers entering from the image border.

    The tip of each wedge is the blob region farthest from the border(s) the
    wedge enters through (e.g. a wedge entering from the left border has its
    tip at the largest x)."""
    H, W = gray.shape[:2]
    dark = (gray < params.metal_gray_max).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, k)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(dark, 8)
    metal = np.zeros_like(dark)
    tips: list[np.ndarray] = []
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] < params.min_metal_area:
            continue
        x, y, w, h = (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
                      stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT])
        blob = labels[y:y + h, x:x + w] == i
        metal[y:y + h, x:x + w][blob] = 255
        edges = []
        if x <= 2:
            edges.append("left")
        if y <= 2:
            edges.append("top")
        if x + w >= W - 2:
            edges.append("right")
        if y + h >= H - 2:
            edges.append("bottom")
        if not edges:                       # scale bar etc. - excluder only
            continue
        ys, xs = np.nonzero(blob)
        xs = xs + x
        ys = ys + y
        d = np.full(xs.shape, np.inf)
        for e in edges:
            de = {"left": xs, "right": W - 1 - xs,
                  "top": ys, "bottom": H - 1 - ys}[e]
            d = np.minimum(d, de.astype(np.float64))
        sel = d >= d.max() - 12.0           # the 12-px cap of the wedge
        tips.append(np.array([xs[sel].mean(), ys[sel].mean()]))
    return metal, tips


# ----------------------------------------------------------------- scale bar
def detect_scale_bar_rect(frame: np.ndarray
                          ) -> Optional[tuple[float, tuple[int, int, int, int]]]:
    """Find the solid black scale bar in the lower-right of the field.
    Returns (mm_per_px assuming a 1-mm bar, (x, y, w, h) in frame coords)."""
    h, w = frame.shape[:2]
    oy, ox = int(h * 0.75), int(w * 0.60)
    roi = frame[oy:, ox:]
    gray = roi if roi.ndim == 2 else cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rh, rw = bw.shape[:2]
    best: Optional[tuple[int, tuple]] = None
    for c in cnts:
        x, y, cw, ch = cv2.boundingRect(c)
        if x <= 2 or y <= 2 or x + cw >= rw - 2 or y + ch >= rh - 2:
            continue                     # letterbox / tweezer entering the ROI
        area = cv2.contourArea(c)
        fill = area / max(cw * ch, 1)
        if cw > 80 and 6 < ch < cw * 0.3 and fill > 0.85:
            if best is None or cw > best[0]:
                best = (cw, (x + ox, y + oy, cw, ch))
    if best is None:
        return None
    return 1.0 / best[0], best[1]


def detect_scale_bar(frame: np.ndarray) -> float | None:
    """mm-per-pixel from the '1 mm' bar, or None (back-compat wrapper)."""
    r = detect_scale_bar_rect(frame)
    return r[0] if r else None


# ----------------------------------------------------------------- segmentation
def segment_crystal(bgr: np.ndarray, gray: np.ndarray, metal: np.ndarray,
                    params: TrackParams,
                    exclude_rect: Optional[tuple] = None) -> np.ndarray:
    """Binary mask of the crystal: thin dark line (black-hat) OR gold colour,
    minus metalware, scale bar and blob-like components."""
    k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (params.blackhat_kernel_px, params.blackhat_kernel_px))
    bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k)
    cand = (bh > params.blackhat_thresh)

    b, _, r = cv2.split(bgr.astype(np.int16))
    cand |= ((r - b) >= params.yellow_rb_min) & (gray < params.crystal_v_hi)

    cand &= gray >= params.crystal_v_lo                 # never metal-dark
    mask = cand.astype(np.uint8) * 255

    md = cv2.dilate(metal, cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (params.metal_margin_px * 2 + 1, params.metal_margin_px * 2 + 1)))
    mask[md > 0] = 0

    # printed overlays (scale bar, "1 mm" text) are near-black blobs whose
    # antialiased outlines would otherwise leak into the crystal mask -
    # blank every sizeable very-dark blob plus a margin
    ink = (gray < 60).astype(np.uint8) * 255
    num_i, lab_i, st_i, _ = cv2.connectedComponentsWithStats(ink, 8)
    stamp = np.zeros_like(ink)
    for i in range(1, num_i):
        if st_i[i, cv2.CC_STAT_AREA] >= 800:
            x, y, w, h = (st_i[i, cv2.CC_STAT_LEFT], st_i[i, cv2.CC_STAT_TOP],
                          st_i[i, cv2.CC_STAT_WIDTH],
                          st_i[i, cv2.CC_STAT_HEIGHT])
            stamp[y:y + h, x:x + w][lab_i[y:y + h, x:x + w] == i] = 255
    if stamp.any():
        stamp = cv2.dilate(stamp, cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (17, 17)))
        mask[stamp > 0] = 0

    if exclude_rect is not None:
        ex, ey, ew, eh = exclude_rect
        mask[max(ey, 0):ey + eh, max(ex, 0):ex + ew] = 0

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    lut = np.zeros(num, np.uint8)
    for i in range(1, num):
        a = int(stats[i, cv2.CC_STAT_AREA])
        if a < params.min_component_px:
            continue
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        diag = float(np.hypot(w, h))
        if diag < params.min_extent_px:                 # tiny speck
            continue
        if a / diag > params.max_thickness_px:          # blobby, not a line
            continue
        lut[i] = 255
    return lut[labels]


# alias kept for older scripts
def segment_needle(frame: np.ndarray, params: TrackParams) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    metal, _ = detect_metalware(gray, params)
    return segment_crystal(frame, gray, metal, params)


# ----------------------------------------------------------------- skeleton
def skeletonize(mask: np.ndarray) -> np.ndarray:
    """One-pixel-wide centreline (cropped to the foreground bbox for speed)."""
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return np.zeros_like(mask)
    m = 2
    x0 = max(int(xs.min()) - m, 0)
    y0 = max(int(ys.min()) - m, 0)
    x1 = min(int(xs.max()) + m + 1, mask.shape[1])
    y1 = min(int(ys.max()) + m + 1, mask.shape[0])
    crop = mask[y0:y1, x0:x1]
    skel_crop = _skeletonize_impl(crop)
    out = np.zeros_like(mask)
    out[y0:y1, x0:x1] = skel_crop
    return out


def _skeletonize_impl(mask: np.ndarray) -> np.ndarray:
    try:
        import cv2.ximgproc as xip  # type: ignore
        return xip.thinning(mask, thinningType=xip.THINNING_ZHANGSUEN)
    except Exception:
        pass
    try:
        from skimage.morphology import skeletonize as _sk_skel  # type: ignore
        return _sk_skel(mask > 0).astype(np.uint8) * 255
    except Exception:
        skel = np.zeros_like(mask)
        m = mask.copy()
        k = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        while cv2.countNonZero(m):
            opened = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
            skel = cv2.bitwise_or(skel, cv2.subtract(m, opened))
            m = cv2.erode(m, k)
        return skel


# ----------------------------------------------------------------- ordering
def _longest_path(pts: np.ndarray) -> np.ndarray:
    """Ordered longest path (graph diameter) through one skeleton component,
    using 8-connected pixel adjacency and a double BFS.  Side spurs shorter
    than the main line are pruned automatically."""
    n = len(pts)
    if n < 3:
        return pts.copy()
    index = {(int(x), int(y)): i for i, (x, y) in enumerate(pts)}
    nbrs: list[list[int]] = [[] for _ in range(n)]
    for i, (x, y) in enumerate(pts):
        x = int(x); y = int(y)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                j = index.get((x + dx, y + dy))
                if j is not None:
                    nbrs[i].append(j)

    def _bfs(src: int) -> tuple[int, np.ndarray]:
        parent = np.full(n, -1, np.int64)
        seen = np.zeros(n, bool)
        seen[src] = True
        frontier = [src]
        last = src
        while frontier:
            nxt: list[int] = []
            for u in frontier:
                for v in nbrs[u]:
                    if not seen[v]:
                        seen[v] = True
                        parent[v] = u
                        nxt.append(v)
            if nxt:
                last = nxt[-1]
            frontier = nxt
        return last, parent

    u, _ = _bfs(0)
    v, parent = _bfs(u)
    path = [v]
    while parent[path[-1]] != -1:
        path.append(int(parent[path[-1]]))
    return pts[path[::-1]]


def order_components(skel: np.ndarray) -> list[np.ndarray]:
    """Ordered (x, y) point chains, one per skeleton connected component.
    Each chain is the component's longest path (spurs pruned)."""
    num, labels = cv2.connectedComponents(skel, connectivity=8)
    if num <= 1:
        return []
    ys, xs = np.nonzero(labels)
    labs = labels[ys, xs]
    chains: list[np.ndarray] = []
    for i in range(1, num):
        sel = labs == i
        if int(sel.sum()) < 5:
            continue
        chain = _longest_path(np.column_stack([xs[sel], ys[sel]]))
        if len(chain) >= 5:
            chains.append(chain)
    return chains


def _end_dir(seg: np.ndarray, end: str, k: int = 12) -> np.ndarray:
    """Unit OUTWARD tangent at one end of an ordered chain."""
    k = min(k, len(seg) - 1)
    if k < 1:
        return np.zeros(2)
    v = (seg[-1] - seg[-1 - k]) if end == "tail" else (seg[0] - seg[k])
    n = float(np.hypot(v[0], v[1]))
    return v / n if n > 0 else np.zeros(2)


def _extend_tail(merged: np.ndarray, segs: list[np.ndarray],
                 max_gap: float, cos_thr: float, k: int = 12
                 ) -> tuple[np.ndarray, bool]:
    """Append the nearest tangent-continuous fragment onto the tail (bridges
    probe/metalware occlusion but never folds back across the bend mouth)."""
    tdir = _end_dir(merged, "tail", k)
    tail = merged[-1]
    best = None
    for i, s in enumerate(segs):
        for oriented in (s, s[::-1]):
            e0 = oriented[0]
            gv = e0 - tail
            gap = float(np.hypot(gv[0], gv[1]))
            if gap > max_gap:
                continue
            kk = min(k, len(oriented) - 1)
            fv = oriented[kk] - oriented[0]
            fn = float(np.hypot(fv[0], fv[1]))
            fdir = fv / fn if fn > 0 else np.zeros(2)
            if gap > 3.0 and float(np.dot(gv / gap, tdir)) < cos_thr:
                continue
            if float(np.dot(fdir, tdir)) < cos_thr:
                continue
            if best is None or gap < best[0]:
                best = (gap, i, oriented)
    if best is None:
        return merged, False
    _, i, oriented = best
    segs.pop(i)
    return np.vstack([merged, oriented]), True


def merge_centerline(chains: list[np.ndarray], max_gap: float = 400.0,
                     cos_thr: float = 0.2) -> Optional[np.ndarray]:
    """Stitch ordered chains into one tip-to-tip centreline."""
    segs = [c.astype(np.float64) for c in chains if len(c) >= 2]
    if not segs:
        return None
    merged = segs.pop(int(np.argmax([len(s) for s in segs])))
    changed = True
    while segs and changed:
        merged, c1 = _extend_tail(merged, segs, max_gap, cos_thr)
        merged = merged[::-1]
        merged, c2 = _extend_tail(merged, segs, max_gap, cos_thr)
        merged = merged[::-1]
        changed = c1 or c2
    return merged


# ----------------------------------------------------------------- geometry
def chord_sagitta_radius(L: float, h_max: float) -> float:
    """Radius of the arc with chord ``L`` and sagitta ``h_max`` (paper eq. 2):
    R = ((L/2)^2 + h_max^2) / (2 h_max).  inf for a straight (h=0) crystal."""
    if h_max <= 0.0:
        return float("inf")
    return ((L * 0.5) ** 2 + h_max ** 2) / (2.0 * h_max)


def compute_strain(t: float, R: float) -> float:
    """Elastic bending strain (%), paper eq. 3: eps = t / (2 R) * 100."""
    if R <= 0.0 or not np.isfinite(R):
        return 0.0
    return (t / (2.0 * R)) * 100.0


def measure_thickness_fwhm(gray: np.ndarray, chains: list[np.ndarray],
                           n_samples: int = 160, half_len: int = 20
                           ) -> float:
    """Crystal thickness (px) as the median full-width-at-half-depth of the
    dark line profile, sampled perpendicular to the centreline.  Measures the
    *undeformed optical width* on the raw image, so it is unaffected by the
    dilation of the segmentation mask."""
    H, W = gray.shape[:2]
    samples: list[tuple[float, float, float, float]] = []   # x, y, nx, ny
    total = sum(len(c) for c in chains)
    if total == 0:
        return 0.0
    stride_all = max(total // n_samples, 1)
    for c in chains:
        c = np.asarray(c, np.float64)
        if len(c) < 11:
            continue
        for i in range(5, len(c) - 5, stride_all):
            t = c[i + 5] - c[i - 5]
            n = float(np.hypot(t[0], t[1]))
            if n < 1e-6:
                continue
            samples.append((c[i][0], c[i][1], -t[1] / n, t[0] / n))
    if not samples:
        return 0.0
    S = np.asarray(samples)
    offs = np.arange(-half_len, half_len + 1, dtype=np.float32)
    map_x = (S[:, 0:1] + S[:, 2:3] * offs).astype(np.float32)
    map_y = (S[:, 1:2] + S[:, 3:4] * offs).astype(np.float32)
    prof = cv2.remap(gray, map_x, map_y, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE).astype(np.float64)
    widths: list[float] = []
    mid = half_len
    for row in prof:
        bg = min(float(np.median(row[:4])), float(np.median(row[-4:])))
        lo = float(row.min())
        depth = bg - lo
        if depth < 25.0:                    # no real dark line under this pt
            continue
        half_level = bg - 0.5 * depth
        below = row < half_level
        i0 = int(np.argmin(row))
        if not below[i0]:
            continue
        a = i0
        while a > 0 and below[a - 1]:
            a -= 1
        b = i0
        while b < len(row) - 1 and below[b + 1]:
            b += 1
        if a == 0 or b == len(row) - 1:     # clipped: neighbouring structure
            continue
        widths.append(float(b - a + 1))
    if len(widths) < 8:
        return 0.0
    return float(np.median(widths))


def estimate_thickness(mask: np.ndarray, chains: list[np.ndarray]) -> float:
    """Crystal thickness (px) = 2 x median distance-transform value sampled
    along the skeleton (robust to fragmentation and stray components)."""
    if not chains:
        return 0.0
    dt = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    pts = np.concatenate(chains, axis=0)
    vals = dt[pts[:, 1].astype(int), pts[:, 0].astype(int)]
    vals = vals[vals > 0]
    if vals.size == 0:
        return 0.0
    return float(2.0 * np.median(vals))


def _farthest_pair(pts: np.ndarray) -> tuple[int, int]:
    """Indices of the two farthest-apart points (pts is small)."""
    best = (-1.0, 0, len(pts) - 1)
    for i in range(len(pts)):
        d = np.hypot(pts[:, 0] - pts[i, 0], pts[:, 1] - pts[i, 1])
        j = int(np.argmax(d))
        if d[j] > best[0]:
            best = (float(d[j]), i, j)
    return best[1], best[2]


def _signed_perp(P: np.ndarray, A: np.ndarray, AB: np.ndarray, L: float
                 ) -> np.ndarray:
    """Signed perpendicular distance of point(s) P from the line through A
    along AB (positive on one side, negative on the other)."""
    P = np.atleast_2d(P)
    return (AB[0] * (P[:, 1] - A[1]) - AB[1] * (P[:, 0] - A[0])) / max(L, 1e-9)


# ----------------------------------------------------------------- result
@dataclass
class FrameResult:
    ok: bool
    frame_idx: int
    status: str = ""                 # tracked | straight | no-crystal | ...
    needle_px: int = 0
    mask: Optional[np.ndarray] = None
    arc_pts: Optional[np.ndarray] = None        # centreline samples (global)
    skeleton_pts: Optional[np.ndarray] = None
    cx: float = 0.0
    cy: float = 0.0
    R_px: float = 0.0
    L_px: float = 0.0
    h_max_px: float = 0.0
    theta_deg: float = 0.0
    thickness_px: float = 0.0
    fit_resid_px: float = 0.0
    chord_p1: tuple[int, int] = (0, 0)
    chord_p2: tuple[int, int] = (0, 0)
    apex: tuple[int, int] = (0, 0)
    apex_foot: tuple[int, int] = (0, 0)
    R_mm: float = 0.0
    D_mm: float = 0.0
    L_mm: float = 0.0
    h_max_mm: float = 0.0
    thickness_mm: float = 0.0
    strain_pct: float = 0.0
    n_arc_samples: int = 0
    n_skel_samples: int = 0
    n_contacts: int = 0
    contacts: list = field(default_factory=list)     # crystal contact points
    tips: list = field(default_factory=list)         # metalware tip points
    probe_tip: Optional[tuple] = None
    mask_packed: Optional[np.ndarray] = None         # np.packbits(mask)
    mask_shape: Optional[tuple] = None

    def pack_mask(self) -> None:
        """Compress the full-res mask 8x (for caching many frames)."""
        if self.mask is not None:
            self.mask_shape = self.mask.shape
            self.mask_packed = np.packbits(self.mask > 0)
            self.mask = None

    def get_mask(self) -> Optional[np.ndarray]:
        if self.mask is not None:
            return self.mask
        if self.mask_packed is None or self.mask_shape is None:
            return None
        n = int(np.prod(self.mask_shape))
        return (np.unpackbits(self.mask_packed, count=n)
                .reshape(self.mask_shape) * 255).astype(np.uint8)


# ----------------------------------------------------------------- main entry
def analyze_frame(frame: np.ndarray, frame_idx: int, mm_per_px: float,
                  params: Optional[TrackParams] = None,
                  prev: Optional[FrameResult] = None) -> FrameResult:
    """Run the full pipeline on one frame.  All px values are in *original
    frame* coordinates.  ``prev`` (the previous frame's result) is used only
    to reject implausible support-contact jumps."""
    p = params or TrackParams()
    H, W = frame.shape[:2]

    gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cx0, cy0, cw, ch = content_rect(gray_full)
    if p.roi is not None:
        rx, ry, rw, rh = [int(v) for v in p.roi]
        nx0 = max(cx0, rx)
        ny0 = max(cy0, ry)
        nx1 = min(cx0 + cw, rx + rw)
        ny1 = min(cy0 + ch, ry + rh)
        if nx1 - nx0 > 50 and ny1 - ny0 > 50:
            cx0, cy0, cw, ch = nx0, ny0, nx1 - nx0, ny1 - ny0
    off = np.array([cx0, cy0], dtype=np.float64)

    sub = frame[cy0:cy0 + ch, cx0:cx0 + cw]
    gray = gray_full[cy0:cy0 + ch, cx0:cx0 + cw]

    # ---- metalware + scale bar ------------------------------------------------
    metal, tips = detect_metalware(gray, p)
    bar = detect_scale_bar_rect(gray)
    exclude = None
    if bar is not None:
        _, (bx, by, bw_, bh_) = bar
        # pad the bar bbox to also cover the "1 mm" text above it
        exclude = (bx - bh_ * 2, by - bh_ * 4, bw_ + bh_ * 4, bh_ * 6)

    # ---- crystal ---------------------------------------------------------------
    mask_sub = segment_crystal(sub, gray, metal, p, exclude)
    npx = int(np.count_nonzero(mask_sub))

    mask = np.zeros((H, W), np.uint8)
    mask[cy0:cy0 + ch, cx0:cx0 + cw] = mask_sub

    def _fail(status: str, **kw) -> FrameResult:
        return FrameResult(ok=False, frame_idx=frame_idx, status=status,
                           needle_px=npx, mask=mask,
                           tips=[(float(t[0] + cx0), float(t[1] + cy0))
                                 for t in tips], **kw)

    if npx < 200:
        return _fail("no-crystal")

    skel = skeletonize(mask_sub)
    chains = order_components(skel)
    if not chains:
        return _fail("no-crystal")
    largest = max(len(c) for c in chains)
    sig = [c for c in chains
           if len(c) >= min(p.min_chain_len, max(int(0.2 * largest), 10))]
    if not sig:
        sig = [max(chains, key=len)]
    pts = np.concatenate(sig, axis=0).astype(np.float64)
    skel_g = pts + off
    if len(pts) < p.min_arc_samples:
        return _fail("no-crystal", skeleton_pts=skel_g,
                     n_skel_samples=len(pts))

    # ---- contacts ---------------------------------------------------------------
    contacts: list[tuple[np.ndarray, np.ndarray]] = []   # (crystal pt, tip)
    for tip in tips:
        d = np.hypot(pts[:, 0] - tip[0], pts[:, 1] - tip[1])
        j = int(np.argmin(d))
        if d[j] <= p.contact_dist_px:
            contacts.append((pts[j], tip))

    # single stitched centreline (bridges probe occlusion, refuses to fold
    # back across a fracture, where the two pieces meet at a sharp angle)
    merged = merge_centerline(sig)

    probe_tip: Optional[np.ndarray] = None
    if len(contacts) >= 2:
        cpts = np.array([c[0] for c in contacts])
        i, j = _farthest_pair(cpts)
        A, B = cpts[i].copy(), cpts[j].copy()
        for k_, (c_, t_) in enumerate(contacts):
            if k_ not in (i, j):
                probe_tip = t_
    else:
        # after a detected fracture, the relaxed loose pieces must not be
        # measured as if they were the loaded crystal
        if prev is not None and prev.status == "fractured":
            return _fail("fractured", skeleton_pts=skel_g,
                         n_skel_samples=len(pts))
        if merged is None or len(merged) < p.min_arc_samples:
            return _fail("no-crystal", skeleton_pts=skel_g,
                         n_skel_samples=len(pts))
        A, B = merged[0].copy(), merged[-1].copy()

    AB = B - A
    L = float(np.hypot(AB[0], AB[1]))
    if L < 1.0:
        return _fail("no-chord", skeleton_pts=skel_g, n_skel_samples=len(pts))

    # ---- sagitta: max deflection between the supports --------------------------
    perp = _signed_perp(pts, A, AB, L)
    tpar = ((pts[:, 0] - A[0]) * AB[0] + (pts[:, 1] - A[1]) * AB[1]) / (L * L)
    inside = (tpar > 0.01) & (tpar < 0.99)
    if not inside.any():
        return _fail("no-arc", skeleton_pts=skel_g, n_skel_samples=len(pts))

    # bulge side = the side of the chord where the crystal actually deflects
    # (the probe tip overlaps the crystal, so its own side is unreliable)
    side = float(np.sign(perp[inside][int(np.argmax(np.abs(perp[inside])))])) or 1.0

    defl = perp * side
    cand_i = np.where(inside & (defl > 0))[0]
    thickness_px = (measure_thickness_fwhm(gray, sig)
                    or estimate_thickness(mask_sub, sig))

    if cand_i.size == 0 or float(defl[cand_i].max()) / L < p.straight_frac:
        # effectively straight - report zero strain rather than failing
        h_px = float(defl[cand_i].max()) if cand_i.size else 0.0
        return FrameResult(
            ok=True, frame_idx=frame_idx, status="straight",
            needle_px=npx, mask=mask,
            arc_pts=skel_g.astype(np.int32), skeleton_pts=skel_g,
            L_px=L, h_max_px=h_px, thickness_px=thickness_px,
            chord_p1=(int(A[0] + cx0), int(A[1] + cy0)),
            chord_p2=(int(B[0] + cx0), int(B[1] + cy0)),
            L_mm=L * mm_per_px, h_max_mm=h_px * mm_per_px,
            thickness_mm=thickness_px * mm_per_px,
            n_arc_samples=len(pts), n_skel_samples=len(pts),
            n_contacts=len(contacts),
            contacts=[(float(c[0][0] + cx0), float(c[0][1] + cy0))
                      for c in contacts],
            tips=[(float(t[0] + cx0), float(t[1] + cy0)) for t in tips],
        )

    apex_i = cand_i[int(np.argmax(defl[cand_i]))]
    h_px = float(defl[apex_i])
    apex = pts[apex_i]
    foot = A + float(tpar[apex_i]) * AB

    # ---- closed-form circle (paper eq. 1/2) -------------------------------------
    R_px = chord_sagitta_radius(L, h_px)
    theta = 2.0 * float(np.arcsin(min(1.0, (L * 0.5) / R_px)))
    M = 0.5 * (A + B)
    n_vec = np.array([-AB[1], AB[0]]) / L
    if np.dot(apex - M, n_vec) > 0:
        n_vec = -n_vec
    centre = M + n_vec * (R_px - h_px)

    arc_sel = inside & (defl > -0.02 * L)
    arc_pts = pts[arc_sel]
    rms = float(np.sqrt(np.mean(
        (np.hypot(arc_pts[:, 0] - centre[0], arc_pts[:, 1] - centre[1])
         - R_px) ** 2))) if len(arc_pts) else 0.0

    # a bent-but-intact crystal deviates only mildly from the bend circle;
    # a broken one (two pieces meeting at a cusp) deviates massively
    if len(arc_pts) and rms > p.max_rms_frac * R_px:
        return _fail("fractured", skeleton_pts=skel_g,
                     n_skel_samples=len(pts))

    R_mm = R_px * mm_per_px
    t_mm = thickness_px * mm_per_px
    strain = compute_strain(t_mm, R_mm)

    ok = (L >= p.min_chord_px) and (strain <= p.max_strain_pct)
    status = "tracked" if ok else \
        ("short-chord" if L < p.min_chord_px else "bad-strain")

    return FrameResult(
        ok=ok, frame_idx=frame_idx, status=status,
        needle_px=npx, mask=mask,
        arc_pts=(arc_pts + off).astype(np.int32),
        skeleton_pts=skel_g,
        cx=float(centre[0] + cx0), cy=float(centre[1] + cy0), R_px=R_px,
        L_px=L, h_max_px=h_px, theta_deg=float(np.degrees(theta)),
        thickness_px=thickness_px, fit_resid_px=rms,
        chord_p1=(int(A[0] + cx0), int(A[1] + cy0)),
        chord_p2=(int(B[0] + cx0), int(B[1] + cy0)),
        apex=(int(apex[0] + cx0), int(apex[1] + cy0)),
        apex_foot=(int(foot[0] + cx0), int(foot[1] + cy0)),
        R_mm=R_mm, D_mm=2.0 * R_mm, L_mm=L * mm_per_px,
        h_max_mm=h_px * mm_per_px, thickness_mm=t_mm, strain_pct=strain,
        n_arc_samples=len(arc_pts), n_skel_samples=len(pts),
        n_contacts=len(contacts),
        contacts=[(float(c[0][0] + cx0), float(c[0][1] + cy0))
                  for c in contacts],
        tips=[(float(t[0] + cx0), float(t[1] + cy0)) for t in tips],
        probe_tip=(None if probe_tip is None
                   else (float(probe_tip[0] + cx0), float(probe_tip[1] + cy0))),
    )


# ----------------------------------------------------------------- overlay
def _draw_dashed_circle(img: np.ndarray, center: tuple[int, int], radius: int,
                        color: tuple[int, int, int], thickness: int = 4,
                        dash_deg: int = 10, gap_deg: int = 6) -> None:
    cx, cy = center
    a = 0
    while a < 360:
        end = min(a + dash_deg, 360)
        cv2.ellipse(img, (cx, cy), (radius, radius), 0, a, end,
                    color, thickness, cv2.LINE_AA)
        a = end + gap_deg


def draw_overlay(frame: np.ndarray, res: FrameResult, *,
                 show_mask: bool = True,
                 show_circle: bool = True,
                 show_chord: bool = True,
                 show_arc_pts: bool = True,
                 show_panel: bool = True,
                 t_seconds: float | None = None) -> np.ndarray:
    """Draw the analysis result on top of the frame for the GUI / export."""
    out = frame.copy()
    th = max(2, frame.shape[0] // 900)

    mask = res.get_mask()
    if mask is not None and show_mask:
        tint = np.zeros_like(out)
        tint[mask > 0] = (0, 255, 255)
        out = cv2.addWeighted(out, 1.0, tint, 0.45, 0)

    # metalware tips (always useful for QA)
    for t in res.tips:
        cv2.drawMarker(out, (int(t[0]), int(t[1])), (255, 80, 255),
                       cv2.MARKER_TILTED_CROSS, 26, th, cv2.LINE_AA)

    if not res.ok and res.status in ("no-crystal", "no-chord", "no-arc",
                                     "fractured"):
        msg = ("CRYSTAL FRACTURED" if res.status == "fractured"
               else f"{res.status} ({res.needle_px} px)")
        cv2.putText(out, f"frame {res.frame_idx}: {msg}",
                    (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.4,
                    (0, 0, 255), 3, cv2.LINE_AA)
        return out

    if show_arc_pts and res.arc_pts is not None and len(res.arc_pts):
        for x, y in res.arc_pts[::3]:
            cv2.circle(out, (int(x), int(y)), 2, (0, 255, 0), -1, cv2.LINE_AA)

    straight = res.status == "straight"

    if show_circle and not straight and res.R_px > 0 \
            and res.R_px < 4 * max(frame.shape):
        center = (int(round(res.cx)), int(round(res.cy)))
        radius = int(round(res.R_px))
        _draw_dashed_circle(out, center, radius, (0, 0, 255),
                            thickness=max(3, frame.shape[0] // 720),
                            dash_deg=12, gap_deg=8)
        cv2.drawMarker(out, center, (0, 0, 255), cv2.MARKER_CROSS,
                       28, th, cv2.LINE_AA)
        if frame.shape[0] > 400:
            lbl = f"D = {res.D_mm:.3f} mm"
            (tw, th_txt), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX,
                                              1.3, 3)
            x0 = int(np.clip(center[0] - tw // 2, 20,
                             frame.shape[1] - tw - 20))
            y0 = int(np.clip(center[1] - radius - 24, th_txt + 20,
                             frame.shape[0] - 20))
            cv2.rectangle(out, (x0 - 8, y0 - th_txt - 12),
                          (x0 + tw + 12, y0 + 8), (255, 255, 255), -1)
            cv2.rectangle(out, (x0 - 8, y0 - th_txt - 12),
                          (x0 + tw + 12, y0 + 8), (0, 0, 0), 2)
            cv2.putText(out, lbl, (x0, y0), cv2.FONT_HERSHEY_SIMPLEX,
                        1.3, (0, 0, 0), 3, cv2.LINE_AA)

    if show_chord:
        cv2.line(out, res.chord_p1, res.chord_p2, (0, 200, 0), th, cv2.LINE_AA)
        cv2.circle(out, res.chord_p1, 10, (0, 200, 0), -1, cv2.LINE_AA)
        cv2.circle(out, res.chord_p2, 10, (0, 200, 0), -1, cv2.LINE_AA)
        # support contacts (cyan rings)
        for c in res.contacts:
            cv2.circle(out, (int(c[0]), int(c[1])), 16, (255, 255, 0), th,
                       cv2.LINE_AA)
        if not straight and res.apex != (0, 0):
            cv2.line(out, res.apex_foot, res.apex, (0, 165, 255), th,
                     cv2.LINE_AA)
            cv2.circle(out, res.apex, 8, (0, 165, 255), -1, cv2.LINE_AA)
            if frame.shape[0] > 400:
                hx, hy = res.apex_foot
                cv2.putText(out, f"h={res.h_max_mm:.3f}mm  L={res.L_mm:.3f}mm",
                            (hx + 14, hy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            (0, 165, 255), 2, cv2.LINE_AA)

    if show_panel:
        panel_w, panel_h = 760, 400
        overlay = out.copy()
        cv2.rectangle(overlay, (20, 20), (20 + panel_w, 20 + panel_h),
                      (0, 0, 0), -1)
        out = cv2.addWeighted(overlay, 0.55, out, 0.45, 0)
        ts = f"  t={t_seconds:.2f}s" if t_seconds is not None else ""
        lines = [
            f"frame  {res.frame_idx:>4d}{ts}   [{res.status}]",
            f"R      {res.R_mm:.3f} mm   ({res.R_px:.1f} px)",
            f"L      {res.L_mm:.3f} mm    h {res.h_max_mm:.3f} mm",
            f"angle  {res.theta_deg:.1f} deg",
            f"t      {res.thickness_mm*1000:.1f} um",
            f"strain {res.strain_pct:.3f} %",
            f"contacts {res.n_contacts}   skel {res.n_skel_samples}",
            f"fit RMS {res.fit_resid_px:.2f} px",
        ]
        for i, txt in enumerate(lines):
            cv2.putText(out, txt, (40, 70 + i * 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (255, 255, 255), 2, cv2.LINE_AA)
    return out
