"""Curvature-aware needle tracker for elastic-strain measurement.

Pipeline
--------
1.  HSV-segment the gold needle (drop tweezer / background pixels).
2.  Keep all connected components whose area is comparable to the largest one
    -- at peak bend the central tweezer bisects the needle into two arcs that
    both lie on the same osculating circle.
3.  Skeletonize every component, then walk each one with a greedy
    nearest-neighbour ordering rooted at the PCA-extreme pixel.
4.  Score every ordered sample by the LOCAL radius of curvature obtained
    from a Kasa fit over a +/- N neighbourhood.  The straight tails that
    stick out past the tweezers have R_local -> infinity, the bent arc
    sits at finite R_local.  We keep the connected stretch whose
    R_local lies within a tolerance of the median bend radius.
5.  Pool the kept samples from every component and fit the FINAL circle
    using Taubin's algebraic method, then refine geometrically with
    Levenberg-Marquardt and one RANSAC pass to drop residual outliers.
6.  From the final circle and the kept-arc samples we compute chord L,
    sagitta h_max, subtended angle theta, needle thickness t, and the
    elastic strain epsilon = t / (2 R) * 100 (% per crystal).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np


# ----------------------------------------------------------------- parameters
@dataclass
class TrackParams:
    """Tunable knobs for analyze_frame.  Defaults are calibrated for test.mp4
    (2908x2160, gold needle, dark tweezers, ~487 px / mm)."""

    # Primary HSV mask for the gold/yellow needle.  Wide enough to survive the
    # large lightness change between rest pose and peak bend.
    hsv_lo: tuple = (8, 18, 35)
    hsv_hi: tuple = (50, 255, 215)
    # Secondary range that catches dimmer shadowed parts of the needle.
    hsv_lo2: tuple = (8, 12, 25)
    hsv_hi2: tuple = (50, 255, 175)
    # Background rejection: drop pixels that are very dark (tweezer) or very
    # bright (white paper).
    v_min: int = 30
    v_max: int = 220

    # Keep components with area >= keep_frac * largest_area, and at least
    # min_component_px pixels (rejects speckle noise / horizontal scan lines).
    keep_frac: float = 0.08
    min_component_px: int = 120

    # Bridging: dilate the mask before connected-components so two arcs that
    # are separated only by the central tweezer are merged into a single
    # walkable chain.  Set to 0 to disable.
    bridge_px: int = 0

    # Skeleton smoothing window (samples).  Each ordered chain is smoothed by
    # this rectangular moving average BEFORE local curvature is measured, so
    # the local R reflects geometric curvature instead of single-pixel
    # stair-stepping.  Small values keep edge fidelity.
    smooth_window: int = 21

    # Curvature window: number of skeleton samples on either side used for the
    # local Kasa fit.  Larger window = smoother curvature, smaller = more
    # localised but noisier.
    curv_window: int = 60

    # Arc isolation: keep skeleton samples whose local radius of curvature is
    # within +/- arc_tol_frac of the global bend scale.
    arc_tol_frac: float = 0.45

    # Bend-scale quantile.  We take the most-curved (smallest-R) fraction of
    # samples and use their median R as the target bend radius.
    bend_quantile: float = 0.20

    # RANSAC: maximum geometric residual (px) for an inlier on the final fit.
    ransac_tol_px: float = 5.0
    ransac_iters: int = 600

    # RANSAC scoring: each candidate's inlier count is penalised by
    # `(R - R_pref)^2 / R_scale^2` so straight-tail / near-infinite circles
    # never win against a moderately-supported curved circle.  Set
    # `max_R_px` to a hard upper bound on the bend radius in pixels.
    max_R_px: float = 4000.0
    R_pref_px: float = 800.0
    R_scale_px: float = 1500.0

    # Minimum chain length for a component to participate in curvature scoring
    # (shorter ones are usually noise off the central tweezer).
    min_chain_len: int = 60

    # Reject the frame if the kept arc is shorter than this many skeleton
    # samples (the needle is probably not in view yet).
    min_arc_samples: int = 25

    # If the sagitta-to-chord ratio h_max / L falls below this, the crystal is
    # treated as effectively straight and the (unstable) radius is not
    # reported - this is the divide-by-near-zero guard from the paper pipeline.
    straight_frac: float = 0.01

    # Max gap (px) across which fragmented skeleton pieces are stitched into a
    # single tip-to-tip centreline.  Joins must also be tangent-continuous, so
    # this can be generous enough to bridge probe / metalware occlusion of the
    # needle without ever short-cutting across the open mouth of the bend.
    stitch_gap_px: float = 400.0

    # Needle arcs are kept for the chord/sagitta measurement when they are at
    # least this fraction of the longest arc (drops segmentation specks while
    # retaining both halves of a probe-split needle).
    tip_min_chain_frac: float = 0.2
    tip_min_chain_px: int = 40

    # Physical-validity guards.  A frame is rejected (ok=False, no strain
    # reported) when the chord is implausibly short or the computed strain is
    # beyond any credible elastic value - this stops fragmented / blurred
    # frames from emitting nonsense like 700 % strain.
    min_chord_px: float = 250.0
    max_strain_pct: float = 25.0

    # --- Three-point contact detection (dark metalware / probe) --------------
    # The metalware tweezers are near-black wedges entering from the frame
    # edges.  h_max is measured to the point where the PROBE (the metalware
    # touching the crystal from the opposite side to the two supports) contacts
    # the crystal - not to the geometric apex.
    metal_gray_max: int = 95          # pixels darker than this are metalware
    min_metal_area: int = 4000        # min blob area (px) for a real tweezer
    contact_dist_px: float = 70.0     # max metalware-tip-to-crystal gap = touch
    probe_exclude_px: float = 130.0   # crystal ends within this of the probe
                                      # contact are occlusion ends, not supports

    # Restrict analysis to this ROI (x, y, w, h) in pixel coords if given.
    roi: Optional[tuple] = None


# ----------------------------------------------------------------- segmentation
def segment_needle(frame: np.ndarray, params: TrackParams) -> np.ndarray:
    """Binary mask of the gold needle (full frame; ROI handled by caller).
    Combines two HSV ranges (bright + shadowed needle) and rejects the very
    bright background and very dark tweezers via a V-channel band-pass."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, np.array(params.hsv_lo, np.uint8),
                     np.array(params.hsv_hi, np.uint8))
    m2 = cv2.inRange(hsv, np.array(params.hsv_lo2, np.uint8),
                     np.array(params.hsv_hi2, np.uint8))
    mask = cv2.bitwise_or(m1, m2)
    band = cv2.inRange(hsv[..., 2], params.v_min, params.v_max)
    mask = cv2.bitwise_and(mask, band)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    if params.bridge_px > 0:
        d = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (params.bridge_px * 2 + 1, params.bridge_px * 2 + 1))
        mask = cv2.dilate(mask, d)
        mask = cv2.erode(mask, k)

    # Drop components whose bounding box is much wider than tall AND short
    # (horizontal scan lines from the background mat).
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if num <= 1:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    if areas.size == 0:
        return mask
    max_area = int(areas.max())
    cutoff = max(int(max_area * params.keep_frac), params.min_component_px)
    # Build a label -> keep lookup once, then remap in a single vectorised
    # gather instead of scanning the full-resolution label image per component.
    lut = np.zeros(num, np.uint8)
    for i, a in enumerate(areas, start=1):
        if a < cutoff:
            continue
        x, y, w, h = (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
                      stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT])
        # Drop scan-line-like horizontal artefacts: very flat, very wide, low fill.
        fill = a / float(max(w * h, 1))
        if h <= 4 and w > 25 * h and fill < 0.35:
            continue
        lut[i] = 255
    out = lut[labels]
    return out


def skeletonize(mask: np.ndarray) -> np.ndarray:
    """One-pixel-wide centreline.

    The mask is first cropped to the bounding box of its foreground (plus a
    small margin) so the thinning routine only touches the needle region
    rather than the whole multi-megapixel frame, then the result is pasted
    back into a full-size canvas.

    Preference order (fastest first):
      1. ``cv2.ximgproc.thinning`` (C++ Zhang-Suen) when opencv-contrib is
         installed.
      2. ``skimage.morphology.skeletonize`` (compiled Cython) - fast and
         widely available.
      3. Pure-OpenCV morphological thinning fallback (slow; last resort)."""
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
        skel = _sk_skel(mask > 0)
        return (skel.astype(np.uint8) * 255)
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
def _walk_skeleton(pts: np.ndarray, max_gap_px: float = 8.0) -> np.ndarray:
    """Greedy nearest-neighbour ordering of a SINGLE connected component's
    skeleton.  Roots the walk at the PCA-extreme pixel for stability and
    breaks on gaps larger than max_gap_px (a one-pixel-wide skeleton has
    nearest-neighbour distance ~1 px)."""
    n = len(pts)
    if n < 3:
        return pts.copy()
    centered = pts.astype(np.float64) - pts.mean(0)
    cov = np.cov(centered.T)
    _, evecs = np.linalg.eigh(cov)
    axis = evecs[:, -1]
    proj = centered @ axis
    start = int(np.argmin(proj))

    pts_f = pts.astype(np.float64)
    visited = np.zeros(n, dtype=bool)
    visited[start] = True
    order = [start]
    cur = pts_f[start]
    gap_sq = max_gap_px * max_gap_px
    while True:
        d = np.sum((pts_f - cur) ** 2, axis=1)
        d[visited] = np.inf
        nxt = int(np.argmin(d))
        if not np.isfinite(d[nxt]) or d[nxt] > gap_sq:
            break
        visited[nxt] = True
        order.append(nxt)
        cur = pts_f[nxt]
    return pts[order]


def _end_dir(seg: np.ndarray, end: str, k: int = 12) -> np.ndarray:
    """Unit OUTWARD tangent at one end of an ordered chain.
    ``'tail'`` -> direction leaving ``seg[-1]``; ``'head'`` -> leaving seg[0]."""
    k = min(k, len(seg) - 1)
    if k < 1:
        return np.zeros(2)
    v = (seg[-1] - seg[-1 - k]) if end == "tail" else (seg[0] - seg[k])
    n = float(np.hypot(v[0], v[1]))
    return v / n if n > 0 else np.zeros(2)


def _extend_tail(merged: np.ndarray, segs: list[np.ndarray],
                 max_gap: float, cos_thr: float, k: int = 12
                 ) -> tuple[np.ndarray, bool]:
    """Try to append one segment onto the tail of ``merged``, accepting a join
    only when the next fragment continues in roughly the tail's heading (so a
    wide but tangent-continuous occlusion gap is bridged, while a fold-back
    across the open mouth of the bend is rejected)."""
    tdir = _end_dir(merged, "tail", k)            # outward heading at the tail
    tail = merged[-1]
    best = None                                   # (gap, idx, oriented)
    for i, s in enumerate(segs):
        for oriented in (s, s[::-1]):
            e0 = oriented[0]
            gv = e0 - tail
            gap = float(np.hypot(gv[0], gv[1]))
            if gap > max_gap:
                continue
            # forward heading as we enter the candidate from its first point
            kk = min(k, len(oriented) - 1)
            fv = oriented[kk] - oriented[0]
            fn = float(np.hypot(fv[0], fv[1]))
            fdir = fv / fn if fn > 0 else np.zeros(2)
            # tangent-continuity gates (skip the gap-direction gate for tiny
            # gaps where its direction is dominated by pixel noise)
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


def _merge_centerline(chains: list[np.ndarray],
                      max_gap: float = 400.0,
                      cos_thr: float = 0.2) -> Optional[np.ndarray]:
    """Stitch ordered skeleton chains into one centreline that walks the
    crystal tip-to-tip.

    Growth starts from the longest chain and extends from BOTH ends, each step
    adding the nearest tangent-continuous fragment.  Because joins must follow
    the local heading of the curve, a wide gap where the probe / metalware
    occludes the needle is bridged, but the open mouth of a C-shaped bend
    (whose two free tips would require a ~180 deg fold-back) is never joined -
    so the final endpoints stay the two physical crystal ends."""
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


def order_components(skel: np.ndarray) -> list[np.ndarray]:
    """Return a list of ordered (px, py) point chains, one per skeleton
    connected component."""
    num, labels = cv2.connectedComponents(skel, connectivity=8)
    if num <= 1:
        return []
    # Gather every skeleton pixel and its label in a single full-image pass,
    # then group by label.  Avoids scanning the whole label image once per
    # component (the previous ``np.where(labels == i)`` per-component loop).
    ys, xs = np.nonzero(labels)
    labs = labels[ys, xs]
    chains: list[np.ndarray] = []
    for i in range(1, num):
        sel = labs == i
        if int(sel.sum()) < 5:
            continue
        chain = _walk_skeleton(np.column_stack([xs[sel], ys[sel]]))
        if len(chain) >= 5:
            chains.append(chain)
    return chains


# ----------------------------------------------------------------- circle fits
def _kasa(pts: np.ndarray) -> tuple[float, float, float, float]:
    """Plain Kasa algebraic fit, returns (cx, cy, R, rms residual px)."""
    x = pts[:, 0].astype(np.float64)
    y = pts[:, 1].astype(np.float64)
    A = np.column_stack([2 * x, 2 * y, np.ones_like(x)])
    b = x * x + y * y
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy, c = sol
    R = float(np.sqrt(max(c + cx * cx + cy * cy, 0.0)))
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) - R
    return float(cx), float(cy), R, float(np.sqrt(np.mean(r * r)))


def fit_circle_taubin(pts: np.ndarray) -> tuple[float, float, float, float]:
    """Taubin (1991) algebraic circle fit.  Asymptotically unbiased for short
    arcs, unlike Kasa.  Returns (cx, cy, R, rms residual px)."""
    if len(pts) < 3:
        return 0.0, 0.0, 0.0, float("inf")
    x = pts[:, 0].astype(np.float64)
    y = pts[:, 1].astype(np.float64)
    cx0, cy0 = x.mean(), y.mean()
    u = x - cx0
    v = y - cy0
    z = u * u + v * v
    Mxx = (u * u).mean()
    Myy = (v * v).mean()
    Mxy = (u * v).mean()
    Mxz = (u * z).mean()
    Myz = (v * z).mean()
    Mzz = (z * z).mean()
    Mz = Mxx + Myy
    cov_xy = Mxx * Myy - Mxy * Mxy
    A3 = 4.0 * Mz
    A2 = -3.0 * Mz * Mz - Mzz
    A1 = Mzz * Mz + 4.0 * cov_xy * Mz - Mxz * Mxz - Myz * Myz - Mz * Mz * Mz
    A0 = (Mxz * (Mxz * Myy - Myz * Mxy)
          + Myz * (Myz * Mxx - Mxz * Mxy)
          - Mzz * cov_xy)
    # Newton's method on the characteristic quartic, started at x=0.
    xn = 0.0
    for _ in range(99):
        F = A0 + xn * (A1 + xn * (A2 + xn * (A3 + xn * 4.0)))
        Fp = A1 + xn * (2.0 * A2 + xn * (3.0 * A3 + xn * 16.0))
        if Fp == 0.0:
            break
        step = F / Fp
        xn -= step
        if abs(step) < 1e-12:
            break
    det = xn * xn - xn * Mz + cov_xy
    if abs(det) < 1e-20:
        return _kasa(pts)
    uc = (Mxz * (Myy - xn) - Myz * Mxy) / det / 2.0
    vc = (Myz * (Mxx - xn) - Mxz * Mxy) / det / 2.0
    cx = uc + cx0
    cy = vc + cy0
    R = float(np.sqrt(max(uc * uc + vc * vc + Mz, 0.0)))
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) - R
    return float(cx), float(cy), R, float(np.sqrt(np.mean(r * r)))


def _refine_geometric(pts: np.ndarray, cx: float, cy: float, R: float
                      ) -> tuple[float, float, float, float]:
    """Levenberg-Marquardt refinement minimising the geometric distance
    (signed orthogonal residual to the circle).  Fixed iteration count keeps
    the routine free of scipy."""
    x = pts[:, 0].astype(np.float64)
    y = pts[:, 1].astype(np.float64)
    lam = 1e-3
    for _ in range(40):
        dx = x - cx
        dy = y - cy
        d = np.sqrt(dx * dx + dy * dy)
        r = d - R
        ddx = -dx / np.maximum(d, 1e-9)
        ddy = -dy / np.maximum(d, 1e-9)
        ddR = -np.ones_like(d)
        J = np.column_stack([ddx, ddy, ddR])
        H = J.T @ J + lam * np.eye(3)
        g = J.T @ r
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
        cx -= step[0]
        cy -= step[1]
        R -= step[2]
        if np.linalg.norm(step) < 1e-7:
            break
    dx = x - cx
    dy = y - cy
    r = np.sqrt(dx * dx + dy * dy) - R
    rms = float(np.sqrt(np.mean(r * r)))
    return float(cx), float(cy), float(R), rms


def _ransac_refit(pts: np.ndarray, cx: float, cy: float, R: float,
                  tol: float, iters: int,
                  max_R: float = float("inf"),
                  R_pref: float = 0.0,
                  R_scale: float = float("inf"),
                  ) -> tuple[float, float, float, float, np.ndarray]:
    """RANSAC over random 3-point circles plus the supplied prior circle.
    Each candidate is scored as `inlier_count - (R - R_pref)**2 / R_scale**2`
    so straight-line three-point picks (which give near-infinite R) cannot
    win against a moderately curved circle with real inlier support.
    Returns (cx, cy, R, rms, inlier_mask).
    """
    n = len(pts)
    rng = np.random.default_rng(0)

    def _score(r: float, cnt: int) -> float:
        if not np.isfinite(r) or r > max_R:
            return -1e18
        return cnt - ((r - R_pref) ** 2) / max(R_scale ** 2, 1e-9)

    res_prior = np.abs(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) - R)
    best_inliers = res_prior < tol
    best_score = _score(R, int(best_inliers.sum()))
    best_R = R
    for _ in range(iters):
        i = rng.choice(n, 3, replace=False)
        a, b, c = pts[i].astype(np.float64)
        d = 2.0 * ((a[0] - c[0]) * (b[1] - c[1])
                   - (b[0] - c[0]) * (a[1] - c[1]))
        if abs(d) < 1e-9:
            continue
        ux = ((a[0] ** 2 - c[0] ** 2 + a[1] ** 2 - c[1] ** 2)
              * (b[1] - c[1])
              - (b[0] ** 2 - c[0] ** 2 + b[1] ** 2 - c[1] ** 2)
              * (a[1] - c[1])) / d
        uy = ((b[0] ** 2 - c[0] ** 2 + b[1] ** 2 - c[1] ** 2)
              * (a[0] - c[0])
              - (a[0] ** 2 - c[0] ** 2 + a[1] ** 2 - c[1] ** 2)
              * (b[0] - c[0])) / d
        rr = np.hypot(a[0] - ux, a[1] - uy)
        if rr > max_R or not np.isfinite(rr):
            continue
        res = np.abs(np.hypot(pts[:, 0] - ux, pts[:, 1] - uy) - rr)
        inl = res < tol
        cnt = int(inl.sum())
        if cnt < 6:
            continue
        sc = _score(rr, cnt)
        if sc > best_score:
            best_score = sc
            best_inliers = inl
            best_R = rr
    if best_inliers.sum() < max(20, int(0.05 * n)):
        best_inliers = np.ones(n, dtype=bool)
    cx2, cy2, R2, rms2 = _refine_geometric(pts[best_inliers], cx, cy, best_R)
    return cx2, cy2, R2, rms2, best_inliers


# ----------------------------------------------------------------- curvature
def _smooth_chain(chain: np.ndarray, window: int) -> np.ndarray:
    """Rectangular moving average over an ordered chain.  Edges are clamped."""
    n = len(chain)
    if window <= 1 or n < window:
        return chain.astype(np.float64)
    w = int(window)
    kernel = np.ones(w) / float(w)
    xs = np.convolve(chain[:, 0].astype(np.float64), kernel, mode="same")
    ys = np.convolve(chain[:, 1].astype(np.float64), kernel, mode="same")
    half = w // 2
    # Convolution with mode="same" pads with zeros at the edges; replace those
    # endpoints with the raw coordinates so the chain still spans the curve.
    xs[:half] = chain[:half, 0]
    xs[-half:] = chain[-half:, 0]
    ys[:half] = chain[:half, 1]
    ys[-half:] = chain[-half:, 1]
    return np.column_stack([xs, ys])


def _local_curvature_radius(chain: np.ndarray, w: int) -> np.ndarray:
    """For every chain sample i with i-w..i+w in range, fit a Kasa circle to
    that local neighbourhood and return its radius.  Samples without enough
    neighbours get NaN.

    Vectorised: instead of running a separate least-squares solve per sample,
    the Kasa normal equations are assembled from sliding-window sums (via
    cumulative sums) and the whole stack of 3x3 systems is solved in one
    batched ``np.linalg.solve`` call.  This is ~100x faster than the per-sample
    Python loop on long skeletons while producing identical radii."""
    n = len(chain)
    R_local = np.full(n, np.nan)
    L = 2 * w + 1
    if n < L + 2:
        return R_local

    x = chain[:, 0].astype(np.float64)
    y = chain[:, 1].astype(np.float64)
    z = x * x + y * y

    def _ws(a: np.ndarray) -> np.ndarray:
        # Sum of every contiguous length-L window: out[j] = sum(a[j:j+L]).
        c = np.cumsum(np.insert(a, 0, 0.0))
        return c[L:] - c[:-L]

    Sx = _ws(x); Sy = _ws(y)
    Sxx = _ws(x * x); Syy = _ws(y * y); Sxy = _ws(x * y)
    Sxz = _ws(x * z); Syz = _ws(y * z); Sz = _ws(z)
    m = Sx.shape[0]                       # = n - 2w, one per valid centre
    S1 = float(L)

    ATA = np.empty((m, 3, 3))
    ATA[:, 0, 0] = 4.0 * Sxx; ATA[:, 0, 1] = 4.0 * Sxy; ATA[:, 0, 2] = 2.0 * Sx
    ATA[:, 1, 0] = 4.0 * Sxy; ATA[:, 1, 1] = 4.0 * Syy; ATA[:, 1, 2] = 2.0 * Sy
    ATA[:, 2, 0] = 2.0 * Sx;  ATA[:, 2, 1] = 2.0 * Sy;  ATA[:, 2, 2] = S1
    ATb = np.empty((m, 3))
    ATb[:, 0] = 2.0 * Sxz; ATb[:, 1] = 2.0 * Syz; ATb[:, 2] = Sz

    det = np.linalg.det(ATA)
    good = np.abs(det) > 1e-6
    if good.any():
        sol = np.linalg.solve(ATA[good], ATb[good][:, :, None])[:, :, 0]
        cx = sol[:, 0]; cy = sol[:, 1]; c = sol[:, 2]
        R = np.sqrt(np.maximum(c + cx * cx + cy * cy, 0.0))
        centres = np.full(m, np.nan)
        centres[good] = R
        R_local[w:n - w] = centres
    return R_local


def _longest_run(mask: np.ndarray) -> tuple[int, int]:
    """Indices (start, end_exclusive) of the longest contiguous True run."""
    n = len(mask)
    best = (0, 0)
    i = 0
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j < n and mask[j]:
            j += 1
        if j - i > best[1] - best[0]:
            best = (i, j)
        i = j
    return best


def _chain_curvature(chain: np.ndarray, params: TrackParams) -> np.ndarray:
    """Return per-sample local radius of curvature (px) for an ordered chain.
    The chain is smoothed first to prevent single-pixel noise from collapsing
    the local Kasa fit onto a tiny circle."""
    if len(chain) < params.curv_window * 2 + 3:
        return np.full(len(chain), np.nan)
    smoothed = _smooth_chain(chain, params.smooth_window)
    return _local_curvature_radius(smoothed, params.curv_window)


def isolate_arc_samples(chain: np.ndarray, R_target: float,
                        params: TrackParams,
                        R_local: Optional[np.ndarray] = None
                        ) -> tuple[np.ndarray, np.ndarray]:
    """Given an ordered skeleton chain and a target bend radius R_target (px),
    keep the longest contiguous run of samples whose local radius of curvature
    matches R_target within +/- arc_tol_frac.  Returns (kept_pts, kept_mask).

    ``R_local`` may be supplied to reuse a previously computed curvature
    profile for this chain and avoid recomputing it.
    """
    if R_local is None:
        R_local = _chain_curvature(chain, params)
    finite = np.isfinite(R_local)
    if not finite.any() or R_target <= 0:
        return chain.copy(), np.ones(len(chain), dtype=bool)
    tol = params.arc_tol_frac
    keep = finite & (R_local >= R_target * (1.0 - tol)) & (R_local <= R_target * (1.0 + tol))
    s, e = _longest_run(keep)
    if e - s < 5:
        # Fall back to the most-curved portion of this chain.
        order = np.argsort(R_local[finite])
        # take 50% smallest-R indices
        n_keep = max(int(0.5 * finite.sum()), 5)
        idx_finite = np.where(finite)[0]
        chosen = np.zeros(len(chain), dtype=bool)
        chosen[idx_finite[order[:n_keep]]] = True
        return chain[chosen], chosen
    keep_mask = np.zeros(len(chain), dtype=bool)
    keep_mask[s:e] = True
    return chain[keep_mask], keep_mask


# ----------------------------------------------------------------- geometry
def chord_sagitta_radius(L: float, h_max: float) -> float:
    """Radius of the circular arc of chord ``L`` and sagitta ``h_max``.

    Exact closed form from Figure S17 / equation [2] of the reference paper::

        R^2 = (L/2)^2 + (R - h_max)^2             [1]
        =>  R = ((L/2)^2 + h_max^2) / (2 * h_max)  [2]

    Pure geometry (no OpenCV / numpy state) - the one piece of maths that has
    to match the paper exactly, kept tiny and independently testable.  ``L``
    and ``h_max`` share units; ``R`` is returned in those units.  Returns
    ``inf`` for a degenerate (zero-deflection / straight) arc.
    """
    if h_max <= 0.0:
        return float("inf")
    return ((L * 0.5) ** 2 + h_max ** 2) / (2.0 * h_max)


def compute_strain(t: float, R: float) -> float:
    """Elastic bending strain (%), equation [3]:  epsilon = t / (2 R) * 100.
    ``t`` (undeformed thickness) and ``R`` share units."""
    if R <= 0.0 or not np.isfinite(R):
        return 0.0
    return (t / (2.0 * R)) * 100.0


def _principal_endpoints(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The two extreme points of a point cloud along its principal (long)
    axis - for a bent-crystal centreline these are the support contact points
    that define the chord A-B."""
    p = pts.astype(np.float64)
    centered = p - p.mean(0)
    cov = np.cov(centered.T)
    _, evecs = np.linalg.eigh(cov)
    proj = centered @ evecs[:, -1]
    return p[int(np.argmin(proj))], p[int(np.argmax(proj))]


def _tip_endpoints(chains: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """The two crystal ends, taken as the farthest-apart pair among all chain
    endpoints.  When the probe / metalware splits the needle into several arcs,
    the junction ends sit in the middle and are close together, so the extreme
    pair is the two physical support contacts that define the chord."""
    eps = []
    for c in chains:
        eps.append(c[0])
        eps.append(c[-1])
    eps = np.asarray(eps, dtype=np.float64)
    if len(eps) <= 2:
        return eps[0], eps[-1]
    best = (-1.0, eps[0], eps[1])
    for i in range(len(eps)):
        d = np.hypot(eps[:, 0] - eps[i, 0], eps[:, 1] - eps[i, 1])
        j = int(np.argmax(d))
        if d[j] > best[0]:
            best = (float(d[j]), eps[i], eps[j])
    return best[1], best[2]


def chord_and_sagitta(pts: np.ndarray, ordered: bool = False,
                      A: Optional[np.ndarray] = None,
                      B: Optional[np.ndarray] = None
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                 np.ndarray, float, float]:
    """Measure the paper's geometry from centreline points.

    Returns ``(A, B, apex, foot, L, h_max)``: chord endpoints ``A``/``B``
    (the two crystal ends / support contacts), the most-deflected point
    ``apex`` (the probe / third-contact point), its perpendicular projection
    ``foot`` onto chord A-B, the chord length ``L`` and the sagitta ``h_max`` -
    all in pixels.

    Chord endpoints are chosen as: the explicit ``A``/``B`` if given, else the
    first/last samples when ``ordered`` is True, else the principal-axis
    extremes.  ``h_max`` is the largest perpendicular distance of ``pts`` from
    chord A-B (so it lands on the third / probe contact point per Fig S17).
    """
    p = pts.astype(np.float64)
    if A is None or B is None:
        if ordered and len(p) >= 2:
            A, B = p[0], p[-1]
        else:
            A, B = _principal_endpoints(p)
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    AB = B - A
    L = float(np.hypot(AB[0], AB[1]))
    if L < 1e-9:
        return A, B, A.copy(), A.copy(), 0.0, 0.0
    perp = (AB[0] * (p[:, 1] - A[1]) - AB[1] * (p[:, 0] - A[0])) / L
    apex_i = int(np.argmax(np.abs(perp)))
    h_max = float(abs(perp[apex_i]))
    apex = p[apex_i]
    tproj = float(np.dot(apex - A, AB) / (L * L))
    foot = A + tproj * AB
    return A, B, apex, foot, L, h_max


def _circle_from_arc(A: np.ndarray, B: np.ndarray, apex: np.ndarray,
                     L: float, h_max: float, R: float) -> np.ndarray:
    """Centre of the bend circle given a chord A-B, its apex and the closed-
    form radius: a point on the chord's far side from the apex, a distance
    (R - h_max) from the chord midpoint."""
    M = 0.5 * (A + B)
    AB = B - A
    n = np.array([-AB[1], AB[0]]) / L
    if np.dot(apex - M, n) > 0:        # make n point AWAY from the apex
        n = -n
    return M + n * (R - h_max)


def detect_metalware_tips(frame: np.ndarray, params: TrackParams
                          ) -> list[np.ndarray]:
    """Locate the tips of the dark metalware tweezers (the two supports and the
    probe).  Each tweezer is a near-black wedge entering from a frame edge; its
    tip is the blob pixel reaching farthest INTO the frame (toward the image
    centre).  Returns a list of (x, y) tip points in frame coords."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dark = (gray < params.metal_gray_max).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, k)
    num, labels, stats, cent = cv2.connectedComponentsWithStats(dark, 8)
    H, W = gray.shape[:2]
    ctr = np.array([W * 0.5, H * 0.5])
    tips: list[np.ndarray] = []
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] < params.min_metal_area:
            continue
        x, y, w, h = (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
                      stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT])
        touches = (x <= 2 or y <= 2 or x + w >= W - 2 or y + h >= H - 2)
        if not touches:                 # real tweezers reach in from an edge
            continue
        ys, xs = np.where(labels[y:y + h, x:x + w] == i)
        pts = np.column_stack([xs + x, ys + y]).astype(np.float64)
        # tip = blob pixel closest to the frame centre (deepest reach inward)
        tip = pts[int(np.argmin(np.hypot(pts[:, 0] - ctr[0],
                                         pts[:, 1] - ctr[1])))]
        tips.append(tip)
    return tips


def detect_tweezer_contacts(frame: np.ndarray, crystal: np.ndarray,
                            params: TrackParams) -> list[np.ndarray]:
    """Find where the dark metalware tweezers touch the crystal.

    Each tweezer is a near-black blob entering from a frame edge.  For every
    such blob we find the blob pixel closest to the crystal point cloud; if
    that gap is small enough the corresponding crystal point is a contact.
    This is far more reliable than locating a geometric "tip", and directly
    yields the points the chord / sagitta are built from.  Returns a
    deduplicated list of (x, y) crystal-contact points."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dark = (gray < params.metal_gray_max).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, k)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(dark, 8)
    H, W = gray.shape[:2]
    cpts = crystal.astype(np.float64)
    contacts: list[np.ndarray] = []
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] < params.min_metal_area:
            continue
        x, y, w, h = (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
                      stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT])
        if not (x <= 2 or y <= 2 or x + w >= W - 2 or y + h >= H - 2):
            continue                                   # not edge-anchored
        ys, xs = np.where(labels[y:y + h, x:x + w] == i)
        bx = (xs + x).astype(np.float64)
        by = (ys + y).astype(np.float64)
        if bx.size > 600:                              # subsample for speed
            sel = np.linspace(0, bx.size - 1, 600).astype(int)
            bx, by = bx[sel], by[sel]
        # nearest crystal point to this blob
        dx = bx[:, None] - cpts[None, :, 0]
        dy = by[:, None] - cpts[None, :, 1]
        d2 = dx * dx + dy * dy
        bi, cj = np.unravel_index(int(np.argmin(d2)), d2.shape)
        if np.sqrt(d2[bi, cj]) > params.contact_dist_px:
            continue
        cp = cpts[cj]
        if all(np.hypot(*(cp - c)) > params.probe_exclude_px for c in contacts):
            contacts.append(cp)
    return contacts


def measure_three_point_geometry(frame: np.ndarray, sig: list[np.ndarray],
                                 params: TrackParams) -> Optional[dict]:
    """Figure-S17 three-point geometry, driven by the tweezer contacts.

    The chord L is defined by WHERE THE TWEEZERS TOUCH THE CRYSTAL (exactly as
    in the figure - the green L line joins the two support contacts, and the
    crystal may stick out past them).  This is far more reliable than guessing
    crystal endpoints from a fragmented skeleton:

      1. Detect where the dark tweezers touch the crystal -> contact points.
      2. The two farthest-apart contacts are the SUPPORTS  -> chord A,B.
         The remaining contact is the PROBE -> apex for h_max.
      3. h_max = perpendicular distance from chord A-B to the probe contact;
         R = ((L/2)^2 + h_max^2)/(2 h_max).

    Falls back gracefully when fewer than three contacts are found.
    """
    crystal = np.concatenate(sig, axis=0).astype(np.float64)
    if len(crystal) < params.min_arc_samples:
        return None

    contacts = detect_tweezer_contacts(frame, crystal, params)

    def _perp(P, A, B):
        AB = B - A
        L = np.hypot(AB[0], AB[1])
        return abs(AB[0] * (P[1] - A[1]) - AB[1] * (P[0] - A[0])) / max(L, 1e-9)

    if len(contacts) >= 3:
        A, B = _farthest_pair(np.asarray(contacts))
        others = [c for c in contacts
                  if not (np.allclose(c, A) or np.allclose(c, B))]
        apex = max(others, key=lambda c: _perp(c, A, B))
        has_probe = True
    elif len(contacts) == 2:
        A, B = contacts[0], contacts[1]
        AB = B - A
        L0 = max(float(np.hypot(AB[0], AB[1])), 1e-9)
        perp = (AB[0] * (crystal[:, 1] - A[1])
                - AB[1] * (crystal[:, 0] - A[0])) / L0
        apex = crystal[int(np.argmax(np.abs(perp)))]
        has_probe = False
    else:
        arc = max(sig, key=len)
        A, B = arc[0].astype(np.float64), arc[-1].astype(np.float64)
        AB = B - A
        L0 = max(float(np.hypot(AB[0], AB[1])), 1e-9)
        perp = (AB[0] * (crystal[:, 1] - A[1])
                - AB[1] * (crystal[:, 0] - A[0])) / L0
        apex = crystal[int(np.argmax(np.abs(perp)))]
        has_probe = False

    A = np.asarray(A, np.float64)
    B = np.asarray(B, np.float64)
    apex = np.asarray(apex, np.float64)
    AB = B - A
    L = float(np.hypot(AB[0], AB[1]))
    if L < 1.0:
        return None
    h_max = float(abs(AB[0] * (apex[1] - A[1])
                      - AB[1] * (apex[0] - A[0])) / L)
    if h_max <= 0.0:
        return None
    if h_max / L < params.straight_frac:
        return {"straight": True, "A": A, "B": B, "L_px": L, "h_max_px": h_max}

    tproj = float(np.dot(apex - A, AB) / (L * L))
    foot = A + tproj * AB
    R = chord_sagitta_radius(L, h_max)
    centre = _circle_from_arc(A, B, apex, L, h_max, R)
    ccx, ccy = float(centre[0]), float(centre[1])
    rms = float(np.sqrt(np.mean(
        (np.hypot(crystal[:, 0] - ccx, crystal[:, 1] - ccy) - R) ** 2)))
    theta = 2.0 * float(np.arcsin(min(1.0, (L * 0.5) / R)))
    return {"straight": False, "A": A, "B": B, "apex": apex, "foot": foot,
            "cx": ccx, "cy": ccy, "R_px": R, "L_px": L, "h_max_px": h_max,
            "theta_deg": float(np.degrees(theta)), "rms": rms,
            "has_probe": has_probe, "contacts": contacts}


def _detect_probe_contact(frame: np.ndarray, crystal: np.ndarray,
                          A: np.ndarray, B: np.ndarray, params: TrackParams
                          ) -> Optional[np.ndarray]:
    """Crystal point where the probe touches it: the crystal point nearest a
    metalware tip that lands in the crystal interior (far from both chord ends)
    rather than at a support end."""
    tips = detect_metalware_tips(frame, params)
    best = None
    for tip in tips:
        d = np.hypot(crystal[:, 0] - tip[0], crystal[:, 1] - tip[1])
        j = int(np.argmin(d))
        if d[j] > params.contact_dist_px:
            continue
        cp = crystal[j]
        de = min(float(np.hypot(*(cp - A))), float(np.hypot(*(cp - B))))
        if de < params.probe_exclude_px:        # this is a support, not probe
            continue
        if best is None or de > best[1]:
            best = (cp, de)
    return best[0] if best is not None else None


def _farthest_pair(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The pair of points with maximum separation (brute force; pts is small)."""
    if len(pts) <= 2:
        return pts[0], pts[-1]
    best = (-1.0, pts[0], pts[1])
    for i in range(len(pts)):
        d = np.hypot(pts[:, 0] - pts[i, 0], pts[:, 1] - pts[i, 1])
        j = int(np.argmax(d))
        if d[j] > best[0]:
            best = (float(d[j]), pts[i], pts[j])
    return best[1], best[2]


def measure_paper_geometry(sig: list[np.ndarray], params: TrackParams
                           ) -> Optional[dict]:
    """Figure-S17 geometry from the dominant bent arc.

    The longest gold arc is taken to be the crystal of interest.  Its two
    ordered endpoints are the chord A-B (the support contacts), the apex is the
    centreline point of largest perpendicular deflection (the probe / third
    contact), and the radius follows from the exact chord/sagitta closed form
    R = ((L/2)^2 + h_max^2)/(2 h_max).  Because R is a property of the bend
    circle, a clean arc gives the correct R even if the probe occludes the rest
    of the needle - so we deliberately do NOT try to gather other skeleton
    pieces, which on real footage drag in stray fragments or a second crystal.

    Returns a dict of px-space geometry, or None when no usable arc exists.
    """
    arc = max(sig, key=len)
    if len(arc) < params.min_arc_samples:
        return None
    A, B, apex, foot, L, h = chord_and_sagitta(arc, ordered=True)
    if L < 1.0 or h <= 0.0:
        return None
    if h / L < params.straight_frac:                 # effectively straight
        return {"straight": True, "A": A, "B": B, "L_px": L, "h_max_px": h}

    R = chord_sagitta_radius(L, h)
    centre = _circle_from_arc(A, B, apex, L, h, R)
    cx, cy = float(centre[0]), float(centre[1])
    P = arc.astype(np.float64)
    rms = float(np.sqrt(np.mean(
        (np.hypot(P[:, 0] - cx, P[:, 1] - cy) - R) ** 2)))
    theta = 2.0 * float(np.arcsin(min(1.0, (L * 0.5) / R)))
    return {"straight": False, "A": A, "B": B, "apex": apex, "foot": foot,
            "cx": cx, "cy": cy, "R_px": R, "L_px": L, "h_max_px": h,
            "theta_deg": float(np.degrees(theta)), "rms": rms}


def measure_geometry(arc_pts: np.ndarray, cx: float, cy: float, R: float
                     ) -> dict:
    """Chord, sagitta, subtended angle from the *kept arc samples* and the
    fitted circle.  Chord endpoints are the arc-angle extrema."""
    p = arc_pts.astype(np.float64)
    ang = np.arctan2(p[:, 1] - cy, p[:, 0] - cx)
    # Unwrap to avoid the +/-pi seam splitting the arc.
    a = np.sort(ang)
    gaps = np.diff(np.concatenate([a, [a[0] + 2 * np.pi]]))
    seam = int(np.argmax(gaps))
    start = a[(seam + 1) % len(a)]
    a_unwrapped = np.where(ang < start, ang + 2 * np.pi, ang)
    i_lo = int(np.argmin(a_unwrapped))
    i_hi = int(np.argmax(a_unwrapped))
    p1 = p[i_lo]
    p2 = p[i_hi]
    L = float(np.linalg.norm(p2 - p1))
    theta = float(abs(a_unwrapped[i_hi] - a_unwrapped[i_lo]))
    # Sagitta: distance from the chord midpoint to the arc apex (far side
    # from the centre).  Geometric value is R * (1 - cos(theta/2)).
    h_max = float(R * (1.0 - np.cos(theta / 2.0)))
    return {
        "L_px": L, "h_max_px": h_max,
        "theta_rad": theta, "theta_deg": float(np.degrees(theta)),
        "chord_p1": p1.tolist(), "chord_p2": p2.tolist(),
    }


def estimate_thickness(mask: np.ndarray, total_skeleton_px: int) -> float:
    """Mean perpendicular needle thickness in px = mask_area / centreline_len."""
    if total_skeleton_px <= 0:
        return 0.0
    return float(np.count_nonzero(mask)) / float(total_skeleton_px)


# ----------------------------------------------------------------- main entry
@dataclass
class FrameResult:
    ok: bool
    frame_idx: int
    needle_px: int = 0
    mask: Optional[np.ndarray] = None
    arc_pts: Optional[np.ndarray] = None         # final inliers (kept arc)
    skeleton_pts: Optional[np.ndarray] = None    # all skeleton samples
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


def analyze_frame(frame: np.ndarray, frame_idx: int, mm_per_px: float,
                  params: Optional[TrackParams] = None) -> FrameResult:
    """Run the full pipeline on one frame.  All px values are in *original
    frame* coordinates even when a ROI is set."""
    p = params or TrackParams()
    H, W = frame.shape[:2]

    if p.roi is not None:
        rx, ry, rw, rh = p.roi
        rx = max(0, int(rx)); ry = max(0, int(ry))
        rw = min(W - rx, int(rw)); rh = min(H - ry, int(rh))
        sub = frame[ry:ry + rh, rx:rx + rw]
    else:
        rx = ry = 0
        sub = frame

    sub_mask = segment_needle(sub, p)
    npx = int(np.count_nonzero(sub_mask))

    # Lift back to full-frame mask for visualisation
    mask = np.zeros((H, W), np.uint8)
    if p.roi is not None:
        mask[ry:ry + sub.shape[0], rx:rx + sub.shape[1]] = sub_mask
    else:
        mask = sub_mask

    if npx < 200:
        return FrameResult(ok=False, frame_idx=frame_idx,
                           needle_px=npx, mask=mask)

    skel = skeletonize(sub_mask)
    chains_local = order_components(skel)
    if not chains_local:
        return FrameResult(ok=False, frame_idx=frame_idx,
                           needle_px=npx, mask=mask)

    skel_pts_local = np.concatenate(chains_local, axis=0)

    # Keep the significant needle arcs (drop segmentation specks).  The probe /
    # metalware frequently splits the crystal into several arcs lying on the
    # same circle; we keep them all and pool their points.
    largest = max(len(c) for c in chains_local)
    sig = [c for c in chains_local
           if len(c) >= max(p.tip_min_chain_px,
                            int(p.tip_min_chain_frac * largest))]
    if not sig:
        sig = [max(chains_local, key=len)]
    pre_pts = np.concatenate(sig, axis=0)
    if len(pre_pts) < p.min_arc_samples:
        return FrameResult(ok=False, frame_idx=frame_idx,
                           needle_px=npx, mask=mask,
                           skeleton_pts=skel_pts_local + np.array([rx, ry]),
                           n_skel_samples=len(skel_pts_local))

    # ---- Paper-faithful geometry (Figure S17) --------------------------------
    # Chord L connects the two crystal ends; h_max is measured to where the
    # probe (dark metalware from the opposite side) contacts the crystal; R is
    # the exact chord/sagitta closed form.  See measure_three_point_geometry.
    geo = measure_three_point_geometry(sub, sig, p)
    if geo is None:
        return FrameResult(ok=False, frame_idx=frame_idx,
                           needle_px=npx, mask=mask,
                           skeleton_pts=skel_pts_local + np.array([rx, ry]),
                           n_skel_samples=len(skel_pts_local))
    if geo.get("straight"):
        sA, sB = geo["A"], geo["B"]
        return FrameResult(ok=False, frame_idx=frame_idx,
                           needle_px=npx, mask=mask,
                           skeleton_pts=skel_pts_local + np.array([rx, ry]),
                           arc_pts=pre_pts + np.array([rx, ry]),
                           L_px=geo["L_px"], h_max_px=geo["h_max_px"],
                           chord_p1=(int(sA[0] + rx), int(sA[1] + ry)),
                           chord_p2=(int(sB[0] + rx), int(sB[1] + ry)),
                           n_arc_samples=len(pre_pts),
                           n_skel_samples=len(skel_pts_local))

    A, B, apex, foot = geo["A"], geo["B"], geo["apex"], geo["foot"]
    cx, cy, R = geo["cx"], geo["cy"], geo["R_px"]
    L_px, h_max_px, theta_deg, rms = (geo["L_px"], geo["h_max_px"],
                                      geo["theta_deg"], geo["rms"])

    arc_pts_local = pre_pts
    t_px = estimate_thickness(sub_mask, len(skel_pts_local))

    R_mm = R * mm_per_px
    L_mm = L_px * mm_per_px
    h_mm = h_max_px * mm_per_px
    t_mm = t_px * mm_per_px
    strain = compute_strain(t_mm, R_mm)

    # Physical-validity guards: reject fragmented / blurred frames that yield an
    # implausibly short chord or a nonphysical strain, rather than reporting it.
    if L_px < p.min_chord_px or strain > p.max_strain_pct:
        return FrameResult(ok=False, frame_idx=frame_idx,
                           needle_px=npx, mask=mask,
                           skeleton_pts=skel_pts_local + np.array([rx, ry]),
                           arc_pts=pre_pts + np.array([rx, ry]),
                           L_px=L_px, h_max_px=h_max_px,
                           chord_p1=(int(A[0] + rx), int(A[1] + ry)),
                           chord_p2=(int(B[0] + rx), int(B[1] + ry)),
                           n_arc_samples=len(pre_pts),
                           n_skel_samples=len(skel_pts_local))

    # Lift everything back to full-frame coordinates.
    off = np.array([rx, ry])
    arc_pts_global = arc_pts_local + off
    skel_pts_global = skel_pts_local + off
    cxg, cyg = cx + rx, cy + ry
    p1 = (int(A[0] + rx), int(A[1] + ry))
    p2 = (int(B[0] + rx), int(B[1] + ry))
    apex_g = (int(apex[0] + rx), int(apex[1] + ry))
    foot_g = (int(foot[0] + rx), int(foot[1] + ry))

    return FrameResult(
        ok=True, frame_idx=frame_idx, needle_px=npx, mask=mask,
        arc_pts=arc_pts_global, skeleton_pts=skel_pts_global,
        cx=cxg, cy=cyg, R_px=R,
        L_px=L_px, h_max_px=h_max_px, theta_deg=theta_deg,
        thickness_px=t_px, fit_resid_px=rms,
        chord_p1=p1, chord_p2=p2, apex=apex_g, apex_foot=foot_g,
        R_mm=R_mm, D_mm=2.0 * R_mm, L_mm=L_mm, h_max_mm=h_mm,
        thickness_mm=t_mm, strain_pct=strain,
        n_arc_samples=len(arc_pts_local),
        n_skel_samples=len(skel_pts_local),
    )


# ----------------------------------------------------------------- overlay
def _draw_dashed_circle(img: np.ndarray, center: tuple[int, int], radius: int,
                        color: tuple[int, int, int], thickness: int = 4,
                        dash_deg: int = 10, gap_deg: int = 6) -> None:
    """Dashed full circle drawn as a sequence of ellipse arcs."""
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
    if res.mask is not None and show_mask:
        tint = np.zeros_like(out)
        tint[res.mask > 0] = (0, 255, 255)   # yellow on the gold needle
        out = cv2.addWeighted(out, 1.0, tint, 0.40, 0)

    if not res.ok:
        if show_panel:
            cv2.putText(out, f"frame {res.frame_idx}: no needle ({res.needle_px} px)",
                        (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.4,
                        (0, 0, 255), 3, cv2.LINE_AA)
        return out

    if show_arc_pts and res.arc_pts is not None and len(res.arc_pts):
        for x, y in res.arc_pts[::3]:
            cv2.circle(out, (int(x), int(y)), 2, (0, 255, 0), -1, cv2.LINE_AA)

    if show_circle:
        center = (int(round(res.cx)), int(round(res.cy)))
        radius = int(round(res.R_px))
        _draw_dashed_circle(out, center, radius, (0, 0, 255),
                            thickness=max(3, frame.shape[0] // 720),
                            dash_deg=12, gap_deg=8)
        cv2.drawMarker(out, center, (0, 0, 255), cv2.MARKER_CROSS,
                       28, max(2, frame.shape[0] // 900), cv2.LINE_AA)
        # Diameter label near the circle
        if frame.shape[0] > 400:
            lbl = f"D = {res.D_mm:.3f} mm"
            label_pos = (center[0] - radius, max(40, center[1] - radius - 20))
            (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 1.3, 3)
            x0, y0 = label_pos
            cv2.rectangle(out, (x0 - 8, y0 - th - 12),
                          (x0 + tw + 12, y0 + 8), (255, 255, 255), -1)
            cv2.rectangle(out, (x0 - 8, y0 - th - 12),
                          (x0 + tw + 12, y0 + 8), (0, 0, 0), 2)
            cv2.putText(out, lbl, (x0, y0),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                        cv2.LINE_AA)

    if show_chord:
        cv2.line(out, res.chord_p1, res.chord_p2, (0, 200, 0), 2, cv2.LINE_AA)
        cv2.circle(out, res.chord_p1, 8, (0, 200, 0), -1, cv2.LINE_AA)
        cv2.circle(out, res.chord_p2, 8, (0, 200, 0), -1, cv2.LINE_AA)
        # h_max sagitta segment (apex -> its foot on the chord), like Fig S17.
        if res.apex != (0, 0):
            cv2.line(out, res.apex_foot, res.apex, (0, 165, 255),
                     max(2, frame.shape[0] // 900), cv2.LINE_AA)
            cv2.circle(out, res.apex, 8, (0, 165, 255), -1, cv2.LINE_AA)
            if frame.shape[0] > 400:
                hx, hy = res.apex_foot
                cv2.putText(out, f"h={res.h_max_mm:.3f}mm  L={res.L_mm:.3f}mm",
                            (hx + 12, hy), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            (0, 165, 255), 2, cv2.LINE_AA)

    if show_panel:
        panel_w, panel_h = 760, 360
        overlay = out.copy()
        cv2.rectangle(overlay, (20, 20), (20 + panel_w, 20 + panel_h),
                      (0, 0, 0), -1)
        out = cv2.addWeighted(overlay, 0.55, out, 0.45, 0)
        ts = f"  t={t_seconds:.2f}s" if t_seconds is not None else ""
        lines = [
            f"frame  {res.frame_idx:>4d}{ts}",
            f"R      {res.R_mm:.3f} mm   ({res.R_px:.1f} px)",
            f"D      {res.D_mm:.3f} mm",
            f"angle  {res.theta_deg:.1f} deg",
            f"t      {res.thickness_mm*1000:.1f} um",
            f"strain {res.strain_pct:.3f} %",
            f"arc N  {res.n_arc_samples} / skel {res.n_skel_samples}",
            f"fit RMS {res.fit_resid_px:.2f} px",
        ]
        for i, txt in enumerate(lines):
            cv2.putText(out, txt, (40, 70 + i * 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (255, 255, 255), 2, cv2.LINE_AA)
    return out


# ----------------------------------------------------------------- calibration
def detect_scale_bar(frame: np.ndarray) -> float | None:
    """Detect the '1 mm' bar in the lower-right of the field.  Returns
    mm-per-pixel, or None on failure."""
    h, w = frame.shape[:2]
    roi = frame[int(h * 0.83):int(h * 0.97), int(w * 0.70):w]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = 0
    for c in cnts:
        x, y, cw, ch = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        fill = area / max(cw * ch, 1)
        if cw > 80 and 6 < ch < cw * 0.4 and fill > 0.7 and cw > best:
            best = cw
    return (1.0 / best) if best else None
