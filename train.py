"""
train.py - the brush ladder: straight strokes at fixed depths (12 depths x LADDER_REPEATS). It homes the machine, finds the paper,
paints every stroke with WATER on the magic paper, photographs it wet, measures where it marks and how wide, waits until it has
dried, and saves the PROFILE: ladder_dataset/<name>.json. paint.py then paints every mark at the depth where the real brush
makes that width.

  python train.py --name S4Z32        the usual run (numbers from settings.py: TOUCH_Z, PAINT_FEED, LADDER_*)
  python train.py --no_home           the machine is already homed
  python train.py --new_paper         a fresh sheet: forget where the earlier strokes were
Moved or re-mounted the brush? Run it again. Do not touch the brush between this and painting.
(The older modes train / trace / paint / shape are in arhiv/ - they need the old config.py set.)
"""
import json
import math
import random
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

import brush_profile as BP
import machine as _machine
import settings as S
from machine import Machine, park
from monitor import MON, CameraStream
from settings import (CANVAS_CORNERS_MM, FEED_TRAVEL, ACCEL_TRAVEL, X_OFFSET, Y_OFFSET, DIP_X, DIP_Y, DIP_Z, DIP_JITTER,
                      DIP_SPIRAL_LOOPS, DIP_SPIRAL_R, DIP_WIPE_R, Z_SAFE_DIP, Z_WIPE_TRAVEL, PLUNGE_FEED, PARK_X, PARK_Y)
from vision import pixel_to_mm, mm_to_pixel, detect_paper

# heights: logical Z = mm above the touch height (0 = touch); machine Z = logical Z + TOUCH_Z
Z_GLOBAL_OFF = S.TOUCH_Z
Z_PAINT_MAX = 0.0
Z_PAINT_MIN = -(Z_GLOBAL_OFF - 0.2) if S.TRAIN_DEPTH_MM is None else -float(S.TRAIN_DEPTH_MM)
ACCEL_PAINT = S.PAINT_ACCEL

_machine.VERBOSE = False                       # terminal stays quiet: no G-code echo
_machine.LINE_HOOK = MON.line_hook             # the monitor watches the G-code silently

_builtin_print = print

def print(*args, **kw):                    # every status message also feeds the monitor window
    msg = " ".join(str(a) for a in args).split(" (widths:")[0]   # drop long width lists
    MON.feed(msg)
    if "Spine Pt" not in msg:
        _builtin_print(msg[:140], **kw)

# The trainer never opens its own windows any more; the monitor owns the display.
class _CV2Proxy:
    def __getattr__(self, n): return getattr(_cv2, n)
    imshow = staticmethod(MON.imshow)
    waitKey = staticmethod(MON.waitKey)
    destroyAllWindows = staticmethod(MON.noop)

_cv2 = cv2

cv2 = _CV2Proxy()

def mz(z_logical):
    """Convert a logical/paper-relative Z (as used everywhere in this file and
    in settings.py, e.g. DIP_Z) into the actual machine Z coordinate
    to send over serial. Lets logical Z go negative (pressing into paper) even
    though the physical mount was lowered so machine Z=0 sits above paper."""
    return z_logical + Z_GLOBAL_OFF

# Tuning Constants for Magic Water Paper
MIN_DRY_TIME = 10.0          # Minimum seconds to let water sit before checking dry state

DRY_CHECK_INTERVAL = 1.0     # Time between camera snapshots

DRY_STABLE_WINDOW = 5        # Number of stable frames needed to confirm pure white paper

HOME_AT_START = True        # every session homes the machine first (Z, then XY); switch off with --no_home

def _home(machine):
    """Homes the machine so every absolute position is right. Skipped with --no_home (machine already homed)."""
    if HOME_AT_START:
        print("🏠 Homing the machine (Z first, then XY). Use --no_home if it is already homed and you want to skip this.")
        machine.home_all()
    else:
        print("🏠 Not homing (--no_home): the machine must already be homed.")

def _park_fast(machine, timeout=30):
    """Lift and go to the park position at travel speed AND travel acceleration (see machine.park)."""
    park(machine, PARK_X, PARK_Y, mz(Z_SAFE_DIP), timeout=timeout)


def _generate_dip_gcode(target_x, target_y):
    """Generates the high-clearance water dish dip and spiral wipe sequence."""
    lines = [
        "\n; --- HIGH-CLEARANCE PETRI DIP ---",
        f"G0 Z{mz(Z_SAFE_DIP):.3f} F3000",
        "M400",
        f"M204 P{ACCEL_TRAVEL} T{ACCEL_TRAVEL}",
        f"G0 X{DIP_X:.3f} Y{DIP_Y:.3f} F{FEED_TRAVEL}"
    ]
    ax = DIP_X + random.uniform(-DIP_JITTER, DIP_JITTER)
    ay = DIP_Y + random.uniform(-DIP_JITTER, DIP_JITTER)
    
    lines.append(f"G0 X{ax:.3f} Y{ay:.3f}")
    lines.append(f"G1 Z{mz(DIP_Z):.3f} F1000")

    steps = int(DIP_SPIRAL_LOOPS * 8)
    for i in range(steps + 1):
        ang = i * (2 * math.pi / steps)
        r   = DIP_SPIRAL_R * (i / steps)
        lines.append(f"G1 X{ax + r*math.cos(ang):.3f} Y{ay + r*math.sin(ang):.3f} F1500")

    dx, dy = target_x - DIP_X, target_y - DIP_Y
    dist = math.hypot(dx, dy)
    if dist > 0:
        wx = DIP_X + (dx / dist) * DIP_WIPE_R
        wy = DIP_Y + (dy / dist) * DIP_WIPE_R
    else:
        wx, wy = DIP_X + DIP_WIPE_R, DIP_Y

    lines.append(f"G1 Z{mz(Z_WIPE_TRAVEL):.3f} F1500")
    lines.append(f"G1 X{wx:.3f} Y{wy:.3f} F1000")
    lines.append(f"G1 Z{mz(Z_SAFE_DIP):.3f} F800")
    lines.append("M400")
    lines.append(f"G0 X{target_x:.3f} Y{target_y:.3f} F{FEED_TRAVEL}")
    return lines

# ── stroke placement: spread strokes over the canvas ─────────────────────────────
# Painting every stroke in the same place soaks and wears the paper (water pools where the
# path ends). Each stroke gets a random shift so it lands inside the four calibration dots
# (keeping it in the camera view), preferring the spot farthest from all earlier strokes.
PLACE_MARGIN_MM = 6.0      # keep strokes at least this far inside the four dots

PLACE_FILE      = Path(__file__).with_name("placed_strokes.json")   # remembers spots between runs

_PLACED = []               # machine-mm point arrays (N x 2) of strokes already painted on this paper

def _init_placement(new_paper=False):
    """Load the spots used so far on this sheet (or forget them when a fresh sheet is loaded)."""
    global _PLACED
    _PLACED = []
    if new_paper:
        if PLACE_FILE.exists():
            PLACE_FILE.unlink()
        print("  🧻 Fresh paper: stroke history cleared")
        return
    try:
        _PLACED = [np.array(p, float) for p in json.load(open(PLACE_FILE))]
        if _PLACED:
            print(f"  🧻 {len(_PLACED)} earlier strokes remembered on this paper (use --new_paper after changing sheets)")
    except Exception:
        _PLACED = []

def _pick_shift(pts_xy, tries=200):
    """Returns (dx, dy, gap_mm): a random shift of the stroke `pts_xy` (N x 2, machine mm) that keeps
    it inside the four dots (minus PLACE_MARGIN_MM) and, among `tries` random candidates, is as far
    as possible from every stroke painted before. gap_mm = distance to the nearest earlier stroke."""
    P = np.asarray(pts_xy, float)
    x0, x1, y0, y1 = _limits()
    lo = np.array([x0 - P[:, 0].min(), y0 - P[:, 1].min()])
    hi = np.array([x1 - P[:, 0].max(), y1 - P[:, 1].max()])
    base = np.where(hi >= lo, lo, (lo + hi) / 2)          # stroke larger than the canvas: just centre it
    span = np.maximum(hi - lo, 0.0)
    cand = base + np.random.uniform(size=(tries, 2)) * span
    gap = None
    if _PLACED:
        sub = P[::max(1, len(P) // 20)]
        prev = np.vstack([q[::max(1, len(q) // 20)] for q in _PLACED[-300:]])
        prev = prev[::max(1, len(prev) // 600)]
        d = np.linalg.norm((sub[None, :, None, :] + cand[:, None, None, :]) - prev[None, None, :, :], axis=3)
        dmin = d.min(axis=(1, 2))
        k = int(np.argmax(dmin)); gap = float(dmin[k])
    else:
        k = 0
    best = cand[k]
    _PLACED.append(P + best)
    try:
        json.dump([np.round(q[::max(1, len(q) // 20)], 1).tolist() for q in _PLACED[-300:]], open(PLACE_FILE, "w"))
    except Exception:
        pass
    return float(best[0]), float(best[1]), gap

# ── stroke location + drying: independent of the calibration ────────────────────
MAX_DRY_TIME  = 240.0    # give up waiting after this many seconds

DRY_DONE_RESIDUAL_FRAC = 0.10   # dry = at most this fraction of the first darkness is left ...

DRY_NOISE_FLOOR = 1.5          # ... or this many gray levels (the camera's own noise: a faint stroke cannot be measured below it)

DRY_PLATEAU_LOOKS = 20         # a stroke that has not faded for this many looks (seconds) and is already mostly gone counts as dry

_BEFORE = None           # photo of the BLANK paper (toolhead parked) taken before each stroke

def _park_and_snapshot(machine, cam):
    """Park the toolhead out of view and photograph the blank paper (reference for the stroke)."""
    global _BEFORE
    _park_fast(machine, timeout=30)
    machine.send_line("M400")
    machine.wait_motion_complete()
    _BEFORE = cam.capture(flush=10)
    if _AREA is None:
        _set_training_area(_BEFORE, cam)

def _find_new_ink(wet, before, expected_px=None):
    """Mask of paint that appeared between the blank-paper photo and now = the wet stroke.
    Uses two thresholds (hysteresis): a high one finds sure paint, a low one then grows it along faint
    parts (light-touch thin sections, fading ends), so faint parts stay attached while isolated shadows
    and noise (no sure paint inside) are dropped. If the expected path (pixels) is given, every blob near
    it is kept - a stroke broken by a gap stays whole - and blobs far from it (shadows) are ignored."""
    gw = cv2.GaussianBlur(cv2.cvtColor(wet, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    gb = cv2.GaussianBlur(cv2.cvtColor(before, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    d32 = gb.astype(np.float32) - gw.astype(np.float32)
    d32 -= np.median(d32)                            # ignore global brightness change (auto-exposure)
    diff = np.clip(d32, 0, 255).astype(np.uint8)
    if int(diff.max()) < 14:
        return None
    t, _ = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    t_hi = max(t * 0.75, 14.0)                       # sure paint
    t_lo = max(0.4 * t_hi, 7.0)                      # faint paint, accepted only when attached to sure paint
    n, lab = cv2.connectedComponents((diff > t_lo).astype(np.uint8) * 255, connectivity=8)
    seeds = np.unique(lab[diff > t_hi]); seeds = seeds[seeds > 0]
    if len(seeds) == 0:
        return None
    mask = (np.isin(lab, seeds)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return None
    if expected_px is not None and len(expected_px) > 1:
        band = np.zeros(mask.shape, np.uint8)
        thick = int(max(20, 2 * 40))                 # +-40 px around the expected path (about 12 mm)
        cv2.polylines(band, [np.asarray(expected_px, np.int32).reshape(-1, 1, 2)], False, 255, thick)
        keep = [k for k in range(1, n) if st[k, cv2.CC_STAT_AREA] >= 40 and (band[lab == k] > 0).any()]
        if keep:
            return np.where(np.isin(lab, keep), 255, 0).astype(np.uint8)
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    if st[k, cv2.CC_STAT_AREA] < 60:
        return None
    return np.where(lab == k, 255, 0).astype(np.uint8)

def _wait_for_dry(cam, roi_bbox, ink_mask, wet_gray_roi):
    """Waits until the stroke has faded back to the blank-paper look.
    How dark the stroke is NOW compared with the blank-paper photo is measured on the CORE of the stroke (the pixels that really
    were dark, not the blurred edge of the mask), corrected for lighting drift with the MEDIAN of the paper around it.
    'Dry' = the median of the last 5 looks is below max(10% of the first darkness, 1.5 gray levels) - the second number is the
    camera's own noise, which a faint (light-touch or thin) stroke cannot be measured below. A stroke that has stopped fading
    for 20 s and is already mostly gone also counts (some paper keeps a faint shade for good)."""
    x1, y1, x2, y2 = roi_bbox
    m = ink_mask > 0
    if _BEFORE is None or m.sum() < 20:
        MON.min_wait(MIN_DRY_TIME + 20)
        return
    ring = cv2.dilate(ink_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))) == 0
    bef = cv2.GaussianBlur(cv2.cvtColor(_BEFORE, cv2.COLOR_BGR2GRAY), (5, 5), 0)[y1:y2, x1:x2].astype(np.float32)
    wet_dark = bef - cv2.GaussianBlur(wet_gray_roi, (5, 5), 0).astype(np.float32)
    core = m & (wet_dark >= 0.5 * float(np.percentile(wet_dark[m], 90)))
    if core.sum() < 15:
        core = m

    def darkness(g):
        d = bef - g.astype(np.float32)
        drift = float(np.median(d[ring])) if ring.sum() > 50 else 0.0           # median: not pulled by specks or the odd bright pixel
        return float(d[core].mean()) - drift

    initial = max(darkness(wet_gray_roi), 3.0)
    thr = max(DRY_DONE_RESIDUAL_FRAC * initial, DRY_NOISE_FLOOR)
    MON.min_wait(MIN_DRY_TIME)
    hist, t0 = [], time.time()
    while True:
        g = cv2.GaussianBlur(cv2.cvtColor(cam.capture(flush=3), cv2.COLOR_BGR2GRAY), (5, 5), 0)[y1:y2, x1:x2]
        hist.append(darkness(g))
        last = hist[-DRY_STABLE_WINDOW:]
        med = float(np.median(last))
        frac = 1.0 - med / initial
        stable = sum(1 for v in last if v <= thr)
        done = len(hist) >= DRY_STABLE_WINDOW and med <= thr
        settled = (len(hist) >= DRY_PLATEAU_LOOKS + DRY_STABLE_WINDOW and med <= 0.4 * initial
                   and float(np.median(hist[-DRY_STABLE_WINDOW - DRY_PLATEAU_LOOKS:-DRY_PLATEAU_LOOKS])) - med < 0.3)
        if MON.set_dry(stable, DRY_STABLE_WINDOW, frac, roi_bbox):
            print("  Drying wait skipped by user"); break
        if done:
            print(f"  Verified dry ({frac*100:.0f}% faded, {med:.1f} of {initial:.1f} gray levels left). Next stroke..."); break
        if settled:
            print(f"  Stopped fading at {frac*100:.0f}% ({med:.1f} of {initial:.1f} gray levels left) - counts as dry. Next stroke..."); break
        if MON.abort:
            break
        if time.time() - t0 > MAX_DRY_TIME:
            print(f"  ⚠️ Still {frac*100:.0f}% faded after {MAX_DRY_TIME:.0f}s ({med:.1f} of {initial:.1f} gray levels left) - continuing anyway"); break
        MON.waitKey(int(DRY_CHECK_INTERVAL * 1000))

_AREA = None                 # (x0, x1, y0, y1) machine mm where strokes may be painted; set once from the first photo of the paper

TRAIN_MARGIN_MM = 12.0       # keep strokes this far inside the paper edge

def _set_training_area(frame, cam):
    """Where may a test stroke go? 1) TRAIN_AREA_MM = (x0, x1, y0, y1) in settings.py, 2) the paper that the camera finds in the
    photo (inscribed rectangle minus TRAIN_MARGIN_MM), 3) the four calibration corners - with a warning, because a calibration
    made over a bigger area than the paper would paint on the wood."""
    global _AREA
    a = S.TRAIN_AREA_MM
    if a:
        _AREA = tuple(float(v) for v in a)
        print(f"  🧻 test strokes go inside X {_AREA[0]:.0f}..{_AREA[1]:.0f}, Y {_AREA[2]:.0f}..{_AREA[3]:.0f} mm (TRAIN_AREA_MM from settings.py)")
        return
    det = None
    try:
        det = detect_paper(frame, cam.H, cam.H_inv, calib_mm=CANVAS_CORNERS_MM)
    except Exception as e:
        print(f"  (paper search failed: {e})")
    if det is not None and min(det["size_mm"]) >= 45.0 and det["fill"] >= 0.85:
        c = np.array(det["corners_mm"], float)
        xs, ys = np.sort(c[:, 0]), np.sort(c[:, 1])
        m = TRAIN_MARGIN_MM
        x0, x1, y0, y1 = xs[1] + m, xs[2] - m, ys[1] + m, ys[2] - m          # the rectangle that fits inside the paper, minus the margin
        if x1 - x0 > 5 and y1 - y0 > 5:
            _AREA = (float(x0), float(x1), float(y0), float(y1))
            print(f"  🧻 paper found by the camera: {det['size_mm'][0]:.0f} x {det['size_mm'][1]:.0f} mm. Test strokes go inside "
                  f"X {x0:.0f}..{x1:.0f}, Y {y0:.0f}..{y1:.0f} mm ({m:.0f} mm from its edge)")
            return
    raise SystemExit(
        "The paper was NOT found in the photo, so the trainer does not know where it may paint (it would paint on the wood).\n"
        "  - Is the paper fully visible, with the toolhead parked out of the picture, and is the centre of the calibration area on the paper?\n"
        "  - Or set the area yourself in settings.py (machine mm, inside the paper):  TRAIN_AREA_MM = (x0, x1, y0, y1)\n"
        "    e.g. the old behaviour, the calibration corners:  TRAIN_AREA_MM = (86.0, 194.0, 356.0, 424.0)")

def _limits():
    """(x0, x1, y0, y1) in machine mm where a stroke may go: the paper area (see _set_training_area), else the four dots minus PLACE_MARGIN_MM."""
    if _AREA is not None:
        return _AREA
    xs = [c[0] for c in CANVAS_CORNERS_MM]; ys = [c[1] for c in CANVAS_CORNERS_MM]
    return (min(xs) + PLACE_MARGIN_MM, max(xs) - PLACE_MARGIN_MM,
            min(ys) + PLACE_MARGIN_MM, max(ys) - PLACE_MARGIN_MM)

# ══ LADDER MODE: straight strokes, each at ONE fixed Z -> how much of it leaves a mark, and how wide ══
def _ink_diff(wet, before):
    """How much darker the paper is now than before the stroke (a little blur only, so edges stay sharp)."""
    gw = cv2.GaussianBlur(cv2.cvtColor(wet, cv2.COLOR_BGR2GRAY), (3, 3), 0).astype(np.float32)
    gb = cv2.GaussianBlur(cv2.cvtColor(before, cv2.COLOR_BGR2GRAY), (3, 3), 0).astype(np.float32)
    d = gb - gw
    return d - float(np.median(d))

def _ladder_measure(comp, diff, cam, p0, p2, margin_mm=5.0, step_mm=1.0, search_mm=7.0, min_w_mm=0.5):
    """comp = full-frame mask of the new paint (None = nothing seen), diff = how much darker each pixel became.
    p0, p2 = the commanded line in canvas mm. For every mm along the line (not the first/last margin_mm: touch-down and
    lift-off): the paint across the line - its width at HALF of its own darkness (so the blur and the generous mask do not
    widen it) - or 'no mark'. Returns [(s_mm, visible, width_mm)]."""
    from scipy.ndimage import map_coordinates
    dxl, dyl = p2[0] - p0[0], p2[1] - p0[1]
    L = math.hypot(dxl, dyl)
    n = max(1, int((L - 2 * margin_mm) / step_mm) + 1)
    out = []
    for k in range(n):
        s = margin_mm + k * step_mm
        t = s / L
        x, y = p0[0] + t * dxl, p0[1] + t * dyl
        if comp is None or diff is None:
            out.append((s, False, 0.0)); continue
        hh, ww = comp.shape[:2]
        u, v = mm_to_pixel(cam.H_inv, x + X_OFFSET, y + Y_OFFSET)
        u2, v2 = mm_to_pixel(cam.H_inv, x + X_OFFSET + dxl / L, y + Y_OFFSET + dyl / L)        # 1 mm further along the line
        tx, ty = u2 - u, v2 - v
        mag = math.hypot(tx, ty)
        if mag < 1e-9:
            out.append((s, False, 0.0)); continue
        nx, ny = -ty / mag, tx / mag                                                           # across the line
        pm0, pm1 = pixel_to_mm(cam.H, u, v), pixel_to_mm(cam.H, u + nx, v + ny)
        mpp = math.hypot(pm1[0] - pm0[0], pm1[1] - pm0[1])                                    # mm per pixel ACROSS the line
        R = max(4, int(search_mm / mpp))
        offs = np.arange(-R, R + 0.01, 0.5)
        xs, ys = u + offs * nx, v + offs * ny
        inside = (xs >= 0) & (xs <= ww - 1) & (ys >= 0) & (ys <= hh - 1)
        if not inside.any():
            out.append((s, False, 0.0)); continue
        prof = np.zeros(len(offs), bool)
        prof[inside] = comp[np.round(ys[inside]).astype(int), np.round(xs[inside]).astype(int)] > 0
        d = np.diff(np.r_[0, prof.astype(np.int8), 0])
        starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
        if len(starts) == 0:
            out.append((s, False, 0.0)); continue
        centres = 0.5 * (starts + ends - 1) * 0.5 - R                                          # in px (0.5 px steps)
        j = int(np.argmin(np.abs(centres)))
        vals = map_coordinates(diff, [np.clip(ys, 0, hh - 1), np.clip(xs, 0, ww - 1)], order=1, mode="nearest")
        peak = float(np.percentile(vals[starts[j]:ends[j]], 90))                               # the darkness of the stroke itself
        if peak < 6.0:
            out.append((s, False, 0.0)); continue
        half = 0.5 * peak
        c = int((starts[j] + ends[j] - 1) // 2)
        lo = c
        while lo > 0 and vals[lo] >= half:
            lo -= 1
        hi = c
        while hi < len(vals) - 1 and vals[hi] >= half:
            hi += 1
        xl = lo + (half - vals[lo]) / max(vals[lo + 1] - vals[lo], 1e-6) if vals[lo] < half else float(lo)   # where the darkness crosses half,
        xr = hi - 1 + (vals[hi - 1] - half) / max(vals[hi - 1] - vals[hi], 1e-6) if vals[hi] < half else float(hi)  # interpolated between samples
        w_mm = float((xr - xl) * 0.5 * mpp)                                                    # full width at half maximum
        out.append((s, w_mm >= min_w_mm, w_mm if w_mm >= min_w_mm else 0.0))
    return out

def _ladder_overlay(wet, cam, p0, p2, meas, z, margin_mm=5.0):
    """The wet photo with the measurement drawn on it: a label with the width every few mm, a red cross where there is no mark."""
    img = wet.copy()
    dxl, dyl = p2[0] - p0[0], p2[1] - p0[1]
    L = math.hypot(dxl, dyl)
    ws = [w for _, v, w in meas if v]
    for i, (s, vis, w) in enumerate(meas):
        x, y = p0[0] + s / L * dxl, p0[1] + s / L * dyl
        u, v = mm_to_pixel(cam.H_inv, x + X_OFFSET, y + Y_OFFSET)
        u, v = int(round(u)), int(round(v))
        if vis:
            cv2.circle(img, (u, v), 2, (255, 0, 0), -1)
            if i % 4 == 0:
                cv2.putText(img, f"{w:.1f}mm", (u + 8, v + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 0, 180), 1)
        elif i % 2 == 0:
            cv2.drawMarker(img, (u, v), (0, 0, 255), cv2.MARKER_CROSS, 8, 2)
    head = f"Z {z:.2f} (machine {mz(z):.2f}):  marked {100 * len(ws) / max(len(meas), 1):.0f}%" + (f",  width median {np.median(ws):.2f} mm" if ws else ",  NO MARK")
    cv2.putText(img, head, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 5)
    cv2.putText(img, head, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    return img

def run_ladder_session(levels=None, repeats=None, length_mm=None, feed=None, z_top=None, z_bottom=None, name=None):
    """Straight strokes at fixed depths from just above the contact height down to the deepest Z, in random order.
    Result: data in ladder_dataset/ and a table Z -> % marked, width, plus the numbers for paint.py."""
    levels = int(levels if levels is not None else S.LADDER_LEVELS)
    repeats = int(repeats if repeats is not None else S.LADDER_REPEATS)
    length_mm = float(length_mm if length_mm is not None else S.LADDER_LENGTH_MM)
    feed = float(feed if feed is not None else S.PAINT_FEED)          # the profile is measured at the speed you paint at
    z_lo, z_hi = min(Z_PAINT_MIN, Z_PAINT_MAX), max(Z_PAINT_MIN, Z_PAINT_MAX)
    explicit_range = z_top is not None and z_bottom is not None
    z_top = z_hi + float(S.LADDER_ABOVE_MM) if z_top is None else z_top
    if z_bottom is None:
        z_bottom = z_lo
        if mz(z_bottom) < 0.2:                                   # the default must never go down to the endstop
            z_bottom = 0.2 - Z_GLOBAL_OFF
            print(f"  ℹ️ the bottom of your Z range would be machine Z {mz(z_lo):.2f} (the endstop): the ladder stops at machine Z 0.20 instead")
    if z_top <= z_bottom:
        raise SystemExit("--z_top must be higher than --z_bottom")
    print(f"  settings.py: TOUCH_Z = {Z_GLOBAL_OFF}  (machine Z = logical Z + this),  TRAIN_DEPTH_MM = {S.TRAIN_DEPTH_MM}")
    if (z_top - z_bottom) < 1.0 and not explicit_range:
        raise SystemExit(
            f"The ladder would only cover machine Z {mz(z_bottom):.2f} .. {mz(z_top):.2f} = {z_top - z_bottom:.2f} mm: too little to see anything.\n"
            f"  It comes from settings.py: TOUCH_Z = {Z_GLOBAL_OFF}, TRAIN_DEPTH_MM = {S.TRAIN_DEPTH_MM}.\n"
            f"  Set TOUCH_Z to the machine Z where the tip just touches the paper (for example 3.0) and TRAIN_DEPTH_MM = None.\n"
            f"  Or give the range yourself in logical Z (0 = touch): --z_top 0.5 --z_bottom -2.8")
    if mz(z_bottom) < 0.1:
        raise SystemExit(f"--z_bottom {z_bottom} would send machine Z {mz(z_bottom):.2f}: too close to the endstop (use at least 0.1)")
    zs = [round(float(z), 3) for z in np.linspace(z_top, z_bottom, levels)]
    order = [(z, r) for z in zs for r in range(repeats)]
    random.shuffle(order)
    out_dir = Path("ladder_dataset"); out_dir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = (name or f"ladder_{stamp}").replace(".json", "")
    if name and (out_dir / f"{base}.json").exists():                       # never overwrite a profile
        base = f"{base}_{stamp}"
        print(f"  ℹ️ '{name}.json' exists already - this one is saved as {base}.json")
    MON.set_mode('LADDER', 'LADDER: mark and width vs brush depth (Z)', [
        'Straight strokes, each at ONE fixed Z,', 'from just above contact down to the deepest Z.',
        'Per millimetre: is there a mark, how wide?', 'Result: contact height + width curve,', 'and the numbers for paint.py.'],
        len(order), chart_label='width along stroke (mm)')
    samples = []
    print(f"🪜 LADDER: {levels} depths from Z={z_top:.2f} down to Z={z_bottom:.2f} (machine Z {mz(z_top):.2f} .. {mz(z_bottom):.2f}), "
          f"{repeats} strokes each = {len(order)} strokes of {length_mm:.0f} mm at F{feed:.0f}")
    with Machine() as machine, CameraStream(MON) as cam:
        _home(machine)
        machine.set_absolute()
        machine.set_units_mm()
        for idx, (z, rep_i) in enumerate(order):
            if MON.abort:
                print('  Stopped by user (ESC)'); break
            _park_and_snapshot(machine, cam)
            MON.new_attempt(idx + 1, len(order))              # the monitor's yellow trail starts again with every stroke
            MON.paint_n = 21
            print(f"\n🪜 --- LADDER STROKE {idx + 1}/{len(order)}:  Z = {z:.2f}  (machine Z {mz(z):.2f}) ---")
            x_mid = (CANVAS_CORNERS_MM[0][0] + CANVAS_CORNERS_MM[1][0]) / 2
            y_mid = (CANVAS_CORNERS_MM[0][1] + CANVAS_CORNERS_MM[2][1]) / 2
            p0, p2 = (x_mid - length_mm / 2, y_mid), (x_mid + length_mm / 2, y_mid)
            _dx, _dy, _gap = _pick_shift(np.linspace(p0, p2, 20))
            p0, p2 = (p0[0] + _dx, p0[1] + _dy), (p2[0] + _dx, p2[1] + _dy)
            MON.set_params(z_logical=z, z_machine=mz(z), feed_mm_min=feed, stroke_len_mm=length_mm)
            ax, ay = p0[0] + X_OFFSET, p0[1] + Y_OFFSET
            gcode = _generate_dip_gcode(ax, ay)
            gcode.extend([f"M204 P{ACCEL_TRAVEL} T{ACCEL_TRAVEL}", f"G0 Z{mz(Z_SAFE_DIP):.3f} F3000", f"G0 X{ax:.3f} Y{ay:.3f}", "M400",
                          f"M204 P{ACCEL_PAINT} T{ACCEL_PAINT}", f"G1 Z{mz(z):.3f} F{PLUNGE_FEED}"])
            path_points = []
            for t in np.linspace(0, 1, 21):
                x, y = (1 - t) * p0[0] + t * p2[0], (1 - t) * p0[1] + t * p2[1]
                gcode.append(f"G1 X{x + X_OFFSET:.3f} Y{y + Y_OFFSET:.3f} Z{mz(z):.3f} F{int(feed)}")
                path_points.append({'x': x, 'y': y, 'z': z})
            gcode.extend([f"G0 Z{mz(Z_SAFE_DIP):.3f} F3000", "M400"])
            print("  🎨 Dipping and drawing the straight stroke at one fixed depth...")
            machine.send_stroke(gcode)
            machine.wait_motion_complete()
            t_stroke_end = time.time()
            _park_fast(machine, timeout=20)
            machine.send_line("M400")
            machine.wait_motion_complete()
            print("  📸 Photographing the wet stroke...")
            wet = cam.capture(flush=15)
            print(f"  ⏱ the photo was taken {time.time() - t_stroke_end:.1f} s after the stroke ended")
            expected = np.array([mm_to_pixel(cam.H_inv, p['x'] + X_OFFSET, p['y'] + Y_OFFSET) for p in path_points])
            comp = _find_new_ink(wet, _BEFORE, expected) if _BEFORE is not None else None
            meas = _ladder_measure(comp, _ink_diff(wet, _BEFORE) if (comp is not None and _BEFORE is not None) else None, cam, p0, p2)
            for s_mm, vis, w in meas:
                samples.append({'stroke': idx, 'repeat': rep_i, 'z': z, 'z_machine': round(mz(z), 3), 'feed': feed, 's_mm': round(float(s_mm), 1),
                                'visible': bool(vis), 'width_mm': round(float(w), 2)})
            nvis = sum(1 for _, v, _ in meas if v)
            wv = [w for _, v, w in meas if v]
            print(f"  ✔ marked along {100 * nvis / max(len(meas), 1):.0f}% of the stroke" + (f", median width {np.median(wv):.2f} mm" if wv else " (no mark)"))
            if wv:
                MON.set_result(points=nvis, marked_pct=100 * nvis / max(len(meas), 1), w_min=float(min(wv)), w_med=float(np.median(wv)), w_max=float(max(wv)))
            else:
                MON.set_result(points=0, marked_pct=0.0)
            MON.set_chart([w for _, _, w in meas])                          # width along the stroke (0 = no mark)
            MON.set_records(len(samples))
            MON.set_ladder(BP.rows_from_samples(samples), current_z=z)
            MON.add_log(f"Z {z:.2f}: marked {100 * nvis / max(len(meas), 1):.0f}%" + (f", width {np.median(wv):.2f} mm" if wv else ", no mark"))
            cv2.imshow("Ladder measurement", cv2.resize(_ladder_overlay(wet, cam, p0, p2, meas, z), (960, 540)))
            cv2.waitKey(2500)                                                  # the measured points stay on screen for a moment
            json.dump({'setup': {'Z_GLOBAL_OFF': Z_GLOBAL_OFF, 'feed': feed, 'length_mm': length_mm, 'levels': zs, 'repeats': repeats,
                                 'accel_paint': ACCEL_PAINT, 'plunge_feed': PLUNGE_FEED, 'touch_z': S.TOUCH_Z, 'created': stamp}, 'samples': samples},
                      open(out_dir / f"{base}.json", 'w'), indent=1)
            if comp is not None:                                   # wait until the water has gone before the next stroke
                ys_, xs_ = np.where(comp > 0)
                PAD = 25
                x1, x2 = max(0, int(xs_.min()) - PAD), min(wet.shape[1], int(xs_.max()) + PAD)
                y1, y2 = max(0, int(ys_.min()) - PAD), min(wet.shape[0], int(ys_.max()) + PAD)
                _wait_for_dry(cam, (x1, y1, x2, y2), comp[y1:y2, x1:x2].copy(), cv2.cvtColor(wet, cv2.COLOR_BGR2GRAY)[y1:y2, x1:x2])
            else:
                MON.min_wait(MIN_DRY_TIME)
    rows = BP.rows_from_samples(samples)
    rec = BP.analyse(rows)
    report = BP.report_lines(rows, rec)
    for ln in report:
        print(ln)
    open(out_dir / f"{base}_report.txt", 'w', encoding='utf-8').write("\n".join(report) + "\n")
    print(f"\nProfile: ladder_dataset/{base}.json   Report: ladder_dataset/{base}_report.txt")
    print(f"Use it in paint.py: BRUSH_PROFILE = \"{base}\" in settings.py   (or: python paint.py artwork.png --ladder {base})")
    return rows, rec


def main():
    import argparse
    global HOME_AT_START
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default=None, help="file name of the profile, e.g. S4Z32 -> ladder_dataset/S4Z32.json (default: ladder_<date>_<time>)")
    ap.add_argument("--levels", type=int, default=None, help="number of depths (default LADDER_LEVELS in settings.py)")
    ap.add_argument("--repeats", type=int, default=None, help="strokes per depth (default LADDER_REPEATS in settings.py)")
    ap.add_argument("--length", type=float, default=None, help="stroke length in mm (default LADDER_LENGTH_MM)")
    ap.add_argument("--feed", type=float, default=None, help="painting speed mm/min (default PAINT_FEED): the profile is only valid at the speed you paint at")
    ap.add_argument("--z_top", type=float, default=None, help="highest depth, logical Z (0 = touch). Default: LADDER_ABOVE_MM above the touch height")
    ap.add_argument("--z_bottom", type=float, default=None, help="deepest depth, logical Z. Default: down to machine Z 0.2")
    ap.add_argument("--new_paper", action="store_true", help="forget earlier stroke positions (new sheet loaded)")
    ap.add_argument("--no_home", action="store_true", help="do not home the machine at the start (it must already be homed)")
    a = ap.parse_args()
    HOME_AT_START = not a.no_home
    _init_placement(a.new_paper)
    run_ladder_session(levels=a.levels, repeats=a.repeats, length_mm=a.length, feed=a.feed, z_top=a.z_top, z_bottom=a.z_bottom, name=a.name)


if __name__ == "__main__":
    main()
