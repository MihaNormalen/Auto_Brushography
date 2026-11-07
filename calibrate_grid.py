"""
calibrate_grid.py - measures the LENS of the camera from ONE photo of a sheet with a ruled grid.

    python calibrate_grid.py preview_snapshot.png            (a raw camera picture: python preview.py, F = hide text, S = save)

You need this only ONCE for a camera (or again if the lens / the zoom / the camera changes). The lens shape does not depend on where
the camera sits, so every later calibration (calibrate_dots.py) uses the same lens.json and needs only dots on ANY white paper.

What it does: finds the sheet, the grid lines and their crossings (sub-pixel, every WHOLE line is fitted as a curve), numbers the
crossings, and fits a homography + one lens term to them. The grid itself does not have to be perfect or exactly 10 mm: hand-drawn
lines are off by a few tenths of a millimetre, which is why the result is checked on lines the model has not seen (cross-validation).
Writes lens.json and calib_grid_check.png (crossings in green, the leftover error x30 in red).
"""
import argparse
import json
import math
import os
import sys
import time

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

import settings as S

LENS_FILE = getattr(S, "LENS_FILE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "lens.json"))


# ------------------------------------------------------------------------------------------------ finding the sheet and the ink
def sheet_mask(im):
    """The sheet = the biggest bright, colourless area."""
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    m = ((hsv[..., 1] < 60) & (hsv[..., 2] > 170)).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    if n < 2:
        raise SystemExit("no white sheet found in the picture")
    k = 1 + int(np.argmax([st[i, cv2.CC_STAT_AREA] for i in range(1, n)]))
    return (lab == k).astype(np.uint8) * 255


def ink_map(im, sheet):
    """How much darker than the paper around it each pixel is (the pen lines), only inside the sheet."""
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32)
    paper = float(np.median(g[sheet > 0]))
    gf = g.copy()
    gf[sheet == 0] = paper                                                  # the wood must not pull the local background
    gs = cv2.GaussianBlur(gf, (0, 0), 0.8)
    bg = cv2.GaussianBlur(gf, (0, 0), 12)
    ink = np.clip(bg - gs, 0, None)
    inner = cv2.erode(sheet, np.ones((25, 25), np.uint8))                   # stay clear of the sheet edge
    ink[inner == 0] = 0
    return ink


def rough_crossings(ink, k=19):
    thr = max(18.0, 0.25 * float(np.percentile(ink[ink > 0], 99)))
    mask = (ink > thr).astype(np.uint8) * 255
    vm = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, k)))
    hm = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1)))
    k5 = np.ones((5, 5), np.uint8)
    inter = cv2.bitwise_and(cv2.dilate(vm, k5), cv2.dilate(hm, k5))
    n, lab, st, cen = cv2.connectedComponentsWithStats(inter)
    pts = []
    yy, xx = np.mgrid[0:19, 0:19]
    for c in cen[1:]:
        x0, y0 = int(round(c[0])) - 9, int(round(c[1])) - 9
        w = ink[y0:y0 + 19, x0:x0 + 19]
        if w.shape != (19, 19) or w.sum() < 1:
            continue
        w = w * (w > 0.25 * w.max())
        pts.append((x0 + (w * xx).sum() / w.sum(), y0 + (w * yy).sum() / w.sum()))
    return np.array(pts), mask


# ------------------------------------------------------------------------------------------------ numbering the crossings
def build_lattice(pts, center):
    """Grow integer grid indices (i, j) outwards from the crossing nearest the centre, following the local spacing."""
    tree = cKDTree(pts)
    d, _ = tree.query(pts, k=2)
    s = float(np.median(d[:, 1]))
    c0 = int(np.argmin(np.linalg.norm(pts - center, axis=1)))
    _, i9 = tree.query(pts[c0], k=9)
    vec = pts[i9[1:]] - pts[c0]
    ln = np.linalg.norm(vec, axis=1)
    vec = vec[(ln > 0.7 * s) & (ln < 1.3 * s)]
    if len(vec) < 2:
        raise SystemExit("could not find the grid directions at the centre of the sheet")
    ang = np.arctan2(vec[:, 1], vec[:, 0])
    a = vec[np.argmin(np.abs(ang))]                                         # towards +x of the picture
    b = vec[np.argmin(np.abs(ang - math.pi / 2))]                           # towards +y (down)
    lat = {(0, 0): c0}
    queue = [(0, 0)]
    used = {c0}
    while queue:
        i, j = queue.pop(0)
        p = pts[lat[(i, j)]]
        for di, dj, base in ((1, 0, a), (-1, 0, -a), (0, 1, b), (0, -1, -b)):
            key = (i + di, j + dj)
            if key in lat:
                continue
            prev = (i - di, j - dj)
            step = (p - pts[lat[prev]]) if prev in lat else base            # follow the local spacing (the lens bends it)
            dd, ii = tree.query(p + step)
            if dd < 0.3 * s and ii not in used:
                lat[key] = int(ii)
                used.add(int(ii))
                queue.append(key)
    return lat, s


def refine_lines(ink, mask, pts, lat, band=5, excl=8, deg=3):
    """Fits a smooth curve to every WHOLE line (vertical: x = f(y), horizontal: y = g(x)) from the ink pixels between the
    crossings, then intersects the curves. The noise averages out along hundreds of pixels."""
    keys = list(lat)
    I = sorted({k[0] for k in keys})
    J = sorted({k[1] for k in keys})
    near = np.zeros(mask.shape, np.uint8)
    for p in pts:
        cv2.circle(near, (int(round(p[0])), int(round(p[1]))), excl, 255, -1)            # leave out the crossings themselves
    usable = (mask > 0) & (near == 0)

    def fit_line(chain, vertical):
        chain = np.array(chain)
        tmp = np.zeros(mask.shape, np.uint8)
        cv2.polylines(tmp, [np.round(chain).astype(np.int32).reshape(-1, 1, 2)], False, 255, 2 * band + 1)
        ys, xs = np.nonzero((tmp > 0) & usable)
        if len(ys) < 30:
            return None
        w = ink[ys, xs].astype(float)
        a, b = (ys, xs) if vertical else (xs, ys)                           # independent variable a, dependent b
        mid, half = 0.5 * (a.min() + a.max()), 0.5 * (a.max() - a.min()) + 1e-9
        t = (a - mid) / half
        keep = np.ones(len(t), bool)
        for _ in range(3):
            c = np.polynomial.polynomial.polyfit(t[keep], b[keep], deg, w=w[keep])
            r = b - np.polynomial.polynomial.polyval(t, c)
            sd = 1.4826 * np.median(np.abs(r[keep])) + 1e-6
            keep = np.abs(r) < max(2.5 * sd, 1.0)
        return (c, mid, half)

    def ev(f, a):
        c, mid, half = f
        return np.polynomial.polynomial.polyval((a - mid) / half, c)

    V, Hh = {}, {}
    for i in I:
        chain = [pts[lat[(i, j)]] for j in J if (i, j) in lat]
        if len(chain) >= 3:
            V[i] = fit_line(chain, True)
    for j in J:
        chain = [pts[lat[(i, j)]] for i in I if (i, j) in lat]
        if len(chain) >= 3:
            Hh[j] = fit_line(chain, False)
    out = pts.copy()
    for (i, j), n in lat.items():
        fv, fh = V.get(i), Hh.get(j)
        if fv is None or fh is None:
            continue
        x, y = pts[n]
        for _ in range(6):
            y = ev(fh, x)
            x = ev(fv, y)
        if np.hypot(x - pts[n][0], y - pts[n][1]) < 6:
            out[n] = (x, y)
    return out


# ------------------------------------------------------------------------------------------------ the fit
def _undist(P, c, R, a1, r2max):
    d = P - c
    r2 = (d ** 2).sum(-1, keepdims=True) / R ** 2
    return c + d * (1.0 + a1 * np.minimum(r2, r2max))


def fit_model(P, G, c, R, r2max, with_lens=True):
    """pixels P -> grid positions G (in cells): homography on the undistorted pixels. Returns (params, predict)."""
    Hm, _ = cv2.findHomography(P, G, 0)
    x0 = np.r_[Hm.flatten()[:8] / Hm[2, 2], [0.0] if with_lens else []]

    def predict(th, Q):
        pu = _undist(Q, c, R, th[8], r2max) if with_lens else Q
        h = np.append(th[:8], 1.0).reshape(3, 3)
        q = np.c_[pu, np.ones(len(pu))] @ h.T
        return q[:, :2] / q[:, 2:3]

    xs = np.r_[np.abs(x0[:8]) + 1e-3, [0.1] if with_lens else []]
    res = least_squares(lambda th: (predict(th, P) - G).ravel(), x0, loss="soft_l1", f_scale=0.05, x_scale=xs)
    return res.x, predict


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("photo", help="a raw camera picture of the sheet with the grid")
    ap.add_argument("--pitch", type=float, default=10.0, help="line spacing in mm, only used to give the errors in mm (default 10)")
    ap.add_argument("--out", default=LENS_FILE, help="where to write the lens (default LENS_FILE in settings.py)")
    ap.add_argument("--no_save", action="store_true", help="only measure and report, do not write lens.json")
    import h4xx0r as HX
    HX.add_flag(ap)
    a = ap.parse_args()
    HX.from_args(a)                                   # --h4xx0r: the terminal in the theme (this tool has no window)
    im = cv2.imread(a.photo)
    if im is None:
        raise SystemExit(f"cannot read the picture {a.photo!r}")
    h, w = im.shape[:2]
    center = np.array([w / 2.0, h / 2.0])
    R = 0.573 * float(np.hypot(w, h))                       # normalisation of the radius (1100 px for a 1920x1080 picture)
    sheet = sheet_mask(im)
    ink = ink_map(im, sheet)
    pts, mask = rough_crossings(ink)
    print(f"picture {w}x{h}: {len(pts)} crossings found")
    if len(pts) < 60:
        raise SystemExit("too few crossings: is the grid sharp, dark enough and fully inside the picture? (take the picture with preview.py: F hides the text, S saves)")
    lat, cell = build_lattice(pts, center)
    keys = list(lat)
    IJ = np.array(keys, float)
    if len(keys) < 0.7 * len(pts):
        raise SystemExit(f"only {len(keys)} of {len(pts)} crossings fit a regular grid: is the sheet flat and the grid regular?")
    ni, nj = int(IJ[:, 0].max() - IJ[:, 0].min() + 1), int(IJ[:, 1].max() - IJ[:, 1].min() + 1)
    print(f"grid: {ni} x {nj} crossings, {len(keys)} numbered, cell {cell:.1f} px")
    ref = refine_lines(ink, mask, pts, lat)
    Px = ref[[lat[k] for k in keys]]
    G = IJ * 1.0                                            # grid positions in cells

    # cross-validation: fit on most lines, test on lines the model has not seen
    folds = 6
    cv = {False: [], True: []}
    for f in range(folds):
        te = (IJ[:, 0].astype(int) % folds == f) | (IJ[:, 1].astype(int) % folds == f)
        for lens in (False, True):
            th, pr = fit_model(Px[~te], G[~te], center, R, 0.5, lens)
            cv[lens] += list(np.linalg.norm(pr(th, Px[te]) - G[te], axis=1) * a.pitch)
    cv = {k: np.array(v) for k, v in cv.items()}
    rho2_max = float(((np.linalg.norm(Px - center, axis=1) / R) ** 2).max())
    r2max = round(max(0.5, 1.2 * rho2_max), 2)              # beyond this the correction stops growing (nothing was measured there)
    th, pr = fit_model(Px, G, center, R, r2max, True)
    a1 = float(th[8])
    res = np.linalg.norm(pr(th, Px) - G, axis=1) * a.pitch
    print(f"\nlens term a1 = {a1:+.4f}   (0 = no distortion; the correction is capped at r2 = {r2max})")
    print(f"error on lines the model has NOT seen (includes the pen's own few-tenths-of-a-mm error, at {a.pitch:g} mm per cell):")
    print(f"   homography only : RMS {np.sqrt((cv[False] ** 2).mean()):.2f} mm, 95% {np.percentile(cv[False], 95):.2f}, max {cv[False].max():.2f}")
    print(f"   with the lens   : RMS {np.sqrt((cv[True] ** 2).mean()):.2f} mm, 95% {np.percentile(cv[True], 95):.2f}, max {cv[True].max():.2f}")
    print(f"   all crossings, fitted: RMS {np.sqrt((res ** 2).mean()):.2f} mm")
    gain = np.sqrt((cv[False] ** 2).mean()) - np.sqrt((cv[True] ** 2).mean())
    ok = len(keys) >= 100 and -0.2 < a1 < 0.3 and gain > -0.01
    if not ok:
        print("\nWARNING: the lens fit looks unreliable (few crossings, an extreme value, or no improvement). lens.json is NOT written.")
    # check picture: crossings green, leftover error x30 red
    out = im.copy()
    pred = pr(th, Px)
    for p, q, g in zip(Px, pred, G):
        e = (q - g) * a.pitch                               # mm, grid frame
        cv2.circle(out, (int(round(p[0])), int(round(p[1]))), 2, (0, 200, 0), -1)
        cv2.line(out, (int(round(p[0])), int(round(p[1]))), (int(round(p[0] + e[0] * 30 / 0.35)), int(round(p[1] + e[1] * 30 / 0.35))), (0, 0, 255), 1)
    x0, y0, x1, y1 = [int(v) for v in (Px[:, 0].min() - 40, Px[:, 1].min() - 40, Px[:, 0].max() + 40, Px[:, 1].max() + 40)]
    cv2.imwrite("calib_grid_check.png", out[max(0, y0):y1, max(0, x0):x1])
    print("check picture: calib_grid_check.png")
    if ok and not a.no_save:
        json.dump({"cx": float(center[0]), "cy": float(center[1]), "R": float(R), "a1": a1, "r2max": r2max, "image_size": [w, h],
                   "crossings": len(keys), "cv_rms_homography_mm": float(np.sqrt((cv[False] ** 2).mean())),
                   "cv_rms_lens_mm": float(np.sqrt((cv[True] ** 2).mean())), "assumed_pitch_mm": a.pitch, "photo": a.photo,
                   "created": time.strftime("%Y-%m-%d %H:%M:%S")}, open(a.out, "w"), indent=1)
        print(f"lens written: {a.out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
