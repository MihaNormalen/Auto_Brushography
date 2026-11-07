"""
calibrate_dots.py - calibrates the camera against the MACHINE: the brush paints a grid of dots on a white paper at known machine
positions, the camera finds them, and the fit gives the picture <-> machine mm conversion (with the lens of lens.json).

    python calibrate_dots.py                 homes, photographs the paper, plans the dots, asks, paints, measures, asks, saves
    python calibrate_dots.py --plan_only     only the plan and its picture; the machine is not touched
    python calibrate_dots.py --analyse       measure again from the saved pictures (calib_dots_before.png / _after.png / calib_dots.json)

Any WHITE paper will do (it needs no grid). Put it where you paint, tape it flat, and make sure the paint dish of CAL_DOT_COLOR has
paint. It needs a rough calibration.npz (the old one is fine: it only has to be right to a few mm) to find the paper and the dish;
the new calibration is computed from the dots alone. The old file is kept as calibration_backup_<time>.npz.
Settings (optional, in settings.py): CAL_DOTS_X / CAL_DOTS_Y / CAL_DOT_MARGIN_MM / CAL_DOT_PRESS_MM / CAL_DOT_COLOR / CAL_DOTS_PER_DIP.
"""
import argparse
import json
import os
import shutil
import time

import cv2
import numpy as np
from scipy.optimize import least_squares

import brush_profile as BP
import paint as PT
import settings as S
import vision as V

BEFORE, AFTER, PLAN = "calib_dots_before.png", "calib_dots_after.png", "calib_dots.json"
# the settings of this tool, with defaults: an older settings.py that does not have them yet keeps working
CAL_DOTS_X = getattr(S, "CAL_DOTS_X", 6)
CAL_DOTS_Y = getattr(S, "CAL_DOTS_Y", 5)
CAL_DOT_MARGIN_MM = getattr(S, "CAL_DOT_MARGIN_MM", 15.0)
CAL_DOT_PRESS_MM = getattr(S, "CAL_DOT_PRESS_MM", 1.0)
CAL_DOT_COLOR = getattr(S, "CAL_DOT_COLOR", "auto")
CAL_DOTS_PER_DIP = getattr(S, "CAL_DOTS_PER_DIP", 4)
# the lowest a dot may be pressed: the machine's lowest allowed Z (zero.g frame: Z 0 is the paper, below it is fine),
# else 0.3 (a machine whose Z 0 is the bottom of its travel)
Z_FLOOR = float(getattr(S, "MACHINE_Z_MIN", 0.3))
LENS_FILE = getattr(S, "LENS_FILE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "lens.json"))


# ------------------------------------------------------------------------------------------------ planning
def contact_height():
    """Machine Z where the brush starts to mark: from the brush profile, else Z_OFFSET / TOUCH_Z of settings.py."""
    if S.Z_OFFSET is not None:
        return float(S.Z_OFFSET)
    if S.BRUSH_PROFILE:
        return float(BP.load(BP.find(S.BRUSH_PROFILE))["z_offset"])
    return float(S.TOUCH_Z)


def inner_rect(corners_mm, margin):
    """An axis-aligned rectangle (machine mm) inside the (slightly tilted) paper, `margin` away from its edges."""
    xs = sorted(c[0] for c in corners_mm)
    ys = sorted(c[1] for c in corners_mm)
    x0, x1 = xs[1] + margin, xs[2] - margin
    y0, y1 = ys[1] + margin, ys[2] - margin
    if x1 - x0 < 40 or y1 - y0 < 40:
        raise SystemExit(f"the paper is too small for dots with a {margin:g} mm margin ({x1 - x0:.0f} x {y1 - y0:.0f} mm left)")
    return x0, x1, y0, y1


def plan_dots(rect, nlong, nshort):
    """Dots in a regular grid over the rectangle, in a serpentine order (short travel). Returns [(x, y)] in machine mm."""
    x0, x1, y0, y1 = rect
    if (y1 - y0) >= (x1 - x0):
        nx, ny = nshort, nlong                                  # the long side of the paper runs along machine Y
    else:
        nx, ny = nlong, nshort
    xs = np.linspace(x0, x1, nx)
    ys = np.linspace(y0, y1, ny)
    out = []
    for i, x in enumerate(xs):
        for y in (ys if i % 2 == 0 else ys[::-1]):
            out.append((float(x), float(y)))
    return out


def pick_dish(pools, color, cfg):
    """The biggest pool of the colour that is a real DISH: the same rules paint.py uses (round enough, big enough, near the dip
    position, inside the machine's travel), so a streak or a splash of that colour is not taken for the dish.
    color "auto": the darkest paint there is (the black dish when there is one, as the probes do; dots show best dark)."""
    near = lambda p: (np.hypot(p["center_mm"][0] - cfg["dip_x"], p["center_mm"][1] - cfg["dip_y"]) <= cfg["palette_radius"]
                      and PT.reachable(p))
    if color in ("auto", "black"):
        dishes = [p for p in pools if p["roundness"] >= cfg["min_roundness"] and p["area_mm2"] >= cfg["min_pool_area"] and near(p)]
        dark = sorted([p for p in dishes if p.get("dark")], key=lambda p: -p["area_mm2"])
        if dark:
            print(f"dots painted with black (the dark dish at X {dark[0]['center_mm'][0]:.0f} Y {dark[0]['center_mm'][1]:.0f})")
            return dark[0]["center_mm"][0], dark[0]["center_mm"][1], dark[0]["radius_mm"]
        cups = getattr(S, "CUPS", None) or {}
        if "black" in cups:                                     # a fixed holder: black is where its cup is (no colour to find)
            bx, by = cups["black"]
            print(f"dots painted with black (the black cup of CUPS at X {bx:.0f} Y {by:.0f})")
            return float(bx), float(by), 0.5 * float(getattr(S, "CUP_WIDTH_MM", 16.2))
        if color == "auto":
            color = "blue" if any(PT.hue_name(p["hue"]) == "blue" for p in dishes) else (PT.hue_name(dishes[0]["hue"]) if dishes else "blue")
            print(f"no black dish: the dots are painted with {color}")
    same = [p for p in pools if PT.hue_name(p["hue"]) == color]
    cands = [p for p in same if p["roundness"] >= cfg["min_roundness"] and p["area_mm2"] >= cfg["min_pool_area"] and near(p)]
    if not cands:
        seen = "\n".join(f"    {PT.hue_name(p['hue']):7s} X {p['center_mm'][0]:5.0f} Y {p['center_mm'][1]:5.0f}  area {p['area_mm2']:5.0f} mm2  roundness {p['roundness']:.2f}"
                         + ("" if PT.hue_name(p['hue']) == color else "   (other colour)") for p in pools) or "    nothing"
        raise SystemExit(f"no {color} paint dish found: the camera needs a round, filled pool of {color} paint "
                         f"(at least {cfg['min_pool_area']:.0f} mm2, roundness {cfg['min_roundness']:.1f}). It sees:\n{seen}\n"
                         f"  Put paint in the dish, or use another colour with --color.")
    p = cands[0]
    return p["center_mm"][0], p["center_mm"][1], p["radius_mm"]


def build_gcode(dots, dish, cfg, per_dip, press, dwell_ms):
    """The painter's own dip (spiral, wipe) + a vertical press for every dot. Returns the G-code lines."""
    P = PT.Painter(cfg)
    z_dot = max(Z_FLOOR, cfg["z_paint"] - press)
    for k, (x, y) in enumerate(dots):
        if k % per_dip == 0:
            P._dip_and_travel(x, y, dish)                       # ends above (x, y) at travel height
        else:
            P.gcode.append(f"G0 X{x:.3f} Y{y:.3f} Z{cfg['z_low']:.3f} F{cfg['feed']}")
        P.gcode.append(f"; dot {k + 1}/{len(dots)} at X{x:.2f} Y{y:.2f}")
        P.gcode.append(f"G1 Z{z_dot:.3f} F{cfg['plunge_feed']}")
        P.gcode.append(f"G4 P{int(dwell_ms)}")
        P.gcode.append(f"G0 Z{cfg['z_low']:.3f} F3000")
    P.gcode.append(f"G0 Z{cfg['z_high']:.3f} F3000")
    return P.gcode


# ------------------------------------------------------------------------------------------------ measuring
def find_dots(frame1, frame2, paper_poly_px):
    """The new dots = what became darker between the picture before and after (the paper, its printing and any grid cancel out)."""
    g1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY).astype(np.float32)
    poly = np.zeros(g1.shape, np.uint8)
    cv2.fillPoly(poly, [np.round(np.array(paper_poly_px)).astype(np.int32)], 255)
    inside = cv2.erode(poly, np.ones((9, 9), np.uint8)) > 0
    g2 = g2 * (float(np.median(g1[inside])) / max(float(np.median(g2[inside])), 1.0))          # same exposure
    d = cv2.GaussianBlur(g1 - g2, (0, 0), 0.8)
    d[~inside] = 0
    d = np.clip(d, 0, None)
    thr = max(25.0, 0.3 * float(np.percentile(d[inside], 99.9)))
    bw = (d > thr).astype(np.uint8) * 255
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    n, lab, st, cen = cv2.connectedComponentsWithStats(bw)
    out = []
    for k in range(1, n):
        area = int(st[k, cv2.CC_STAT_AREA])
        if area < 8 or area > 700:
            continue
        m = cv2.dilate((lab == k).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        ys, xs = np.nonzero(m)
        wgt = d[ys, xs]
        out.append({"x": float((xs * wgt).sum() / wgt.sum()), "y": float((ys * wgt).sum() / wgt.sum()), "area": area, "peak": float(d[m].max())})
    return out


def match_dots(blobs, dots_mm, cal0, spacing):
    """Pairs every planned dot with the nearest blob (the old calibration only has to be right to a fraction of the spacing)."""
    tol = min(0.4 * spacing, 18.0)
    px = np.array([[b["x"], b["y"]] for b in blobs]).reshape(-1, 2)
    est = V.px_to_mm_array(cal0["H"], px) if len(px) else np.zeros((0, 2))
    pairs, used = [], set()
    for k, (x, y) in enumerate(dots_mm):
        if len(est) == 0:
            break
        d = np.linalg.norm(est - np.array([x, y]), axis=1)
        for j in np.argsort(d):
            if d[j] > tol:
                break
            if int(j) not in used:
                used.add(int(j))
                pairs.append((k, int(j)))
                break
    return pairs


def fit_homography(Pu, M):
    """Undistorted pixels -> machine mm, robust. Returns (H, residuals in mm)."""
    H0, _ = cv2.findHomography(Pu, M, 0)
    x0 = H0.flatten()[:8] / H0[2, 2]

    def pred(th, Q):
        q = np.c_[Q, np.ones(len(Q))] @ np.append(th, 1.0).reshape(3, 3).T
        return q[:, :2] / q[:, 2:3]

    r = least_squares(lambda th: (pred(th, Pu) - M).ravel(), x0, loss="soft_l1", f_scale=0.2, x_scale=np.abs(x0) + 1e-3)
    H = np.append(r.x, 1.0).reshape(3, 3)
    return H, np.linalg.norm(pred(r.x, Pu) - M, axis=1)


def refit_lens(P_raw, M, lens):
    """The lens centre and strength fitted together with the homography, from the dots themselves. lens.json was measured on a
    smaller area with the centre fixed in the middle of the picture; on a big sheet that left a smooth error of about 1 mm at the
    corners (2026-10-28: the dots put the real centre ~230 px left, ~135 px up). Returns a lens tuple (cx, cy, R, a1, r2max)."""
    cx, cy, R, a1, _ = lens
    r2_dots = float((((P_raw - [cx, cy]) ** 2).sum(1) / R ** 2).max())
    cap = round(max(0.5, 1.2 * r2_dots), 2)                     # beyond the dots the correction stops growing (nothing was measured there)

    def und(th):
        return V.undistort_px(P_raw, (th[8], th[9], R, th[10], cap))

    H0, _ = cv2.findHomography(V.undistort_px(P_raw, lens), M, 0)
    x0 = np.r_[H0.flatten()[:8] / H0[2, 2], cx, cy, a1]

    def f(th):
        q = np.c_[und(th), np.ones(len(M))] @ np.append(th[:8], 1.0).reshape(3, 3).T
        return (q[:, :2] / q[:, 2:3] - M).ravel()
    r = least_squares(f, x0, loss="soft_l1", f_scale=0.3, x_scale=np.r_[np.abs(x0[:8]) + 1e-3, 50.0, 50.0, 0.05])
    return (float(r.x[8]), float(r.x[9]), float(R), float(r.x[10]), cap)


def solve(P_raw, M, lens):
    """Fits, drops outliers (a blob that is not a dot) and returns the calibration plus its quality numbers,
    and the lens model used (lens.json, or the lens refitted from the dots when that predicts the dots better)."""
    if lens is not None and len(M) >= 15:
        fitted = refit_lens(P_raw, M, lens)
        first = _solve_with(P_raw, M, lens)
        second = _solve_with(P_raw, M, fitted)
        rms = lambda s: float(np.sqrt((s[3] ** 2).mean()))
        print(f"lens: from lens.json -> leave-one-out RMS {rms(first):.2f} mm;  refitted from the dots (centre {fitted[0]:.0f}, "
              f"{fitted[1]:.0f}, strength {fitted[3]:+.3f}) -> {rms(second):.2f} mm")
        return (*second, fitted) if rms(second) < rms(first) else (*first, lens)
    return (*_solve_with(P_raw, M, lens), lens)


def _solve_with(P_raw, M, lens):
    Pu = V.undistort_px(P_raw, lens) if lens else P_raw
    keep = np.ones(len(M), bool)
    for _ in range(3):
        H, res = fit_homography(Pu[keep], M[keep])
        q = np.c_[Pu, np.ones(len(Pu))] @ H.T
        allres = np.linalg.norm(q[:, :2] / q[:, 2:3] - M, axis=1)
        new_keep = allres < max(1.0, 4.0 * float(np.median(allres[keep])))
        if (new_keep == keep).all():
            break
        keep = new_keep
    Hf, resk = fit_homography(Pu[keep], M[keep])
    # leave-one-out: how well is a dot predicted by a fit that has never seen it?
    idx = np.nonzero(keep)[0]
    loo = []
    for i in idx:
        sel = idx[idx != i]
        Hi, _ = fit_homography(Pu[sel], M[sel])
        q = np.array([[*Pu[i], 1.0]]) @ Hi.T
        loo.append(float(np.linalg.norm(q[0, :2] / q[0, 2] - M[i])))
    return Hf, keep, resk, np.array(loo)


# ------------------------------------------------------------------------------------------------ pictures and saving
def plan_picture(frame, det, dish, dots, cal0, path):
    out = frame.copy()
    cv2.polylines(out, [np.array(det["corners_px"], np.int32)], True, (0, 255, 0), 3, cv2.LINE_AA)
    c = V.mm_to_px_array(cal0["H_inv"], [(dish[0], dish[1])])[0]
    cv2.circle(out, (int(c[0]), int(c[1])), max(6, int(dish[2] * 2.9)), (255, 128, 0), 3, cv2.LINE_AA)
    for k, p in enumerate(V.mm_to_px_array(cal0["H_inv"], dots), 1):
        x, y = int(round(p[0])), int(round(p[1]))
        cv2.drawMarker(out, (x, y), (0, 0, 255), cv2.MARKER_CROSS, 18, 2)
        cv2.putText(out, str(k), (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.imwrite(path, out)


def result_picture(frame, P_raw, M, keep, cal_new, path):
    out = frame.copy()
    pred = V.px_to_mm_array(cal_new["H"], P_raw)
    for p, m, q, ok in zip(P_raw, M, pred, keep):
        x, y = int(round(p[0])), int(round(p[1]))
        cv2.circle(out, (x, y), 6, (0, 200, 0) if ok else (0, 0, 255), 1, cv2.LINE_AA)
        e = (q - m)                                                                  # mm; shown x20 in the picture's directions
        ex = V.mm_to_px_array(cal_new["H_inv"], [tuple(m + e * 20)])[0] - V.mm_to_px_array(cal_new["H_inv"], [tuple(m)])[0]
        cv2.line(out, (x, y), (int(round(x + ex[0])), int(round(y + ex[1]))), (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imwrite(path, out)


def load_lens():
    if not os.path.isfile(LENS_FILE):
        print(f"NOTE: no {os.path.basename(LENS_FILE)} (run calibrate_grid.py once): fitting a plain homography, without lens correction.")
        return None
    d = json.load(open(LENS_FILE))
    return (d["cx"], d["cy"], d["R"], d["a1"], d["r2max"])


def measure_and_save(frame1, frame2, plan, cal0, args):
    lens = load_lens()
    dots = np.array(plan["dots_mm"], float)
    spacing = float(plan["spacing_mm"])
    blobs = find_dots(frame1, frame2, plan["paper_px"])
    print(f"\nthe camera found {len(blobs)} new marks on the paper; {len(dots)} dots were planned")
    pairs = match_dots(blobs, dots, cal0, spacing)
    print(f"matched {len(pairs)} of {len(dots)} dots")
    need = max(12, int(np.ceil(0.5 * len(dots))))
    if len(pairs) < need:
        raise SystemExit(f"too few dots found ({len(pairs)}; at least {need} are needed so that the whole paper is covered).\n"
                         "  Look at calib_dots_plan.png: is the blue circle ON the paint dish? If it is off, the old calibration.npz is too far off\n"
                         "  at the dish (run calibrate.py --guided first). Is there paint in the dish, and are the dots dark enough? See calib_dots_after.png.")
    P_raw = np.array([[blobs[j]["x"], blobs[j]["y"]] for _, j in pairs])
    M = np.array([dots[k] for k, _ in pairs])
    # how wrong was the OLD calibration at these dots?
    old = np.linalg.norm(V.px_to_mm_array(cal0["H"], P_raw) - M, axis=1)
    Hf, keep, res, loo, lens = solve(P_raw, M, lens)
    print(f"\nthe OLD calibration was off by: RMS {np.sqrt((old ** 2).mean()):.2f} mm, 95% {np.percentile(old, 95):.2f}, max {old.max():.2f}")
    print(f"the NEW fit, {int(keep.sum())} dots used ({int((~keep).sum())} dropped as outliers):")
    print(f"   error at the dots   : RMS {np.sqrt((res ** 2).mean()):.2f} mm, max {res.max():.2f}")
    print(f"   leave-one-out check : RMS {np.sqrt((loo ** 2).mean()):.2f} mm, max {loo.max():.2f}   (a dot predicted without having been used)")
    rms_loo = float(np.sqrt((loo ** 2).mean()))
    verdict = "good" if rms_loo < 0.35 else "usable" if rms_loo < 0.7 else "POOR - do not use"
    if len(pairs) < 0.8 * len(dots) and verdict == "good":
        verdict = "good where the dots are, but only {}% of them were found: the places without dots are less certain".format(int(100 * len(pairs) / len(dots)))
    print(f"   verdict: {verdict}")
    cal_new = {"H": V.CalMat(Hf, {"lens": lens} if lens else None), "H_inv": V.CalMat(np.linalg.inv(Hf), {"lens": lens} if lens else None)}
    result_picture(frame2, P_raw, M, keep, cal_new, "calib_dots_result.png")
    print("picture: calib_dots_result.png (green = used, red line = the error x20)")
    if rms_loo >= 0.7:
        raise SystemExit("the fit is too poor to save. Look at calib_dots_result.png: a dot far from the others, a crumpled paper, or the camera moved?")
    if not args.yes:
        ans = input("\nSave this as the new calibration.npz? (the old one is kept as a backup)  [y/n] ").strip().lower()
        if ans not in ("y", "yes", "j", "ja", "d", "da"):
            print("not saved.")
            return
    if os.path.isfile(S.CALIBRATION_FILE):
        bak = os.path.join(os.path.dirname(S.CALIBRATION_FILE), time.strftime("calibration_backup_%Y%m%d_%H%M%S.npz"))
        shutil.copy2(S.CALIBRATION_FILE, bak)
        print(f"old calibration kept: {bak}")
    mm_pts = np.array(cal0["mm_pts"], float)
    pixel_pts = V.mm_to_px_array(cal_new["H_inv"], mm_pts)
    V.save_calibration(Hf, pixel_pts, mm_pts, S.CALIBRATION_FILE, image_size=(frame2.shape[1], frame2.shape[0]), lens=lens,
                       dots_px=P_raw, dots_mm=M, dots_loo_rms=np.array([rms_loo]))
    print("new calibration saved. Check it with:  python calibration_verify.py")


# ------------------------------------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan_only", action="store_true", help="plan and show, never touch the machine")
    ap.add_argument("--yes", action="store_true", help="do not ask")
    ap.add_argument("--analyse", action="store_true", help="measure again from the saved pictures, no camera, no machine")
    ap.add_argument("--color", default=CAL_DOT_COLOR, help=f"colour of the paint dish to use (default {CAL_DOT_COLOR})")
    ap.add_argument("--no_preview", action="store_true", help="do not open the plan picture")
    import h4xx0r as HX
    HX.add_flag(ap)
    a = ap.parse_args(argv)
    HX.from_args(a)                                   # --h4xx0r: the terminal in the theme (this tool has no window)
    cal0 = V.load_calibration()
    if a.analyse:
        plan = json.load(open(PLAN))
        f1, f2 = cv2.imread(BEFORE), cv2.imread(AFTER)
        if f1 is None or f2 is None:
            raise SystemExit(f"{BEFORE} / {AFTER} not found in this folder")
        measure_and_save(f1, f2, plan, cal0, a)
        return

    cfg = PT.default_config()
    contact = contact_height()
    cfg["z_offset"] = contact
    for key in ("z_paint", "z_low", "z_high", "z_wipe_exit", "dip_z"):        # machine Z = height above the contact + the contact
        cfg[key] += contact
    machine = None
    if not a.plan_only:
        machine = PT.connect_and_home(cfg)
        PT.go_to_snapshot(machine, cfg)
    cap = V.open_camera()
    frame1, live = PT.capture_fresh(cap)
    cap.release()
    cv2.imwrite(BEFORE, frame1)
    print(f"camera picture taken {time.strftime('%H:%M:%S')}" + ("" if live else "  - WARNING: frozen or old picture! Close other programs using the camera."))
    H, H_inv = cal0["H"], cal0["H_inv"]
    det = V.detect_paper(frame1, H, H_inv, calib_mm=cal0["mm_pts"])
    if det is None:
        raise SystemExit("no paper found in the picture (is the toolhead in view?)")
    rect = inner_rect(det["corners_mm"], CAL_DOT_MARGIN_MM)
    dots = plan_dots(rect, CAL_DOTS_X, CAL_DOTS_Y)
    ux = sorted({round(p[0], 3) for p in dots})
    uy = sorted({round(p[1], 3) for p in dots})
    spacing = min(np.diff(ux).min() if len(ux) > 1 else 1e9, np.diff(uy).min() if len(uy) > 1 else 1e9)
    pools = V.find_pools(frame1, H, det["corners_px"])
    dish = pick_dish(pools, a.color, cfg)
    code = "\n".join(build_gcode(dots, dish, cfg, CAL_DOTS_PER_DIP, CAL_DOT_PRESS_MM, S.DOT_DWELL_MS))
    z_dot = max(Z_FLOOR, cfg["z_paint"] - CAL_DOT_PRESS_MM)
    print(f"paper {det['size_mm'][0]:.0f} x {det['size_mm'][1]:.0f} mm; {len(dots)} dots in X {rect[0]:.0f}..{rect[1]:.0f}, Y {rect[2]:.0f}..{rect[3]:.0f} "
          f"(spacing {spacing:.0f} mm); dish {a.color} at X {dish[0]:.0f} Y {dish[1]:.0f}")
    print(f"dots are pressed at machine Z {z_dot:.2f} (contact {contact:.2f} - {CAL_DOT_PRESS_MM:g} mm); a dip every {CAL_DOTS_PER_DIP} dots")
    plan = {"dots_mm": [list(d) for d in dots], "spacing_mm": float(spacing), "paper_px": det["corners_px"], "dish": list(dish), "contact": contact}
    json.dump(plan, open(PLAN, "w"), indent=1)
    plan_picture(frame1, det, dish, dots, cal0, "calib_dots_plan.png")
    print("plan picture: calib_dots_plan.png (green = paper, blue circle = dish, red crosses = the dots)")
    if not a.no_preview:
        PT.show_picture("calib_dots_plan.png")
    if a.plan_only:
        print("plan only: the machine was not touched.")
        return
    if not a.yes and input("\nPaint the dots now?  Check calib_dots_plan.png: the blue circle must sit ON the paint dish, the paper must be flat.  [y/n] ").strip().lower() not in ("y", "yes", "j", "ja", "d", "da"):
        print("not started. The machine stays homed.")
        return
    cfg["z_high"] = max(cfg["z_high"], cfg["z_low"] + 1.0)
    if not PT.stream_gcode(machine, code, cfg):
        raise SystemExit("painting was stopped: no calibration was made.")
    PT.go_to_snapshot(machine, cfg)
    time.sleep(1.5)
    cap = V.open_camera()
    frame2, live = PT.capture_fresh(cap)
    cap.release()
    cv2.imwrite(AFTER, frame2)
    measure_and_save(frame1, frame2, plan, cal0, a)


if __name__ == "__main__":
    main()
