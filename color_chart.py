"""
color_chart.py - the overlap chart for TRANSPARENT colours: what does each colour give over the paper, and over each other?

A mesh of bands, painted in two passes on one sheet:
    pass 1  HORIZONTAL bands, one per colour, bright to dark
    ...     drying (CHART_DRY_MIN minutes; Enter skips it)
    pass 2  VERTICAL bands, the same colours on top, bright to dark
Every crossing is "row colour UNDER column colour", so each pair is there in BOTH orders (yellow under red AND red under
yellow), the diagonal is each colour twice (a double glaze), and the band between the crossings is each colour alone.

    python color_chart.py make yellow red blue     # ko/chart_h.png, ko/chart_v.png, ko/chart.json
    python color_chart.py paint                    # pass 1, drying, pass 2 - all automatic (probes, dips, rinse, repair)
    python color_chart.py measure                  # photograph and measure -> ko/chart_result.json + ko/chart_swatches.png
    (python color_chart.py paint h / paint v       # one pass only, e.g. to go on after a stop)

Colour names are paint.py's hue families: red, orange, yellow, green, cyan, blue, purple, pink. Each needs a dish with
that paint (found by the camera like for any colour artwork), and a water dish for rinsing in between (WATER_X / WATER_Y).
The colours are painted BRIGHT TO DARK (a dark colour's trace in a bright dish shows, the other way round much less);
`make --keep_order` keeps the order you typed.

SIZE: the chart takes CHART_SHARE of the paper (its shorter side, inside the margin), so on a bigger sheet the bands are
bigger. Both passes are placed by the same rule on the same sheet: the sheet must NOT move between them.

measure: colours are taken relative to the paper in the same photo (the camera's white balance and exposure drop out),
in linear light. For every pair it also prints what simply MULTIPLYING the two single colours predicts (an ideal
transparent glaze: each layer lets through a share of the light) and how far the real overlap is from that (Delta E):
small = the colours can be planned from the singles alone; large = the chart itself must be used.
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

CHART_DIR = os.path.join(HERE, "ko")
LAYOUT = os.path.join(CHART_DIR, "chart.json")
RESULT = os.path.join(CHART_DIR, "chart_result.json")
PX = 10                             # picture pixels per layout unit
BAND = 14.0                         # band width : gap between bands = 14 : 10 (layout units; the real size follows the paper)
GAP = 10.0
CHART_SHARE = 0.6                   # the chart (bands and the gaps around them) takes this share of the paper's shorter side
MARGIN_MM = 10.0                    # paint.py's --margin: paper kept free around the picture
CHART_DRY_MIN = 15.0                # drying between pass 1 and pass 2 (the vertical bands go OVER the horizontal ones)
CHART_STYLE = "lines"               # how the bands are filled (paint.py INFILL_STYLE): plain parallel lines = an even glaze
SAMPLE_SHARE = 0.5                  # a patch is measured over its middle part only (this share of its width), away from edges

# the colour each name is drawn in: the middle of paint.py's hue range, fully saturated (paint.py turns it into that layer)
HUE_OF = {"red": 0, "orange": 14, "yellow": 28, "green": 60, "cyan": 92, "blue": 117, "purple": 148, "pink": 166,
          "black": None}            # black (a transparent black too): drawn dark grey, paint.py makes it the black layer (darkest dish)
# how bright the paint usually is (for the bright-to-dark order; the measured chart knows better later)
BRIGHT = {"yellow": 8, "orange": 7, "cyan": 6, "pink": 5, "green": 4, "red": 3, "purple": 2, "blue": 1, "black": 0}


def name_rgb(name, value=235):
    if HUE_OF[name] is None:
        return (30, 30, 30)
    hsv = np.uint8([[[HUE_OF[name], 255, value]]])
    return tuple(int(v) for v in cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0, 0])


# --------------------------------------------------------------------------------------------------------- make
def make(colors, keep_order=False):
    bad = [c for c in colors if c not in HUE_OF]
    if bad:
        raise SystemExit(f"unknown colour {', '.join(bad)}: use {', '.join(HUE_OF)}")
    if len(set(colors)) != len(colors) or len(colors) < 2:
        raise SystemExit("at least two colours, each once")
    if not keep_order:
        colors = sorted(colors, key=lambda c: -BRIGHT[c])
    n = len(colors)
    chart = GAP + n * (BAND + GAP)                                   # the chart itself, layout units
    total = chart / CHART_SHARE                                     # the picture: the chart with white paper around it
    pad = (total - chart) / 2.0
    pos = [pad + GAP + k * (BAND + GAP) for k in range(n)]
    L = {"colors": colors, "picture_units": total, "chart_units": chart, "pad_units": pad, "band_units": BAND,
         "gap_units": GAP, "rows": [[p, p + BAND] for p in pos], "cols": [[p, p + BAND] for p in pos],
         "margin_mm": MARGIN_MM, "share": CHART_SHARE}
    n_px = int(round(total * PX))
    h = np.full((n_px, n_px, 3), 255, np.uint8)
    v = h.copy()
    lo, hi = int(round((pad + GAP) * PX)), int(round((pad + chart - GAP) * PX))      # the bands run from the first gap to the last
    for k, (c, (a, b)) in enumerate(zip(colors, L["rows"])):
        # paint.py paints the layers brightest first (by the picture's own brightness): each next colour 5 steps darker keeps our order
        rgb = name_rgb(c, 250 - 5 * k)
        h[int(round(a * PX)):int(round(b * PX)), lo:hi] = rgb
        v[lo:hi, int(round(a * PX)):int(round(b * PX))] = rgb
    os.makedirs(CHART_DIR, exist_ok=True)
    cv2.imwrite(os.path.join(CHART_DIR, "chart_h.png"), cv2.cvtColor(h, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(CHART_DIR, "chart_v.png"), cv2.cvtColor(v, cv2.COLOR_RGB2BGR))
    look = (h.astype(float) / 255.0) * (v.astype(float) / 255.0) * 255.0                    # a rough "multiply" look
    cv2.imwrite(os.path.join(CHART_DIR, "chart_look.png"), cv2.cvtColor(look.astype(np.uint8), cv2.COLOR_RGB2BGR))
    # flat, even patches: plain parallel lines (INFILL_STYLE "lines") whatever style the artworks use, through paint.py's
    # <picture>.styles.json (one box over the whole picture). A horizontal band gets horizontal lines, a vertical one vertical
    for which in ("h", "v"):
        json.dump({"regions": [{"box": [0, 0, n_px, n_px], "style": CHART_STYLE, "infill_type": "lines"}]},
                  open(os.path.join(CHART_DIR, f"chart_{which}.styles.json"), "w"), indent=1)
    json.dump(L, open(LAYOUT, "w"), indent=1)
    a4 = (210.0 - 2 * MARGIN_MM) / total                            # mm per unit on an A4 sheet (short side 210 mm)
    print(f"chart: {n} colours, painted bright to dark: {', '.join(colors)}; {n * n} crossings, each colour alone both ways")
    print(f"   size follows the paper: {100 * CHART_SHARE:.0f}% of its shorter side. On A4 the chart is about {chart * a4:.0f} mm, "
          f"each band {BAND * a4:.0f} mm wide with {GAP * a4:.0f} mm paper between")
    print("   ko/chart_h.png (pass 1, horizontal), ko/chart_v.png (pass 2, vertical, on top), ko/chart_look.png (rough idea)")
    print("next:  python color_chart.py paint")


def load_layout():
    if not os.path.isfile(LAYOUT):
        raise SystemExit("no ko/chart.json: run  python color_chart.py make <colours>  first")
    L = json.load(open(LAYOUT))
    if "picture_units" not in L:
        raise SystemExit("ko/chart.json is from an older color_chart.py: run  make  again")
    return L


# -------------------------------------------------------------------------------------------------------- paint
def wait_dry(minutes, what, screen=None):
    """A countdown. Enter (or R) goes on at once; + (or P) adds DRY_EXTEND_MIN minutes when the sheet is not dry yet, - takes
    them off again; Ctrl+C stops. screen(left_s, added_min) is called a few times a second (a live window: color_paint's
    drying view); it may return a key pressed in that window, which counts like one pressed here.
    -> True when you pressed Enter (go on now), False when the time ran out."""
    step = float(getattr(__import__("settings"), "DRY_EXTEND_MIN", 5.0))
    print(f"\nDRYING {minutes:g} min before {what}. Here: Enter (or R) = go on now, + (or P) = {step:g} min more, "
          f"- = {step:g} min less, Ctrl+C = stop.")
    try:
        import msvcrt
    except ImportError:
        msvcrt = None
    while msvcrt is not None and msvcrt.kbhit():
        msvcrt.getwch()
    end, next_print, added = time.time() + 60.0 * minutes, 0.0, 0.0
    while time.time() < end:
        if time.time() >= next_print:
            print(f"   drying: {(end - time.time()) / 60.0:4.1f} min left" + (f"  ({added:+g} min by you)" if added else ""))
            next_print = time.time() + 60.0
        keys = []
        if msvcrt is not None:
            while msvcrt.kbhit():
                keys.append(msvcrt.getwch())
        if screen is not None:
            try:
                k = screen(end - time.time(), added)
            except Exception as e:                                  # the drying must not depend on the window
                print(f"   (drying window not available: {e})")
                screen, k = None, None
            if k:
                keys.append(k)
        for k in keys:
            k = k.lower()
            if k in ("\r", "\n", "r"):
                print("   going on (you pressed Enter)")
                return True
            if k in ("+", "p", "="):
                end += 60.0 * step
                added += step
            elif k == "-":
                cut = min(60.0 * step, max(0.0, end - time.time()))
                end -= cut
                added -= cut / 60.0
            else:
                continue
            print(f"   drying: {max(0.0, end - time.time()) / 60.0:4.1f} min left now ({added:+g} min by you)")
            next_print = time.time() + 60.0
        time.sleep(0.1 if screen is not None else 0.2)
    print("   drying time over")
    return False


def run_pass(which, L, extra):
    import settings
    settings.SIGNATURE = None           # a chart has no signature (and the second pass must not paint one again)
    import paint as PT
    img = os.path.join("ko", f"chart_{which}.png")
    argv = [img, "--rotate", "0", "--margin", f"{L['margin_mm']:g}", "--color", "auto"] + extra
    if which == "v" and os.path.isfile(os.path.join(HERE, "probe_result.json")) and "--probe_file" not in extra:
        # the same sheet: the probe of the first pass is used again (no new probe lines; paint.py refuses it if the sheet
        # moved or the result is over 6 hours old). Faint colour probes were not recognised and painted again (2026-11-12)
        argv += ["--probe_file", os.path.join(HERE, "probe_result.json")]
    print(f"\ncolor_chart: PASS {'1, horizontal bands' if which == 'h' else '2, vertical bands on top'} "
          f"({', '.join(L['colors'])}):  paint.py " + " ".join(argv))
    if which == "v":
        print("   the sheet must not have moved since pass 1; its probe is reused (no new probe lines)")
    PT.LAST_JOB_DONE = False
    PT.main(argv)
    if not PT.LAST_JOB_DONE:                                # stopped (probes failed, N, Ctrl+C): no drying, no next pass
        raise SystemExit("\nthis pass was NOT painted to the end (the reason is above): stopped here. "
                         f"After fixing it, go on with this pass:  python color_chart.py paint {which}")


def paint(which, extra, dry_min):
    L = load_layout()
    import rinse as RN
    if RN.load() is None and not getattr(__import__("settings"), "WATER_X", None):
        print("NOTE: the rinse container is not calibrated (python rinse.py calibrate): the brush goes from one colour to the "
              "next WITHOUT rinsing")
    if which in ("all", "h"):
        run_pass("h", L, extra)
    if which == "all":
        wait_dry(dry_min, "pass 2 (the vertical bands go over the horizontal ones)")
    if which in ("all", "v"):
        run_pass("v", L, extra)
        print("\nnext, when dry:  python color_chart.py measure")
    else:
        print(f"\nnext, when dry:  python color_chart.py paint v")


# ------------------------------------------------------------------------------------------------------ measure
def srgb_to_lin(c):
    c = np.asarray(c, float) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def lin_to_srgb(c):
    c = np.clip(np.asarray(c, float), 0.0, 1.0)
    return 255.0 * np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055)


M_XYZ = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
WHITE = M_XYZ @ np.ones(3)


def refl_to_lab(r):
    """Reflectance relative to the paper (linear RGB, paper = 1,1,1) -> CIELAB with the paper as white."""
    xyz = (M_XYZ @ np.asarray(r, float)) / WHITE
    f = np.where(xyz > (6 / 29) ** 3, np.cbrt(xyz), xyz / (3 * (6 / 29) ** 2) + 4 / 29)
    return np.array([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])


def chart_to_camera(frame, L):
    """Layout units (x right, y down) -> camera pixels, by the placement paint.py used: the picture as large as fits on the
    paper photographed now, same margin, not turned. Returns (function, mm per unit)."""
    import paint as PT
    import vision as V
    cal = V.load_calibration()
    det = V.detect_paper(frame, cal["H"], cal["H_inv"], calib_mm=cal["mm_pts"])
    if det is None:
        raise SystemExit("no paper found in the picture")
    det = V.clip_to_reach(det)
    pf = V.paper_frame(det, cal["H"])
    n_px = int(round(L["picture_units"] * PX))
    pl = V.fit_artwork(pf, n_px, n_px, margin_mm=L["margin_mm"], rotate=0)
    to_mm = PT.machine_mapper({"placement": pl})
    unit = pl["width_mm"] / L["picture_units"]
    T = L["picture_units"]

    def f(pts):                                         # artwork plane: x right, y UP (the picture's rows are flipped)
        mm = np.array([to_mm(x * unit, (T - y) * unit) for x, y in np.asarray(pts, float)])
        return V.mm_to_px_array(cal["H_inv"], mm)
    return f, unit


def measure(frame_path):
    L = load_layout()
    if frame_path:
        frame = cv2.imread(frame_path)
        if frame is None:
            raise SystemExit(f"cannot read {frame_path}")
    else:
        import vision as V
        import paint as PT
        cap = V.open_camera()
        frame, _ = PT.capture_fresh(cap)
        cap.release()
        cv2.imwrite(os.path.join(CHART_DIR, "chart_photo.png"), frame)
    to_px, unit = chart_to_camera(frame, L)
    colors, rows, cols = L["colors"], L["rows"], L["cols"]
    lo, hi = L["pad_units"] + L["gap_units"], L["pad_units"] + L["chart_units"] - L["gap_units"]
    n = len(colors)
    print(f"chart on this sheet: bands {L['band_units'] * unit:.1f} mm wide, {L['gap_units'] * unit:.1f} mm apart")

    # the painted bands may sit a little off the planned place (paper detection now vs then): the shift that puts the most
    # COLOURED pixels inside the planned bands is used (within 3 mm)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    colored = ((hsv[..., 1] > 60) & (hsv[..., 2] > 40)).astype(np.float32)
    plan = np.zeros(frame.shape[:2], np.float32)
    for a, b in rows:
        cv2.fillPoly(plan, [np.round(to_px([(lo, a), (hi, a), (hi, b), (lo, b)])).astype(np.int32)], 1.0)
    for a, b in cols:
        cv2.fillPoly(plan, [np.round(to_px([(a, lo), (b, lo), (b, hi), (a, hi)])).astype(np.int32)], 1.0)
    px_mm = float(np.linalg.norm(to_px([(10.0 / unit, 0.0)])[0] - to_px([(0.0, 0.0)])[0])) / 10.0
    # the shift within 3 mm that puts the most coloured pixels inside the planned bands (searched directly: the chart
    # repeats, so a whole-picture correlation can lock onto the NEXT band, 2026-11-12)
    ys_, xs_ = np.nonzero(plan)
    r_ = int(np.ceil(3.0 * px_mm))
    y0_, y1_ = max(0, ys_.min() - r_), min(plan.shape[0], ys_.max() + r_ + 1)
    x0_, x1_ = max(0, xs_.min() - r_), min(plan.shape[1], xs_.max() + r_ + 1)
    pc, cc = plan[y0_:y1_, x0_:x1_], colored[y0_:y1_, x0_:x1_]
    best_ = (-1.0, 0, 0)
    for dy in range(-r_, r_ + 1):
        for dx in range(-r_, r_ + 1):
            if dx * dx + dy * dy <= r_ * r_:
                sc = float((np.roll(np.roll(pc, dy, 0), dx, 1) * cc).sum())
                if sc > best_[0]:
                    best_ = (sc, dx, dy)
    sx, sy = float(best_[1]), float(best_[2])
    if np.hypot(sx, sy) > 3.0 * px_mm:
        print(f"   (the paint sits {np.hypot(sx, sy) / px_mm:.1f} mm from the planned place: too far to be a small shift, NOT corrected)")
        sx = sy = 0.0
    elif np.hypot(sx, sy) > 0.3 * px_mm:
        print(f"   the paint sits {np.hypot(sx, sy) / px_mm:.1f} mm from the planned place: measured where it is")

    lin = srgb_to_lin(frame[..., ::-1])
    out_img = frame.copy()

    def patch(x0, x1, y0, y1, label):
        """Median linear RGB over the middle SAMPLE_SHARE of a chart rectangle (layout units)."""
        cx, cy, hw, hh = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) * SAMPLE_SHARE / 2, (y1 - y0) * SAMPLE_SHARE / 2
        q = np.round(to_px([(cx - hw, cy - hh), (cx + hw, cy - hh), (cx + hw, cy + hh), (cx - hw, cy + hh)]) + [sx, sy]).astype(np.int32)
        m = np.zeros(frame.shape[:2], np.uint8)
        cv2.fillPoly(m, [q], 1)
        cv2.polylines(out_img, [q], True, (255, 255, 255), 1)
        sel = m.astype(bool)
        if sel.sum() < 10:
            raise SystemExit(f"patch {label} is outside the picture")
        return np.median(lin[sel], axis=0)

    inner_x = [(cols[k - 1][1], cols[k][0]) for k in range(1, n)]                   # paper BETWEEN two columns (inside the chart)
    inner_y = [(rows[k - 1][1], rows[k][0]) for k in range(1, n)]
    paper = np.median([patch(xa, xb, ya, yb, "paper") for xa, xb in inner_x for ya, yb in inner_y], axis=0)

    def rel(c):
        return c / np.maximum(paper, 1e-6)

    single_h = [rel(np.median([patch(xa, xb, a, b, f"{colors[i]} alone") for xa, xb in inner_x], axis=0)) for i, (a, b) in enumerate(rows)]
    single_v = [rel(np.median([patch(a, b, ya, yb, f"{colors[j]} alone") for ya, yb in inner_y], axis=0)) for j, (a, b) in enumerate(cols)]
    cross = [[rel(patch(cols[j][0], cols[j][1], rows[i][0], rows[i][1], f"{colors[i]} under {colors[j]}")) for j in range(n)] for i in range(n)]

    res = {"colors": colors, "measured": time.strftime("%Y-%m-%d %H:%M:%S"), "band_mm": L["band_units"] * unit,
           "paper_rgb_camera": [float(v) for v in lin_to_srgb(paper)], "single": {}, "pairs": []}
    print(f"\nmeasured (Lab, the paper = L 100; each colour alone, the mean of its horizontal and vertical band):")
    for i, c in enumerate(colors):
        t = np.sqrt(single_h[i] * single_v[i])
        lab = refl_to_lab(t)
        res["single"][c] = {"refl": [float(v) for v in t], "lab": [float(v) for v in lab],
                            "lab_h": [float(v) for v in refl_to_lab(single_h[i])], "lab_v": [float(v) for v in refl_to_lab(single_v[i])]}
        dpass = float(np.linalg.norm(refl_to_lab(single_h[i]) - refl_to_lab(single_v[i])))
        print(f"   {c:7s} L {lab[0]:5.1f} a {lab[1]:+6.1f} b {lab[2]:+6.1f}   (pass 1 vs pass 2 differ by dE {dpass:.1f})")
    print("\noverlaps (UNDER + OVER): measured Lab | predicted by multiplying the two singles | dE between them")
    errs = []
    for i in range(n):
        for j in range(n):
            m = cross[i][j]
            pred = single_h[i] * single_v[j]
            lab_m, lab_p = refl_to_lab(m), refl_to_lab(pred)
            de = float(np.linalg.norm(lab_m - lab_p))
            errs.append(de)
            res["pairs"].append({"under": colors[i], "over": colors[j], "refl": [float(v) for v in m],
                                 "lab": [float(v) for v in lab_m], "lab_multiply": [float(v) for v in lab_p], "dE_multiply": de})
            tag = "double" if i == j else ""
            print(f"   {colors[i]:7s} + {colors[j]:7s} {tag:6s} L {lab_m[0]:5.1f} a {lab_m[1]:+6.1f} b {lab_m[2]:+6.1f} | "
                  f"L {lab_p[0]:5.1f} a {lab_p[1]:+6.1f} b {lab_p[2]:+6.1f} | dE {de:5.1f}")
    order = []
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(refl_to_lab(cross[i][j]) - refl_to_lab(cross[j][i])))
            order.append(d)
            print(f"   order: {colors[i]} under {colors[j]}  vs  {colors[j]} under {colors[i]}:  dE {d:.1f}")
    res["dE_multiply_median"] = float(np.median(errs))
    res["dE_order_median"] = float(np.median(order)) if order else None
    print(f"\nmultiplying the singles predicts the overlaps to dE {np.median(errs):.1f} (median; under 5 = hard to see, over 10 = clearly off)"
          + (f"; the layer order changes the colour by dE {np.median(order):.1f}" if order else ""))
    json.dump(res, open(RESULT, "w"), indent=1)

    # swatches: what the camera saw (relative to the paper), next to what multiplying predicted
    cell = 60
    sw = np.full(((n + 2) * cell, (2 * n + 3) * cell, 3), 255, np.uint8)

    def put(r, c, refl):
        rgb = lin_to_srgb(np.clip(refl, 0, 1)).astype(np.uint8)
        sw[r * cell + 4:(r + 1) * cell - 4, c * cell + 4:(c + 1) * cell - 4] = rgb[::-1]
    for j in range(n):
        put(0, j + 1, single_v[j])
        put(0, n + 3 + j, single_v[j])
    for i in range(n):
        put(i + 1, 0, single_h[i])
        put(i + 1, n + 2, single_h[i])
        for j in range(n):
            put(i + 1, j + 1, cross[i][j])
            put(i + 1, n + 3 + j, single_h[i] * single_v[j])
    cv2.putText(sw, "measured", (cell, (n + 2) * cell - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(sw, "multiplied", ((n + 3) * cell, (n + 2) * cell - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(os.path.join(CHART_DIR, "chart_swatches.png"), sw)
    cv2.imwrite(os.path.join(CHART_DIR, "chart_measured.png"), out_img)
    print("ko/chart_result.json, ko/chart_swatches.png (rows = under, columns = over), ko/chart_measured.png (where it measured)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make", help="write the chart pictures for these colours")
    m.add_argument("colors", nargs="+")
    m.add_argument("--keep_order", action="store_true", help="paint in the order typed (default: bright to dark)")
    p = sub.add_parser("paint", help="pass 1, drying, pass 2 (or only h / only v)")
    p.add_argument("which", nargs="?", default="all", choices=["all", "h", "v"])
    p.add_argument("--dry", type=float, default=CHART_DRY_MIN, help=f"minutes of drying between the passes (default {CHART_DRY_MIN:g})")
    me = sub.add_parser("measure", help="photograph the dry chart and measure every patch")
    me.add_argument("--frame", help="a saved camera picture instead of a new photo (e.g. paint_final.png)")
    a, extra = ap.parse_known_args(argv)
    os.chdir(HERE)
    if a.cmd == "make":
        make(a.colors, a.keep_order)
    elif a.cmd == "paint":
        paint(a.which, extra, a.dry)
    else:
        measure(a.frame)


if __name__ == "__main__":
    main()
