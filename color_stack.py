"""
color_stack.py - the STACK chart: transparent colours glazed over each other in up to four passes, every cell measured, so
the program knows which stack of glazes gives which shade.

    python color_stack.py make orange red blue          # ko/stack_p1.png ... ko/stack.json (3 passes)
    python color_stack.py make orange red blue --passes 4
    python color_stack.py paint                         # all passes, drying in between (probes, dips, rinse, repair: automatic)
    python color_stack.py paint 3                       # from pass 3 on (e.g. after a stop)
    python color_stack.py measure                       # photo -> ko/stack_result.json, ko/stack_swatches.png
    python color_stack.py find 7a3020                   # which measured stacks come closest to this colour (hex RGB)

THE PASSES (bands never overlap WITHIN a pass: paint.py paints one picture, and wet paint must not be painted over;
every overlap comes from a later pass, after drying). At most TWO bands of a colour in each direction:
    pass 1  VERTICAL bands, each colour once
    pass 2  HORIZONTAL bands, each colour once        -> every pair, horizontal OVER vertical
    pass 3  VERTICAL bands again, each colour shifted half a band over its own pass-1 band
            -> across each: A alone | A twice (two coats) | A alone but painted later, so at the crossings every pair
               comes in BOTH orders, and A + B + A where both vertical coats lie
    pass 4  (--passes 4) HORIZONTAL bands again, shifted, each over the band of the NEXT colour: pairs and 4-layer stacks
Some stacks appear more than once: their spread shows how even the painting is.

The colours of one pass are painted bright to dark, like any artwork. The chart takes STACK_SHARE of the usable sheet and is
landscape (its long side on the sheet's long side), so a bigger sheet gives bigger cells. All passes are placed by the same
rule on the same sheet: the sheet must NOT move between them.

measure: every cell's colour relative to the paper beside the chart, in linear light and Lab (color_chart.py's method),
the repeats of a stack combined, and compared with MULTIPLYING the single coats (an ideal glaze).
find: the stacks nearest to a target colour (dE), for planning a painting from this table.
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
import color_chart as CC                                   # noqa: E402  (colours, drying, light / Lab maths)

DIR = os.path.join(HERE, "ko")
LAYOUT = os.path.join(DIR, "stack.json")
RESULT = os.path.join(DIR, "stack_result.json")
PX = 10                     # picture pixels per layout unit
BAND = 12.0                 # band width (layout units); the half-band shift makes cells BAND / 2 wide
STACK_SHARE = 0.92          # the chart takes this share of the usable sheet (the rest is paper around it: the white reference)
ASPECT = 1.4                # picture width : height (A4 landscape inside a 10 mm margin is about 1.4)
MARGIN_MM = 10.0
DRY_MIN = 15.0
SAMPLE_SHARE = 0.5          # a cell is measured over its middle part only
MIN_CELL_MM = 5.0           # a warning when the cells come out smaller than this on A4


# ---------------------------------------------------------------------------------------------------------- make
def axis_slots(colors, second=False, rot=0):
    """Bands along one axis: each colour ONCE in the first pass; with `second`, each colour once more in a later pass, shifted
    half a band, over the band of the colour `rot` places before it (rot 0 = over itself: two coats). Between the slots half a
    band of paper. Returns (bands_first, bands_second, used): bands = [(start, end, colour)]."""
    period = 2.0 * BAND                                       # band, half-band shift, half a band of paper
    n = len(colors)
    first = [(k * period, k * period + BAND, colors[k]) for k in range(n)]
    sec = [(k * period + BAND / 2, k * period + 1.5 * BAND, colors[(k + rot) % n]) for k in range(n)] if second else []
    return first, sec, n * period - (BAND / 2 if second else BAND)


def make(colors, passes=3, keep_order=False):
    bad = [c for c in colors if c not in CC.HUE_OF]
    if bad:
        raise SystemExit(f"unknown colour {', '.join(bad)}: use {', '.join(CC.HUE_OF)}")
    if len(set(colors)) != len(colors) or not 2 <= len(colors) <= 5:
        raise SystemExit("2 to 5 colours, each once")
    if passes not in (3, 4):
        raise SystemExit("--passes 3 or 4")
    if not keep_order:
        colors = sorted(colors, key=lambda c: -CC.BRIGHT[c])
    v1, v2, w_used = axis_slots(colors, True, 0)                     # vertical: each colour, then the same again (two coats)
    h1, h2, h_used = axis_slots(colors, passes == 4, 1)               # horizontal: each colour (pass 4: the next colour over it)
    # the vertical bands run one band past the horizontal block at the top and bottom, the horizontal ones past the vertical
    # block left and right: there each direction lies alone (single coats, pairs of one direction), the corners stay paper
    ext = BAND
    chart_w, chart_h = w_used + 2 * ext, h_used + 2 * ext
    W = max(chart_w / STACK_SHARE, chart_h / STACK_SHARE * ASPECT)            # the picture: the chart with paper around it
    Hh = W / ASPECT
    ox, oy = (W - chart_w) / 2 + ext, (Hh - chart_h) / 2 + ext
    shift = lambda bands, o: [(a + o, b + o, c) for a, b, c in bands]
    v1, v2, h1, h2 = shift(v1, ox), shift(v2, ox), shift(h1, oy), shift(h2, oy)
    x_span, y_span = (ox - ext, ox + w_used + ext), (oy - ext, oy + h_used + ext)
    # pass order: 1 vertical, 2 horizontal, 3 vertical shifted, 4 horizontal shifted
    plan = [("v", v1), ("h", h1), ("v", v2)] + ([("h", h2)] if passes == 4 else [])
    L = {"colors": colors, "passes": [{"dir": d, "bands": [[a, b, c] for a, b, c in bands]} for d, bands in plan],
         "picture_units": [W, Hh], "x_span": x_span, "y_span": y_span, "margin_mm": MARGIN_MM, "band_units": BAND}
    wpx, hpx = int(round(W * PX)), int(round(Hh * PX))
    os.makedirs(DIR, exist_ok=True)
    look = np.ones((hpx, wpx, 3))
    rank = {c: k for k, c in enumerate(colors)}
    for p, (d, bands) in enumerate(plan, 1):
        img = np.full((hpx, wpx, 3), 255, np.uint8)
        for a, b, c in bands:
            rgb = CC.name_rgb(c, 250 - 5 * rank[c])               # each next colour a little darker: paint.py keeps our order
            if d == "v":
                sl = (slice(int(round(y_span[0] * PX)), int(round(y_span[1] * PX))), slice(int(round(a * PX)), int(round(b * PX))))
            else:
                sl = (slice(int(round(a * PX)), int(round(b * PX))), slice(int(round(x_span[0] * PX)), int(round(x_span[1] * PX))))
            img[sl] = rgb
        cv2.imwrite(os.path.join(DIR, f"stack_p{p}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        json.dump({"regions": [{"box": [0, 0, wpx, hpx], "style": CC.CHART_STYLE, "infill_type": "lines"}]},
                  open(os.path.join(DIR, f"stack_p{p}.styles.json"), "w"), indent=1)
        look *= 0.45 + 0.55 * img.astype(float) / 255.0                    # a half-strength glaze, roughly
    cv2.imwrite(os.path.join(DIR, "stack_look.png"), cv2.cvtColor((look * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    json.dump(L, open(LAYOUT, "w"), indent=1)
    cells = cells_of(L)
    stacks = {}
    for c in cells:
        if not c["stack"]:
            continue
        stacks.setdefault(tuple(c["stack"]), 0)
        stacks[tuple(c["stack"])] += 1
    a4 = min((297 - 2 * MARGIN_MM) / W, (210 - 2 * MARGIN_MM) / Hh)          # mm per unit on A4 landscape
    print(f"stack chart: {len(colors)} colours ({', '.join(colors)}), {passes} passes, {len(cells)} cells, "
          f"{len(stacks)} different stacks ({sum(1 for v in stacks.values() if v > 1)} of them more than once)")
    by_n = {}
    for s in stacks:
        by_n[len(s)] = by_n.get(len(s), 0) + 1
    print("   layers per stack: " + ", ".join(f"{k}: {v}" for k, v in sorted(by_n.items())))
    print(f"   on A4 the chart is about {chart_w * a4:.0f} x {chart_h * a4:.0f} mm; bands {BAND * a4:.1f} mm, cells "
          f"{BAND / 2 * a4:.1f} mm" + ("   WARNING: small cells, use fewer colours or a bigger sheet" if BAND / 2 * a4 < MIN_CELL_MM else ""))
    print("   ko/stack_p1.png ... (one picture per pass), ko/stack_look.png (rough idea)")
    print("next:  python color_stack.py paint")


def cells_of(L):
    """Every cell of the chart: the rectangle (layout units) where the same passes overlap, and its stack (colours in the
    order they were painted)."""
    xs, ys = set(L["x_span"]), set(L["y_span"])
    for p in L["passes"]:
        for a, b, _ in p["bands"]:
            (xs if p["dir"] == "v" else ys).update((a, b))
    xs, ys = sorted(xs), sorted(ys)
    out = []
    for x0, x1 in zip(xs, xs[1:]):
        for y0, y1 in zip(ys, ys[1:]):
            if x1 - x0 < BAND / 4 or y1 - y0 < BAND / 4:
                continue
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            if not (L["x_span"][0] < cx < L["x_span"][1] and L["y_span"][0] < cy < L["y_span"][1]):
                continue
            st = []
            for p in L["passes"]:
                for a, b, c in p["bands"]:
                    if a < (cx if p["dir"] == "v" else cy) < b:
                        st.append(c)
            out.append({"box": [x0, x1, y0, y1], "stack": st})
    return out


def load_layout():
    if not os.path.isfile(LAYOUT):
        raise SystemExit("no ko/stack.json: run  python color_stack.py make <colours>  first")
    return json.load(open(LAYOUT))


# --------------------------------------------------------------------------------------------------------- paint
def run_pass(p, L, extra):
    import settings
    settings.SIGNATURE = None
    import paint as PT
    img = os.path.join("ko", f"stack_p{p}.png")
    argv = [img, "--rotate", "0", "--margin", f"{L['margin_mm']:g}", "--color", "auto"] + extra
    if p > 1 and os.path.isfile(os.path.join(HERE, "probe_result.json")) and "--probe_file" not in extra:
        # the same sheet: the probe of the first pass is used again (no new probe lines; paint.py refuses it if the sheet
        # moved or the result is over 6 hours old). Faint colour probes were not recognised and painted again (2026-11-12)
        argv += ["--probe_file", os.path.join(HERE, "probe_result.json")]
    d = L["passes"][p - 1]["dir"]
    print(f"\ncolor_stack: PASS {p} of {len(L['passes'])}, {'vertical' if d == 'v' else 'horizontal'} bands:  paint.py " + " ".join(argv))
    if p > 1:
        print("   the sheet must not have moved since pass 1; its probe is reused")
    PT.LAST_JOB_DONE = False
    PT.main(argv)
    if not PT.LAST_JOB_DONE:                                # stopped (probes failed, N, Ctrl+C): no drying, no next pass
        raise SystemExit("\nthis pass was NOT painted to the end (the reason is above): stopped here. "
                         f"After fixing it, go on with this pass:  python color_stack.py paint {p}")
    final = os.path.join(HERE, "paint_final.png")                # the next pass overwrites it: a copy per pass
    if os.path.isfile(final):
        import shutil
        shutil.copyfile(final, os.path.join(DIR, f"stack_after_p{p}.png"))
        print(f"   photo of the sheet after pass {p}: ko/stack_after_p{p}.png")


def paint(start, extra, dry_min):
    L = load_layout()
    n = len(L["passes"])
    for p in range(start, n + 1):
        if p > start:
            CC.wait_dry(dry_min, f"pass {p}")
        run_pass(p, L, extra)
    print("\nnext, when dry:  python color_stack.py measure")


# ------------------------------------------------------------------------------------------------------- measure
def stack_to_camera(frame, L):
    """Layout units -> camera pixels, by the placement paint.py uses (the picture as large as fits, same margin, not turned)."""
    import paint as PT
    import vision as V
    cal = V.load_calibration()
    det = V.detect_paper(frame, cal["H"], cal["H_inv"], calib_mm=cal["mm_pts"])
    if det is None:
        raise SystemExit("no paper found in the picture")
    det = V.clip_to_reach(det)
    pf = V.paper_frame(det, cal["H"])
    W, Hh = L["picture_units"]
    pl = V.fit_artwork(pf, int(round(W * PX)), int(round(Hh * PX)), margin_mm=L["margin_mm"], rotate=0)
    to_mm = PT.machine_mapper({"placement": pl})
    unit = pl["width_mm"] / W

    def f(pts):                                           # artwork plane: x right, y UP (the picture's rows are flipped)
        mm = np.array([to_mm(x * unit, (Hh - y) * unit) for x, y in np.asarray(pts, float)])
        return V.mm_to_px_array(cal["H_inv"], mm)
    return f, unit


def measure(frame_path):
    L = load_layout()
    if frame_path:
        frame = cv2.imread(frame_path)
        if frame is None:
            raise SystemExit(f"cannot read {frame_path}")
    else:
        import paint as PT
        import vision as V
        cap = V.open_camera()
        frame, _ = PT.capture_fresh(cap)
        cap.release()
        cv2.imwrite(os.path.join(DIR, "stack_photo.png"), frame)
    to_px, unit = stack_to_camera(frame, L)
    cells = cells_of(L)
    print(f"stack chart on this sheet: bands {L['band_units'] * unit:.1f} mm, cells {L['band_units'] / 2 * unit:.1f} mm, {len(cells)} cells")
    rect = lambda x0, x1, y0, y1: np.round(to_px([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])).astype(np.int32)

    # small shift between plan and paint (paper found now vs then): where the coloured pixels fit the planned bands best
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    colored = ((hsv[..., 1] > 60) & (hsv[..., 2] > 40)).astype(np.float32)
    plan = np.zeros(frame.shape[:2], np.float32)
    for c in cells:
        if c["stack"]:
            cv2.fillPoly(plan, [rect(*c["box"])], 1.0)
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
        print(f"   (the paint sits {np.hypot(sx, sy) / px_mm:.1f} mm from the planned place: too far for a small shift, NOT corrected)")
        sx = sy = 0.0
    elif np.hypot(sx, sy) > 0.3 * px_mm:
        print(f"   the paint sits {np.hypot(sx, sy) / px_mm:.1f} mm from the planned place: measured where it is")

    lin = CC.srgb_to_lin(frame[..., ::-1])
    shown = frame.copy()

    def sample(x0, x1, y0, y1):
        cx, cy, hw, hh = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) * SAMPLE_SHARE / 2, (y1 - y0) * SAMPLE_SHARE / 2
        q = np.round(to_px([(cx - hw, cy - hh), (cx + hw, cy - hh), (cx + hw, cy + hh), (cx - hw, cy + hh)]) + [sx, sy]).astype(np.int32)
        m = np.zeros(frame.shape[:2], np.uint8)
        cv2.fillPoly(m, [q], 1)
        cv2.polylines(shown, [q], True, (255, 255, 255), 1)
        sel = m.astype(bool)
        return np.median(lin[sel], axis=0) if sel.sum() >= 6 else None

    # the paper: the white ring around the chart, left / right / above / below it
    W, Hh = L["picture_units"]
    (xa, xb), (ya, yb) = L["x_span"], L["y_span"]
    ring = [(xa * 0.2, xa * 0.8, ya, yb), (xb + (W - xb) * 0.2, xb + (W - xb) * 0.8, ya, yb),
            (xa, xb, ya * 0.2, ya * 0.8), (xa, xb, yb + (Hh - yb) * 0.2, yb + (Hh - yb) * 0.8)]
    ps = [s for s in (sample(*r) for r in ring if r[1] - r[0] > 1 and r[3] - r[2] > 1) if s is not None]
    blank = [sample(*c["box"]) for c in cells if not c["stack"]]
    ps += [s for s in blank if s is not None]
    if not ps:
        raise SystemExit("no paper around the chart to measure against")
    paper = np.median(ps, axis=0)

    got = {}
    for c in cells:
        if not c["stack"]:
            continue
        s = sample(*c["box"])
        if s is not None:
            got.setdefault(tuple(c["stack"]), []).append(s / np.maximum(paper, 1e-6))
    one = {k[0]: np.median(v, axis=0) for k, v in got.items() if len(k) == 1}
    res = {"colors": L["colors"], "measured": time.strftime("%Y-%m-%d %H:%M:%S"), "cell_mm": L["band_units"] / 2 * unit,
           "paper_rgb_camera": [float(v) for v in CC.lin_to_srgb(paper)], "stacks": []}
    errs, spreads = [], []
    for st, vals in sorted(got.items(), key=lambda kv: (len(kv[0]), kv[0])):
        r = np.median(vals, axis=0)
        lab = CC.refl_to_lab(r)
        spread = float(np.median([np.linalg.norm(CC.refl_to_lab(v) - lab) for v in vals])) if len(vals) > 1 else None
        pred = np.prod([one[c] for c in st], axis=0) if all(c in one for c in st) else None
        de = float(np.linalg.norm(CC.refl_to_lab(pred) - lab)) if pred is not None and len(st) > 1 else None
        if de is not None:
            errs.append(de)
        if spread is not None:
            spreads.append(spread)
        res["stacks"].append({"stack": list(st), "refl": [float(v) for v in r], "lab": [float(v) for v in lab],
                              "hex": "%02x%02x%02x" % tuple(int(v) for v in CC.lin_to_srgb(r)), "cells": len(vals),
                              "spread_dE": spread, "dE_multiply": de})
    json.dump(res, open(RESULT, "w"), indent=1)
    print(f"\n{len(res['stacks'])} stacks measured (Lab, the paper = L 100):")
    for s in res["stacks"]:
        print(f"   {' + '.join(s['stack']):32s} L {s['lab'][0]:5.1f} a {s['lab'][1]:+6.1f} b {s['lab'][2]:+6.1f}  #{s['hex']}  "
              f"x{s['cells']}" + (f"  spread dE {s['spread_dE']:.1f}" if s["spread_dE"] is not None else "")
              + (f"  multiply dE {s['dE_multiply']:.1f}" if s["dE_multiply"] is not None else ""))
    if spreads:
        print(f"\nrepeats agree to dE {np.median(spreads):.1f} (median): how even the painting is")
    if errs:
        print(f"multiplying the single coats predicts the stacks to dE {np.median(errs):.1f} (median; under 5 = hard to see)")
    swatches(res)
    cv2.imwrite(os.path.join(DIR, "stack_measured.png"), shown)
    print("ko/stack_result.json, ko/stack_swatches.png, ko/stack_measured.png (where it measured)")


def swatches(res, path=None, cell=56):
    """All measured stacks as patches, lightest to darkest, each with its recipe under it."""
    st = sorted(res["stacks"], key=lambda s: -s["lab"][0])
    cols = 8
    rows = (len(st) + cols - 1) // cols
    img = np.full((rows * (cell + 34), cols * (cell + 70), 3), 255, np.uint8)
    for k, s in enumerate(st):
        r, c = divmod(k, cols)
        x, y = c * (cell + 70) + 4, r * (cell + 34) + 4
        img[y:y + cell, x:x + cell] = CC.lin_to_srgb(np.clip(s["refl"], 0, 1)).astype(np.uint8)[::-1]
        short = "+".join(n[:3] for n in s["stack"])
        cv2.putText(img, short, (x, y + cell + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(img, f"L{s['lab'][0]:.0f}", (x, y + cell + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (90, 90, 90), 1, cv2.LINE_AA)
    cv2.imwrite(path or os.path.join(DIR, "stack_swatches.png"), img)


# ---------------------------------------------------------------------------------------------------------- find
def find(target_hex, n=8):
    if not os.path.isfile(RESULT):
        raise SystemExit("no ko/stack_result.json: paint and measure the stack chart first")
    res = json.load(open(RESULT))
    t = target_hex.strip("#")
    rgb = np.array([int(t[i:i + 2], 16) for i in (0, 2, 4)], float)
    lab_t = CC.refl_to_lab(CC.srgb_to_lin(rgb))
    ranked = sorted(res["stacks"], key=lambda s: np.linalg.norm(np.array(s["lab"]) - lab_t))
    print(f"#{t}: L {lab_t[0]:.1f} a {lab_t[1]:+.1f} b {lab_t[2]:+.1f}; nearest measured stacks (painted in this order):")
    for s in ranked[:n]:
        print(f"   dE {np.linalg.norm(np.array(s['lab']) - lab_t):5.1f}   {' + '.join(s['stack']):32s} #{s['hex']}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make", help="write one picture per pass for these colours")
    m.add_argument("colors", nargs="+")
    m.add_argument("--passes", type=int, default=3, help="3 (default) or 4 (pairs down the sheet too)")
    m.add_argument("--keep_order", action="store_true", help="the colours in the order typed (default: bright to dark)")
    p = sub.add_parser("paint", help="all passes with drying between them")
    p.add_argument("start", nargs="?", type=int, default=1, help="start at this pass (default 1)")
    p.add_argument("--dry", type=float, default=DRY_MIN, help=f"minutes of drying between passes (default {DRY_MIN:g})")
    me = sub.add_parser("measure", help="photograph the dry chart and measure every cell")
    me.add_argument("--frame", help="a saved camera picture instead of a new photo")
    f = sub.add_parser("find", help="the measured stacks nearest to a colour")
    f.add_argument("hex")
    a, extra = ap.parse_known_args(argv)
    os.chdir(HERE)
    if a.cmd == "make":
        make(a.colors, a.passes, a.keep_order)
    elif a.cmd == "paint":
        paint(a.start, extra, a.dry)
    elif a.cmd == "measure":
        measure(a.frame)
    else:
        find(a.hex)


if __name__ == "__main__":
    main()
