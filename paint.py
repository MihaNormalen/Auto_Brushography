"""
paint.py - artwork -> G-code for the brush plotter, and (by default) painting it.

    python paint.py artwork.png out.gcode            (or set ARTWORK in settings.py and run: python paint.py)

It homes the machine, moves the toolhead out of the way, photographs the table, and from the picture it finds by itself:
  * the paper, and places + scales the artwork on it (centred, upright, 10 mm margin)
  * the colours of the artwork, each one becomes a layer (lightest first, black last)
  * the paint dish of every colour (saturated pools near the dip position; specks and LEDs are ignored)
It writes the G-code, a preview per layer and an overlay picture, opens it, and asks "Start painting now?" before the first
stroke. Then it sends the G-code itself, with a live camera window. If a colour has no dish it stops BEFORE painting.

Every number you may want to change is in settings.py. The brush profile (BRUSH_PROFILE, made by train.py) decides how deep each
mark is painted: a thin mark at the touch height, a wide mark deeper, so the REAL brush makes that width.
--plan_only = never touch the machine (still uses the camera).  --yes = do not ask.  Ctrl+C = lift the brush and stop.
Other options: --list_colors, --color blue, --frame shot.png, --paper paper.json, --rotate 0/90/auto, --margin, --width, --infill, --seed.
"""
import argparse
import math
import os
import re
import random
import time

import cv2
import numpy as np
from PIL import Image
from skimage.morphology import skeletonize

import brush_profile as BP
import machine as mc
import settings as S
from vision import mm_to_px_array, px_to_mm_array, load_calibration

RES = 8.0        # pixels per mm used for planning


# ----------------------------------------------------------------------------------------------- helpers
def disk(r_px):
    r = max(1, int(round(r_px)))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


# Hue ranges in OpenCV's 0..179 hue scale (pure blue is 120, pure red is 0)
COLOR_HUES = {"red": [(0, 8), (172, 180.5)], "orange": [(9, 20)], "yellow": [(21, 35)], "green": [(36, 85)],
              "cyan": [(86, 98)], "blue": [(99, 135)], "purple": [(136, 160)], "pink": [(161, 171)]}
MIN_SAT, MIN_VAL = 64, 50          # below this a pixel is grey / white / black, not "a colour"


def color_mask(rgb, color, thr=140, tol=40.0):
    """Boolean mask of the artwork pixels that belong to `color`.
    'dark'  = every dark pixel, whatever its colour (what mama.py did)
    'any'   = every coloured or dark pixel (anything that is not paper)
    a name from COLOR_HUES ('blue', 'red', ...), or 'r,g,b' (closest colours within `tol` Lab distance)."""
    if color == "dark":
        gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
        return gray < thr
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    chromatic = (s >= MIN_SAT) & (v >= MIN_VAL)
    if color == "any":
        return chromatic | (v < MIN_VAL) | (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2] < thr)
    if color in COLOR_HUES:
        hue_ok = np.zeros(h.shape, bool)
        for lo, hi in COLOR_HUES[color]:
            hue_ok |= (h >= lo) & (h <= hi)
        return chromatic & hue_ok
    try:
        target = np.array([int(t) for t in color.split(",")], np.uint8).reshape(1, 1, 3)
    except ValueError:
        raise SystemExit(f"unknown colour '{color}': use dark, any, {', '.join(COLOR_HUES)} or r,g,b")
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    t = cv2.cvtColor(target, cv2.COLOR_RGB2LAB).astype(np.float32)
    return np.linalg.norm(lab - t, axis=2) < tol


def load_mask(path, width_mm, res=RES, thr=140, rotate=0, color="dark", tol=40.0):
    """True = paint. Image rows are flipped so that image-up is +y of the artwork plane (same as mama.py)."""
    img = Image.open(path).convert("RGB")
    if rotate == 90:
        img = img.rotate(90, expand=True)
    img = img.transpose(Image.FLIP_TOP_BOTTOM)
    w = int(width_mm * res)
    h = int(width_mm * (img.height / img.width) * res)
    img = img.resize((w, h), Image.Resampling.LANCZOS)          # smooth edges first, classify afterwards
    return color_mask(np.array(img), color, thr, tol)


def describe_colors(path):
    """Prints which colours the artwork contains (share of the picture)."""
    img = Image.open(path).convert("RGB")
    img = img.resize((300, max(1, int(300 * img.height / img.width))), Image.Resampling.LANCZOS)
    rgb = np.array(img)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    total = rgb.shape[0] * rgb.shape[1]
    paper = (hsv[..., 1] < 40) & (hsv[..., 2] > 200)
    black = (hsv[..., 2] < MIN_VAL) | ((hsv[..., 1] < MIN_SAT) & (hsv[..., 2] < 110))
    vprint(f"colours in {path}  ({img.width} x {img.height} px sample)")
    vprint(f"   paper / white : {100 * paper.sum() / total:5.1f}%")
    vprint(f"   black / dark  : {100 * black.sum() / total:5.1f}%")
    for name in COLOR_HUES:
        mk = color_mask(rgb, name)
        if mk.any():
            mean = rgb[mk].mean(axis=0).astype(int).tolist()
            vprint(f"   {name:13s} : {100 * mk.sum() / total:5.1f}%   typical RGB {mean}")


def locate_regions(mask, cfg, label, res=RES, min_mm=3.0, top=8):
    """Prints where the regions of this colour are: size, machine X/Y, and distance from the paper's left/bottom edge."""
    n, lab, st, cen = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    to_mm = machine_mapper(cfg)
    pf = cfg.get("paper_frame")
    regs = [(st[k, cv2.CC_STAT_AREA], k) for k in range(1, n) if max(st[k, cv2.CC_STAT_WIDTH], st[k, cv2.CC_STAT_HEIGHT]) / res >= min_mm]
    regs.sort(reverse=True)
    vprint(f"{label}: {len(regs)} region(s) of at least {min_mm:.0f} mm" + (f" (showing the biggest {top})" if len(regs) > top else ""))
    for i, (area, k) in enumerate(regs[:top], 1):
        x, y, w, h = st[k, :4]
        cx, cy = cen[k]
        X, Y = to_mm(cx / res, cy / res)
        s = f"   {label} region {i}: {w / res:.0f} x {h / res:.0f} mm, machine X {X:.0f} Y {Y:.0f}"
        if pf:
            c, u, v = np.array(pf["origin_mm"]), np.array(pf["u"]), np.array(pf["v"])
            d = np.array([X, Y]) - c
            s += f"   = {(d @ u) - pf['a_left']:.0f} mm from the paper's left edge, {(d @ v) - pf['b_bot']:.0f} mm from its bottom edge"
        vprint(s)


def load_rgb(path, width_mm, res=RES, rotate=0):
    """The artwork as an RGB array at planning resolution (rows flipped: row 0 = bottom, same as load_mask)."""
    img = Image.open(path).convert("RGB")
    if rotate == 90:
        img = img.rotate(90, expand=True)
    img = img.transpose(Image.FLIP_TOP_BOTTOM)
    w = int(width_mm * res)
    h = int(width_mm * (img.height / img.width) * res)
    return np.array(img.resize((w, h), Image.Resampling.LANCZOS))


def load_style_map(img_path, shape, rotate=0):
    """<artwork>.styles.json next to the artwork: boxes on the picture, each planned in its own INFILL_STYLE (e.g. a test sheet
    with every style, ko/styles_sheet.png). Format: {"regions": [{"box": [x0, y0, x1, y1], "style": "leaf"}, ...]}, the box in
    pixels of the picture as drawn, optionally "infill_type": "rings". Returns (labels at planning resolution, turned and flipped
    like load_rgb: 0 = in no box, i = regions[i - 1]; regions), or None without the file."""
    import json
    side = os.path.splitext(img_path)[0] + ".styles.json"
    if not os.path.exists(side):
        return None
    regions = json.load(open(side, encoding="utf-8"))["regions"]
    w0, h0 = Image.open(img_path).size
    lab = np.zeros((h0, w0), np.uint8)
    for i, r in enumerate(regions, 1):
        x0, y0, x1, y1 = (int(v) for v in r["box"])
        lab[max(0, y0):max(0, y1), max(0, x0):max(0, x1)] = i
    im = Image.fromarray(lab)
    if rotate == 90:
        im = im.rotate(90, expand=True)
    im = im.transpose(Image.FLIP_TOP_BOTTOM).resize((shape[1], shape[0]), Image.Resampling.NEAREST)
    return np.array(im), regions


def find_layers(rgb, thr=140, min_share=0.003, min_mm=3.0, res=RES):
    """Splits the artwork into colour layers, each pixel in exactly one layer:
    coloured pixels go to their hue family (blue, red, ...), dark neutral pixels to 'black'.
    A black-and-white artwork therefore gives one layer, exactly like mama.py. Light colours come first, black last."""
    from vision import circular_mean_hue
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hh, ss, vv = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    chromatic = (ss >= MIN_SAT) & (vv >= MIN_VAL)
    # a DARK colour with little colour in it (black ink with a blue cast, a photographed drawing: hue 113, sat 125, value 100) is
    # black, not "blue": chroma = sat x value / 255 below DARK_CHROMA. A real dark blue (sat 240, value 165: chroma 155) stays a colour
    chroma = ss.astype(float) * vv.astype(float) / 255.0
    # The same for washed-out tints (the bluish fringe of ink strokes on a scanned page: sat 76): weak colour is no colour
    chromatic &= ~((chroma < float(getattr(S, "DARK_CHROMA", 70.0))) & ((vv < 170) | (ss < 90)))
    # the colours the artwork really has: peaks of its hue histogram (circular, 0..179). Every coloured pixel goes to the NEAREST
    # peak, so a colour sitting on the border of two fixed hue ranges (a purple at hue 135) stays one clean layer, not two speckled ones
    hist = np.bincount(hh[chromatic].ravel(), minlength=180).astype(float)[:180]
    k = np.exp(-0.5 * (np.arange(-6, 7) / 2.5) ** 2)
    sm = np.convolve(np.r_[hist[-6:], hist, hist[:6]], k / k.sum(), "same")[6:-6]
    total_c = max(1.0, hist.sum())
    peaks = [i for i in range(180) if sm[i] >= sm[i - 1] and sm[i] > sm[(i + 1) % 180] and sm[i] > 0]
    peaks = [p for p in peaks if hist[[(p + d) % 180 for d in range(-6, 7)]].sum() >= 0.01 * total_c]   # at least 1 % of the coloured pixels
    peaks.sort(key=lambda p: -hist[[(p + d) % 180 for d in range(-6, 7)]].sum())
    keep = []
    for p in peaks:                                                  # peaks closer than 10 hue steps are one colour
        if all(min(abs(p - q), 180 - abs(p - q)) >= 10 for q in keep):
            keep.append(p)
    cand = []
    if keep:
        d = np.stack([np.minimum(np.abs(hh.astype(int) - p), 180 - np.abs(hh.astype(int) - p)) for p in keep])
        near = np.argmin(d, axis=0)
        used = {}
        for j, p in enumerate(keep):
            name = nearest_hue_name(p)
            used[name] = used.get(name, 0) + 1
            if used[name] > 1:
                name = f"{name}{used[name]}"                          # two different blues: blue, blue2
            cand.append((name, chromatic & (near == j)))
    black = ((~chromatic) & (gray < thr)) | (vv < MIN_VAL)
    faint_thr = getattr(S, "FAINT_LINE_THR", None)
    if faint_thr is not None and float(faint_thr) > thr:
        # thin lines drawn light grey (a 1 px line of a small picture, blurred when it is enlarged: grey 180..230) are lighter than
        # thr and were never planned, so the camera check could not miss them either (ko/48: the mast and the diagonal rope,
        # 2026-11-06). Their centre lines are added, thin, where they lie away from the dark strokes; wide grey areas (shading,
        # toned paper) are not lines and stay out
        faint = ((~chromatic) & (gray < float(faint_thr))).astype(np.uint8)
        faint &= ~(cv2.morphologyEx(faint, cv2.MORPH_OPEN, disk(0.5 * res)) > 0)          # only what is narrower than about 1 mm
        sk = skeletonize(faint > 0)
        far = cv2.distanceTransform((~black).astype(np.uint8), cv2.DIST_L2, 3) > 2.0
        add = (sk & far).astype(np.uint8)
        n_, lab_, st_, _ = cv2.connectedComponentsWithStats(add, connectivity=8)
        long_ = np.zeros(n_, bool)
        long_[1:] = np.maximum(st_[1:, cv2.CC_STAT_WIDTH], st_[1:, cv2.CC_STAT_HEIGHT]) >= res    # specks under 1 mm are paper grain
        add = long_[lab_].astype(np.uint8)
        black |= cv2.dilate(add, disk(1)) > 0
    cand.append(("black", black))
    total = rgb.shape[0] * rgb.shape[1]
    layers = []
    for name, mk in cand:
        if mk.sum() < min_share * total:
            continue
        n, lab, st, _ = cv2.connectedComponentsWithStats(mk.astype(np.uint8), connectivity=8)
        if not any(max(st[k, cv2.CC_STAT_WIDTH], st[k, cv2.CC_STAT_HEIGHT]) >= min_mm * res for k in range(1, n)):
            continue                                          # only specks: not worth a layer (a thin LINE is)
        layers.append({"name": name, "mask": mk, "share": float(mk.sum() / total),
                       "hue": None if name == "black" else float(circular_mean_hue(hh[mk])), "value": float(vv[mk].mean()),
                       "sat": float(np.median(ss[mk]))})
    # a whole "colour" layer that is dark and only weakly coloured (the navy cast of black ink) is black: merged into the black layer
    min_ch = float(getattr(S, "LAYER_MIN_CHROMA", 90.0))
    weak = [l for l in layers if l["hue"] is not None and l["value"] < 150
            and float(np.median(chroma[l["mask"]])) < min_ch]
    # ... and so is a "colour" that only hugs the black: the tinted halo along the edges of black strokes (ko/8.png: a blue
    # layer of 0.8 % in ~2000 specks, 91 % of it within 1 mm of black). A real colour area lies mostly away from the black
    blk0 = next((l for l in layers if l["name"] == "black"), None)
    if blk0 is not None:
        kk = int(2 * 1.0 * res) | 1
        near_blk = cv2.dilate(blk0["mask"].astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kk, kk))) > 0
        weak += [l for l in layers if l["hue"] is not None and l not in weak
                 and float(near_blk[l["mask"]].mean()) >= float(getattr(S, "LAYER_FRINGE_SHARE", 0.8))]
    if weak:
        blk = next((l for l in layers if l["name"] == "black"), None)
        mk = (blk["mask"] if blk else np.zeros(hh.shape, bool)).copy()
        for l in weak:
            mk |= l["mask"]
            layers.remove(l)
        if blk:
            layers.remove(blk)
        layers.append({"name": "black", "mask": mk, "share": float(mk.sum() / total), "hue": None,
                       "value": float(vv[mk].mean()), "sat": float(np.median(ss[mk]))})
    layers.sort(key=lambda l: (l["name"] == "black", -l["value"]))
    return layers


def nearest_hue_name(hue):
    """Name of the colour family whose CENTRE is nearest (a hue on the border of two ranges gets the closer family)."""
    best, bd = "?", 1e9
    for name, ranges in COLOR_HUES.items():
        a = [math.radians(2.0 * (lo + hi) / 2.0) for lo, hi in ranges]       # circular centre of the family's ranges
        w = [hi - lo for lo, hi in ranges]
        c = (math.degrees(math.atan2(sum(wi * math.sin(ai) for ai, wi in zip(a, w)), sum(wi * math.cos(ai) for ai, wi in zip(a, w)))) % 360.0) / 2.0
        d = min(abs(hue - c), 180.0 - abs(hue - c))
        if d < bd:
            best, bd = name, d
    return best


def hue_name(hue):
    """Name of the colour family for an OpenCV hue (display only)."""
    for name, ranges in COLOR_HUES.items():
        if any(lo <= hue <= hi for lo, hi in ranges):
            return name
    return "?"


def assign_dishes(layers, pools, cfg, max_hue_distance=22.0):
    """Which paint pool (found by the camera) belongs to which layer. Matching is by hue, because paint in a dish
    never looks exactly like the artwork colour. Black has no colour to find: it uses the configured dip position."""
    from vision import hue_distance
    default = (cfg["dip_x"], cfg["dip_y"], None, "configured dip position (DIP_X / DIP_Y)")
    # black: the darkest round pool the camera found (vision.find_pools marks those "dark"); only without one the configured position
    darks = sorted([p for p in pools if p.get("dark")], key=lambda p: -p["area_mm2"] * p.get("roundness", 1.0))
    black_dish = ((darks[0]["center_mm"][0], darks[0]["center_mm"][1], darks[0]["radius_mm"], "dark paint pool found by the camera")
                  if darks else None)
    pools = [p for p in pools if not p.get("dark")]                     # a colour never dips into the black
    chroma = [l for l in layers if l["hue"] is not None]
    cands = []
    for i, l in enumerate(chroma):
        for j, p in enumerate(pools):
            d = hue_distance(l["hue"], p["hue"])
            # hue alone cannot tell a strong blue from a pale lilac next to it (artwork blue 120, blue paint 111, lilac paint 128):
            # how strong the colour is counts too (0.1 per step of saturation difference)
            cost = d + 0.1 * abs(float(l.get("sat", p.get("sat", 0.0))) - float(p.get("sat", 0.0)))
            cands.append(((int(cost // 8), -p["area_mm2"], cost), d, i, j))   # colour first (steps of 8), then the BIGGER pool: a dish beats a speck
    cands.sort()
    out, used = {}, set()
    for _, d, i, j in cands:
        name = chroma[i]["name"]
        if d > max_hue_distance or name in out or j in used:
            continue
        p = pools[j]
        out[name] = (p["center_mm"][0], p["center_mm"][1], p["radius_mm"], f"paint pool found by the camera (hue distance {d:.0f})")
        used.add(j)
    missing = []
    for l in layers:
        if l["name"] in out:
            continue
        if l["hue"] is None:
            out[l["name"]] = black_dish or default
            if not black_dish:
                print(f"\n   *** WARNING: no dish of dark paint found for '{l['name']}': it dips at DIP_X / DIP_Y (X {cfg['dip_x']:.0f} Y {cfg['dip_y']:.0f}) "
                      f"from settings.py. Check in the overlay that this IS your black dish, or answer N. ***\n")
        elif len(layers) == 1:
            out[l["name"]] = default
        else:
            missing.append(l["name"])
    if missing:
        found = ", ".join(f"hue {p['hue']:.0f} at X {p['center_mm'][0]:.0f} Y {p['center_mm'][1]:.0f}" for p in pools) or "none"
        raise SystemExit(f"no paint dish found for: {', '.join(missing)}.\n"
                         f"  paint pools the camera sees outside the paper: {found}\n"
                         f"  (is the paint in a dish in the camera view, and the toolhead out of the picture? "
                         f"white, black and clear paints cannot be found by colour)")
    return out


def simplify(pts, eps_px, closed=False):
    """Ramer-Douglas-Peucker: merges straight stretches."""
    a = np.asarray(pts, np.float32).reshape(-1, 1, 2)
    if len(a) < 3:
        return [tuple(p) for p in np.asarray(pts, float)]
    out = cv2.approxPolyDP(a, eps_px, closed).reshape(-1, 2)
    return [tuple(p) for p in out.astype(float)]


BUILD = "2026-10-18 h: clean version (one settings.py, brush_profile.py, no press)"


def apply_probe(cfg, res, a):
    """Uses what the probes measured instead of the profile's numbers: the contact height (machine Z), the width-versus-depth curve of
    THIS paint on THIS paper, and the tilt of the paper (every stroke's Z follows it). The dish heights stay where they are."""
    contact = float(res["contact_center"]) + float(getattr(S, "PROBE_Z_EXTRA_MM", 0.0))   # PROBE_Z_EXTRA_MM: your own correction on top of the measured contact (+ = higher, - = deeper)
    old = cfg["z_paint"]
    curve = (list(res["curve"][0]), list(res["curve"][1]))
    cfg["z_offset"] = cfg["z_paint"] = contact
    cfg["z_low"] = max(cfg["z_low"], contact + float(S.TRAVEL_MM))             # never closer to the paper than before
    cfg["z_high"] = max(cfg["z_high"], contact + float(S.HIGH_MM))
    cfg["curve"] = curve
    cfg["aniso"] = res.get("aniso") if getattr(S, "PROBE_USE_ANISO", False) else None   # see PROBE_USE_ANISO in settings.py
    pf = cfg.get("paper_frame") or {}
    if pf.get("u") and pf.get("v"):
        cfg["aniso_axes"] = (tuple(pf["u"][:2]), tuple(pf["v"][:2]))
    room = contact - 0.2                                                       # how far below the contact the machine can go (Z 0.2)
    cfg["line_depth"] = round(max(min(curve[0][-1], room), 0.0), 2)
    mine = getattr(a, "from_profile", set())                                   # numbers the PROFILE put there are replaced by the measured ones; yours stay
    if a.line_depth is not None and "line_depth" not in mine:                  # your own limit (LINE_DEPTH in settings.py / --line_depth) still holds
        cfg["line_depth"] = round(min(cfg["line_depth"], float(a.line_depth)), 2)
    if a.brush is None or "brush" in mine:
        cfg["brush_w"] = float(res["brush"])
    if a.max_stroke_width is None:
        cfg["thin_max"] = max(round(float(np.interp(cfg["line_depth"], curve[0], curve[1])), 2), cfg["brush_w"] * 1.1)
    tilt = None
    if res.get("plane_spread_mm", 0.0) >= 0.05:
        tilt = (float(res["tilt_x"]), float(res["tilt_y"]), float(res["center_mm"][0]), float(res["center_mm"][1]))
    cfg["tilt"] = tilt
    cfg["tool_offset"] = None
    if "tool_offset_mm" in res and getattr(S, "PROBE_APPLY_OFFSET", True):
        import probe as PB
        ox, oy = res["tool_offset_mm"]
        mag, scat = float(np.hypot(ox, oy)), res.get("tool_offset_scatter_mm")
        if mag >= 0.4 and PB.offset_usable((ox, oy), scat):                    # a real, consistent shift of the tip (not noise, not a wild reading)
            cfg["tool_offset"] = (float(ox), float(oy))
        elif mag >= 0.4:
            cfg["offset_rejected_mm"] = mag                                    # the paint will land this far off: the gap repair must not trust the photo
            print(f"\n   *** WARNING: brush tip offset ({ox:+.2f}, {oy:+.2f}) mm NOT used: larger than {PB.PROBE_OFFSET_MAX_MM:g} mm, or the probe "
                  f"lines disagree by more than {PB.PROBE_OFFSET_SCATTER_MM:g} mm (they disagree by {scat} mm).\n"
                  f"   *** The painting will land about {mag:.1f} mm away from the overlay. Answer N, check the brush, or raise "
                  f"PROBE_OFFSET_MAX_MM in settings.py. ***\n")
    vprint(f"\nprobe result used: contact at machine Z {contact:.2f} (the profile / settings said {old:.2f}: {contact - old:+.2f} mm), "
          f"thinnest line {cfg['brush_w']:.1f} mm, widest {curve[1][-1]:.2f} mm at {curve[0][-1]:.1f} mm deeper (using {cfg['line_depth']:.1f})")
    vprint("   paper tilt: " + (f"{res['plane_spread_mm']:.2f} mm across the probes: every stroke's Z follows the plane" if tilt else "negligible: one height for the whole paper"))
    if cfg.get("tool_offset"):
        print(f"   brush tip: the paint lands {cfg['tool_offset'][0]:+.2f} mm in X / {cfg['tool_offset'][1]:+.2f} mm in Y from where it is told; every X / Y in this job is shifted "
              f"by the opposite, so the strokes land where the overlay shows them")
    if room < curve[0][-1] - 1e-6:
        print(f"   WARNING: the deepest line is limited by the machine's Z 0.2: only {room:.1f} mm below the contact are possible, the widest "
              f"lines the brush could make ({curve[1][-1]:.2f} mm) are out of reach. A brush mounted about {curve[0][-1] - room:.1f} mm longer would give them.")


def shift_gcode_xy(code, dx, dy):
    """Adds (dx, dy) to the X and Y of every G0 / G1 move (the brush tip is not where the calibration thinks it is)."""
    def fix(m):
        line = m.group(0)
        line = re.sub(r"(?<=\bX)(-?\d+\.\d+)", lambda k: f"{float(k.group(1)) + dx:.3f}", line)
        return re.sub(r"(?<=\bY)(-?\d+\.\d+)", lambda k: f"{float(k.group(1)) + dy:.3f}", line)
    return re.sub(r"^G[01] [^\n;]*", fix, code, flags=re.M)


def first_layer_color(img_path, cfg):
    """The colour the probes should be painted with: the darkest colour of the artwork that shows well on white (not yellow / white)."""
    try:
        if cfg.get("color", "auto") != "auto":
            return cfg["color"]
        rgb = load_rgb(img_path, cfg["target_width"], rotate=cfg.get("rotate", 0))
        layers = [l for l in find_layers(rgb) if l["mask"].any()]
        usable = [l for l in layers if l["name"] not in ("yellow", "white")]
        if usable:
            # the DARKEST colour (luminance on the white paper). Not the brightness channel: saturated red and blue both
            # have brightness 255, so that choice was a tie. The job still paints every colour; only the probe uses this one.
            lum = 0.299 * rgb[..., 0].astype(float) + 0.587 * rgb[..., 1].astype(float) + 0.114 * rgb[..., 2].astype(float)
            vals = {l["name"]: float(lum[l["mask"]].mean()) for l in usable}
            pick = min(vals, key=vals.get)
            vprint("probe colour: " + pick + "   (darkness on white, lower = darker: "
                  + ", ".join(f"{k} {v:.0f}" for k, v in sorted(vals.items(), key=lambda kv: kv[1])) + ")")
            return pick
    except Exception:
        pass
    return None


def load_settings():
    """The values in settings.py that are not None (None = 'not set here': the profile / the default decides)."""
    return {k: v for k, v in vars(S).items() if k.isupper() and v is not None}


class Stroke(list):
    """A polyline that remembers what it is: 'line' = skeleton of a part thinner than the brush, 'fill' = infill, rings,
    outline or repair. The G-code uses it to paint lines deeper than the fill (--line_depth)."""
    def __init__(self, pts=(), kind="fill"):
        super().__init__(pts)
        self.kind = kind
        self.widths = None            # line width in mm at every point (only for 'line' strokes, used for the depth)
        self.tier = 9                 # painting order within a layer: voronoi blobs 0, 1, 2 (biggest first), everything else after
        self.fixed_dir = False        # True = painted from its first point to its last, never turned round (calligraphic blobs)
        self.group = None             # the blob / leaf this stroke belongs to (for the strokes picture)



def cut_spur(s, res):
    """A stroke that turns round and comes back over its own earlier points paints that line twice. Keep it only up to the turn
    (the farthest point reached), and drop the way back. Widths are cut along with the points."""
    P = np.asarray(list(s), float)
    n = len(P)
    if n < 6:
        return s
    tol = 0.3 * res
    for i in range(4, n):
        d = np.hypot(P[:i - 3, 0] - P[i, 0], P[:i - 3, 1] - P[i, 1])
        if d.size and d.min() < tol:
            j = int(np.argmin(d))
            if float((P[i] - P[i - 1]) @ (P[j + 1] - P[j])) > 0:   # over its earlier points in the SAME direction: a closed ring
                k = i + 1                                 # (an O, a ring) closing on itself, not a turn back: keep it all the way
                                                          # round, without the second time over (cut at the farthest point = half)
            else:
                far = np.hypot(P[:i, 0] - P[0, 0], P[:i, 1] - P[0, 1])
                k = int(np.argmax(far)) + 1
            new = Stroke([tuple(p) for p in P[:k]], getattr(s, "kind", "fill"))
            if getattr(s, "widths", None) is not None:
                new.widths = list(s.widths)[:k]
            return new
    return s


def path_length(p):
    return float(sum(math.hypot(p[i + 1][0] - p[i][0], p[i + 1][1] - p[i][1]) for i in range(len(p) - 1)))


# ----------------------------------------------------------------------------------------------- thin parts
VERBOSE = bool(getattr(S, "VERBOSE", False))      # False = only the key lines (photos, drying, repair passes, GIF, warnings)


def vprint(*args, **kwargs):
    if VERBOSE:
        print(*args, **kwargs)


def _to_tips(xy, wpx, dist, free, stop_px, look_px=6.0, step=0.5):
    """The skeleton stops about half a mark short of a shape's pointed end: the stroke would start and stop already thick.
    Each FREE end (not at a fork) is carried on in its own direction to the tip, while the shape there is still at least stop_px
    wide; the width keeps coming from the shape, so it falls towards the tip and the brush lifts out of it (thin - thick - thin)."""
    H, W = dist.shape

    def width_at(p):
        x, y = int(round(p[0])), int(round(p[1]))
        return 2.0 * float(dist[y, x]) if 0 <= x < W and 0 <= y < H else 0.0

    for at_end in (False, True):
        if not free[1 if at_end else 0] or len(xy) < 3:
            continue
        q = xy[::-1] if at_end else xy                                # q[0] = this end
        s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(q, axis=0).T))]
        k = min(int(np.searchsorted(s, look_px)), len(q) - 1)
        d = q[0] - q[k]
        n = float(np.hypot(*d))
        if n < 1e-9:
            continue
        d = d / n
        pts, ws, t = [], [], step
        while True:
            p = q[0] + d * t
            w = width_at(p)
            if w < stop_px:
                break
            pts.append(p); ws.append(w)
            t += step
        if not pts:
            continue
        ext, wext = np.array(pts), np.array(ws)
        if at_end:
            xy, wpx = np.vstack([xy, ext]), np.r_[wpx, wext]
        else:
            xy, wpx = np.vstack([ext[::-1], xy]), np.r_[wext[::-1], wpx]
    return xy, wpx


def trace_skeleton(sk, min_len_px, dist=None, tips=False, tip_px=2.0):
    """Skeleton pixels -> list of polylines (branches between junctions).
    With dist (distance transform of the artwork) every polyline also gets .widths_px = the line width at each point.
    tips=True: free ends are carried on to the shape's pointed ends (_to_tips), down to where it is tip_px wide."""
    sk = sk.astype(np.uint8)
    nb = cv2.filter2D(sk, -1, np.ones((3, 3), np.float32), borderType=cv2.BORDER_CONSTANT) - sk
    junction = (sk > 0) & (nb >= 3)
    branch = (sk > 0) & ~junction
    n, lab = cv2.connectedComponents(branch.astype(np.uint8), connectivity=8)
    jy, jx = np.nonzero(junction)
    jset = set(zip(jy.tolist(), jx.tolist()))
    by, bx = np.nonzero(branch)                                   # group the pixels of all branches once
    lb = lab[by, bx]
    order_ = np.argsort(lb, kind="stable")
    by, bx, lb = by[order_], bx[order_], lb[order_]
    groups = np.split(np.arange(len(lb)), np.flatnonzero(np.diff(lb)) + 1) if len(lb) else []
    paths = []
    for g in groups:
        ys, xs = by[g], bx[g]
        if len(ys) < min_len_px:
            continue
        pix = set(zip(ys.tolist(), xs.tolist()))
        nbrs = {p: [(p[0] + dy, p[1] + dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                    if (dy or dx) and (p[0] + dy, p[1] + dx) in pix] for p in pix}
        ends = [p for p, v in nbrs.items() if len(v) <= 1]
        cur = ends[0] if ends else next(iter(pix))
        seen, order = {cur}, [cur]
        while True:
            nxt = [q for q in nbrs[cur] if q not in seen]
            if not nxt:
                break
            cur = min(nxt, key=lambda q: abs(q[0] - cur[0]) + abs(q[1] - cur[1]))   # prefer 4-neighbours
            seen.add(cur); order.append(cur)
        def touch(p):                                                         # a junction pixel next to this end, if any
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if (p[0] + dy, p[1] + dx) in jset:
                        return (p[0] + dy, p[1] + dx)
            return None
        a, b = touch(order[0]), touch(order[-1])
        if a:
            order.insert(0, a)
        if b:
            order.append(b)
        xy = np.array([(x, y) for y, x in order], float)
        wpx = np.array([2.0 * dist[y, x] for y, x in order], float) if dist is not None else None
        sig = float(getattr(S, "LINE_SMOOTH_PX", 2.0))
        if len(xy) >= 7:
            from scipy.ndimage import gaussian_filter1d
            if sig > 0:                                                       # smooth the staircase of the pixel grid
                xy = gaussian_filter1d(xy, sig, axis=0, mode="nearest")
            if wpx is not None:
                wpx = gaussian_filter1d(wpx, 3.0, mode="nearest")             # the width along a line changes smoothly
        if dist is not None and tips:
            xy, wpx = _to_tips(xy, wpx, dist, free=(a is None, b is None), stop_px=tip_px)
        st = Stroke([tuple(p) for p in xy], "thin")
        st.widths_px = None if wpx is None else wpx.tolist()
        paths.append(st)
    return join_branches(paths, reach_px=RES, look_px=max(4.0, 2.0 * min_len_px))


def join_branches(paths, reach_px, look_px, max_turn_deg=40.0):
    """The skeleton is cut into branches at every fork, even a tiny side spur: a line that just goes on through the fork would be
    painted as two or three strokes. Branches whose ends meet (within reach_px) and go on in the same direction (turning less than
    max_turn_deg) are joined into one stroke, straightest pair first; widths are joined along."""
    from scipy.spatial import cKDTree
    if len(paths) < 2:
        return paths
    P = [np.asarray(p, float) for p in paths]
    W = [None if getattr(p, "widths_px", None) is None else np.asarray(p.widths_px, float) for p in paths]

    def out_dir(a, at_end):                                       # the direction the line leaves this end, over look_px
        q = a[::-1] if at_end else a
        s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(q, axis=0).T))]
        k = int(np.searchsorted(s, min(look_px, s[-1])))
        d = q[0] - q[min(k, len(q) - 1)]
        n = float(np.hypot(*d))
        return d / n if n > 1e-9 else None

    ends = [(i, e) for i in range(len(P)) for e in (0, 1) if len(P[i]) >= 2]
    xy = np.array([P[i][-1] if e else P[i][0] for i, e in ends])
    dirs = [out_dir(P[i], e) for i, e in ends]
    cands = []
    lim = -math.cos(math.radians(max_turn_deg))
    for a, b in cKDTree(xy).query_pairs(reach_px):
        if ends[a][0] == ends[b][0] or dirs[a] is None or dirs[b] is None:
            continue
        dot = float(dirs[a] @ dirs[b])
        gap = xy[b] - xy[a]
        side = max(abs(float(dirs[a][0] * gap[1] - dirs[a][1] * gap[0])), abs(float(dirs[b][0] * gap[1] - dirs[b][1] * gap[0])))
        if dot < lim and side <= 2.0:                             # the two ends face each other: one line goes on in the other
            cands.append((dot, a, b))
    parent = list(range(len(P)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    link = {}
    for _, a, b in sorted(cands):
        if a in link or b in link or root(ends[a][0]) == root(ends[b][0]):
            continue                                              # each end joins once, and no closed rings
        link[a], link[b] = b, a
        parent[root(ends[a][0])] = root(ends[b][0])
    if not link:
        return paths
    idx = {(i, e): k for k, (i, e) in enumerate(ends)}
    used, out = set(), []
    for i in range(len(P)):
        if i in used:
            continue
        cur, e_in = i, 0                                          # walk back to the first branch of this chain
        while True:
            k = idx.get((cur, e_in))
            if k is None or k not in link:
                break
            j, e = ends[link[k]]
            if j == i:
                break
            cur, e_in = j, 1 - e
        chain, cur, e_start = [], cur, e_in
        while True:                                               # then forwards, end to end
            used.add(cur)
            chain.append((cur, e_start == 1))                     # reversed when it is entered at its far end
            k = idx.get((cur, 1 - e_start))
            if k is None or k not in link:
                break
            cur, e_start = ends[link[k]]
        pts = np.vstack([P[c][::-1] if rev else P[c] for c, rev in chain])
        st = Stroke([tuple(p) for p in pts], getattr(paths[i], "kind", "thin"))
        if all(W[c] is not None for c, _ in chain):
            st.widths_px = np.concatenate([W[c][::-1] if rev else W[c] for c, rev in chain]).tolist()
        else:
            st.widths_px = None
        out.append(st)
    return out


def attach_widths(q, orig, w_px, step_px):
    """q = simplified polyline (vertices are points of `orig`). Returns a 'line' Stroke whose segments are cut to at most
    step_px, each point with its width in mm - so the depth can follow the width even along a long straight piece."""
    O = np.asarray(orig, float)
    W = np.asarray(w_px, float)
    s_o = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(O, axis=0), axis=1))]
    sq, j = [], 0
    for v in q:                                                               # position along the original line of every vertex
        d = np.linalg.norm(O[j:] - np.asarray(v), axis=1)
        j += int(np.argmin(d))
        sq.append(s_o[j])
    pts, ss = [tuple(q[0])], [sq[0]]
    for k in range(len(q) - 1):
        a, b = np.asarray(q[k], float), np.asarray(q[k + 1], float)
        n = max(1, int(math.ceil(float(np.linalg.norm(b - a)) / step_px)))
        for i in range(1, n + 1):
            t = i / n
            pts.append(tuple(a + (b - a) * t))
            ss.append(sq[k] + (sq[k + 1] - sq[k]) * t)
    st = Stroke(pts, "line")
    st.widths = [float(np.interp(s, s_o, W)) / RES for s in ss]
    return st


# ----------------------------------------------------------------------------------------------- infill
def lines_in(comp, step_px, angle_deg, min_seg_px=2.0):
    """Parallel lines through a boolean mask. The first and last line sit exactly on the edges of the mask and the
    spacing in between is never larger than step_px. Returns 2-point paths in pixel coordinates."""
    ys, xs = np.nonzero(comp)
    cx, cy = xs.mean(), ys.mean()
    diag = math.hypot(xs.max() - xs.min() + 4, ys.max() - ys.min() + 4) / 2.0 + 2
    th = math.radians(angle_deg)
    c, s = math.cos(th), math.sin(th)
    d_pix = -(xs - cx) * s + (ys - cy) * c                       # position of every pixel across the line direction
    dmin, dmax = float(d_pix.min()), float(d_pix.max())
    n = max(1, int(math.ceil((dmax - dmin) / step_px))) if dmax - dmin > 0.5 else 0
    offsets = [0.5 * (dmin + dmax)] if n == 0 else [dmin + i * (dmax - dmin) / n for i in range(n + 1)]
    t = np.arange(-diag, diag, 0.5)
    h, w = comp.shape
    out = []
    for d in offsets:
        x = cx + t * c - d * s
        y = cy + t * s + d * c
        ix, iy = np.round(x).astype(int), np.round(y).astype(int)
        ok = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
        inside = np.zeros(len(t), bool)
        inside[ok] = comp[iy[ok], ix[ok]]
        if not inside.any():
            continue
        edges = np.diff(np.r_[0, inside.astype(int), 0])
        for a_, b_ in zip(np.nonzero(edges == 1)[0], np.nonzero(edges == -1)[0] - 1):
            if (b_ - a_) * 0.5 >= min_seg_px:
                out.append([(float(x[a_]), float(y[a_])), (float(x[b_]), float(y[b_]))])
    return out


def rings_in(comp, step_px):
    """Concentric contours, outside in. Every ring follows the real boundary."""
    out = []
    cur = comp.astype(np.uint8)
    while cur.any():
        cnts, _ = cv2.findContours(cur, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        for c in cnts:
            pts = c.reshape(-1, 2).astype(float)
            if len(pts) >= 3 and cv2.arcLength(c, True) >= step_px:
                out.append([tuple(p) for p in pts] + [tuple(pts[0])])
        nxt = cv2.erode(cur, disk(step_px))
        if not nxt.any():                                                    # leftover centre narrower than one step
            half = cv2.erode(cur, disk(step_px / 2.0))
            if half.any():
                cnts, _ = cv2.findContours(half, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
                for c in cnts:
                    pts = c.reshape(-1, 2).astype(float)
                    out.append([tuple(p) for p in pts] + [tuple(pts[0])])
            break
        cur = nxt
    return out


def choose_infill(comp):
    """'rings' for compact round shapes, 'lines' otherwise. Returns (mode, long-axis angle in degrees)."""
    m = cv2.moments(comp.astype(np.uint8), binaryImage=True)
    angle = 0.5 * math.degrees(math.atan2(2 * m["mu11"], m["mu20"] - m["mu02"])) if m["m00"] > 0 else 0.0
    cnts, _ = cv2.findContours(comp.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return "lines", angle
    c = max(cnts, key=cv2.contourArea)
    area, per = cv2.contourArea(c), cv2.arcLength(c, True)
    (_, _), (rw, rh), _ = cv2.minAreaRect(c)
    circ = 4 * math.pi * area / max(per * per, 1.0)
    rect_fill = area / max(rw * rh, 1.0)
    return ("rings" if (circ > 0.72 and rect_fill < 0.88) else "lines"), angle


def painterly_strokes(inner, res, cfg, angle_deg, rng):
    """Version 3: evenly spaced streamlines (after Jobard and Lefer). The strokes are thin centre lines, and a new one is only
    started about one brush width away from every line already laid (the seed). A stroke is traced in both directions along the
    shape's direction, and stops when it comes within a fraction of a brush of another stroke, or at the edge of the shape.
    So strokes are long, parallel where the form is parallel, and the spacing is one brush width everywhere.
    Returns polylines in pixel coordinates."""
    P = min(1.0, max(0.0, float(cfg.get("painterliness", 0.5))))
    lmin, lmax = cfg.get("painterly_len_mm", (11.0, 23.0))
    lmin_px, lmax_px = lmin * res, lmax * res
    jitter = math.radians(2.0 + 10.0 * P)
    brush = max(2.0, cfg["brush_w"] * res)                  # spacing between centre lines = one brush width
    d_sep = brush
    d_test = 0.7 * brush                                    # a stroke stops this close to another centre line
    step = max(1.0, 0.25 * res)
    H, W = inner.shape
    total = int(inner.sum())
    if total <= 0:
        return []
    occ = np.zeros((H, W), np.uint8)                        # centre lines laid so far (1 px wide)
    occ[~inner] = 1                                         # the outside of the shape counts as an obstacle
    paths, length_px = [], 0.0
    max_strokes = int(cfg.get("painterly_max_strokes", 4000))

    def free_distance():
        return cv2.distanceTransform((occ == 0).astype(np.uint8), cv2.DIST_L2, 3)

    def trace(x0, y0, a0, dist):
        line = []
        for sign in (-1.0, 1.0):
            x, y, done = float(x0), float(y0), 0.0
            ang = a0 if sign > 0 else a0 + math.pi
            side = []
            while done < lmax_px / 2.0:
                x += math.cos(ang) * step
                y += math.sin(ang) * step
                done += step
                xi, yi = int(round(x)), int(round(y))
                if not (0 <= xi < W and 0 <= yi < H) or occ[yi, xi]:
                    break
                if dist[yi, xi] < d_test:
                    break
                side.append((x, y))
            line.append(side)
        return list(reversed(line[0])) + [(float(x0), float(y0))] + line[1]

    dist = free_distance()
    tries = 0
    while len(paths) < max_strokes and tries < 20 * max_strokes:
        tries += 1
        free = inner & (dist >= 0.5 * d_sep)
        if not free.any():
            break
        ring = inner & (dist >= 0.9 * d_sep) & (dist <= 1.1 * d_sep)
        pool = ring if ring.any() else free                 # seeds exactly one brush from the lines already laid
        ys, xs = np.nonzero(pool)
        k = rng.randrange(len(xs))
        x0, y0 = int(xs[k]), int(ys[k])
        a0 = math.radians(angle_deg) + rng.gauss(0.0, jitter)
        path = trace(x0, y0, a0, dist)
        if len(path) * step < lmin_px * 0.5:                # too short to be a stroke: mark the seed and go on
            occ[y0, x0] = 1
            dist = free_distance()
            continue
        paths.append(path)
        length_px += len(path) * step
        arr = np.round(np.array(path)).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(occ, [arr], False, 1, 1)
        dist = free_distance()
    if paths:
        vprint(f"   streamline fill v3: {len(paths)} strokes, total path {length_px / res / 1000.0:.2f} m")
    return paths


def band_strokes(comp, res, angle_deg, band_px, min_len_px):
    """Parallel strokes along the shape's own direction, one band apart (band_px). Each stroke is traced through the shape
    and its width at every point is the shape's width there, capped at band_px, so the depth goes up and down with the
    shape. Returns (points, widths_px) pairs in pixel coordinates."""
    comp8 = comp.astype(np.uint8)
    half = cv2.distanceTransform(comp8, cv2.DIST_L2, 5)
    ys, xs = np.nonzero(comp8)
    if xs.size == 0:
        return []
    cx, cy = float(xs.mean()), float(ys.mean())
    a = math.radians(angle_deg)
    d = np.array([math.cos(a), math.sin(a)])
    nrm = np.array([-d[1], d[0]])
    H, W = comp8.shape
    R = float(math.hypot(H, W))
    step = 0.5 * res
    t = np.arange(-R, R, step)
    out = []
    for o in np.arange(-R, R + band_px, band_px):
        P = np.array([cx, cy]) + o * nrm + t[:, None] * d[None, :]
        xi = np.round(P[:, 0]).astype(int)
        yi = np.round(P[:, 1]).astype(int)
        ok = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
        valid = np.zeros(len(t), bool)
        valid[ok] = comp8[yi[ok], xi[ok]] > 0
        edges = np.flatnonzero(np.diff(np.r_[0, valid.astype(int), 0]))
        for i0, i1 in zip(edges[::2], edges[1::2]):
            if (i1 - i0) * step < min_len_px:
                continue
            seg = P[i0:i1]
            hw = half[yi[i0:i1], xi[i0:i1]]
            w = np.minimum(band_px, 2.0 * hw)
            out.append((seg, w))
    return out


def shape_field(comp, res, sigma_mm=3.0):
    """Direction of the shape at every pixel, from the structure of its edges: the strokes run ALONG the form, so they
    turn where the form turns. Returns (cos, sin) arrays of the unit direction (sign-free, doubled-angle smoothing)."""
    m = comp.astype(np.float32)
    sm = cv2.GaussianBlur(m, (0, 0), sigma_mm * res)
    gx = cv2.Sobel(sm, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(sm, cv2.CV_32F, 0, 1, ksize=3)
    jxx = cv2.GaussianBlur(gx * gx, (0, 0), sigma_mm * res)
    jyy = cv2.GaussianBlur(gy * gy, (0, 0), sigma_mm * res)
    jxy = cv2.GaussianBlur(gx * gy, (0, 0), sigma_mm * res)
    # edge normal angle (doubled); the stroke runs perpendicular to the dominant normal
    th2 = np.arctan2(2 * jxy, jxx - jyy)
    theta = 0.5 * th2 + math.pi / 2.0
    return np.cos(theta).astype(np.float32), np.sin(theta).astype(np.float32)


def field_strokes(comp, res, cfg, angle_deg, rng):
    """Articulated strokes: each one is traced along the shape's own direction field (it bends with the form), spaced about
    one brush width from the strokes already laid, tapered at both ends, and as wide as the shape is there (depth follows the
    width). Returns (points, widths_mm) pairs in crop coordinates."""
    H, W = comp.shape
    brush = max(2.0, cfg["brush_w"] * res)
    d_sep, d_test = brush, 0.45 * brush
    step = max(1.0, 0.5 * res)
    lmin, lmax = cfg.get("painterly_len_mm", (11.0, 23.0))
    fx, fy = shape_field(comp, res)
    inside_u8 = comp.astype(np.uint8)
    half = cv2.distanceTransform(inside_u8, cv2.DIST_L2, 5)     # half width of the shape at every pixel
    occ = np.zeros((H, W), np.uint8)
    occ[~comp] = 1
    out = []
    dist = cv2.distanceTransform((occ == 0).astype(np.uint8), cv2.DIST_L2, 3)

    def unit_at(x, y, ref=None):
        xi, yi = int(round(x)), int(round(y))
        vx, vy = float(fx[yi, xi]), float(fy[yi, xi])
        if ref is not None and vx * ref[0] + vy * ref[1] < 0:
            vx, vy = -vx, -vy
        return vx, vy

    def trace(x0, y0, sgn_ref, grow_px):
        pts = []
        for sign in (-1.0, 1.0):
            x, y, done = float(x0), float(y0), 0.0
            v = unit_at(x, y, (sign * sgn_ref[0], sign * sgn_ref[1]))
            seg = []
            while done < grow_px:
                # midpoint (RK2) step along the field, keeping the same orientation
                mx, my = x + 0.5 * step * sign * v[0], y + 0.5 * step * sign * v[1]
                if not (0 <= int(round(mx)) < W and 0 <= int(round(my)) < H):
                    break
                v2 = unit_at(mx, my, v)
                x += step * sign * v2[0]
                y += step * sign * v2[1]
                done += step
                xi, yi = int(round(x)), int(round(y))
                if not (0 <= xi < W and 0 <= yi < H) or occ[yi, xi]:
                    break
                if done > res and dist[yi, xi] < d_test:
                    break
                v = unit_at(x, y, v)
                seg.append((x, y))
            pts.append(seg)
        return list(reversed(pts[0])) + [(float(x0), float(y0))] + pts[1]

    tries = 0
    since = 0
    while tries < 20000:
        tries += 1
        since += 1
        if since >= 8:
            dist = cv2.distanceTransform((occ == 0).astype(np.uint8), cv2.DIST_L2, 3)
            since = 0
        ring = comp & (dist >= 0.9 * d_sep) & (dist <= 1.3 * d_sep)
        pool = ring if ring.any() else (comp & (dist >= 0.5 * d_sep))
        if not pool.any():
            break
        ys, xs = np.nonzero(pool)
        k = rng.randrange(len(xs))
        x0, y0 = float(xs[k]), float(ys[k])
        ref = unit_at(x0, y0)
        grow = rng.uniform(lmin, lmax) * res / 2.0
        path = trace(x0, y0, ref, grow)
        if len(path) < 2 or len(path) * step < 0.5 * lmin * res:
            occ[int(y0), int(x0)] = 1
            dist = cv2.distanceTransform((occ == 0).astype(np.uint8), cv2.DIST_L2, 3)
            since = 0
            continue
        arr = np.array(path)
        # width = local shape width, capped by the spacing; tapered over the first and last 15 % of the length
        hw = half[np.clip(np.round(arr[:, 1]).astype(int), 0, H - 1), np.clip(np.round(arr[:, 0]).astype(int), 0, W - 1)]
        w_px = np.minimum(2.0 * hw, 1.3 * d_sep)
        n = len(arr)
        t = np.linspace(0.0, 1.0, n)
        taper = np.clip(np.minimum(t, 1 - t) / 0.15, 0.25, 1.0)
        w_mm = (w_px * taper) / res
        out.append((arr, w_mm))
        occ_px = np.round(arr).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(occ, [occ_px], False, 1, 1)
        dist = cv2.distanceTransform((occ == 0).astype(np.uint8), cv2.DIST_L2, 3)
        since = 0
    return out


def axis_field(comp, res):
    """Direction field along the LONG AXIS of each shape (principal direction of its pixels), not along its edges.
    This is what stops the strokes from running round the outline in rings."""
    H, W = comp.shape
    fx = np.zeros((H, W), np.float32); fy = np.zeros((H, W), np.float32)
    n, lab = cv2.connectedComponents(comp.astype(np.uint8), connectivity=8)
    ys, xs = np.nonzero(comp)
    ids = lab[ys, xs]
    order = np.argsort(ids, kind="stable")
    ids_s, ys_s, xs_s = ids[order], ys[order], xs[order]
    cuts = np.flatnonzero(np.diff(ids_s)) + 1
    for yy, xx in zip(np.split(ys_s, cuts), np.split(xs_s, cuts)):
        if xx.size < 3:
            continue
        c = np.cov(np.vstack([xx, yy]).astype(float))
        vals, vecs = np.linalg.eigh(c)
        d = vecs[:, int(np.argmax(vals))]
        fx[yy, xx] = d[0]
        fy[yy, xx] = d[1]
    return fx, fy


def cell_strokes(comp, res, cfg, rng):
    """Multi-scale brush strokes. The widest stroke is the widest the probe says the brush makes (thin_max). The first pass
    uses the BIGGEST cells (seed spacing = 2 stroke widths) and lays long strokes along the shape's direction wherever there is
    room; each later pass uses smaller cells and fills only what is still missing. A stroke stops where its side would reach
    paint already laid, at the outline, or at its length budget (cell_len_mm). Width = the widest stroke where the shape allows,
    narrowing towards the outline and tapered at both ends. Returns (points, widths_mm) pairs in crop coordinates."""
    H, W = comp.shape
    brush = max(2.0, cfg["brush_w"] * res)
    cap = max(brush, float(cfg.get("thin_max") or cfg["brush_w"] * 2.0) * res)     # widest stroke (px)
    lmin, lmax = cfg.get("cell_len_mm", (10.0, 25.0))
    nlab_s, lab_s, st_s, _ = cv2.connectedComponentsWithStats(comp.astype(np.uint8), connectivity=8)
    area_mm2 = st_s[:, cv2.CC_STAT_AREA].astype(float) / (res * res)
    sl = tuple(cfg.get("cell_len_small", (4.0, 10.0))); ll = tuple(cfg.get("cell_len_large", (15.0, 40.0)))
    a_s, a_l = float(cfg.get("shape_small_mm2", 50.0)), float(cfg.get("shape_large_mm2", 600.0))
    clear_f = float(cfg.get("blob_clear", 0.4))          # how close a blob may come to the paint beside it (0.4 = slight overlap, 0.5 = just touching)
    blob_min = float(cfg.get("blob_min_mm", 6.0))                    # shortest blob kept, mm (BLOB_MIN_MM in settings)
    lmin_px, lmax_px = lmin * res, lmax * res
    step = max(1.0, 0.5 * res)
    fx, fy = axis_field(comp, res) if cfg.get("field", "axis") == "axis" else shape_field(comp, res)
    half = cv2.distanceTransform(comp.astype(np.uint8), cv2.DIST_L2, 5)
    cov = np.zeros((H, W), np.uint8)
    out = []
    ys_c, xs_c = np.nonzero(comp)
    if xs_c.size == 0:
        return out

    def free_map():
        return cv2.distanceTransform((cov == 0).astype(np.uint8), cv2.DIST_L2, 3)

    def unit(x, y, ref=None):
        xi, yi = int(round(x)), int(round(y))
        vx, vy = float(fx[yi, xi]), float(fy[yi, xi])
        if ref is not None and vx * ref[0] + vy * ref[1] < 0:
            vx, vy = -vx, -vy
        return vx, vy

    def inside(x, y):
        xi, yi = int(round(x)), int(round(y))
        return 0 <= xi < W and 0 <= yi < H and comp[yi, xi]

    def trace(x0, y0, budget_px, free, clear):
        """Both ways along the field. Stops at the outline, where the free room is below `clear` (paint is near), or at the budget."""
        ref0 = unit(x0, y0)
        sides = []
        for sign in (-1.0, 1.0):
            x, y, done = float(x0), float(y0), 0.0
            v = unit(x, y, (sign * ref0[0], sign * ref0[1]))
            seg = []
            while done < 0.5 * budget_px:
                mx, my = x + 0.5 * step * sign * v[0], y + 0.5 * step * sign * v[1]
                if not inside(mx, my):
                    break
                v2 = unit(mx, my, v)
                nx, ny = x + step * sign * v2[0], y + step * sign * v2[1]
                if not inside(nx, ny) or free[int(round(ny)), int(round(nx))] < clear:
                    break
                x, y, done = nx, ny, done + step
                v = unit(x, y, v)
                seg.append((x, y))
            sides.append(seg)
        return list(reversed(sides[0])) + [(float(x0), float(y0))] + sides[1]

    def finish(path, wp):
        arr = np.array(path, float)
        hw = half[np.clip(np.round(arr[:, 1]).astype(int), 0, H - 1), np.clip(np.round(arr[:, 0]).astype(int), 0, W - 1)]
        w_px = np.minimum(2.0 * hw, wp)
        if len(arr) >= 4:                                             # taper both ends of a real stroke; a dot keeps its width
            t = np.linspace(0.0, 1.0, len(arr))
            w_px = w_px * np.clip(np.minimum(t, 1 - t) / 0.3, 0.45, 1.0)
        return arr, w_px / res

    # passes: the widest stroke with the biggest cells first; then narrower strokes in smaller cells fill what is still missing
    # the big blobs go down first; smaller passes only if BLOB_PASSES asks for them (1 = widest only, the repair fills the rest)
    for wp in sorted({cap, 0.75 * cap, 0.5 * cap, brush}, reverse=True)[:max(1, int(cfg.get("blob_passes", 1)))]:
        spacing = 2.0 * wp
        off = (rng.uniform(0, spacing), rng.uniform(0, spacing))
        gx, gy = np.meshgrid(np.arange(off[0] - spacing, W + spacing, spacing), np.arange(off[1] - spacing, H + spacing, spacing))
        seeds = np.c_[gx.ravel(), gy.ravel()]
        seeds = seeds + np.array([rng.uniform(-0.25, 0.25) * spacing for _ in range(2 * len(seeds))]).reshape(-1, 2)
        order = list(range(len(seeds)))
        rng.shuffle(order)
        free = free_map()
        n_since = 0
        for k in order:
            x, y = seeds[k]
            xi, yi = int(round(x)), int(round(y))
            if not (0 <= xi < W and 0 <= yi < H) or not comp[yi, xi]:
                continue
            if free[yi, xi] < clear_f * wp:                              # no room for a stroke of this width here
                continue
            ar = float(area_mm2[lab_s[yi, xi]]) if lab_s[yi, xi] > 0 else a_s
            t = min(1.0, max(0.0, (math.log(max(ar, 1e-6)) - math.log(a_s)) / max(math.log(a_l) - math.log(a_s), 1e-6)))
            lo_mm = sl[0] + t * (ll[0] - sl[0]); hi_mm = sl[1] + t * (ll[1] - sl[1])
            budget = rng.uniform(lo_mm, max(lo_mm, hi_mm)) * res
            path = trace(x, y, budget, free, clear_f * wp)
            if len(path) < 2 or len(path) * step < min(blob_min, lo_mm) * res:
                continue
            arr, w = finish(path, wp)
            out.append((arr, w))
            pts = np.round(arr).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(cov, [pts], False, 1, max(1, int(round(wp))))
            n_since += 1
            if n_since >= 10:
                free = free_map()
                n_since = 0
        free = free_map()
    # last pass: any paper still left inside the shape gets a short brush-width stroke (or a dot when there is no room for one)
    left = comp & (cov == 0) & (half >= 0.5 * brush)
    if cfg.get("blob_leftover", False) and left.any():
        ly, lx = np.nonzero(left)
        free = free_map()
        for k in rng.sample(range(len(lx)), len(lx)):
            x, y = float(lx[k]), float(ly[k])
            if cov[int(y), int(x)]:
                continue
            path = trace(x, y, 2.0 * brush, free, 0.5 * brush)
            if len(path) * step < 0.5 * blob_min * res:              # too short to be a blob: leave it, no dot
                continue
            arr, w = finish(path, brush)
            out.append((arr, w))
            pts = np.round(arr).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(cov, [pts], False, 1, max(1, int(round(brush))))
    return out


# ----------------------------------------------------------------------------------------------- Voronoi blobs
def voronoi_field(area, res, sigma_mm=5.0, axis_w=0.3):
    """Direction along the shape at every pixel: across the slope of the distance to the outline (so it follows the form, also
    round a bend), smoothed over sigma_mm. Where the local direction is unclear the piece's long axis takes over (axis_w)."""
    d = cv2.distanceTransform(area.astype(np.uint8), cv2.DIST_L2, 5)
    gx = cv2.Sobel(d, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(d, cv2.CV_32F, 0, 1, ksize=3)
    tx, ty = -gy, gx                                                  # along the outline, not towards it
    m = area.astype(np.float32)
    s = max(1.0, sigma_mm * res)
    c2 = cv2.GaussianBlur((tx * tx - ty * ty) * m, (0, 0), s)        # doubled angle: a direction and its reverse add up, not cancel
    s2 = cv2.GaussianBlur(2 * tx * ty * m, (0, 0), s)
    e = np.maximum(cv2.GaussianBlur((tx * tx + ty * ty) * m, (0, 0), s), 1e-6)
    c2, s2 = c2 / e, s2 / e                                           # length 0..1 = how clear the local direction is
    ys, xs = np.nonzero(area)
    if xs.size >= 3:
        vals, vecs = np.linalg.eigh(np.cov(np.vstack([xs, ys]).astype(float)))
        ax, ay = vecs[:, 1]
        k = axis_w * (1.0 - vals[0] / max(vals[1], 1e-9))             # only a long piece has a long axis worth following
        c2 = c2 + k * (ax * ax - ay * ay)
        s2 = s2 + k * 2 * ax * ay
    th = 0.5 * np.arctan2(s2, c2)
    return np.cos(th).astype(np.float32), np.sin(th).astype(np.float32)


def voronoi_cells(area, fx, fy, len_px, w_px, rng, relax=3):
    """Splits `area` into cells about len_px long (along the field) and w_px wide (across it): seeds kept at least one cell apart
    in that stretched distance, then each pixel goes to the nearest seed in the same stretched distance (an anisotropic Voronoi),
    then `relax` Lloyd steps move every seed to the middle of its cell. Returns (labels 1..M, 0 = outside, M)."""
    from scipy import ndimage
    from scipy.spatial import cKDTree
    H, W = area.shape
    ys, xs = np.nonzero(area)
    if xs.size == 0:
        return np.zeros(area.shape, np.int32), 0
    lo, hi = len_px

    def frame(pts):
        xi = np.clip(np.round(pts[:, 0]).astype(int), 0, W - 1)
        yi = np.clip(np.round(pts[:, 1]).astype(int), 0, H - 1)
        return np.c_[fx[yi, xi], fy[yi, xi]]

    # seeds: candidates on a grid a third of a cell width apart, in random order; one is kept when no kept seed is within one cell
    g = max(1, int(w_px / 3))
    sel = (xs % g == 0) & (ys % g == 0)
    cand = np.c_[xs[sel], ys[sel]].astype(float)
    if len(cand) == 0:
        cand = np.c_[xs[:1], ys[:1]].astype(float)
    cand = cand[np.array(rng.sample(range(len(cand)), len(cand)))]
    t_c = frame(cand)
    bucket = max(1.0, hi)
    grid, seeds, lens = {}, [], []
    for p, t in zip(cand, t_c):
        L = rng.uniform(lo, hi)
        bx, by = int(p[0] // bucket), int(p[1] // bucket)
        ok = True
        for j in (j for gx in (bx - 1, bx, bx + 1) for gy in (by - 1, by, by + 1) for j in grid.get((gx, gy), ())):
            dx, dy = p[0] - seeds[j][0], p[1] - seeds[j][1]
            a = (dx * t[0] + dy * t[1]) / (0.5 * (L + lens[j]))
            b = (-dx * t[1] + dy * t[0]) / w_px
            if a * a + b * b < 1.0:
                ok = False
                break
        if ok:
            grid.setdefault((bx, by), []).append(len(seeds))
            seeds.append(p)
            lens.append(L)
    seeds, lens = np.array(seeds), np.array(lens)
    P = np.c_[xs, ys].astype(np.float32)

    def label(seeds, lens):
        ts = frame(seeds)
        k = min(len(seeds), 24)
        tree = cKDTree(seeds)
        out = np.empty(len(P), np.int32)
        for i0 in range(0, len(P), 100000):
            p = P[i0:i0 + 100000]
            _, idx = tree.query(p, k=k)
            idx = idx.reshape(len(p), k)
            d = p[:, None, :] - seeds[idx]
            t = ts[idx]
            a = (d[..., 0] * t[..., 0] + d[..., 1] * t[..., 1]) / (0.5 * lens[idx])
            b = (-d[..., 0] * t[..., 1] + d[..., 1] * t[..., 0]) / (0.5 * w_px)
            out[i0:i0 + len(p)] = idx[np.arange(len(p)), np.argmin(a * a + b * b, axis=1)]
        return out

    for _ in range(max(0, int(relax))):
        own = label(seeds, lens)
        cnt = np.bincount(own, minlength=len(seeds))
        cx = np.bincount(own, P[:, 0], len(seeds)) / np.maximum(cnt, 1)
        cy = np.bincount(own, P[:, 1], len(seeds)) / np.maximum(cnt, 1)
        live = cnt > 0
        new = np.c_[cx, cy]
        xi, yi = np.round(new[:, 0]).astype(int).clip(0, W - 1), np.round(new[:, 1]).astype(int).clip(0, H - 1)
        inside_ = area[yi, xi]                                        # a bent cell's middle can lie outside the shape: keep that seed
        seeds = np.where(inside_[:, None], new, seeds)[live]
        lens = lens[live]
    lab = np.zeros(area.shape, np.int32)
    lab[ys, xs] = label(seeds, lens) + 1
    # a cell must be one piece: bits cut off from it (across a notch in the shape) go to the cell they touch
    orphan = np.zeros(area.shape, bool)
    for i, sl in enumerate(ndimage.find_objects(lab), 1):
        if sl is None:
            continue
        cell = (lab[sl] == i).astype(np.uint8)
        n, cc = cv2.connectedComponents(cell, connectivity=4)
        if n > 2:
            keep = 1 + int(np.argmax(np.bincount(cc.ravel())[1:]))
            orphan[sl] |= (cell > 0) & (cc != keep)
    if orphan.any():
        lab[orphan] = 0
        iy, ix = ndimage.distance_transform_edt(lab == 0, return_distances=False, return_indices=True)
        lab[orphan] = lab[iy, ix][orphan]
    return lab, len(seeds)


def calligraphy_profile(n, entry=0.45, peak=0.4, tail=0.25):
    """Share of the full width along a calligraphic blob of n points: it starts at `entry`, swells to full width at `peak` of its
    length and thins out to `tail` at the end (the brush is set down lightly, pressed, then lifted while it still moves)."""
    t = np.linspace(0.0, 1.0, max(n, 1))
    up = entry + (1.0 - entry) * np.sin(0.5 * np.pi * np.clip(t / peak, 0.0, 1.0))
    down = tail + (1.0 - tail) * np.cos(0.5 * np.pi * np.clip((t - peak) / (1.0 - peak), 0.0, 1.0))
    return np.where(t <= peak, up, down)


def voronoi_strokes(area, dmask, res, cfg, rng, debug=None):
    """Blob fill, biggest blobs first. Tier by tier (VORO_TIERS = shares of the widest mark the brush REALLY paints), the part of the
    shape still open where a blob of that size fits is split into Voronoi cells (voronoi_cells) 2.5-5x as long as wide
    (VORO_BLOB_ASPECT); each cell gets ONE stroke along its centre line, all painted the same way round (VORO_HAND_DEG). A blob may
    reach past its cell up to the outline; what it leaves open (corners, notches, gaps where blobs meet) goes to the next, smaller
    tier - so the small blobs fill in, they never line the outline (no rings). Width: calligraphic (calligraphy_profile) within what
    the cell allows. Returns (points px, widths mm, tier) triples."""
    from scipy import ndimage
    H, W = area.shape
    out = []
    if not area.any():
        return out
    brush = max(2.0, cfg["brush_w"] * res)
    cap = max(brush, float(cfg.get("thin_max") or cfg["brush_w"] * 2.0) * res)     # the widest mark asked for (full depth)
    mark = min(1.0, max(0.3, float(cfg.get("voro_mark_share", 0.7))))             # the brush really paints this share of what is asked
    tiers = tuple(cfg.get("voro_tiers", (1.0, 0.6, 0.35)))
    asp = tuple(cfg.get("voro_blob_aspect", (2.5, 5.0)))
    hd = math.radians(float(cfg.get("voro_hand_deg", 45.0)))
    hand = (math.cos(hd), math.sin(hd))                                           # every blob is painted this way round (y = up)
    callig = bool(cfg.get("voro_calligraphy", True))
    ov = float(cfg.get("voro_overlap_mm", 0.3)) * res
    cell_w = float(cfg.get("voro_cell_width", 0.85))
    step = max(1.0, 0.5 * res)
    end_px = 0.25 * brush
    fx, fy = voronoi_field(area, res, float(cfg.get("voro_field_sigma_mm", 5.0)), float(cfg.get("voro_axis_weight", 0.3)))
    covered = np.zeros((H, W), np.uint8)                                          # what the blobs so far REALLY paint

    def at(img, x, y):
        return img[min(max(int(round(y)), 0), H - 1), min(max(int(round(x)), 0), W - 1)]

    for tier, share in enumerate(tiers):
        last = tier == len(tiers) - 1
        big = cap * share                                                         # the widest mark this tier asks for
        w_px = max(brush, big * mark * cell_w)                                    # cell width = what one blob of this tier really covers
        todo = cv2.morphologyEx((area & (covered == 0)).astype(np.uint8), cv2.MORPH_OPEN, disk(0.35 * w_px)) > 0   # slivers: too thin for this tier
        if not last:
            todo &= dmask >= 0.5 * w_px                                           # only where a blob of this size fits inside the outline
        if todo.sum() < 0.5 * w_px * w_px:
            continue
        hi = asp[1] * w_px
        lab, M = voronoi_cells(todo, fx, fy, (asp[0] * w_px, hi), w_px, rng, relax=int(cfg.get("voro_relax", 3)))
        if M == 0:
            continue
        if debug is not None:                                                     # for pictures of the method (blob_method.png)
            debug.setdefault("cells", []).append((tier, lab.copy(), todo.copy()))
        edge = ~area                                                              # the outline, and where two cells of this tier meet
        for dy, dx in ((0, 1), (1, 0)):
            a, b = lab[: H - dy, : W - dx], lab[dy:, dx:]
            diff = (a != b) & (a > 0) & (b > 0)
            edge[: H - dy, : W - dx] |= diff
            edge[dy:, dx:] |= diff
        dcell = cv2.distanceTransform((~edge).astype(np.uint8), cv2.DIST_L2, 5)
        starts = ndimage.maximum_position(np.where(lab > 0, dcell, 0), lab, index=np.arange(1, M + 1))
        # each cell's own long axis (from its second moments): the stroke starts along it and never turns more than 45 degrees off it
        cy_, cx_ = np.nonzero(lab)
        li = lab[cy_, cx_]
        cnt = np.maximum(np.bincount(li, minlength=M + 1), 1).astype(float)
        mx, my = np.bincount(li, cx_, M + 1) / cnt, np.bincount(li, cy_, M + 1) / cnt
        sxx = np.bincount(li, cx_ * cx_.astype(float), M + 1) / cnt - mx * mx
        syy = np.bincount(li, cy_ * cy_.astype(float), M + 1) / cnt - my * my
        sxy = np.bincount(li, cx_ * cy_.astype(float), M + 1) / cnt - mx * my
        ang = 0.5 * np.arctan2(2 * sxy, sxx - syy)
        for i, (y0, x0) in enumerate(starts, 1):
            arr = _cell_centre_line(i, x0, y0, (math.cos(ang[i]), math.sin(ang[i])), lab, dcell, fx, fy, at, step, end_px, brush, hi)
            arr = _extend_ends(arr, 0.6 * w_px, step, lambda x, y: 0 <= x <= W - 1 and 0 <= y <= H - 1 and at(dmask, x, y) >= 0.5 * brush)
            if tier > 0 and float(np.hypot(*np.diff(arr, axis=0).T).sum()) < max(1.5 * res, w_px):
                continue                                                          # a dot, not a blob: left to the repair
            if (arr[-1, 0] - arr[0, 0]) * hand[0] + (arr[-1, 1] - arr[0, 1]) * hand[1] < 0:
                arr = arr[::-1]                                                   # one hand: every blob runs the same way
            dc = np.array([at(dcell, x, y) for x, y in arr], float)
            dm = np.array([at(dmask, x, y) for x, y in arr], float)
            w = np.minimum(big, np.minimum(2.0 * dm, 2.0 * dc + 2.0 * ov) / mark)   # asked for: what it must REALLY cover, / mark
            if callig:
                w = w * calligraphy_profile(len(arr))
            w = np.maximum(w, brush)
            out.append((arr, w / res, tier))
            p = np.round(arr).astype(np.int32)
            for j in range(len(p) - 1):
                cv2.line(covered, tuple(map(int, p[j])), tuple(map(int, p[j + 1])), 1, max(1, int(round(0.5 * mark * (w[j] + w[j + 1])))))
    return out


def _extend_ends(arr, ext, step, ok):
    """The centre line carried on straight past both ends by up to ext px while ok(x, y) holds: the thin entry and tail of a
    calligraphic blob overlap its neighbours instead of leaving a gap where two blob ends meet."""
    if len(arr) < 2 or ext <= 0:
        return arr
    out = [arr]
    for end, prev, front in ((arr[0], arr[1], True), (arr[-1], arr[-2], False)):
        d = end - prev
        n = float(np.hypot(*d))
        if n < 1e-9:
            continue
        d = d / n
        pts, s = [], step
        while s <= ext and ok(end[0] + d[0] * s, end[1] + d[1] * s):
            pts.append(end + d * s)
            s += step
        if pts:
            out = ([np.array(pts[::-1])] + out) if front else (out + [np.array(pts)])
    return np.vstack(out)


def _cell_centre_line(i, x0, y0, ax_, lab, dcell, fx, fy, at, step, end_px, brush, hi):
    """Centre line of cell i from its middle (x0, y0) both ways: along the field, never more than 45 degrees off the cell's own axis
    ax_, kept to the middle of the cell, until it leaves the cell or comes within end_px of its edge."""
    H, W = lab.shape
    sides = []
    for sgn in (-1.0, 1.0):
        x, y, v, seg = float(x0), float(y0), (sgn * ax_[0], sgn * ax_[1]), []
        for _ in range(int(1.5 * hi / step) + 1):
            u = (float(at(fx, x, y)), float(at(fy, x, y)))
            if abs(u[0] * ax_[0] + u[1] * ax_[1]) < 0.7:                  # the field runs across this cell: follow the cell instead
                u = ax_
            if u[0] * v[0] + u[1] * v[1] < 0:
                u = (-u[0], -u[1])
            v = u
            nx, ny = x + step * v[0], y + step * v[1]
            n = (-v[1], v[0])                                           # keep to the middle of the cell: climb its ridge sideways
            dp = float(at(dcell, nx + n[0], ny + n[1])) - float(at(dcell, nx - n[0], ny - n[1]))
            nx, ny = nx + 0.5 * n[0] * max(-1.0, min(1.0, dp)), ny + 0.5 * n[1] * max(-1.0, min(1.0, dp))
            if not (0 <= nx <= W - 1 and 0 <= ny <= H - 1) or at(lab, nx, ny) != i or at(dcell, nx, ny) < end_px:
                break
            x, y = nx, ny
            seg.append((x, y))
        sides.append(seg)
    path = list(reversed(sides[0])) + [(float(x0), float(y0))] + sides[1]
    if len(path) < 2:                                                   # a round little cell: a short dab across its middle
        r = 0.25 * brush
        path = [(x0 - r * ax_[0], y0 - r * ax_[1]), (x0 + r * ax_[0], y0 + r * ax_[1])]
    return np.array(path, float)


def piece_elongation(area):
    """How long and thin a piece is: its length / its width along its own main axis (1 = round, 8 = a ripple or a blade)."""
    ys, xs = np.nonzero(area)
    if xs.size < 6:
        return 1.0
    vals = np.linalg.eigvalsh(np.cov(np.vstack([xs, ys]).astype(float)))
    return float(math.sqrt(max(vals[1], 1e-9) / max(vals[0], 1e-9)))


def _plan_piece(job):
    """Plans one piece of the thick parts (runs in its own process when there are several pieces). Returns its strokes, its
    band strokes and the counts, with every point moved back to the full picture."""
    comp = job["crop"]
    ox, oy = job["off"]
    cfg, res, step_px = job["cfg"], job["res"], job["step_px"]
    rng = random.Random(job["seed"])
    out_strokes, out_bands = [], []
    kinds = {"lines": 0, "rings": 0, "outline": 0}

    def sh(p):
        return [(float(x) + ox, float(y) + oy) for x, y in p]

    mode, angle = choose_infill(comp)
    if cfg["infill_type"] in ("lines", "rings"):
        mode = cfg["infill_type"]
    if cfg["infill_angle"] is not None:
        angle = cfg["infill_angle"]
    style = cfg.get("infill_style")
    if style == "leaf" and job.get("area") is not None and job["area"].any():
        # a style per region (REGION_STYLES): a long thin piece (water ripple, blade of grass) gets e.g. dashes, the rest leaves
        to_style = (cfg.get("region_styles") or {}).get("elongated")
        if to_style and piece_elongation(job["area"]) >= float(cfg.get("elongated_ratio", 5.0)):
            style = to_style
    if mode == "rings":
        for p in rings_in(comp, step_px):
            out_strokes.append(("rings", sh(p))); kinds["rings"] += 1
    elif style == "voronoi" and job.get("area") is not None and job["area"].sum() >= 4.0 * res * res:
        for n_b, (arr, w, tier) in enumerate(voronoi_strokes(job["area"], job["dmask"], res, cfg, rng)):
            out_bands.append((arr + np.array([ox, oy]), w * res, tier, (job["seed"], n_b))); kinds["lines"] += 1
    elif style == "leaf" and job.get("area") is not None and job["area"].sum() >= 4.0 * res * res:
        import leaf_style
        for arr, w, tier, leaf in leaf_style.leaf_strokes(job["area"], job["dmask"], res, cfg, rng, _cell_centre_line, _extend_ends, voronoi_field):
            out_bands.append((arr + np.array([ox, oy]), w * res, tier, (job["seed"], leaf))); kinds["lines"] += 1
    elif cfg.get("infill_style") == "cells" and comp.sum() >= 4.0 * res * res:
        for arr, w in cell_strokes(comp, res, cfg, rng):
            out_bands.append((arr + np.array([ox, oy]), w * res)); kinds["lines"] += 1
    elif cfg.get("infill_style") == "wide" and comp.sum() >= 4.0 * res * res:
        # fixed-width strokes: as wide as the probe says the brush can make, one stroke width apart, along the piece's main axis.
        # The centre lines sit one half-width inside the shape, so the stroke edges reach the outline and nothing is left out.
        w_mm = float(cfg.get("thin_max") or cfg["brush_w"] * 2.0)
        w_px = max(2.0, w_mm * res)
        ys_w, xs_w = np.nonzero(comp)
        ev_w, evec_w = np.linalg.eigh(np.cov(np.vstack([xs_w, ys_w]).astype(float)))
        main_w = evec_w[:, 1]
        wide_angle = float(np.degrees(np.arctan2(main_w[1], main_w[0])))
        core = cv2.distanceTransform(comp.astype(np.uint8), cv2.DIST_L2, 5) >= w_px / 2.0
        if core.any():
            for p in lines_in(core, 0.98 * w_px, wide_angle):
                pts_w = np.asarray(p, float) + np.array([ox, oy])
                out_bands.append((pts_w, np.full(len(pts_w), w_px))); kinds["lines"] += 1
    elif cfg.get("infill_style") == "field" and comp.sum() >= 15.0 * res * res:
        for arr, w in field_strokes(comp, res, cfg, angle, rng):
            out_bands.append((arr + np.array([ox, oy]), w)); kinds["lines"] += 1
    elif cfg.get("infill_style") == "bands" and comp.sum() >= float(cfg.get("bands_min_mm2", 150.0)) * res * res:   # small fragments keep the plain fill
        cap_mm = max(float(cfg.get("thin_max") or 0.0), cfg["brush_w"] * 1.1)
        band_px = max(step_px * 0.5, 0.9 * cap_mm * res)
        ys_b, xs_b = np.nonzero(comp)
        cov_b = np.cov(np.vstack([xs_b, ys_b]).astype(float))
        ev, evec = np.linalg.eigh(cov_b)
        main = evec[:, 1]                                       # the piece's own main axis: bands run along the form
        band_angle = float(np.degrees(np.arctan2(main[1], main[0])))
        for seg, w in band_strokes(comp, res, band_angle, band_px, min_len_px=max(3.0, cfg["brush_w"] * res * 0.8)):
            out_bands.append((np.asarray(seg, float) + np.array([ox, oy]), w)); kinds["lines"] += 1
    else:
        inner = comp
        if cfg["outline"]:
            cnts, _ = cv2.findContours(comp.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
            for c in cnts:
                pts = c.reshape(-1, 2).astype(float)
                if len(pts) >= 3:
                    out_strokes.append(("outline", sh(pts) + sh(pts[:1]))); kinds["outline"] += 1
            inner = cv2.erode(comp.astype(np.uint8), disk(step_px)) > 0
        if inner.any():
            if cfg.get("infill_style") == "painterly":
                for p in painterly_strokes(inner, res, cfg, angle, rng):
                    out_strokes.append(("lines", sh(p))); kinds["lines"] += 1
            else:
                for p in lines_in(inner, step_px, angle):
                    out_strokes.append(("lines", sh(p))); kinds["lines"] += 1
    return {"strokes": out_strokes, "bands": out_bands, "kinds": kinds}


def _plan_pieces(jobs):
    """Plans all pieces: in parallel processes when there are several (PLAN_WORKERS in settings.py, 0 = automatic), otherwise here.
    The result is in the same order either way, and each piece has its own seed, so the output does not depend on the number of cores."""
    import os
    workers = int(getattr(S, "PLAN_WORKERS", 0) or 0) or min(os.cpu_count() or 1, 8)
    if workers > 1 and len(jobs) >= 4:
        try:
            import multiprocessing as mp
            with mp.get_context("spawn").Pool(min(workers, len(jobs))) as pool:
                return pool.map(_plan_piece, jobs, chunksize=max(1, len(jobs) // (4 * workers)))
        except Exception as e:
            vprint(f"   (parallel planning not used: {e}; planning on one core)")
    return [_plan_piece(j) for j in jobs]


# ----------------------------------------------------------------------------------------------- planner
def plan_strokes(mask, cfg, res=RES):
    """mask (bool, True = paint) -> list of strokes (pixel polylines) + counts per kind."""
    rb = cfg["brush_w"] / 2.0 * res
    step_px = max(1.0, cfg["brush_w"] * (1 - cfg["overlap"]) * res)
    m8 = mask.astype(np.uint8)
    strokes, kinds = [], {"thin": 0, "lines": 0, "rings": 0, "outline": 0, "patch": 0}
    bands_out = []
    # the edge of the picture is an outline too: art that runs to the border must not be painted past it (widths, blob passes)
    dist = cv2.distanceTransform(np.pad(m8, 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
    thin_max = cfg.get("thin_max") or cfg["brush_w"] * 1.1
    if cfg.get("infill_style") == "skeleton":
        # INFILL_STYLE "skeleton": EVERY shape is centre-line strokes, whatever its width; the depth follows the width (probe
        # curve, at most LINE_DEPTH). No infill: what a shape wider than the brush still leaves open, the pass below (3.)
        # paints along the centre lines of those open parts
        core = np.zeros(mask.shape, bool)
        sk = skeletonize(mask)
    elif thin_max > cfg["brush_w"] * 1.1 + 1e-6:
        # marks narrower than thin_max: centre-line strokes (the depth follows the width). What is wider keeps the infill.
        # a line that only gets a little wider than the widest mark (LINE_WIDTH_SLACK) stays one line: it is not cut into
        # line - leaf - line pieces where its width wobbles round thin_max (the slightly wider bit is painted at the widest mark)
        rt = thin_max / 2.0 * res * max(1.0, float(cfg.get("line_width_slack", 1.0)))
        thick = cv2.morphologyEx(m8, cv2.MORPH_OPEN, disk(rt)) > 0              # where a disc of thin_max (x the slack) fits
        core = cv2.erode(thick.astype(np.uint8), disk(rb)) > 0
        sk = skeletonize(mask) & (dist < rt + 0.5) & ~thick
    else:
        core = cv2.erode(m8, disk(rb)) > 0                # where the brush centre may go without leaving the shape
        # 1. thin parts: the skeleton where the shape is narrower than the brush
        sk = skeletonize(mask) & (dist < rb + 0.5)
    tip_mm = float(cfg.get("line_tip_mm", 0.0))
    for p in trace_skeleton(sk, min_len_px=max(3, int(cfg["brush_w"] * res * 0.8)), dist=dist,
                            tips=tip_mm > 0, tip_px=tip_mm * res):
        strokes.append(("thin", p)); kinds["thin"] += 1

    # 2. thick parts: per connected piece of the core. Each piece is planned on its own (several processes when there are several pieces)
    n, lab = cv2.connectedComponents(core.astype(np.uint8), connectivity=8)
    seed0 = int(cfg.get("seed") or 1)
    jobs = []
    vor = cfg.get("infill_style") in ("voronoi", "leaf")
    mg = int(math.ceil(rb)) + 3 if vor else 2                # voronoi: the crop also holds the band between the core and the outline
    for k in range(1, n):
        comp = lab == k
        if comp.sum() < 4:
            continue
        ys_, xs_ = np.nonzero(comp)
        y0, y1 = max(0, ys_.min() - mg), min(comp.shape[0], ys_.max() + mg + 1)
        x0, x1 = max(0, xs_.min() - mg), min(comp.shape[1], xs_.max() + mg + 1)
        job = {"crop": comp[y0:y1, x0:x1].copy(), "off": (int(x0), int(y0)), "cfg": cfg, "res": res,
               "step_px": step_px, "seed": seed0 * 100003 + k}
        if vor:                                                # the cells tile the whole shape (core + brush radius), up to the real outline
            job["area"] = (cv2.dilate(job["crop"].astype(np.uint8), disk(rb + 1)) > 0) & mask[y0:y1, x0:x1]
            job["dmask"] = dist[y0:y1, x0:x1].copy()
        jobs.append(job)
    for r in _plan_pieces(jobs):
        strokes.extend(r["strokes"])
        bands_out.extend(r["bands"])
        for key, v in r["kinds"].items():
            kinds[key] += v

    eps = cfg["simplify_mm"] * res
    simple = []
    for band in bands_out:                                   # band strokes keep their own width at every point (no simplify)
        seg, w = band[0], band[1]
        st = Stroke([tuple(map(float, q)) for q in seg], "line")
        st.widths = [float(x) / res for x in w]
        if vor:                                              # a voronoi blob / leaf vein: its size tier (painted biggest first), painted the way it was planned
            st.tier, st.fixed_dir, st.keep_turns, st.group = band[2], True, True, band[3]
        simple.append(st)
    for kind, p in strokes:
        if kind in ("rings", "outline") and len(p) > 4 and p[0] == p[-1]:
            q = simplify(p[:-1], eps, closed=True)               # closed loop: keep the closing segment
            q = q + [q[0]]
        else:
            q = simplify(p, eps, closed=False)
        if len(q) >= 2 and path_length(q) > 0:
            if kind == "thin" and cfg.get("line_depth", 0) > 0 and getattr(p, "widths_px", None) is not None:
                simple.append(attach_widths(q, p, p.widths_px, step_px=max(1.0, cfg.get("point_step", 0.4) * res)))   # depth follows the width
            else:
                simple.append(Stroke(q, "line" if kind == "thin" else "fill"))

    # skeleton style: a shape up to LINE_WIDTH_SLACK x the widest mark stays ONE stroke (its edges come out a bit thin);
    # only what is wider than that gets side strokes in pass 3.
    reach_max = thin_max * max(1.0, float(cfg.get("line_width_slack", 1.0)))
    if cfg.get("infill_style") == "skeleton":
        for st in simple:                                    # the brush makes at most thin_max: wider parts are left for pass 3.
            if getattr(st, "widths", None) is not None:
                st.reach = [min(float(w), reach_max) for w in st.widths]
                st.widths = [min(float(w), thin_max) for w in st.widths]

    # 3. repair pass: whatever the plan still leaves unpainted (thin gaps down the middle of shapes only slightly
    #    wider than the brush, small corners) is painted along the skeleton of the missed area
    # (skeleton style: again, up to 4 rounds, so a shape several marks wide gets side strokes until it is full)
    for _round in range(4 if cfg.get("infill_style") == "skeleton" else 1):
        swapped = [st for st in simple if getattr(st, "reach", None) is not None]
        for st in swapped:                                   # count the slack as covered (see reach_max)
            st.widths, st.reach = st.reach, st.widths
        painted = render(simple, mask.shape, cfg["brush_w"], res)
        for st in swapped:
            st.widths, st.reach = st.reach, st.widths
        missed = (mask & ~painted).astype(np.uint8)
        missed = cv2.morphologyEx(missed, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        if not missed.any():
            break
        var_depth = cfg.get("line_depth", 0) > 0 and thin_max > cfg["brush_w"] * 1.1 + 1e-6
        dmiss = cv2.distanceTransform(missed, cv2.DIST_L2, 5) if var_depth else None
        n_before = len(simple)
        for p in trace_skeleton(skeletonize(missed > 0), min_len_px=max(3, int(float(cfg.get("blob_min_mm", 6.0)) * res * 0.5)), dist=dmiss):
            q = simplify(p, eps, closed=False)
            if len(q) >= 2 and path_length(q) > 0:
                if var_depth and getattr(p, "widths_px", None) is not None:
                    st = attach_widths(q, p, p.widths_px, step_px=max(1.0, cfg.get("point_step", 0.4) * res))   # same rule: depth follows width
                    if cfg.get("infill_style") == "skeleton":
                        st.reach = [min(float(w), reach_max) for w in st.widths]
                        st.widths = [min(float(w), thin_max) for w in st.widths]
                    simple.append(st)
                else:
                    simple.append(Stroke(q, "fill"))
                kinds["patch"] += 1
        if len(simple) == n_before:
            break
    simple = [q if getattr(q, "keep_turns", False) else cut_spur(q, res) for q in simple]   # no stroke goes out and comes back over itself
    return simple, kinds


# ----------------------------------------------------------------------------------------------- ordering
class PathOptimizer:
    """Fast path ordering (spatial sort for very many strokes, nearest neighbour otherwise)."""
    @staticmethod
    def optimize(paths, start_pos):
        n = len(paths)
        if n == 0:
            return []
        if n > 5000:
            return sorted(paths, key=lambda p: (p[0][0] // 20, p[0][1] if (p[0][0] // 20) % 2 == 0 else -p[0][1]))
        starts = np.array([p[0] for p in paths]); ends = np.array([p[-1] for p in paths])
        fixed = np.array([bool(getattr(p, "fixed_dir", False)) for p in paths])     # these can only be entered at their start
        used = np.zeros(n, bool); out = []; cur = np.array(start_pos, float)
        for _ in range(n):
            ds = np.hypot(*(starts - cur).T)
            d = np.where(fixed, ds, np.minimum(ds, np.hypot(*(ends - cur).T)))
            d[used] = np.inf
            i = int(np.argmin(d)); used[i] = True
            out.append(paths[i])
            enter_at_start = fixed[i] or np.hypot(*(starts[i] - cur)) <= np.hypot(*(ends[i] - cur))
            cur = ends[i] if enter_at_start else starts[i]                      # we leave at the other end
        return out


# ----------------------------------------------------------------------------------------------- G-code
def lead_ramp_zs(base, pts, air, ramp):
    """Z for a blob: it comes down from `air` over the first `ramp` mm and goes back up over the last `ramp` mm, so the brush
    does not dwell at the start or the end (no start blob). Between the ramps it holds the base Z of each point."""
    n = len(pts)
    s = [0.0]
    for i in range(1, n):
        s.append(s[-1] + math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]))
    L = s[-1]
    out = []
    for i in range(n):
        f = 1.0
        if ramp > 0 and L > 0:
            f = max(0.0, min(1.0, s[i] / ramp, (L - s[i]) / ramp))
        out.append(air + (base[i] - air) * f)
    return out


def thin_end_flip(path, min_diff_mm=1e-3, top=None):
    """True = paint this line backwards (its thick part lies in its first half), False = as it is, None = no difference.
    Judged on the whole line, not its tips (2026-11-01: lines taper to the same tip at both ends, so the tips decided by
    noise): where the extra width (above the line's thinnest) is centred along it. The thick part always goes last, the
    brush glides in at the thin part towards it, like a probe line.
    top: the widest the brush is pressed for (thin_max); wider widths count as top, as they are painted."""
    w = np.asarray(path.widths, float)
    if top is not None:
        w = np.minimum(w, top)
    P = np.asarray(path, float)
    s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]
    m = (w - w.min()) * np.gradient(s) if len(s) > 1 else np.zeros(1)     # extra width x the length each point stands for
    if s[-1] <= 0 or m.sum() <= 1e-9:
        return None
    centre = float((s * m).sum() / m.sum())
    if abs(centre - s[-1] / 2.0) < min_diff_mm:
        return None
    return bool(centre < s[-1] / 2.0)


def end_dir(path, look_mm=1.0):
    """Unit vector pointing OUT of the line at its first point (taken over its first look_mm). end_dir(path[::-1]) = at its end."""
    P = np.asarray(path, float)
    s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]
    v = P[0] - P[min(int(np.searchsorted(s, look_mm)), len(P) - 1)]
    n = float(np.hypot(*v))
    return (float(v[0] / n), float(v[1] / n)) if n > 1e-9 else (1.0, 0.0)


class Painter:
    """Writes the G-code. Dip logic is the one from mama.py (random interval, spiral, wipe); the interval is exact,
    and one file can hold several colour layers with a rinse in between."""
    def __init__(self, cfg):
        self.cfg = cfg
        self.gcode = ["G90", "G21"]
        self.dist_since_dip = 0.0
        self.current_pos = (cfg["dip_x"], cfg["dip_y"])
        self.current_max_dist = random.uniform(cfg["min_dist"], cfg["max_dist"])
        self.dips = 0
        self.dish = None

    def _set_speed(self, mode="travel"):
        c = self.cfg
        accel = c["accel_travel"] if mode == "travel" else c["accel_paint"]
        feed = c["feed"] if mode == "travel" else c["feed_paint"]
        self.gcode += ["M400", f"M204 P{accel} T{accel}", f"G1 F{feed}"]

    def _dip_depth(self):
        """Machine Z the brush goes down to in a dish. Off (= PAINT_DIP_Z every time) unless DIP_DEEPEN_MM > 0 in settings.py:
        then every dip goes DIP_DEEPEN_MM lower than the one before, but never below DIP_FLOOR_Z (machine Z, which you measure
        from the bottom of the dish, with a margin)."""
        c = self.cfg
        step = float(getattr(S, "DIP_DEEPEN_MM", 0.0) or 0.0)
        floor = getattr(S, "DIP_FLOOR_Z", None)
        if step <= 0 or floor is None:
            return c["dip_z"]
        return max(float(floor), c["dip_z"] - step * (self.dips - 1))

    def _dip_and_travel(self, tx, ty, dish=None):
        """Dip at `dish` = (x, y, radius or None), wipe, travel to (tx, ty). Spiral and jitter shrink to fit a small pool."""
        c = self.cfg
        dx0, dy0, dr = dish if dish is not None else (c["dip_x"], c["dip_y"], None)
        jitter, spiral = c["dip_jitter"], c["dip_spiral_r"]
        if dr is not None:
            spiral = min(spiral, 0.6 * dr)
            jitter = min(jitter, max(0.0, dr - spiral - 1.0))
        self.dips += 1
        self.gcode.append("\n; --- DIP CYCLE ---")
        self.gcode.append(f"G0 Z{c['z_low']:.3f} F3000")
        self._set_speed("travel")
        ax = dx0 + random.uniform(-jitter, jitter)
        ay = dy0 + random.uniform(-jitter, jitter)
        self.gcode.append(f"G0 X{ax:.3f} Y{ay:.3f} Z{c['z_high']:.3f}")
        dz = self._dip_depth()
        self.gcode.append(f"; dip depth Z {dz:.3f} (dip {self.dips})")
        self.gcode.append(f"G1 Z{dz:.3f} F3000")
        steps = int(c["dip_spiral_loops"] * 4)
        for i in range(steps):
            ang = i * (math.pi / 2)
            r = (i / steps) * spiral
            self.gcode.append(f"G1 X{ax + r * math.cos(ang):.3f} Y{ay + r * math.sin(ang):.3f} F2500")
        dx, dy = tx - dx0, ty - dy0
        dist = math.hypot(dx, dy)
        wx = dx0 + (dx / dist * c["wipe_r"]) if dist > 0 else dx0 + c["wipe_r"]
        wy = dy0 + (dy / dist * c["wipe_r"]) if dist > 0 else dy0
        self.gcode.append(f"G0 Z{c['z_wipe_exit']:.3f} F3000")
        self.gcode.append(f"G0 X{wx:.3f} Y{wy:.3f} F{c['wipe_feed']}")                 # the wipe: its own speed, not whatever the lift left behind
        self.gcode.append(f"G0 Z{c['z_high']:.3f} F3000")
        self.gcode.append("; travel to the next stroke")                               # the repair reuses the dip cycle up to here, not this trip
        self.gcode.append(f"G0 X{tx:.3f} Y{ty:.3f} Z{c['z_low']:.3f} F{c['feed']}")     # the trip to the next stroke: TRAVEL speed again (the lift set F3000)
        self.dist_since_dip = 0.0
        self.current_max_dist = random.uniform(c["min_dist"], c["max_dist"])
        self.current_pos = (tx, ty)

    def z_for(self, kind):
        """Machine Z while painting: the fill at the touch height, thin lines line_depth deeper."""
        c = self.cfg
        return c["z_paint"] - c.get("line_depth", 0.0) if kind == "line" else c["z_paint"]

    def aniso_factor(self, pts, i):
        """How much wider than average the brush makes a mark in the direction of point i (1.0 = round brush). The footprint along
        the paper's axes comes from the probe (aniso = [along u, along v]); a mark is as wide as the footprint ACROSS its direction."""
        an = self.cfg.get("aniso")
        if not an:
            return 1.0
        n_pts = len(pts)
        j0, j1 = max(0, i - 1), min(n_pts - 1, i + 1)
        dx, dy = pts[j1][0] - pts[j0][0], pts[j1][1] - pts[j0][1]
        L = math.hypot(dx, dy)
        if L < 1e-9:
            return 1.0
        nx, ny = -dy / L, dx / L                                  # across the stroke
        u, v = self.cfg.get("aniso_axes", ((1.0, 0.0), (0.0, 1.0)))
        pu, pv = nx * u[0] + ny * u[1], nx * v[0] + ny * v[1]
        f = an[0] * pu * pu + an[1] * pv * pv
        return float(min(1.6, max(0.6, f)))

    def z_width(self, w_mm):
        """Machine Z for a centre-line mark of width w_mm. With a brush profile (the ladder's curve): the depth at which the REAL brush
        makes a line this wide (thin marks = touch, wide = deeper, never deeper than line_depth). Without: a straight line between
        width_min (touch) and width_ref (full line_depth)."""
        c = self.cfg
        cur = c.get("curve")
        if cur is not None:
            w_mm = min(float(w_mm), float(cur[1][-1]))              # never wider than the brush really makes: no outline-like bands
            d = float(np.interp(w_mm, cur[1], cur[0]))
        else:
            lo = c.get("width_min", 0.0)
            d = c["line_depth"] * min(1.0, max(0.0, (w_mm - lo) / max(c["w_ref"] - lo, 1e-6)))
        return c["z_paint"] - min(d, c["line_depth"])

    def _pen_down(self, z):
        self._set_speed("paint")
        self.gcode.append(f"G1 Z{z:.3f} F{self.cfg.get('plunge_feed', 2500)}")
        self.gcode.append(f"G1 F{self.cfg['feed_paint']}")             # the plunge speed is modal: without this line the whole stroke runs at the plunge speed

    # A line is landed and lifted OUTSIDE itself, while moving: a brush that meets the paper standing still leaves a dot
    # (2026-10-29). The straight-down / straight-up part stays LINE_LAND_HIGH_MM above the paper; the slant to the line's
    # first point (and away from its last) runs LINE_LEAD_IN_MM in line with it. The line itself keeps its own depths.
    def _air_z(self, x, y):
        c, tilt = self.cfg, self.cfg.get("tilt")
        tz = (tilt[0] * (x - tilt[2]) + tilt[1] * (y - tilt[3])) if tilt else 0.0
        return c["z_paint"] + tz + float(getattr(S, "LINE_LAND_HIGH_MM", 2.0))

    def _lead_point(self, at, out):
        lead = float(getattr(S, "LINE_LEAD_IN_MM", 3.0))
        return at[0] + out[0] * lead, at[1] + out[1] * lead

    def _land(self, at, out, z):
        """Onto `at` at depth z, gliding in from the lead point that lies in direction `out` from it."""
        lx, ly = self._lead_point(at, out)
        self.gcode.append(f"G0 X{lx:.3f} Y{ly:.3f} Z{self.cfg['z_low']:.3f}")
        self._pen_down(max(z, self._air_z(lx, ly)))
        self.gcode.append(f"G1 X{at[0]:.3f} Y{at[1]:.3f} Z{z:.3f}")
        self.current_pos = tuple(at)

    def _lift(self, at, out):
        """Off the paper from `at`, gliding on in direction `out` up to the air height, then up to the travel height."""
        lx, ly = self._lead_point(at, out)
        self.gcode.append(f"G1 X{lx:.3f} Y{ly:.3f} Z{self._air_z(lx, ly):.3f}")
        self.gcode.append(f"G0 Z{self.cfg['z_low']:.3f} F3000")
        self.current_pos = (lx, ly)

    def write_layer(self, name, paths_mm, dish, rinse=False):
        """Paints one colour layer. dish = (x, y, radius or None). rinse: wash the brush in the water dish first."""
        c = self.cfg
        if not paths_mm:
            return
        self.dish = dish
        self.gcode.append(f"\n; ===== LAYER {name} =====")
        if rinse and c.get("water"):
            self._dip_and_travel(dish[0], dish[1], dish=c["water"])            # rinse, then stop above the paint dish
        first = paths_mm[0][0]
        self._dip_and_travel(first[0], first[1], dish=self.dish)               # load the colour
        for path in paths_mm:
            kind = getattr(path, "kind", "fill")
            zs = None
            if kind == "line" and getattr(path, "widths", None) is not None and c.get("line_depth", 0) > 0:
                # a skeleton line is pressed for its own width, as the probe measured it (no VORO_MARK_SHARE boost: that 0.7 is for the deep,
                # wide leaf veins; on thin lines it pressed 0.5 mm too deep and painted them 3x as wide as the artwork, 2026-10-27)
                boost = 1.0
                top = float(c.get("thin_max") or 1e9)
                # ONLY the width sets the depth along a line: no ramps or offsets inside it. Landing and lifting happen
                # outside the line (_land / _lift)
                zs = [self.z_width(min(w * boost, top) / self.aniso_factor(path, i)) for i, w in enumerate(path.widths)]   # the depth follows the width in THIS direction
                sd = getattr(S, "SIGN_DEPTH_MM", None)
                if getattr(path, "group", None) == "signature" and sd is not None:
                    # the signature: a fixed depth below the touch height, not through the probe curve. Its 1.5..1.6 mm stretch is
                    # nearly flat, so 1.6 mm landed 0.75 mm deep and the freshly dipped brush made 3-4 mm blobs (2026-11-05)
                    zs = [c["z_paint"] - min(float(sd), c["line_depth"])] * len(path)
            if kind != "line" and getattr(path, "widths", None) is not None and c.get("curve") is not None \
                    and bool(getattr(S, "BLOB_DEPTH_FROM_WIDTH", True)):
                zs = [self.z_width(w / self.aniso_factor(path, i)) for i, w in enumerate(path.widths)]   # a blob: its depth follows its width, like a skeleton stroke
            if kind != "line" and len(path) >= 2:
                ramp = float(getattr(S, "BLOB_RAMP_MM", 2.5))
                if ramp > 0 and zs is None:
                    # a point every 0.4 mm: a simplified stroke of only a few points would stay at the raised end height all the way
                    seg_ = [math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:])]
                    s_ = np.r_[0.0, np.cumsum(seg_)]
                    t_ = np.linspace(0.0, s_[-1], max(2, int(math.ceil(s_[-1] / 0.4)) + 1))
                    new = Stroke(list(zip(np.interp(t_, s_, [p[0] for p in path]), np.interp(t_, s_, [p[1] for p in path]))), kind)
                    new.tier, new.fixed_dir, new.group = path.tier, path.fixed_dir, path.group
                    path = new
                if ramp > 0:
                    air = c["z_paint"] + float(getattr(S, "BLOB_LEAD_AIR_MM", 1.0))           # the brush starts a little above the paper
                    base = zs if zs is not None else [self.z_for(kind)] * len(path)
                    zs = lead_ramp_zs(base, path, air, ramp)
            z_down = self.z_for(kind)
            has_w = zs is not None and getattr(path, "widths", None) is not None
            tilt = c.get("tilt")
            if tilt:                                                           # the paper is not level (measured by the probes): every Z follows it
                base = zs if zs else [z_down] * len(path)
                zs = [z + tilt[0] * (px_ - tilt[2]) + tilt[1] * (py_ - tilt[3]) for z, (px_, py_) in zip(base, path)]
            if kind == "line":                                                 # visible in the G-code: what this stroke does with Z
                if has_w:
                    self.gcode.append(f"; skeleton stroke: {len(path)} points, width {min(path.widths):.2f}-{max(path.widths):.2f} mm, "
                                      f"Z {min(zs):.2f}-{max(zs):.2f} (deeper = wider)")
                else:
                    self.gcode.append(f"; skeleton stroke: {len(path)} points, constant Z {z_down:.2f}")
            cx, cy = self.current_pos
            flip = None
            if kind == "line" and has_w and not getattr(path, "fixed_dir", False):
                flip = thin_end_flip(path, top=c.get("thin_max"))              # a line starts at its thin end: no press mark at the start
            if flip is None:
                flip = not getattr(path, "fixed_dir", False) and \
                    math.hypot(path[-1][0] - cx, path[-1][1] - cy) < math.hypot(path[0][0] - cx, path[0][1] - cy)   # enter at the nearer end
            # DIP_BY_WIDTH: a wide mark uses more paint, so each mm counts by its width against DIP_REF_WIDTH_MM (widths as painted,
            # capped at thin_max); the dip distances DIP_MIN/MAX_MM hold for a line that wide
            wk = None
            if getattr(S, "DIP_BY_WIDTH", False) and has_w and len(path.widths) == len(path):
                w_ = np.minimum(np.asarray(path.widths, float), float(c.get("thin_max") or 99.0))
                wk = np.clip(w_ / float(getattr(S, "DIP_REF_WIDTH_MM", 1.5)), 0.5, 3.0).tolist()
            if flip:
                path = path[::-1]
                zs = zs[::-1] if zs else None
                wk = wk[::-1] if wk else None
            cz = zs[0] if zs else z_down
            glide = kind == "line" and zs is not None and len(path) >= 2 and float(getattr(S, "LINE_LEAD_IN_MM", 0.0)) > 0
            one_stroke = glide and bool(getattr(S, "LINE_ONE_STROKE", False))   # never a dip in the middle of this line
            if glide and self.dist_since_dip > 0:
                # a line that does not fit into the paint left on the brush gets its dip BEFORE it: cut in the middle, it would
                # start again at the cut, often its wide part, with a freshly loaded brush (2026-10-31: 40 of 236 lines)
                L = path_length(path) if not wk else sum(math.hypot(b[0] - a[0], b[1] - a[1]) * 0.5 * (ka + kb)
                                                          for a, b, ka, kb in zip(path, path[1:], wk, wk[1:]))
                # LINE_ONE_STROKE: also a line longer than one dip gets its dip before it (it is then painted uncut, below)
                if self.dist_since_dip + L > self.current_max_dist and (L <= self.current_max_dist or one_stroke):
                    self._dip_and_travel(*self._lead_point(path[0], end_dir(path)), dish=self.dish)
            self._set_speed("travel")
            if glide:
                self._land(path[0], end_dir(path), cz)
            else:
                self.gcode.append(f"G0 X{path[0][0]:.3f} Y{path[0][1]:.3f} Z{c['z_low']:.3f}")
                self._pen_down(cz)
            self.current_pos = path[0]
            for i in range(1, len(path)):
                px, py = path[i]
                zn = zs[i] if zs else None
                k_seg = 0.5 * (wk[i - 1] + wk[i]) if wk else 1.0                 # paint used per mm of this segment
                while True:
                    cx, cy = self.current_pos
                    dx, dy = px - cx, py - cy
                    d = math.hypot(dx, dy)
                    room = self.current_max_dist - self.dist_since_dip
                    if d * k_seg <= room + 1e-9 or one_stroke:                  # one_stroke: the paint is counted, the line not cut
                        self.gcode.append(f"G1 X{px:.3f} Y{py:.3f}" + (f" Z{zn:.3f}" if zs else ""))
                        self.dist_since_dip += d * k_seg
                        self.current_pos = (px, py)
                        cz = zn if zs else cz
                        break
                    t = max(room, 0.0) / (d * k_seg)                            # the interval ends inside this segment
                    cut = (cx + dx * t, cy + dy * t)
                    z_cut = cz + (zn - cz) * t if zs else cz
                    if room > 0.01:
                        self.gcode.append(f"G1 X{cut[0]:.3f} Y{cut[1]:.3f}" + (f" Z{z_cut:.3f}" if zs else ""))
                    cz = z_cut
                    fwd = (dx / d, dy / d)
                    if glide:                                                   # off and back on the same way as at the ends of a line
                        self._lift(cut, fwd)
                        self._dip_and_travel(*self._lead_point(cut, (-fwd[0], -fwd[1])), dish=self.dish)
                        self._land(cut, (-fwd[0], -fwd[1]), cz)
                    else:
                        self.gcode.append(f"G0 Z{c['z_low']:.3f} F3000")
                        self._dip_and_travel(*cut, dish=self.dish)
                        self._pen_down(cz)
            if glide:
                self._lift(path[-1], end_dir(path[::-1]))
            else:
                self.gcode.append(f"G0 Z{c['z_low']:.3f} F3000")
            grp = getattr(path, "group", None)                                  # which leaf this stroke belongs to (stroke photos, STROKE_PHOTOS)
            grp = grp if grp is None or isinstance(grp, str) else '-'.join(map(str, grp))
            self.gcode.append(f"; STROKE END {grp if grp is not None else 'line'}")

    def finish(self):
        self.gcode += ["M400", f"G0 Z{self.cfg['z_high']:.3f} F3000", "M2"]
        return "\n".join(self.gcode)


# ----------------------------------------------------------------------------------------------- preview + report
def render(paths_px, shape, brush_w, res=RES):
    canvas = np.zeros(shape, np.uint8)
    th0 = max(1, int(round(brush_w * res)))
    for p in paths_px:
        w = getattr(p, "widths", None)
        pts = np.round(np.array(p)).astype(np.int32)
        if w and len(w) == len(p) and len(p) >= 2:                           # a centre-line stroke paints as wide as the mark is, point by point
            for i in range(len(p) - 1):
                cv2.line(canvas, (int(pts[i][0]), int(pts[i][1])), (int(pts[i + 1][0]), int(pts[i + 1][1])), 255,
                         max(1, int(round(0.5 * (w[i] + w[i + 1]) * res))), cv2.LINE_8)
        else:
            cv2.polylines(canvas, [pts.reshape(-1, 1, 2)], False, 255, th0, cv2.LINE_8)
    return canvas > 0


def report(label, mask, paths_px, kinds, dips, n_lines, cfg, out_png):
    painted = render(paths_px, mask.shape, cfg["brush_w"])
    rb = cfg["brush_w"] / 2.0 * RES
    m8 = mask.astype(np.uint8)
    thick = cv2.dilate(cv2.erode(m8, disk(rb)), disk(rb)) > 0           # parts a brush-width disc fits into
    thin_zone = cv2.dilate(((mask & ~thick)).astype(np.uint8), disk(rb * 1.6)) > 0
    cov = (painted & mask).sum() / max(mask.sum(), 1)
    spill_all = (painted & ~mask).sum() / max(mask.sum(), 1)
    spill_thick = (painted & ~mask & ~thin_zone).sum() / max(mask.sum(), 1)
    plen = sum(path_length(p) for p in paths_px) / RES
    est = plen / cfg["feed_paint"] * 60 + dips * 8
    vis = np.zeros(mask.shape + (3,), np.uint8)
    vis[mask & painted] = (255, 255, 255)
    vis[mask & ~painted] = (0, 0, 255)         # missed (red)
    vis[~mask & painted] = (255, 120, 0)       # spill (blue)
    cv2.imwrite(out_png, np.hstack([cv2.cvtColor((mask * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR), vis]))
    vprint(f"   {sum(kinds.values())} strokes ({kinds['thin']} skeleton, {kinds['lines']} lines, {kinds['rings']} rings, "
          f"{kinds['outline']} outlines, {kinds['patch']} repair), {plen / 1000:.2f} m, {dips} dips, {n_lines} G-code lines, ~{est / 60:.0f} min")
    vprint(f"   covered {100 * cov:.1f}%, paint outside the target {100 * spill_thick:.1f}% on shapes wider than the brush"
          f" (+{100 * (spill_all - spill_thick):.1f}% around parts thinner than the brush)")
    if spill_all - spill_thick > 0.5 and cfg.get("thin_max", 0) <= cfg["brush_w"] * 1.1 + 1e-6:
        print(f"   WARNING: much of this layer is thinner than the {cfg['brush_w']:.1f} mm brush - fine detail will come out thicker. "
              f"Use a finer brush (--brush) or a larger artwork.")
    return cov, spill_thick


def default_config():
    """The built-in numbers, taken from settings.py. (Heights are mm above the touch height; main() adds the touch height.)"""
    return {
        "target_width": 260, "brush_w": 1.0, "overlap": 0.0,
        "infill_type": getattr(S, "INFILL_TYPE", "lines"), "infill_angle": None, "outline": bool(getattr(S, "OUTLINE", False)), "simplify_mm": 0.1,
        "infill_style": getattr(S, "INFILL_STYLE", "cells"),
        "blob_min_mm": float(getattr(S, "BLOB_MIN_MM", 6.0)),
        "blob_clear": float(getattr(S, "BLOB_CLEAR", 0.4)),
        "cell_len_small": tuple(getattr(S, "CELL_LEN_SMALL", (4.0, 10.0))),
        "cell_len_large": tuple(getattr(S, "CELL_LEN_LARGE", (15.0, 40.0))),
        "shape_small_mm2": float(getattr(S, "SHAPE_SMALL_MM2", 50.0)),
        "shape_large_mm2": float(getattr(S, "SHAPE_LARGE_MM2", 600.0)),
        "blob_passes": int(getattr(S, "BLOB_PASSES", 1)),          # 1 = big blobs only in the first run, 4 = every size (old behaviour)
        "blob_leftover": bool(getattr(S, "BLOB_LEFTOVER", False)),  # small leftover strokes in the first run (False = the repair does them)
        "voro_cell_width": float(getattr(S, "VORO_CELL_WIDTH", 0.85)),
        "voro_mark_share": float(getattr(S, "VORO_MARK_SHARE", 0.7)),
        "voro_tiers": tuple(getattr(S, "VORO_TIERS", (1.0, 0.6, 0.35))),
        "voro_blob_aspect": tuple(getattr(S, "VORO_BLOB_ASPECT", (2.5, 5.0))),
        "voro_hand_deg": float(getattr(S, "VORO_HAND_DEG", 45.0)),
        "voro_calligraphy": bool(getattr(S, "VORO_CALLIGRAPHY", True)),
        "voro_overlap_mm": float(getattr(S, "VORO_OVERLAP_MM", 0.3)),
        "voro_relax": int(getattr(S, "VORO_RELAX", 3)),
        "voro_field_sigma_mm": float(getattr(S, "VORO_FIELD_SIGMA_MM", 5.0)),
        "voro_axis_weight": float(getattr(S, "VORO_AXIS_WEIGHT", 0.3)),
        "infill_style_by_color": dict(getattr(S, "INFILL_STYLE_BY_COLOR", {}) or {}),
        "leaf_target_mm": float(getattr(S, "LEAF_TARGET_MM", 9.0)),
        "leaf_whole_max": int(getattr(S, "LEAF_WHOLE_MAX", 4)),
        "leaf_whole_cover": float(getattr(S, "LEAF_WHOLE_COVER", 0.05)),
        "line_width_slack": float(getattr(S, "LINE_WIDTH_SLACK", 1.0)),
        "line_tip_mm": float(getattr(S, "LINE_TIP_MM", 0.0)),
        "leaf_width_mm": tuple(getattr(S, "LEAF_WIDTH_MM", (4.0, 14.0))),
        "leaf_aspect": tuple(getattr(S, "LEAF_ASPECT", (3.0, 7.0))),
        "leaf_variety": float(getattr(S, "LEAF_VARIETY", 0.25)),
        "leaf_tiers": tuple(getattr(S, "LEAF_TIERS", (1.0, 0.5, 0.3))),
        "leaf_vein_variety": float(getattr(S, "LEAF_VEIN_VARIETY", 0.4)),
        "leaf_direction": str(getattr(S, "LEAF_DIRECTION", "alternate")),
        "leaf_outline": bool(getattr(S, "LEAF_OUTLINE", False)),
        "leaf_detail": float(getattr(S, "LEAF_DETAIL", 0.6)),
        "region_styles": dict(getattr(S, "REGION_STYLES", {}) or {}),
        "elongated_ratio": float(getattr(S, "ELONGATED_RATIO", 5.0)),
        "cell_len_mm": tuple(getattr(S, "CELL_LEN_MM", (10.0, 25.0))), "painterliness": float(getattr(S, "PAINTERLINESS", 0.5)),
        "painterly_len_mm": tuple(getattr(S, "PAINTERLY_LEN_MM", (20.0, 40.0))),
        "infill_coverage": float(getattr(S, "INFILL_COVERAGE", 1.0)),
        "min_dist": S.DIP_MIN_MM, "max_dist": S.DIP_MAX_MM,
        "z_paint": 0.0, "z_low": S.TRAVEL_MM, "z_high": S.HIGH_MM, "z_wipe_exit": S.WIPE_EXIT_MM,
        "x_off": 15.0, "y_off": 90.0,
        "dip_x": S.DIP_X, "dip_y": S.DIP_Y, "dip_z": S.PAINT_DIP_Z, "dip_jitter": S.PAINT_DIP_JITTER,
        "dip_spiral_loops": S.PAINT_DIP_SPIRAL_LOOPS, "dip_spiral_r": S.PAINT_DIP_SPIRAL_R, "wipe_r": S.PAINT_WIPE_R,
        "feed": S.FEED_TRAVEL, "wipe_feed": S.WIPE_FEED, "feed_paint": S.PAINT_FEED, "accel_travel": S.ACCEL_TRAVEL, "accel_paint": S.PAINT_ACCEL,
        "line_depth": None, "z_offset": 0.0, "thin_max": None, "point_step": 0.4, "plunge_feed": S.PLUNGE_FEED,
        "min_roundness": S.POOL_MIN_ROUNDNESS, "min_pool_area": S.POOL_MIN_AREA_MM2, "palette_radius": S.PALETTE_RADIUS_MM,
        "snapshot": (S.SNAPSHOT_X, S.SNAPSHOT_Y),
        "water": (S.WATER_X, S.WATER_Y, S.WATER_R) if S.WATER_X is not None and S.WATER_Y is not None else None,
    }


def machine_mapper(cfg):
    """artwork-plane mm (x right, y up) -> machine mm. Without a paper placement this is mama.py's x_off / y_off."""
    pl = cfg.get("placement")
    if pl:
        o, u, v = np.array(pl["origin_mm"]), np.array(pl["u"]), np.array(pl["v"])
        return lambda x, y: (o[0] + x * u[0] + y * v[0], o[1] + x * u[1] + y * v[1])
    return lambda x, y: (x + cfg["x_off"], y + cfg["y_off"])


def signature_file():
    """SIGNATURE from settings.py, or None when it is off or the picture is not there (a fresh copy of the program has no
    signature picture: it paints without one instead of stopping)."""
    path = getattr(S, "SIGNATURE", None)
    if not path:
        return None
    for p in (path, os.path.join(os.path.dirname(os.path.abspath(__file__)), path)):
        if os.path.isfile(p):
            return p
    print(f"no signature: the picture {path} (SIGNATURE in settings.py) is not there")
    return None


def plan_signature(cfg, res=RES):
    """The signature + year (SIGNATURE in settings.py) as centre-line strokes in artwork mm (x right, y up), under the
    picture's bottom right corner with its top SIGN_GAP_MM below the picture - or, with SIGN_INSIDE, inside the picture's
    bottom right corner, SIGN_INSET_MM in from its edges (the artwork is painted over it). A picture turned by 90 degrees (counter-
    clockwise) has its bottom on the right: the signature turns with it. [] without a signature."""
    path = signature_file()
    if not path:
        return []
    import signature as SG
    mask, w_mm, h_mm = SG.block_mask(path, float(S.SIGN_WIDTH_MM), res, year=bool(getattr(S, "SIGN_YEAR", True)),
                                     year_h_mm=float(getattr(S, "SIGN_YEAR_MM", 4.0)))
    pl = cfg.get("placement") or {}
    art_w = float(pl.get("width_mm") or cfg["target_width"])
    turned = int(pl.get("rotate", cfg.get("rotate", 0)) or 0) == 90
    upright_w = float(pl["height_mm"]) if turned else art_w            # the picture's width as it is meant to be seen
    if bool(getattr(S, "SIGN_INSIDE", False)):
        inset = float(getattr(S, "SIGN_INSET_MM", 5.0))
        x0, y0 = upright_w - mask.shape[1] / res - inset, inset
    else:
        x0, y0 = upright_w - w_mm, -(float(S.SIGN_GAP_MM) + mask.shape[0] / res)
    line_w = float(getattr(S, "SIGN_LINE_MM", 1.3))                     # one width all along: the depth stays even
    out = []
    paths = trace_skeleton(skeletonize(mask), min_len_px=3)
    # a signature is one pen line, but the skeleton cuts it at every crossing and sharp turn and leaves short spurs where the ink
    # fills a cusp: each piece is one more landing in the crowded letters. Spurs (shorter than SIGN_SPUR_MM, touching another
    # piece) go; a short piece on its own (the dot over an i) stays. Then pieces that meet are joined even round sharp turns,
    # straightest pair first (a crossing goes on straight)
    spur = float(getattr(S, "SIGN_SPUR_MM", 1.0)) * res
    if len(paths) > 1:
        from scipy.spatial import cKDTree
        keep = []
        for i, p in enumerate(paths):
            if path_length(p) < spur:
                rest = np.vstack([np.asarray(o, float) for j, o in enumerate(paths) if j != i])
                d, _ = cKDTree(rest).query(np.asarray([p[0], p[-1]], float))
                if d.min() <= 1.5:
                    continue
            keep.append(p)
        paths = join_branches(keep, reach_px=res, look_px=res, max_turn_deg=170.0)
    for p in paths:
        q = simplify(p, cfg["simplify_mm"] * res, closed=False)
        if len(q) < 2 or path_length(q) <= 0:
            continue
        st = attach_widths(q, p, [line_w * res] * len(p), step_px=max(1.0, cfg.get("point_step", 0.4) * res))
        st[:] = [(x0 + x / res, y0 + y / res) for x, y in st]
        if turned:                                                       # upright (x, y) -> turned artwork (art_w - y, x)
            st[:] = [(art_w - y, x) for x, y in st]
        st.group = "signature"
        out.append(st)
    return out


LAYER_BGR = {"blue": (255, 90, 0), "red": (0, 0, 255), "green": (0, 200, 0), "yellow": (0, 230, 255), "orange": (0, 140, 255),
             "cyan": (255, 255, 0), "purple": (200, 0, 160), "pink": (180, 100, 255), "black": (60, 60, 60)}


def draw_stroke(img, pts, widths_mm, ppm, color, th0, aa=cv2.LINE_AA, mask=None):
    """Draws a stroke on the camera picture. With widths it is tapered like the mark; without, one brush-wide line."""
    pts = np.round(pts).astype(np.int32)
    if widths_mm and len(widths_mm) == len(pts) and len(pts) >= 2:
        for i in range(len(pts) - 1):
            t = max(1, int(round(0.5 * (widths_mm[i] + widths_mm[i + 1]) * ppm)))
            a, b = (int(pts[i][0]), int(pts[i][1])), (int(pts[i + 1][0]), int(pts[i + 1][1]))
            cv2.line(img, a, b, color, t, aa)
            if mask is not None:
                cv2.line(mask, a, b, 255, t, aa)
    else:
        cv2.polylines(img, [pts.reshape(-1, 1, 2)], False, color, th0, aa)
        if mask is not None:
            cv2.polylines(mask, [pts.reshape(-1, 1, 2)], False, 255, th0, aa)


def _art_on_photo(art, frame, cfg, H):
    """The original artwork drawn into the camera picture where the plan puts it (None without a placement)."""
    warp = _art_warp(art, frame, cfg, H)
    if warp is None:
        return None
    warped, inside = warp
    out = frame.copy()
    out[inside] = warped[inside]
    return out


def _art_marks_layer(art, frame, cfg, H):
    """For the live preview: the artwork's marks (not its white) in camera pixels, as (image, mask) like the planned strokes."""
    warp = _art_warp(art, frame, cfg, H)
    if warp is None:
        return None
    warped, inside = warp
    hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
    marks = inside & ((hsv[..., 2] < 170) | (hsv[..., 1] > 80))
    return warped, marks


def _art_warp(art, frame, cfg, H):
    """The artwork warped into the camera picture: (image, inside mask), or None without a placement."""
    placed = _art_from_plan(art, frame, cfg, H)
    if placed is None:
        return None
    art_r, map_x, map_y = placed
    ah, aw = art_r.shape[:2]
    inside = (map_x >= 0) & (map_x <= aw - 1) & (map_y >= 0) & (map_y <= ah - 1)   # the maps cover a box around the artwork
    if not inside.any():
        return None
    # the camera sees ~3 px per mm: the artwork is shrunk to about that first (area average), so fine marks are not lost to aliasing
    f = min(1.0, (inside.sum() / float(ah * aw)) ** 0.5 * 2.0)
    if f < 1.0:
        art_r = cv2.resize(art_r, (max(1, int(round(aw * f))), max(1, int(round(ah * f)))), interpolation=cv2.INTER_AREA)
        map_x = np.where(inside, (map_x + 0.5) * art_r.shape[1] / aw - 0.5, -1).astype(np.float32)
        map_y = np.where(inside, (map_y + 0.5) * art_r.shape[0] / ah - 0.5, -1).astype(np.float32)
    warped = cv2.remap(art_r, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    return warped, inside


def draw_overlay(frame, H, H_inv, layers_out, dishes, cfg, det, out_png, art=None):
    """Paper (green), artwork box (yellow), paint dishes and water on the camera picture, and on the paper either the original
    artwork where the plan puts it (OVERLAY_SHOW "art") or the planned strokes in layer colours ("strokes")."""
    import vision as PP
    from vision import pixel_to_mm
    out = frame.copy()
    h, w = out.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    x0, y0 = pixel_to_mm(H, cx, cy); x1, y1 = pixel_to_mm(H, cx + 10, cy); x2, y2 = pixel_to_mm(H, cx, cy + 10)
    px_per_mm = 10.0 / (0.5 * (math.hypot(x1 - x0, y1 - y0) + math.hypot(x2 - x0, y2 - y0)))
    th = max(1, int(round(cfg["brush_w"] * px_per_mm)))
    layer = None
    if art is not None and str(getattr(S, "OVERLAY_SHOW", "art")).lower() == "art":
        layer = _art_on_photo(art, frame, cfg, H)
    if layer is None:
        layer = out.copy()
        for name, paths_mm in layers_out:
            for p in paths_mm:
                pts = mm_to_px_array(H_inv, p)
                draw_stroke(layer, pts, getattr(p, "widths", None), px_per_mm, LAYER_BGR.get(name, (0, 0, 255)), th)
    for name, paths_mm in layers_out:                                          # the signature is not in the artwork picture: always as strokes
        for p in paths_mm:
            if getattr(p, "group", None) == "signature":
                draw_stroke(layer, mm_to_px_array(H_inv, p), p.widths, px_per_mm, (0, 0, 255), th)
    out = cv2.addWeighted(layer, 0.75, out, 0.25, 0)
    if det is not None:
        out = PP.draw_paper(out, det, H_inv, None, cfg.get("placement"))

    def mark(x, y, r, text, col):
        p = mm_to_px_array(H_inv, [(x, y)])[0]
        if -50 <= p[0] < w + 50 and -50 <= p[1] < h + 50:
            cv2.circle(out, (int(p[0]), int(p[1])), max(8, int(r * px_per_mm)), col, 3, cv2.LINE_AA)
            cv2.putText(out, text, (int(p[0]) + 12, int(p[1]) - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 5, cv2.LINE_AA)
            cv2.putText(out, text, (int(p[0]) + 12, int(p[1]) - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2, cv2.LINE_AA)
    for name, (x, y, r, note) in dishes.items():
        mark(x, y, r or 10, f"{name} dip", LAYER_BGR.get(name, (255, 0, 255)))
    for x, y, r, text in cfg.get("pools_ignored", []):
        mark(x, y, r, text, (170, 170, 170))                                   # what the camera saw but did not use
    if cfg.get("water"):
        mark(cfg["water"][0], cfg["water"][1], cfg["water"][2], "water", (255, 200, 0))
    if cfg.get("photo_time"):
        stamp_picture(out, f"camera picture taken {cfg['photo_time']}" + ("" if cfg.get("photo_live", True) else "  - NOT LIVE?"),
                      cfg.get("photo_live", True))
    cv2.putText(out, "paint " + BUILD.split(":")[0], (14, out.shape[0] - 48), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(out_png, out)
    vprint(f"overlay on the camera picture: {out_png}  (strokes in layer colours, green = paper, yellow = artwork box, circles = where it will dip)")


def run(img_path, cfg, out_gcode, frame=None, det=None):
    """Plans every colour layer, finds its paint dish with the camera, writes ONE G-code file."""
    t0 = time.time()
    base = out_gcode.rsplit(".", 1)[0]
    rgb = load_rgb(img_path, cfg["target_width"], rotate=cfg.get("rotate", 0))
    if cfg.get("color", "auto") == "auto":
        layers = find_layers(rgb)
    elif cfg["color"] == "black":
        # by design one paint: everything the artwork has (every colour and the black) is painted from the black dish
        mk = np.zeros(rgb.shape[:2], bool)
        for l in find_layers(rgb):
            mk |= l["mask"]
        layers = [{"name": "black", "mask": mk, "share": float(mk.mean()), "hue": None, "value": 0.0}]
    else:                                                                    # --color: just that one colour
        from vision import circular_mean_hue
        mk = color_mask(rgb, cfg["color"], tol=cfg.get("tol", 40.0))
        hue = None
        if cfg["color"] in COLOR_HUES:
            hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
            hue = float(circular_mean_hue(hsv[..., 0][mk])) if mk.any() else None
        layers = [{"name": cfg["color"], "mask": mk, "share": float(mk.mean()), "hue": hue, "value": 0.0}]
    layers = [l for l in layers if l["mask"].any()]
    if not layers:
        raise SystemExit("nothing to paint found in the artwork (try --list_colors)")
    cfg["layer_hues"] = {l["name"]: l["hue"] for l in layers}              # the repair gives every empty spot the nearest of these colours
    pools = []
    if frame is not None:
        import vision as PL
        from vision import load_calibration
        cal = load_calibration()
        seen = PL.find_pools(frame, cal["H"], det["corners_px"] if det else None)
        # the paint dishes are near the dip position (DIP_X / DIP_Y in settings.py); a red LED, a cable or a label
        # somewhere else in the picture is not a dish, however well its colour matches
        vprint(f"paint pools the camera sees outside the paper (dishes must be within {cfg['palette_radius']:.0f} mm of the dip position "
              f"X {cfg['dip_x']:.0f} Y {cfg['dip_y']:.0f}):")
        cfg["pools_ignored"] = []
        for p in seen:
            d = math.hypot(p["center_mm"][0] - cfg["dip_x"], p["center_mm"][1] - cfg["dip_y"])
            rnd = p.get("roundness", 1.0)
            if d > cfg["palette_radius"]:
                verdict = f"-> IGNORED, {d:.0f} mm from the dip position"
            elif p["area_mm2"] < cfg["min_pool_area"]:
                verdict = f"-> IGNORED, too small ({p['area_mm2']:.0f} mm2 - a speck or splash, not a dish; POOL_MIN_AREA_MM2 in settings.py)"
            elif rnd < cfg["min_roundness"]:
                verdict = f"-> IGNORED, not round (roundness {rnd:.2f}: a streak or spill)"
            else:
                verdict = "-> candidate"
                pools.append(p)
            if not verdict.endswith("candidate"):
                cfg["pools_ignored"].append((p["center_mm"][0], p["center_mm"][1], p["radius_mm"], "ignored"))
            vprint(f"   {hue_name(p['hue']):7s} X {p['center_mm'][0]:5.0f} Y {p['center_mm'][1]:5.0f}  radius {p['radius_mm']:3.0f} mm  "
                  f"area {p['area_mm2']:5.0f} mm2   {verdict}")
        if not seen:
            vprint("   none")
    dishes = assign_dishes(layers, pools, cfg)
    names = ", ".join(f"{l['name']} ({100 * l['share']:.1f}%)" for l in layers)
    vprint(f"layers: {names}")
    for l in layers:
        x, y, r, note = dishes[l["name"]]
        vprint(f"   {l['name']:7s} dips at X {x:.0f} Y {y:.0f}" + (f" (pool radius {r:.0f} mm)" if r else "") + f"  - {note}")
    if cfg.get("line_depth_auto") and cfg["line_depth"] > 0:
        vprint(f"variable depth is ON by default: marks narrower than {cfg['thin_max']:.1f} mm are painted as one centre-line stroke, up to "
              f"{cfg['line_depth']:.2f} mm deeper for the widest (--line_depth 0 switches it off)")
    elif cfg["line_depth"] == 0:
        vprint("variable depth for thin lines is OFF (no Z offset / --line_depth 0): thin lines are painted at the fill Z")
    zf, zl = cfg["z_paint"], cfg["z_paint"] - cfg["line_depth"]
    vprint(f"Z (machine): touch {cfg['z_paint']:.2f}: fill at {zf:.2f}, thin lines " + (f"from {zf:.2f} (thinnest) down to {zl:.2f} (widest)" if cfg["line_depth"] > 0 else f"at {zl:.2f}")
          + f"; travel {cfg['z_low']:.2f}, dip {cfg['dip_z']:.2f}, high {cfg['z_high']:.2f}" + (f"   [Z offset {cfg['z_offset']:+.2f} included]" if cfg["z_offset"] else ""))
    if len(layers) > 1 and not cfg.get("water"):
        vprint("   note: no WATER_X / WATER_Y in settings.py - colours are loaded without rinsing the brush in between")
    to_mm = machine_mapper(cfg)
    painter = Painter(cfg)
    planned = []
    for L in layers:
        vprint(f"\nlayer {L['name']}:")
        locate_regions(L["mask"], cfg, L["name"])
        cfg_l = dict(cfg)                                                   # a style per colour (INFILL_STYLE_BY_COLOR): mix styles on one artwork
        cfg_l["infill_style"] = (cfg.get("infill_style_by_color") or {}).get(L["name"], cfg["infill_style"])
        if cfg_l["infill_style"] != cfg["infill_style"]:
            vprint(f"   {L['name']}: infill style {cfg_l['infill_style']}")
        smap = load_style_map(img_path, L["mask"].shape, cfg.get("rotate", 0))
        if smap is None:
            strokes, kinds = plan_strokes(L["mask"], cfg_l)
        else:                                                               # <artwork>.styles.json: a style per box
            lab, regions = smap
            vprint(f"   {len(regions)} boxes with their own style ({os.path.splitext(img_path)[0]}.styles.json)")
            strokes, kinds = [], {"thin": 0, "lines": 0, "rings": 0, "outline": 0, "patch": 0}
            parts = [(0, cfg_l)] + [(i, dict(cfg_l, infill_style=r["style"], infill_type=r.get("infill_type", cfg_l["infill_type"])))
                                    for i, r in enumerate(regions, 1)]
            for i, cfg_r in parts:
                part = L["mask"] & (lab == i)
                if part.any():
                    s_r, k_r = plan_strokes(part, cfg_r)
                    strokes += s_r
                    for key, v in k_r.items():
                        kinds[key] += v
        try:                                                                # picture of the blobs / leaves and their strokes
            import leaf_style
            grp = [(np.asarray(s, float), np.asarray(s.widths, float), s.group if s.group is not None else ("s", n_s))
                   for n_s, s in enumerate(strokes) if getattr(s, "widths", None) is not None and len(s) >= 2]
            leaf_style.picture(L["mask"], grp, RES, float(cfg.get("voro_mark_share", 0.7)), f"{base}_{L['name']}_strokes.png",
                               f"{L['name']}: {cfg_l['infill_style']}")
            vprint(f"   strokes picture: {base}_{L['name']}_strokes.png")
        except Exception as e:
            vprint(f"   (strokes picture not made: {e})")
        paths_mm = []
        for p in strokes:
            st = Stroke([to_mm(x / RES, y / RES) for x, y in p], getattr(p, "kind", "fill"))
            st.widths = getattr(p, "widths", None)
            st.tier, st.fixed_dir, st.group = getattr(p, "tier", 9), getattr(p, "fixed_dir", False), getattr(p, "group", None)
            paths_mm.append(st)
        planned.append((L, strokes, kinds, paths_mm))
    widths = [w for _, _, _, pm in planned for p in pm if p.kind == "line" and p.widths for w in p.widths]
    widest = max(widths) if widths else 0.0
    cfg["w_ref"] = cfg["width_ref"] if cfg.get("width_ref") else max(widest, 0.5 * cfg["brush_w"])
    if cfg["line_depth"] > 0 and widths:
        q = np.percentile(widths, [10, 50, 90])
        vprint(f"\nwidths of the centre-line marks (mm): 10% narrower than {q[0]:.2f} | median {q[1]:.2f} | 90% narrower than {q[2]:.2f} | widest {widest:.2f}")
        zq = [painter.z_width(v) for v in q]
        if cfg.get("curve") is not None:
            cd, cw = cfg["curve"]
            vprint(f"depth from the brush profile: the brush makes {cw[0]:.2f} mm at the touch height and {float(np.interp(cfg['line_depth'], cd, cw)):.2f} mm at "
                  f"{cfg['line_depth']:.2f} mm deeper; marks outside that range are made as thin / as wide as it can")
            vprint(f"   typical marks ({q[0]:.2f} .. {q[2]:.2f} mm) are painted between Z {zq[2]:.2f} and Z {zq[0]:.2f}")
        else:
            lo = cfg.get("width_min", 0.0)
            vprint(f"depth follows the width: Z {cfg['z_paint']:.2f} (touch) for marks {lo:.2f} mm or narrower, "
                  f"Z {cfg['z_paint'] - cfg['line_depth']:.2f} for marks {cfg['w_ref']:.2f} mm or wider (--width_min / --width_ref)")
            vprint(f"   so the typical marks (10% .. 90%) are painted between Z {zq[2]:.2f} and Z {zq[0]:.2f}  - {abs(zq[0] - zq[2]):.2f} mm of the {cfg['line_depth']:.2f} mm available")
    sign = []
    for p in plan_signature(cfg):                                           # painted FIRST: on the sheet even when the run is stopped
        st = Stroke([to_mm(x, y) for x, y in p], "line")
        st.widths, st.group = p.widths, p.group
        sign.append(st)
    if sign:
        vprint(f"\nsignature: {len(sign)} strokes, painted first (with the first layer, {planned[0][0]['name']})")
    layers_out, summary = [], []
    for idx, (L, strokes, kinds, paths_mm) in enumerate(planned):
        vprint(f"\nlayer {L['name']} plan:")
        ordered, pos = [], dishes[L["name"]][:2]
        if idx == 0 and sign:
            ordered = PathOptimizer.optimize(sign, pos)
            pos = ordered[-1][-1]
        for tier in sorted({p.tier for p in paths_mm}):                     # biggest blobs first, then smaller ones, then the rest
            # a leaf is painted in one go: its veins stay together (in their own order, middle first); the leaves are ordered as units
            units, by_group = [], {}
            for p in paths_mm:
                if p.tier != tier:
                    continue
                if p.group is None:
                    units.append([p])
                elif p.group in by_group:
                    by_group[p.group].append(p)
                else:
                    by_group[p.group] = [p]
                    units.append(by_group[p.group])
            proxies = []
            for u in units:
                pr = Stroke([u[0][0], u[-1][-1]]) if len(u) > 1 else u[0]
                pr.fixed_dir = True if len(u) > 1 else getattr(u[0], "fixed_dir", False)
                pr.unit = u
                proxies.append(pr)
            part = []
            for pr in PathOptimizer.optimize(proxies, pos):
                part += pr.unit
            if part:
                ordered += part
                pos = part[-1][-1] if part[-1].fixed_dir else part[-1][0]   # roughly where the brush ends (the painter picks the nearer end)
        paths_mm = ordered
        n0, d0 = len(painter.gcode), painter.dips
        painter.write_layer(L["name"], paths_mm, dishes[L["name"]][:3], rinse=idx > 0)
        report(L["name"], L["mask"], strokes, kinds, painter.dips - d0, len(painter.gcode) - n0, cfg, f"{base}_{L['name']}_preview.png")
        layers_out.append((L["name"], paths_mm))
        summary.append((L["name"], kinds, sum(1 for p in paths_mm if p.kind == "line" and p.widths and cfg["line_depth"] > 0)))
    allp = np.array([pt for _, ps in layers_out for p in ps for pt in p])
    vprint(f"\nartwork lands at machine X {allp[:, 0].min():.1f}..{allp[:, 0].max():.1f} mm, Y {allp[:, 1].min():.1f}..{allp[:, 1].max():.1f} mm")
    pl, zf, zl = cfg.get("placement"), cfg["z_paint"], cfg["z_paint"] - cfg["line_depth"]
    header = ["; ================= paint.py =================", f"; build {BUILD}",
              f"; written {time.strftime('%Y-%m-%d %H:%M:%S')} from {img_path}",
              "; camera picture taken " + (f"{cfg['photo_time']}" + ("" if cfg.get("photo_live", True) else " (NOT LIVE?)") if cfg.get("photo_time") else "- no camera picture (saved picture or --paper none)"),
              (f"; paper {det['size_mm'][0]:.0f} x {det['size_mm'][1]:.0f} mm; artwork {pl['width_mm']:.0f} x {pl['height_mm']:.0f} mm, rotated {pl['rotate']} deg"
               if det is not None and pl else "; no paper placement (old offsets)"),
              f"; Z offset {cfg['z_offset']:+.2f} mm is INCLUDED in every Z below (machine Z)",
              ("; PROBES USED: " + cfg.get("probe_text", "") if cfg.get("probe_status") == "USED"
               else f"; PROBES NOT USED ({cfg.get('probe_status') or 'not run'}): the contact comes from your settings / profile"),
              f"; fill Z {zf:.2f} | thin lines Z " + (f"{zf:.2f} (thinnest) .. {zl:.2f} (widest, {cfg.get('w_ref', 0):.2f} mm wide): VARIABLE depth, deeper = wider"
                                                    if cfg["line_depth"] > 0 else f"{zl:.2f}: constant (variable depth OFF)")
              + f" | travel {cfg['z_low']:.2f} | dip {cfg['dip_z']:.2f} | wipe {cfg['z_wipe_exit']:.2f} | high {cfg['z_high']:.2f}"]
    for name, kinds, nvar in summary:
        x, y, r, note = dishes[name]
        header.append(f"; layer {name}: {sum(kinds.values())} strokes ({kinds['thin']} skeleton + {kinds['patch']} repair = {nvar} centre-line strokes with variable Z; "
                      f"{kinds['lines']} fill lines, {kinds['rings']} rings, {kinds['outline']} outlines); dips at X {x:.1f} Y {y:.1f}"
                      + (f" (pool radius {r:.0f} mm)" if r else ""))
    if cfg.get("curve") is not None:
        header.append(f"; depth from the brush profile: each mark is pressed to the depth where the real brush makes that width "
                      f"(touch at machine Z {cfg['z_paint']:.2f}, at most {cfg['line_depth']:.2f} mm deeper)")
    else:
        header.append(f"; depth follows width (straight line): touch for marks <= {cfg.get('width_min', 0.0):.2f} mm, full depth {cfg['line_depth']:.2f} mm for marks >= {cfg.get('w_ref', 0):.2f} mm")
    header.append(f"; marks narrower than {cfg['thin_max']:.1f} mm = centre-line strokes ({cfg['point_step']:.2f} mm between points, each with its own Z); "
                  f"wider shapes = infill. Painting F{cfg['feed_paint']} accel {cfg['accel_paint']}, plunge F{cfg['plunge_feed']}")
    header.append("; dip: jitter/spiral shrink to the pool; wipe point is " + f"{cfg['wipe_r']:.0f} mm from the dish centre towards the first stroke")
    if cfg.get("tool_offset"):
        header.append(f"; TOOL OFFSET: the probes measured that the paint lands ({cfg['tool_offset'][0]:+.2f}, {cfg['tool_offset'][1]:+.2f}) mm from where the machine is told; "
                      f"every X / Y below is shifted by ({-cfg['tool_offset'][0]:+.2f}, {-cfg['tool_offset'][1]:+.2f}) mm (dips included)")
    header.append("; =========================================")
    painter.gcode[2:2] = header
    code = painter.finish()
    if cfg.get("tool_offset"):
        code = shift_gcode_xy(code, -cfg["tool_offset"][0], -cfg["tool_offset"][1])
    with open(out_gcode, "w") as f:
        f.write(code)
    print(f"G-code: {out_gcode}   ({len(painter.gcode)} lines, {painter.dips} dips in total)")
    if frame is not None:
        from vision import load_calibration
        cal = load_calibration()
        draw_overlay(frame, cal["H"], cal["H_inv"], layers_out, dishes, cfg, det, f"{base}_overlay.png", art=cv2.imread(img_path))
        import shutil
        overlay_ts = f"{base}_overlay_{time.strftime('%H%M%S')}.png"            # a new name every run: viewers cache a file name
        shutil.copyfile(f"{base}_overlay.png", overlay_ts)
        vprint(f"   (the same picture, with the time in its name: {overlay_ts} - this is the one that opens)")
    print(f"planned in {time.time() - t0:.1f} s")
    plan = None
    if frame is not None:
        from vision import load_calibration
        cal = load_calibration()
        plan = {"layers_out": layers_out, "H_inv": cal["H_inv"], "brush_w": cfg["brush_w"],
                "dishes": {n: tuple(d[:3]) for n, d in dishes.items()}}
        if str(getattr(S, "OVERLAY_SHOW", "art")).lower() == "art":           # the live preview shows the artwork's marks, like the overlay
            plan["art_layer"] = _art_marks_layer(cv2.imread(img_path), frame, cfg, cal["H"])
    return code, (overlay_ts if frame is not None else None), plan


# ----------------------------------------------------------------------------------------------- the machine
def connect_and_home(cfg):
    """Connects to the machine (port and speed from settings.py) and homes it: Z, a lift to HOME_LIFT_Z, then XY."""
    return mc.open_machine(home=True, quiet=True)


def go_to_snapshot(machine, cfg, settle=1.0):
    """Lifts the brush and parks the toolhead where it is not in the camera's way."""
    sx, sy = cfg["snapshot"]
    vprint(f"moving the toolhead out of the camera's view: X {sx:.0f} Y {sy:.0f}")
    mc.park(machine, sx, sy, cfg["z_high"], feed=cfg["feed"], accel=cfg["accel_travel"], timeout=60)
    machine.flush_motion()
    time.sleep(settle)                                         # let the picture settle (exposure, vibration)


def capture_fresh(cap, max_s=6.0):
    """Some cameras hand out OLD frames for the first moments after opening (driver buffer). Two live frames are never
    identical (sensor noise), so read until the picture has visibly changed several times. Returns (frame, live?)."""
    t0, prev, changes, frame = time.time(), None, 0, None
    while time.time() - t0 < max_s:
        ok, f = cap.read()
        if not ok:
            continue
        if prev is not None and float(np.mean(np.abs(f.astype(np.int16) - prev.astype(np.int16)))) > 0.02:
            changes += 1
        prev = frame = f
        if changes >= 8 and time.time() - t0 > 0.8:
            break
    return frame, changes >= 3


def stamp_picture(img, text, ok=True):
    """Writes the time the camera picture was taken into the corner, so an old picture cannot pass as the current one."""
    h = img.shape[0]
    col = (0, 255, 0) if ok else (0, 0, 255)
    cv2.putText(img, text, (14, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(img, text, (14, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2, cv2.LINE_AA)


def show_picture(path):
    """Opens the overlay in the default picture viewer (Windows); elsewhere just says where it is."""
    import os
    import sys
    if sys.platform == "win32":
        try:
            os.startfile(os.path.abspath(path))
        except OSError:
            pass


def command_table(code):
    """[(line, layer, dips so far, stroke end)] for every command, and the estimated cumulative seconds after each one
    (distance / feed; accelerations ignored, so it is only a rough guide). stroke end = the leaf / 'line' whose stroke ends with this
    command (from the '; STROKE END' marks), else None."""
    cmds, layer, dips = [], "", 0
    for raw in code.split("\n"):
        l = raw.strip()
        if l.startswith("; ===== LAYER"):
            layer = l.split("LAYER", 1)[1].strip(" =")
        elif l.startswith("; --- DIP CYCLE"):
            dips += 1
        elif l.startswith("; STROKE END") and cmds:
            cmds[-1] = cmds[-1][:3] + (l.split("STROKE END", 1)[1].strip(),)
        elif l and not l.startswith(";"):
            cmds.append((l, layer, dips, None))
    cum, t, feed, pos = [], 0.0, 3000.0, [0.0, 0.0, 0.0]
    for l, *_ in cmds:
        if l.startswith(("G0", "G1")):
            tok = {k: float(v) for k, v in re.findall(r"([XYZF])(-?[\d.]+)", l)}
            feed = tok.get("F", feed)
            new = [tok.get("X", pos[0]), tok.get("Y", pos[1]), tok.get("Z", pos[2])]
            t += math.dist(pos, new) / max(feed, 1.0) * 60.0
            pos = new
        elif l.startswith("G4"):
            t += float((re.findall(r"P(\d+)", l) or ["0"])[0]) / 1000.0
        cum.append(t)
    return cmds, cum


class LivePreview:
    """Camera window while the machine paints - made for screen recording: the live picture, the planned strokes
    (faint, key O switches them off), the layer, progress, elapsed and remaining time. Q closes only this window."""
    def __init__(self, plan, total, total_est, grabber=None, show=True, record=None, crop=None):
        self.grabber, self.show = grabber, show                    # grabber: the camera is held by check.py and shared; show False = no window at all
        self.record, self.crop = record, crop                      # record: time-lapse video file (TIMELAPSE); crop: (x0, y0, x1, y1) camera px
        self.rec_frames = 0
        self.plan, self.total, self.total_est = plan, total, max(total_est, 1.0)
        self.layer, self.dips, self.est_done, self.i = "", 0, 0.0, 0
        self.t0 = time.time()
        self.stop_flag = self.done = False
        self.show_plan = True
        self.thread = None
        self.latest = None                                         # the newest camera frame (the photos take it from here)
        self.win = "paint - painting (Q closes this window, O planned strokes on/off)"
        import h4xx0r as _hx
        self.hx = _hx if _hx.ON else None                             # the h4xx0r theme (--h4xx0r): its lettering and its glass
        if self.hx:
            self.win = "brushography://live_feed  [Q] close  [O] planned strokes"
            self.glass = _hx.Glass()
        self._disp, self._shown, self._win_open, self._last_pump = None, None, False, 0.0

    def start(self):
        import threading
        import machine as _mc
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        _mc.WAIT_HOOK = self.pump                                  # the main thread shows the window while it waits for the machine

    def update(self, i, layer, dips, est_done):
        self.i, self.layer, self.dips, self.est_done = i, layer, dips, est_done
        self.pump()

    def _ui_thread(self):
        """Is this the thread that draws the window? On macOS the MAIN one (it draws windows nowhere else). Elsewhere the camera
        thread, as before: the serial link has no WAIT_HOOK, so the main thread would leave the window frozen during long moves."""
        import sys, threading
        if sys.platform == "darwin":
            return threading.current_thread() is threading.main_thread()
        return threading.current_thread() is self.thread

    def pump(self):
        """Shows the newest picture and reads the keys, from the drawing thread only (_ui_thread): on macOS the camera thread
        only prepares the picture (self._disp) and this is called for every G-code line and while waiting for the machine."""
        if not self.show or not self._ui_thread():
            return
        now = time.time()
        if now - self._last_pump < 0.03:
            return
        self._last_pump = now
        disp = self._disp
        if disp is None:
            return
        try:
            if not self._win_open:
                cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(self.win, 1280, 720)
                self._win_open = True
            if disp is not self._shown:
                cv2.imshow(self.win, disp)
                self._shown = disp
            k = cv2.waitKey(1) & 0xFF
        except Exception as e:                                     # painting must not depend on the window
            print(f"(live preview not available: {e})")
            self.show = False
            return
        if k in (ord("q"), ord("Q"), 27):
            self.show = False                                       # the window goes; a time-lapse keeps recording
            self._close_window()
        if k in (ord("o"), ord("O")):
            self.show_plan = not self.show_plan

    def _close_window(self):
        if self._win_open and self._ui_thread():
            try:
                cv2.destroyWindow(self.win)
                cv2.waitKey(1)
            except Exception:
                pass
            self._win_open = False

    def finish(self, wait_s=20.0):
        """Painting is over: show 'finished' for a while (or until Q), then close."""
        self.done = True
        if not self.show:
            wait_s = 0.0                                              # no window to look at: just finish the recording
        t = time.time()
        while self.thread and self.thread.is_alive() and self.show and time.time() - t < wait_s:
            self.pump()
            time.sleep(0.03)
        self.stop()

    def stop(self):
        import machine as _mc
        if _mc.WAIT_HOOK == self.pump:
            _mc.WAIT_HOOK = None
        self.stop_flag = True
        if self.thread:
            self.thread.join(timeout=3)
        self._close_window()

    def _plan_layer(self, shape):
        """The planned strokes (or the artwork's marks, OVERLAY_SHOW "art") drawn once on a transparent layer in camera pixels."""
        art = self.plan.get("art_layer")
        if art is not None and art[0].shape[:2] == shape[:2]:
            return art
        h, w = shape[:2]
        img = np.zeros((h, w, 3), np.uint8)
        mask = np.zeros((h, w), np.uint8)
        Hi = self.plan["H_inv"]
        first = self.plan["layers_out"][0][1][0][0]
        a = mm_to_px_array(Hi, [first])[0]
        b = mm_to_px_array(Hi, [(first[0] + 10.0, first[1])])[0]
        ppm = float(np.hypot(*(b - a))) / 10.0                                     # camera pixels per mm
        th0 = max(1, int(round(self.plan["brush_w"] * ppm)))
        for name, paths in self.plan["layers_out"]:
            col = LAYER_BGR.get(name, (0, 0, 255))
            for p in paths:
                pts = mm_to_px_array(Hi, p)
                draw_stroke(img, pts, getattr(p, "widths", None), ppm, col, th0, mask=mask)
        return img, mask > 0

    def _loop(self):
        cap = None
        writer, last_rec, last_small = None, 0.0, None
        every = float(getattr(S, "TIMELAPSE_EVERY_S", 2.0))
        try:
            from vision import open_camera
            cap = None if self.grabber is not None else open_camera(verbose=False)
            layer_img = layer_mask = None
            prev_small, same = None, 0
            while not self.stop_flag:
                ok, frame = (self.grabber.read() if self.grabber is not None else cap.read())
                if not ok:
                    time.sleep(0.05)
                    continue
                self.latest = frame
                if self.record and time.time() - last_rec >= every:
                    # time-lapse: one frame every TIMELAPSE_EVERY_S, the clean camera picture cropped to the paper; a frame that hardly
                    # differs from the last one written (drying, waiting) is left out, so the video shows only the painting
                    last_rec = time.time()
                    x0, y0, x1, y1 = self.crop or (0, 0, frame.shape[1], frame.shape[0])
                    fr = frame[y0:y1, x0:x1]
                    fr = cv2.resize(fr, (fr.shape[1] // 2 * 2, fr.shape[0] // 2 * 2))
                    sm = cv2.resize(fr, (64, 48)).astype(np.int16)
                    if last_small is None or float(np.abs(sm - last_small).mean()) > float(getattr(S, "TIMELAPSE_MIN_CHANGE", 0.4)):
                        if writer is None:
                            writer = cv2.VideoWriter(self.record, cv2.VideoWriter_fourcc(*"mp4v"), float(getattr(S, "TIMELAPSE_FPS", 15)),
                                                     (fr.shape[1], fr.shape[0]))
                        writer.write(fr)
                        self.rec_frames += 1
                        last_small = sm
                if not self.show:
                    time.sleep(0.03)
                    continue
                small = cv2.resize(frame, (96, 54)).astype(np.int16)
                same = same + 1 if (prev_small is not None and float(np.abs(small - prev_small).mean()) < 0.02) else 0
                prev_small = small
                disp = frame.copy()
                if self.plan and self.show_plan:
                    if layer_img is None:
                        layer_img, layer_mask = self._plan_layer(frame.shape)
                    disp[layer_mask] = (0.65 * disp[layer_mask] + 0.35 * layer_img[layer_mask]).astype(np.uint8)
                h, w = disp.shape[:2]
                frac = 1.0 if self.done else min(1.0, self.est_done / self.total_est)
                el = time.time() - self.t0
                left = (el * (1 - frac) / frac) if frac > 0.02 else None
                if self.hx:                                         # the h4xx0r theme: its lettering, then the glass over all of it
                    small = cv2.resize(disp, (1280, int(round(1280 * h / w))), interpolation=cv2.INTER_AREA)
                    info = (f"t+{int(el // 60):02d}:{int(el % 60):02d}" +
                            (f"  eta {int(left // 60):02d}:{int(left % 60):02d}" if left is not None and not self.done else "") +
                            f"  dips {self.dips}  line {self.i}/{self.total}")
                    themed = self.hx.hud(small, "job complete" if self.done else f"layer::{self.layer or '-'}", info,
                                         time.strftime("%H:%M:%S"), frac, done=self.done, frozen=same > 45, t=el)
                    if themed is not None:
                        self._disp = self.glass.apply(themed)
                        self.pump()                                 # draws here except on macOS
                        time.sleep(0.03)
                        continue
                bar = disp.copy()
                cv2.rectangle(bar, (0, 0), (w, 70), (0, 0, 0), -1)
                cv2.rectangle(bar, (0, h - 22), (w, h), (0, 0, 0), -1)
                disp = cv2.addWeighted(bar, 0.6, disp, 0.4, 0)
                cv2.rectangle(disp, (0, h - 22), (int(w * frac), h), (0, 200, 0), -1)
                title = "FINISHED" if self.done else f"layer: {self.layer or '-'}"
                info = (f"{100 * frac:.0f}%   elapsed {int(el // 60):02d}:{int(el % 60):02d}" +
                        (f"   about {int(left // 60):02d}:{int(left % 60):02d} left" if left is not None and not self.done else "") +
                        f"   dips {self.dips}   line {self.i}/{self.total}")
                cv2.putText(disp, title, (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(disp, info, (16, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 255, 200), 2, cv2.LINE_AA)
                cv2.putText(disp, time.strftime("%H:%M:%S"), (w - 170, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
                if same > 45:                                       # about 1.5 s of exactly identical frames
                    cv2.putText(disp, "CAMERA PICTURE FROZEN?", (w // 2 - 230, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 4, cv2.LINE_AA)
                self._disp = disp                                   # shown by pump(): here, or on macOS by the main thread
                self.pump()
                if not self.show and not self.record:
                    break
                time.sleep(0.03)
        except Exception as e:                                           # no window / camera busy: painting must not depend on it
            print(f"(live preview not available: {e})")
        finally:
            if writer is not None:
                for _ in range(int(float(getattr(S, "TIMELAPSE_FPS", 15)) * 3)):    # hold the last picture for 3 s
                    writer.write(fr)
                writer.release()
                print(f"time-lapse: {self.record} ({self.rec_frames} frames)")
            if cap is not None:
                cap.release()
            self._close_window()      # only where this thread draws; on macOS stop() closes it from the main thread


def _gif_frame(frame, paper_poly, n, total):
    """A small picture of the sheet for the GIF: cropped to the paper, 640 px wide, with a counter."""
    from PIL import Image, ImageDraw
    h, w = frame.shape[:2]
    if paper_poly is not None:
        pts = np.round(np.array(paper_poly)).astype(int)
        x0, y0 = max(0, pts[:, 0].min() - 20), max(0, pts[:, 1].min() - 20)
        x1, y1 = min(w, pts[:, 0].max() + 20), min(h, pts[:, 1].max() + 20)
        frame = frame[y0:y1, x0:x1]
    im = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    im = im.resize((640, max(1, int(im.height * 640 / im.width))), Image.LANCZOS)
    d = ImageDraw.Draw(im)
    d.rectangle((0, 0, 150, 22), fill=(255, 255, 255))
    d.text((6, 5), f"photo {n}/{total}", fill=(0, 0, 0))
    return im


def dry_wait(minutes, label):
    """Waits for the paint to dry, with a countdown in this window every minute. Press R (or Enter) here to go on at once.
    Ctrl+C still stops the program."""
    end = time.time() + minutes * 60.0
    print(f"\n   DRYING before the picture of {label}: {minutes:g} min. Press R (or Enter) here to continue now.")
    try:
        import msvcrt                                   # Windows: reads the keyboard without waiting
    except ImportError:
        msvcrt = None
    while msvcrt is not None and msvcrt.kbhit():           # keys pressed earlier (the Enter after "Start painting now?") do not count
        msvcrt.getwch()
    next_print = time.time()
    while True:
        left = end - time.time()
        if left <= 0:
            print("   drying finished: taking the picture")
            return
        if time.time() >= next_print:
            print(f"   drying: {left / 60.0:5.1f} min left")
            next_print = time.time() + 300.0
        if msvcrt is not None and msvcrt.kbhit():
            ch = msvcrt.getwch()
            if ch.lower() == "r" or ch in ("\r", "\n"):
                print("   continued by you")
                return
        time.sleep(0.5)


def repair_stop(gaps):
    """The stop rule of the gap repair, given the empty pixels seen so far (the finished sheet first, then after each pass):
    the reason to stop here, or None (one more pass). 2026-11-08 readings, 1920x1080 camera: 17194 7048 3617 2712 2281 1994
    1904 1872 and 1404 339 179 126 - from the 4th pass on each pass kept 84-98 % (specks the brush does not fill), about
    2-3 min each, and the job had to be stopped by hand."""
    g = gaps[-1]
    if g < int(getattr(S, "REPAIR_GAP_DONE_PX", 150)):
        return "done"
    if len(gaps) >= 2 and g > float(getattr(S, "REPAIR_MIN_PROGRESS", 0.8)) * gaps[-2]:
        return "no real progress"
    if len(gaps) - 1 >= int(getattr(S, "REPAIR_PASSES", 6)):
        return "REPAIR_PASSES reached"
    return None


# ============================ REPAIR BY GAPS (the artwork against the photo) ============================
def _paper_box(hsv):
    pap = ((hsv[..., 1] < 70) & (hsv[..., 2] > 160)).astype(np.uint8)
    pap = cv2.morphologyEx(pap, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(pap)
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    return st[k, :4]


def _paint_seen(frame):
    """The paper box in the photo, and where it carries paint: clearly more coloured OR clearly darker than the bare paper
    (pale paint is still paint)."""
    H_, W_ = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    px, py, pw, ph_ = _paper_box(hsv)
    xs = np.arange(W_)[None, :]; ys = np.arange(H_)[:, None]
    roi = (xs > px) & (xs < px + pw) & (ys > py) & (ys < py + ph_)
    bare = roi & (hsv[..., 1] < 70)
    paper_s = float(np.median(hsv[..., 1][bare])); paper_v = float(np.median(hsv[..., 2][bare]))
    paint_m = ((hsv[..., 1] > paper_s + 35) | (hsv[..., 2] < paper_v - 35)) & roi
    return roi, paint_m, (px, py, pw, ph_)


def _art_from_plan(art, frame, cfg, H):
    """The artwork in the photo exactly where the PLAN put it: every camera pixel -> machine mm (the calibration, lens included)
    -> the artwork plane of the placement -> the artwork pixel. The paint lands there because the G-code is corrected by the
    measured tool offset. Returns (art as planned (rotated), map_x, map_y) or None when there is no placement."""
    pl = cfg.get("placement")
    if not pl:
        return None
    art_r = cv2.rotate(art, cv2.ROTATE_90_COUNTERCLOCKWISE) if int(pl.get("rotate", 0)) == 90 else art   # as load_rgb (PIL rotate 90)
    ah, aw = art_r.shape[:2]
    H_, W_ = frame.shape[:2]
    from vision import placement_corners_mm
    c = mm_to_px_array(np.linalg.inv(H), placement_corners_mm(pl))
    x0, y0 = np.clip(np.floor(c.min(axis=0)).astype(int) - 3, 0, None)
    x1, y1 = np.minimum(np.ceil(c.max(axis=0)).astype(int) + 3, [W_ - 1, H_ - 1])
    ys, xs = np.mgrid[y0:y1 + 1, x0:x1 + 1]
    mm = px_to_mm_array(H, np.c_[xs.ravel(), ys.ravel()])
    d = mm - np.array(pl["origin_mm"])
    a, b = d @ np.array(pl["u"]), d @ np.array(pl["v"])                  # artwork-plane mm: x right, y up (row 0 of load_rgb = bottom)
    map_x = np.full((H_, W_), -1.0, np.float32); map_y = np.full((H_, W_), -1.0, np.float32)
    map_x[y0:y1 + 1, x0:x1 + 1] = (a / pl["width_mm"] * aw - 0.5).reshape(xs.shape)
    map_y[y0:y1 + 1, x0:x1 + 1] = ((pl["height_mm"] - b) / pl["height_mm"] * ah - 0.5).reshape(xs.shape)
    return art_r, map_x, map_y


def _align_art(art, frame):
    """The artwork placed on the paper: scale and position from the paper, refined by matching the ink.
    Only used when there is no placement (the plan is the better answer: _art_from_plan)."""
    H_, W_ = frame.shape[:2]
    roi, paint_m, (px, py, pw, ph_) = _paint_seen(frame)
    ahsv = cv2.cvtColor(art, cv2.COLOR_BGR2HSV)
    ink_a = ((ahsv[..., 1] > 90) & (ahsv[..., 2] > 40)).astype(np.float32)
    d = 2
    P = cv2.resize(paint_m.astype(np.float32), (W_ // d, H_ // d), interpolation=cv2.INTER_AREA) > 0.3
    best = (-1.0, None)
    s0 = (pw * 274.0 / 295.0) / art.shape[1]                     # a first guess only (artwork 274 mm on a 295 mm sheet): the search below fits it
    for fs in np.linspace(0.9, 1.1, 5):
        sc = s0 * fs; aw, ah = art.shape[1] * sc, art.shape[0] * sc
        for dx in range(-40, 41, 8):
            for dy in range(-40, 41, 8):
                x0 = px + (pw - aw) / 2.0 + dx; y0 = py + (ph_ - ah) / 2.0 + dy
                M = np.array([[sc / d, 0, x0 / d], [0, sc / d, y0 / d]], np.float32)
                A = cv2.warpAffine(ink_a, M, (W_ // d, H_ // d)) > 0.5
                iou = (A & P).sum() / max((A | P).sum(), 1)
                if iou > best[0]:
                    best = (iou, (sc, x0, y0))
    sc, x0, y0 = best[1]
    M = np.array([[sc, 0, x0], [0, sc, y0]], np.float32)
    warp0 = cv2.warpAffine(ink_a, M, (W_, H_))
    warp = np.eye(2, 3, dtype=np.float32)
    try:
        _, warp = cv2.findTransformECC(paint_m.astype(np.float32), warp0, warp, cv2.MOTION_AFFINE,
                                       (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-5), None, 5)
        A3 = np.vstack([M, [0, 0, 1]]); W3 = np.vstack([warp, [0, 0, 1]])
        M = (np.linalg.inv(W3) @ A3)[:2].astype(np.float32)
    except cv2.error:
        pass
    return M, roi, paint_m, ink_a


def _colour_of(hue, sat, val, layer_hues=None):
    """Which colour layer an artwork pixel belongs to: the layer of the job whose hue is nearest (layer_hues = {name: hue}).
    Without the job's layers: the old fixed red / blue."""
    if sat < 90 or val < 40:
        return None
    if layer_hues:
        name, hue_l = min(layer_hues.items(), key=lambda kv: min(abs(hue - kv[1]), 180 - abs(hue - kv[1])))
        return name if min(abs(hue - hue_l), 180 - abs(hue - hue_l)) <= 25 else None
    if hue < 20 or hue > 160:
        return "red"
    if 95 <= hue <= 140:
        return "blue"
    return None


def _art_black(art_bgr, thr=140):
    """Where the artwork is black, by the same rule as find_layers: dark and without real colour (black ink with a blue cast too)."""
    hsv = cv2.cvtColor(art_bgr, cv2.COLOR_BGR2HSV)
    ss, vv = hsv[..., 1].astype(float), hsv[..., 2].astype(float)
    gray = cv2.cvtColor(art_bgr, cv2.COLOR_BGR2GRAY)
    chroma = ss * vv / 255.0
    chromatic = (ss >= MIN_SAT) & (vv >= MIN_VAL)
    chromatic &= ~((chroma < float(getattr(S, "DARK_CHROMA", 70.0))) & ((vv < 170) | (ss < 90)))
    return ((~chromatic) & (gray < thr)) | (vv < MIN_VAL)


LAST_GAP = None
LAST_ART = None                                                  # where the artwork has colour in the last gap photo (camera pixels)


def aim_wipe(block, tx, ty, wipe_r):
    """The dip cycle (copied from the job) with its wipe turned towards (tx, ty). The copy wipes towards the JOB's first stroke:
    the repair then went there first and turned the corner to its own stroke, an L-shaped trip (2026-11-07). The wipe is the last
    XY move of the block, the dish centre its first."""
    xy = [i for i, l in enumerate(block) if l.startswith("G0 X")]
    if len(xy) < 2:
        return block
    c = dict(re.findall(r"([XY])(-?[\d.]+)", block[xy[0]]))
    cx, cy = float(c["X"]), float(c["Y"])
    d = math.hypot(tx - cx, ty - cy)
    if d < 1e-6:
        return block
    wx, wy = cx + (tx - cx) / d * wipe_r, cy + (ty - cy) / d * wipe_r
    tail = re.search(r"\sF\S+", block[xy[-1]])
    out = list(block)
    out[xy[-1]] = f"G0 X{wx:.3f} Y{wy:.3f}" + (tail.group(0) if tail else "")
    return out


def taper_repair_stroke(pts, ws, inside, w_min, step=0.4, along=None):
    """A repair stroke that blends into the paint around it instead of sitting on it as a block with square ends (the 'stairs',
    2026-11-08 sketch): its widths are smoothed along it, and each end is carried on in its own direction over the paint already
    there - only while inside(x, y), the artwork's shape - with the width falling smoothly (cosine) to w_min, the thinnest mark.
    The depth follows the width, so the brush glides in and out in an arc. The run is REPAIR_TAPER_RATIO x the stroke's widest
    point, within REPAIR_TAPER_MIN_MM .. REPAIR_TAPER_MAX_MM. along(x, y, d) False = the run would cut ACROSS the painted lines
    there: it stops (in a solid black area painted as parallel lines, a straight run crossed the next lines, 2026-11-08 ko/71).
    pts every `step` mm, ws (mm) one per point. Returns (pts, ws)."""
    from scipy.ndimage import gaussian_filter1d
    ws = np.asarray(ws, float)
    if len(ws) >= 3:
        ws = gaussian_filter1d(ws, max(1.0, 1.0 / step), mode="nearest")                  # no plateaus with corners (sigma 1 mm)
    run = float(np.clip(float(getattr(S, "REPAIR_TAPER_RATIO", 2.5)) * float(ws.max()),
                        float(getattr(S, "REPAIR_TAPER_MIN_MM", 1.5)), float(getattr(S, "REPAIR_TAPER_MAX_MM", 6.0))))

    def carry(p_end, d, w_end):                    # the run beyond one end: points outward from it, with their widths
        xs, n = [], int(run / step)
        for i in range(1, n + 1):
            q = (p_end[0] + d[0] * i * step, p_end[1] + d[1] * i * step)
            if not inside(*q) or (along is not None and not along(q[0], q[1], d)):
                break
            xs.append(q)
        if not xs:
            return [], []
        t = np.arange(1, len(xs) + 1) / len(xs)                                            # 0 at the gap .. 1 at the far tip
        return xs, list(w_min + (w_end - w_min) * 0.5 * (1.0 + np.cos(np.pi * t)))
    pa, wa = carry(pts[0], end_dir(pts), float(ws[0]))
    pb, wb = carry(pts[-1], end_dir(pts[::-1]), float(ws[-1]))
    return pa[::-1] + list(pts) + pb, wa[::-1] + list(ws) + wb


def gap_repair_gcode(frame, art, H, dip_blocks, cfg, extra_mm=0.0, persist=None):
    global LAST_GAP
    """G-code for the EMPTY spots: where the artwork has colour and the photo shows none. Each colour is dipped into its own
    dish (the dip cycle of its own layer). Returns (lines, empty_pixels, {colour: pixels})."""
    global LAST_ART
    H_img, W_img = frame.shape[:2]
    placed = _art_from_plan(art, frame, cfg, H)
    if placed is not None:                                        # the artwork where the plan put it
        art_r, map_x, map_y = placed
        roi, paint_m, _ = _paint_seen(frame)
        ahsv_r = cv2.cvtColor(art_r, cv2.COLOR_BGR2HSV)
        ink_r = ((ahsv_r[..., 1] > 90) & (ahsv_r[..., 2] > 40))
        if "black" in dip_blocks:                                 # a black layer: its dark pixels are ink too (they have no saturation)
            ink_r |= _art_black(art_r)
        ink_r = ink_r.astype(np.float32)
        ink_f = cv2.remap(ink_r, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0.5
    else:                                                         # no placement: fitted to the paint (older method)
        M, roi, paint_m, ink_a = _align_art(art, frame)
        art_r = art
        ink_f = cv2.warpAffine(ink_a, M, (W_img, H_img)) > 0.5
        Minv = cv2.invertAffineTransform(M)
        yy, xx = np.mgrid[0:H_img, 0:W_img].astype(np.float32)
        map_x = (Minv[0, 0] * xx + Minv[0, 1] * yy + Minv[0, 2]).astype(np.float32)
        map_y = (Minv[1, 0] * xx + Minv[1, 1] * yy + Minv[1, 2]).astype(np.float32)
    LAST_ART = ink_f
    empty = ink_f & roi & ~paint_m
    faded = np.zeros_like(empty)
    # FADED black: painted, but much paler than this job's well-painted strokes - a second coat there (2026-11-08). Darkness =
    # how far below the bare paper the brightness is, over about 1 mm (the banding along a stroke is not a fade); the reference
    # is the dark end (20th percentile) of the painted black. REPAIR_FADED_SHARE = 0 / None switches it off
    fade_share = float(getattr(S, "REPAIR_FADED_SHARE", 0.5) or 0.0)
    if fade_share > 0 and "black" in dip_blocks:
        hsv_f = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        bare = roi & ~paint_m & (hsv_f[..., 1] < 70)
        if bare.any():
            paper_v = float(np.median(hsv_f[..., 2][bare]))
            p0, p1 = px_to_mm_array(H, [(W_img / 2.0, H_img / 2.0), (W_img / 2.0 + 1, H_img / 2.0)])
            sig = max(1.0, 0.5 / max(float(np.linalg.norm(p1 - p0)), 1e-9))           # 0.5 mm in camera pixels
            dark = cv2.GaussianBlur(np.clip(paper_v - hsv_f[..., 2].astype(np.float32), 0, None), (0, 0), sig)
            black_f = cv2.remap(_art_black(art_r).astype(np.float32), map_x, map_y, cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0.5
            ref_sel = black_f & paint_m & roi
            if ref_sel.sum() > 200:
                ref = float(np.percentile(dark[ref_sel], 80))                  # the darkest fifth of the painted black
                faded = black_f & paint_m & roi & (dark < fade_share * ref)
                empty |= faded
    cx, cy = W_img / 2.0, H_img / 2.0
    p0, p1 = px_to_mm_array(H, [(cx, cy), (cx + 1, cy)])
    px_per_mm = 1.0 / max(float(np.linalg.norm(p1 - p0)), 1e-9)
    # is the paint where the plan put it? If the whole painting sits shifted (tip offset not applied, paper moved), the "empty"
    # spots are a strip along every shape: repainting them would only paint lines beside the shapes. Then nothing is repainted
    (sdx, sdy), _ = cv2.phaseCorrelate(ink_f.astype(np.float32), (paint_m & roi).astype(np.float32))
    shift_mm = float(np.hypot(sdx, sdy)) / px_per_mm
    limit = float(getattr(S, "REPAIR_MAX_SHIFT_MM", 1.5))
    if shift_mm > limit:
        print(f"   REPAIR SKIPPED: the paint sits {shift_mm:.1f} mm away from where the plan put it (more than REPAIR_MAX_SHIFT_MM = {limit:g}).\n"
              f"   The white strips along the shapes are misplacement, not missing paint. Check the tip offset in the G-code header / the probes.")
        LAST_GAP = None
        return None, int(empty.sum()), {}
    brush_px = float(cfg["brush_w"]) * px_per_mm
    rim_px = float(getattr(S, "REPAIR_RIM_MM", 1.9)) * px_per_mm               # strips thinner than this along an edge are not 'empty' (stops overpainting edges)
    k = max(3, int(round(rim_px)) | 1)                           # thin rims ALONG THE OUTLINE are dropped (no overpainting of edges) ...
    thick_only = cv2.morphologyEx(empty.astype(np.uint8), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) > 0
    # ... but inside a shape a thin white strip (between veins) is a real gap: only the band within REPAIR_RIM_MM of the outline is filtered
    d_in = cv2.distanceTransform(ink_f.astype(np.uint8), cv2.DIST_L2, 5)
    band_px = float(getattr(S, "REPAIR_EDGE_BAND_MM", 0.5)) * px_per_mm
    empty = thick_only | (empty & (d_in > band_px))
    # pinholes (dry brush: many 1 mm2 specks close together) are each too small to repaint, but a patch full of them is thin paint:
    # specks closer than REPAIR_SPECK_MERGE_MM merge into one area, which is then repainted as a whole (a second coat there)
    merge = float(getattr(S, "REPAIR_SPECK_MERGE_MM", 1.5)) * px_per_mm
    if merge >= 1.0:
        km = max(3, int(round(merge)) | 1)
        closed = cv2.morphologyEx(empty.astype(np.uint8), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (km, km))) > 0
        empty = empty | (closed & ink_f & (d_in > band_px))
    min_area = float(getattr(S, "REPAIR_MIN_GAP_MM2", 3.0)) * px_per_mm * px_per_mm      # smaller empty specks are left alone
    nlab, labs, stats, _ = cv2.connectedComponentsWithStats(empty.astype(np.uint8), connectivity=8)
    keep = np.zeros(nlab, bool); keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    empty = keep[labs]
    if persist is not None:                                       # only spots that were ALSO empty in the photo before
        empty = empty & persist
    # missed LINES: a thin line (narrower than REPAIR_LINE_MAX_MM) is all "edge band", so the rules above never see it. Instead: the
    # centre line of every thin line of the artwork, where no paint lies within REPAIR_LINE_NEAR_MM of it, is repainted as a line
    line_max = float(getattr(S, "REPAIR_LINE_MAX_MM", 2.5)) * px_per_mm
    thick_art = cv2.dilate((d_in > 0.5 * line_max).astype(np.uint8),
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(line_max) | 1, int(line_max) | 1))) > 0
    sk = skeletonize(ink_f & roi & ~thick_art)
    nr = max(3, int(round(2 * float(getattr(S, "REPAIR_LINE_NEAR_MM", 0.7)) * px_per_mm)) | 1)
    missed = sk & ~(cv2.dilate(paint_m.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (nr, nr))) > 0)
    nl, ll, sl, _ = cv2.connectedComponentsWithStats(missed.astype(np.uint8), connectivity=8)
    keep_l = np.zeros(nl, bool); keep_l[1:] = sl[1:, cv2.CC_STAT_AREA] >= float(getattr(S, "REPAIR_MIN_LINE_MM", 2.0)) * px_per_mm
    missed = keep_l[ll]
    empty = empty | ((cv2.dilate(missed.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0) & ink_f)
    LAST_GAP = empty.copy()
    ahsv = cv2.cvtColor(art_r, cv2.COLOR_BGR2HSV)
    ys, xs = np.nonzero(empty)
    if xs.size == 0:
        return [], 0, {}
    axi = np.clip(np.round(map_x[ys, xs]).astype(int), 0, art_r.shape[1] - 1)
    ayi = np.clip(np.round(map_y[ys, xs]).astype(int), 0, art_r.shape[0] - 1)
    hs = ahsv[ayi, axi]
    lh = {k: v for k, v in (cfg.get("layer_hues") or {}).items() if k in dip_blocks and v is not None}
    colour = np.array([_colour_of(int(h), int(s_), int(v), lh) or "" for h, s_, v in hs], dtype=object)
    if set(dip_blocks) == {"black"}:                              # one paint (a black job): every empty spot is black
        colour[:] = "black"
    elif "black" in dip_blocks:                                   # black pixels go to the black dish, whatever their tint
        colour[_art_black(art_r)[ayi, axi]] = "black"
    if "black" in dip_blocks:
        lh.setdefault("black", None)                              # black has no hue: without this the repair paints no layer at all
    mm = px_to_mm_array(H, np.c_[xs, ys])                         # the lens correction included (as the calibration)
    cd, cw = cfg["curve"][0], cfg["curve"][1]
    contact = float(cfg["z_paint"])
    z_travel = contact + float(getattr(S, "REPAIR_TRAVEL_MM", getattr(S, "TRAVEL_MM", 3.0)))
    wmax = float(cw[-1])
    feed_p = int(cfg.get("feed_paint", getattr(S, "PAINT_FEED", 3000))); feed_d = int(getattr(S, "PLUNGE_FEED", 2000))
    # the same accelerations as the painting (Painter._set_speed): the dip cycle leaves the fast travel one set
    acc_t = int(cfg.get("accel_travel", getattr(S, "ACCEL_TRAVEL", 12000))); acc_p = int(cfg.get("accel_paint", getattr(S, "PAINT_ACCEL", 200)))
    z_high = float(cfg["z_high"])

    def z_of(w, extra=extra_mm):
        d = float(np.interp(min(float(w), wmax), cw, cd))
        return max(0.3, contact - d - extra)

    cfg2 = dict(cfg); cfg2["outline"] = False
    cfg2["infill_style"] = "lines"                                 # small spots: plain strokes, not leaves (fewer, longer moves)
    if cfg2.get("line_depth") is None:
        cfg2["line_depth"] = round(max(min(float(cd[-1]), contact - 0.2), 0.0), 2)
    # a repair stroke is landed and lifted like every line (Painter._land / _lift: gliding in and out, LINE_LEAD_IN_MM /
    # LINE_LAND_HIGH_MM), never asks for more than REPAIR_WIDTH_SHARE of the widest mark, and specks shorter than
    # REPAIR_MIN_STROKE_MM are left
    min_len = float(getattr(S, "REPAIR_MIN_STROKE_MM", 2.0))
    w_rep = float(getattr(S, "REPAIR_WIDTH_SHARE", 0.9)) * wmax
    lead = float(getattr(S, "LINE_LEAD_IN_MM", 3.0))
    air = contact + float(getattr(S, "LINE_LAND_HIGH_MM", 2.0))
    tilt = cfg.get("tilt")                                         # the paper is not level: every repair Z follows it, like the painting
    feed_t = int(cfg.get("feed", 12000))

    def tilted(z, x, y):
        return z + tilt[0] * (x - tilt[2]) + tilt[1] * (y - tilt[3]) if tilt else z
    layers_r = list(lh) if lh else list(dip_blocks)                # the colours this job painted
    # REPAIR_TAPER: each repair stroke runs on over the paint around it, inside the artwork's shape, thinning out (taper_repair_stroke)
    taper = bool(getattr(S, "REPAIR_TAPER", True))
    pad = 2.0 + (float(getattr(S, "REPAIR_TAPER_MAX_MM", 6.0)) if taper else 0.0)          # room on the grid for the runs
    if taper:
        # one paint (a monochrome job): the runs may also go over any paint already on the sheet (paint over the same paint does no
        # harm; a run that left the artwork's edge on a curve, or along a stroke that spread a little, stopped short - stairs again).
        # Several colours: only inside the artwork's shape, so a run never goes over a different colour
        mono = len(layers_r) == 1
        iy, ix = np.nonzero(((ink_f | paint_m) if mono else ink_f) & roi)
        ink_mm = px_to_mm_array(H, np.c_[ix, iy]) if ix.size else np.zeros((0, 2))
        py_, px_ = np.nonzero(paint_m & roi)                       # the paint on the sheet: which way its lines run (along)
        paint_mm = px_to_mm_array(H, np.c_[px_, py_]) if px_.size else np.zeros((0, 2))
        cross_cos = math.cos(math.radians(float(getattr(S, "REPAIR_TAPER_MAX_TURN_DEG", 30.0))))
    out, counts = [f"; REPAIR BY GAPS: empty spots only, {extra_mm:.2f} mm deeper than the first pass"], {k: 0 for k in layers_r}
    for layer in layers_r:
        sel = colour == layer
        if not sel.any():
            continue
        m = mm[sel]
        origin = m.min(axis=0) - pad
        Wg = int(math.ceil((m[:, 0].max() + pad - origin[0]) * RES)) + 4
        Hg = int(math.ceil((m[:, 1].max() + pad - origin[1]) * RES)) + 4
        grid = np.zeros((Hg, Wg), bool)
        gx = np.clip(((m[:, 0] - origin[0]) * RES).astype(int), 0, Wg - 1)
        gy = np.clip(((m[:, 1] - origin[1]) * RES).astype(int), 0, Hg - 1)
        grid[gy, gx] = True
        grid = cv2.morphologyEx(grid.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)) > 0
        grid_f = np.zeros((Hg, Wg), bool)                          # which of it is FADED paint (not white): no extra depth there
        fsel = faded[ys, xs][sel]
        grid_f[gy[fsel], gx[fsel]] = True
        grid_f = cv2.dilate(grid_f.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        inside = along = None
        if taper:                                                  # the artwork's shape on the same grid: the runs stay in it
            ink_g = np.zeros((Hg, Wg), np.uint8)
            q = ((ink_mm - origin) * RES).astype(int)
            ok_q = (q[:, 0] >= 0) & (q[:, 0] < Wg) & (q[:, 1] >= 0) & (q[:, 1] < Hg)
            ink_g[q[ok_q, 1], q[ok_q, 0]] = 1
            ink_g = cv2.morphologyEx(ink_g, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)) > 0     # camera pixels are coarser than the grid

            def inside(x, y, ink_g=ink_g, origin=origin):
                gx_, gy_ = int((x - origin[0]) * RES), int((y - origin[1]) * RES)
                return 0 <= gx_ < Wg and 0 <= gy_ < Hg and bool(ink_g[gy_, gx_])
            # which way the painted lines run around each point (structure tensor of the paint, about 1.5 mm around), and how clearly
            pg = np.zeros((Hg, Wg), np.float32)
            q = ((paint_mm - origin) * RES).astype(int)
            ok_q = (q[:, 0] >= 0) & (q[:, 0] < Wg) & (q[:, 1] >= 0) & (q[:, 1] < Hg)
            pg[q[ok_q, 1], q[ok_q, 0]] = 1.0
            pg = cv2.GaussianBlur(pg, (0, 0), 0.5 * RES)
            dgx, dgy = cv2.Sobel(pg, cv2.CV_32F, 1, 0), cv2.Sobel(pg, cv2.CV_32F, 0, 1)
            jxx, jyy, jxy = (cv2.GaussianBlur(v, (0, 0), 1.5 * RES) for v in (dgx * dgx, dgy * dgy, dgx * dgy))
            phi = 0.5 * np.arctan2(2 * jxy, jxx - jyy) + 0.5 * np.pi           # the lines' direction (across the gradient)
            coh = np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2) / (jxx + jyy + 1e-9)
            clear = (coh > 0.3) & (jxx + jyy > 0.05 * float((jxx + jyy).max() or 1.0))

            def along(x, y, d, phi=phi, clear=clear, origin=origin):
                gx_, gy_ = int((x - origin[0]) * RES), int((y - origin[1]) * RES)
                if not (0 <= gx_ < Wg and 0 <= gy_ < Hg) or not clear[gy_, gx_]:
                    return True                                    # no clear line direction here: no limit
                return abs(d[0] * math.cos(phi[gy_, gx_]) + d[1] * math.sin(phi[gy_, gx_])) >= cross_cos
        strokes, _ = plan_strokes(grid, cfg2, res=RES)
        # how wide the gap is around each point: a stroke is pressed only that wide, so it fills a narrow gap without painting
        # over the strokes on either side of it
        gap_w = 2.0 * cv2.distanceTransform(grid.astype(np.uint8), cv2.DIST_L2, 5) / RES
        dip = dip_blocks.get(layer)
        if not dip:
            continue
        counts[layer] = int(sel.sum())
        out.append(f"; ----- {layer}: {len(strokes)} gap strokes -----")
        for b in range(0, len(strokes), 8):
            out.append(f"; {layer} batch {b // 8 + 1}")
            t0 = strokes[b][0]
            out.extend(aim_wipe(dip, origin[0] + t0[0] / RES, origin[1] + t0[1] / RES, float(cfg["wipe_r"])))
            after_dip = True
            for st in strokes[b:b + 8]:
                pts = [(origin[0] + x / RES, origin[1] + y / RES) for x, y in st]
                L = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(pts, pts[1:]))
                if len(pts) < 2 or L < min_len:
                    continue
                # mostly over FADED paint: a plain second coat at the planned depth; over white: this pass's extra depth
                on_f = np.mean([grid_f[min(max(int(round(y)), 0), Hg - 1), min(max(int(round(x)), 0), Wg - 1)] for x, y in st])
                ex = 0.0 if on_f > 0.5 else extra_mm
                ws = getattr(st, "widths", None)
                if ws is None or len(ws) != len(pts):
                    ws = [float(gap_w[min(max(int(round(y)), 0), Hg - 1), min(max(int(round(x)), 0), Wg - 1)]) for x, y in st]
                # a point every 0.4 mm: a simplified stroke can be only its two ends, and the depth follows the gap width along it
                s_ = np.r_[0.0, np.cumsum([math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(pts, pts[1:])])]
                t_ = np.linspace(0.0, s_[-1], max(2, int(math.ceil(s_[-1] / 0.4)) + 1))
                ws = list(np.interp(t_, s_, ws)) if ws is not None and len(ws) == len(pts) else None
                pts = list(zip(np.interp(t_, s_, [p[0] for p in pts]), np.interp(t_, s_, [p[1] for p in pts])))
                if ws is not None and inside is not None:                  # blend in: smooth widths, thin runs over the paint around it
                    pts, ws = taper_repair_stroke(pts, ws, inside, float(cw[0]), along=along)
                if ws is not None:                                         # the thin end first, like every line
                    tmp = Stroke(pts, "line"); tmp.widths = ws
                    if thin_end_flip(tmp):
                        pts, ws = pts[::-1], ws[::-1]
                # every repair stroke is as wide as its gap: at least the thinnest mark the brush makes, at most REPAIR_WIDTH_SHARE of the widest
                zs = [z_of(min(max(w, float(cw[0])), w_rep), ex) for w in ws] if ws is not None else [z_of(w_rep, ex)] * len(pts)
                zs = [tilted(z, x, y) for z, (x, y) in zip(zs, pts)]
                o_in, o_out = end_dir(pts), end_dir(pts[::-1])
                a = (pts[0][0] + o_in[0] * lead, pts[0][1] + o_in[1] * lead)
                e = (pts[-1][0] + o_out[0] * lead, pts[-1][1] + o_out[1] * lead)
                # right after a dip: ONE diagonal move from the wipe point (already outside the rim) down to the stroke, like the
                # painting does - not up, across, down (2026-11-08). Lowering it straight down beside the dish had dragged it
                # through the rim (2026-11-03); on the diagonal it only gets low far from the dish. Between strokes the brush is
                # at travel height already: across, then the small step to this stroke's travel height
                out += ["M400", f"M204 P{acc_t} T{acc_t}"]
                if after_dip:
                    out.append(f"G0 X{a[0]:.3f} Y{a[1]:.3f} Z{tilted(z_travel, *a):.3f} F{feed_t}")
                    after_dip = False
                else:
                    out.append(f"G0 X{a[0]:.3f} Y{a[1]:.3f} F{feed_t}")
                    out.append(f"G0 Z{tilted(z_travel, *a):.3f} F3000")
                out += ["M400", f"M204 P{acc_p} T{acc_p}"]
                out.append(f"G1 Z{max(zs[0], tilted(air, *a)):.3f} F{feed_d}")
                out.append(f"G1 F{feed_p}")
                for (x, y), z in zip(pts, zs):
                    out.append(f"G1 X{x:.3f} Y{y:.3f} Z{z:.3f}")
                out.append(f"G1 X{e[0]:.3f} Y{e[1]:.3f} Z{tilted(air, *e):.3f}")
                out.append(f"G0 Z{tilted(z_travel, *e):.3f} F3000")
            # no lift to z_high here: the next dip cycle lifts by itself (its first line, the painting's own lift) - after a
            # lift here that line sent the brush DOWN again over the painting before the trip to the dish (2026-11-08)
    out.append(f"G0 Z{z_high:.3f} F3000")
    return out, int(empty.sum()), counts


class Photos:
    """A picture of the sheet every `every` dips of the job: paint1.jpg, paint2.jpg, ... The toolhead goes to the snapshot position,
    the picture is taken, and the toolhead comes back to where it stopped (lifted). The picture is taken after every `every` complete dips.
    The frame comes from the live window when it is open (the camera is held by that window), otherwise the camera is opened for it.
    A picture that cannot be taken is reported and the painting goes on."""

    def __init__(self, cfg, every, preview=None, code=None, H_inv=None, paper_poly=None, first_frame=None, art_bgr=None):
        self.cfg, self.every, self.preview = cfg, every, preview
        self.n, self.xy = 0, None
        self.code, self.H_inv, self.paper_poly = code, H_inv, paper_poly
        self.first = first_frame
        self.art = art_bgr
        self.gif_frames = []                                 # small pictures for the GIF, in order
        self.prev, self.prev_dips = first_frame, 0           # the sheet before the last block of dips (first: before any paint)
        self.strokes, self.dip_blocks = [], {}
        self.dip_blocks_plan = {}                            # the dip cycles WITHOUT the tool-offset shift (the gap repair shifts its whole text once)
        self._art_outline = None                             # the artwork's outline in the photo where the plan put it (made once)
        if code is not None:
            import checkup as CU
            self.strokes, self.dip_blocks = CU.parse_strokes_full(code)
            self.dip_blocks_plan = self.dip_blocks
            to = cfg.get("tool_offset")
            if to:
                # the G-code is shifted by -tool_offset so the paint lands where it was planned: the points that are judged on the
                # photo go back to where the paint LANDS. "lines" stay as sent (the stroke repair resends them as they are)
                for s in self.strokes:
                    s["pts"] = [(x + to[0], y + to[1]) for x, y in s["pts"]]
                self.dip_blocks_plan = {k: shift_gcode_xy("\n".join(v), to[0], to[1]).split("\n") for k, v in self.dip_blocks.items()}

    def baseline(self, machine):
        """The start picture: toolhead at the snapshot position, before the first stroke (and after any probe lines).
        Every later picture is compared with this one, so the lighting and the head are the same in both."""
        try:
            go_to_snapshot(machine, self.cfg)
            self.first = self._frame()
            self.prev = self.first
            cv2.imwrite("paint0.png", self.first)
            self.overlay(self.first, "paint0.png")
            self.gif_frames.append(("start", self.first.copy()))
            print("photos: start picture paint0.jpg (the comparison for the whole job)")
        except Exception as e:
            print(f"(start picture not taken: {e}; the first comparison uses the picture from the setup)")

    def make_gif(self):
        """The job's GIF: painting_<date>_<time>.gif, one frame per picture (start, every photo, final). Automatic unless
        AUTO_GIF = False in settings.py. A new name every time, so an old GIF is never overwritten."""
        if not getattr(S, "AUTO_GIF", True) or len(self.gif_frames) < 2:
            return
        try:
            frames = [_gif_frame(f, self.paper_poly, i + 1, len(self.gif_frames)) for i, (_, f) in enumerate(self.gif_frames)]
            name = f"painting_{time.strftime('%Y%m%d_%H%M%S')}.gif"
            durations = [700] * len(frames)
            durations[-1] = 3000                              # hold the last picture
            frames[0].save(name, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=True)
            print(f"GIF: {name} ({len(frames)} frames)")
        except Exception as e:
            print(f"(GIF not made: {e}; the pictures are saved)")

    def _finish_by_gaps(self, machine):
        """End of job, gap method: photo of the finished sheet, the empty spots (artwork has colour, photo has none) are painted,
        each colour from its own dish. Repeats (REPAIR_PASSES) while the empty spots keep going down."""
        on = bool(getattr(S, "REPAIR_AT_END", True))
        passes = int(getattr(S, "REPAIR_PASSES", 6)) if on else 0             # repair_stop() ends them: done, no real progress, or this many
        step = float(getattr(S, "REPAIR_STEP_MM", 0.15))
        deepest = float(getattr(S, "REPAIR_MAX_DEEPER_MM", 1.0))
        H = np.linalg.inv(self.H_inv)
        if not self.dip_blocks and passes:
            print("   (no dip cycle in the G-code: nothing repainted)")
            passes = 0
        gaps = []
        self.prev_gap = None
        for k in range(passes + 1):
            name = "paint_final.png" if k == 0 else f"paint_repair{k}.png"
            try:
                dry = float(getattr(S, "DRY_WAIT_MIN", 20) or 0)
                if dry > 0:
                    go_to_snapshot(machine, self.cfg)
                    dry_wait(dry, "the finished sheet" if k == 0 else f"after repair {k}")
                go_to_snapshot(machine, self.cfg)
                frame = self._frame()
                cv2.imwrite(name, frame)
                self.overlay(frame, name)
                self.gif_frames.append(("final" if k == 0 else f"repair{k}", frame.copy()))
                extra = min(deepest, step * (k + 1))                                   # every cycle one REPAIR_STEP_MM deeper, up to REPAIR_MAX_DEEPER_MM
                lines, n_gap, per = gap_repair_gcode(frame, self.art, H, self.dip_blocks_plan, self.cfg, extra,
                                                     persist=(self.prev_gap if k >= 2 else None))
                self.prev_gap = LAST_GAP
                if lines is None:                                         # the paint is misplaced: nothing to repair safely
                    return
                if LAST_GAP is not None:                                  # the picture of what is still empty (yellow)
                    ov = frame.copy()
                    if LAST_ART is not None:                              # magenta = the artwork's outline where the plan put it
                        cs, _ = cv2.findContours(LAST_ART.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                        cv2.drawContours(ov, cs, -1, (255, 0, 255), 1)
                    ov[LAST_GAP] = (0, 255, 255)
                    cv2.imwrite(f"gap_{name[:-4]}_empty.png", ov)
            except Exception as e:
                print(f"   (picture / gap check not done: {e})")
                return
            label = "final" if k == 0 else f"after repair {k}"
            print(f"   {label}: {n_gap} empty pixels left (" + ", ".join(f"{v} {k}" for k, v in per.items()) + ")")
            gaps.append(int(n_gap))
            why = ("repairs off" if passes == 0 else "REPAIR_PASSES reached") if k == passes else repair_stop(gaps)
            if why:
                if k > 0:
                    print(f"   repairs finished: {why} ({gaps[0]} -> {n_gap} empty pixels in {k} pass(es))")
                break
            print(f"REPAIR pass {k + 1}: empty spots painted, {extra:.2f} mm deeper")
            text = "\n".join(lines)
            if self.cfg.get("tool_offset"):                       # the repair lands like the painting: the same measured shift
                text = shift_gcode_xy(text, -self.cfg["tool_offset"][0], -self.cfg["tool_offset"][1])
            if not stream_gcode(machine, text, self.cfg, None, None, photos=None):
                print("   repair stopped by you or by the machine")
                return

    def finish(self, machine):
        """At the end of the job: pictures of the finished sheet, every stroke judged from that picture alone (darker than the paper
        beside it). The strokes still MISSING are painted again, each pass a little deeper (REPAIR_STEP_MM) for those strokes,
        up to REPAIR_MAX_DEEPER_MM, with a fresh dip before every 8. Repeats (at most REPAIR_PASSES) until nothing is missing,
        or the passes make no real progress. Nothing is repainted when the judgement is not reliable."""
        if self.H_inv is None:
            return
        if self.art is not None and getattr(S, "REPAIR_BY_GAPS", True):
            return self._finish_by_gaps(machine)
        if self.paper_poly is None or not self.strokes:
            return
        import checkup as CU
        on = bool(getattr(S, "REPAIR_AT_END", True))
        passes = int(getattr(S, "REPAIR_PASSES", 3)) if on else 0
        step = float(getattr(S, "REPAIR_STEP_MM", 0.15))
        deepest = float(getattr(S, "REPAIR_MAX_DEEPER_MM", 1.0))
        cap = int(getattr(S, "REPAIR_MAX_STROKES", 400))
        if not self.dip_blocks and passes:
            print("   (no dip cycle found in the G-code: nothing repainted)")
            passes = 0
        extra = {}                                           # stroke index -> how much deeper it has been painted, mm
        last_missing = None
        print("\nFINAL: picture of the finished sheet, every stroke judged from that picture (darker than the paper beside it)")
        for k in range(passes + 1):
            name = "paint_final.png" if k == 0 else f"paint_repair{k}.png"
            try:
                dry = float(getattr(S, "DRY_WAIT_MIN", 20) or 0)
                if dry > 0:
                    # gouache is glossy while wet: the camera reads the glare, so no judgement before the paint has dried
                    go_to_snapshot(machine, self.cfg)
                    dry_wait(dry, "the finished sheet" if k == 0 else f"after repair {k}")
                go_to_snapshot(machine, self.cfg)
                frame = self._frame()
                cv2.imwrite(name, frame)
                self.gif_frames.append(("final" if k == 0 else f"repair{k}", frame.copy()))
                if self.first is not None and getattr(S, "JUDGE_BY_DIFFERENCE", True):
                    # what is painted = the difference to the blank start picture; only strokes with little paint on them are repainted
                    res = CU.judge_by_difference(frame, self.first, self.strokes, self.H_inv, self.paper_poly, name[:-4] + "_check")
                else:
                    res = CU.judge_local(frame, self.strokes, self.H_inv, self.paper_poly, name[:-4] + "_check")
            except Exception as e:
                print(f"   (picture / judgement not done: {e})")
                return
            print(f"   {'final' if k == 0 else 'after repair ' + str(k)}: {res['ok']} of {res['strokes']} strokes painted, "
                  f"{res['missing']} still MISSING" + ("" if res["reliable"] else "  (not reliable: nothing repainted)"))
            if not res["reliable"]:
                return
            missing = [i for i, r in enumerate(res["rows"]) if r["status"] == "MISSING"]
            if not missing or k == passes:
                break
            if last_missing is not None and len(missing) >= 0.98 * last_missing:
                print("   no real progress since the last pass: stopping the repairs here (the missing ones are listed in the check file)")
                break
            last_missing = len(missing)
            eligible = [i for i in missing if extra.get(i, 0.0) < deepest - 1e-9][:cap]
            if not eligible:
                vprint(f"   {len(missing)} still missing, all at the deepest allowed ({deepest:g} mm): stopping")
                break
            for i in eligible:
                extra[i] = min(deepest, extra.get(i, 0.0) + step)
            code = CU.repair_code_steps(self.dip_blocks, [(self.strokes[i], extra[i]) for i in eligible], z_high=self.cfg["z_high"])
            print(f"REPAIR pass {k + 1}: {len(eligible)} missing strokes painted again, {step:g} mm deeper "
                  f"(deepest now {max(extra[i] for i in eligible):.2f} mm)")
            if not stream_gcode(machine, code, self.cfg, None, None, photos=None):
                print("   repair stopped by you or by the machine")
                return
        if extra:
            print(f"REPAIR summary: {len(extra)} strokes were painted again, deepest {max(extra.values()):.2f} mm")

    def _check(self, frame, dips_done, name):
        """Judges the centre-line strokes of the block of dips just finished (report only: nothing is changed on the machine)."""
        if self.prev is None or self.H_inv is None or self.paper_poly is None:
            return
        if float(np.abs(frame.astype(np.int16) - self.prev.astype(np.int16)).mean()) < 0.5:
            print(f"   {name} is the same picture as the one before: the camera looks FROZEN. Not judged, the painting goes on.")
            return
        import checkup as CU
        block = [s for s in self.strokes if self.prev_dips < s["dip"] <= dips_done - 1]
        stem = name[:-4] + "_check"
        if block:
            res = CU.check_block(self.prev, frame, block, self.H_inv, self.paper_poly, stem)
            msg = f"   check: {res['strokes']} strokes of the last block of dips, {res['ok']} on the paper, {res['missing']} MISSING"
            vprint(msg + ("" if res["reliable"] else "  (block too faint to judge)"))
            if res["missing"]:
                vprint(f"   missing strokes: {stem}.json, picture {stem}.png (red = missing)")
        self.prev_dips = dips_done - 1
        self.prev = frame

    def due(self, dips_done):
        """dips_done = dip cycles started so far. Dip number every+1 starts when `every` dips are complete."""
        return self.every > 0 and dips_done > 1 and (dips_done - 1) % self.every == 0

    def seen(self, line):
        """Remembers the last XY the strokes went to, so the toolhead can come back there."""
        if not line.startswith(("G0", "G1")):
            return
        tok = dict(re.findall(r"([XY])(-?[\d.]+)", line))
        if self.xy is None:
            if "X" in tok and "Y" in tok:
                self.xy = [float(tok["X"]), float(tok["Y"])]
        else:
            if "X" in tok:
                self.xy[0] = float(tok["X"])
            if "Y" in tok:
                self.xy[1] = float(tok["Y"])

    def _frame(self, settle=1.0):
        pv = self.preview
        thread = getattr(pv, "thread", None) if pv is not None else None
        if thread is not None and thread.is_alive():
            # the live window holds the camera (or is still opening it): wait for its picture. Opening the camera a second time at the
            # same moment crashes Python in the camera driver (access violation)
            t = time.time()
            while getattr(pv, "latest", None) is None and thread.is_alive() and time.time() - t < 20.0:
                time.sleep(0.1)
            if getattr(pv, "latest", None) is not None:
                time.sleep(settle)                           # the head has just stopped: let the live view show the settled sheet
                return pv.latest.copy()
            if thread.is_alive():
                raise RuntimeError("the live window has the camera but no picture yet")
        from vision import open_camera
        cap = open_camera(verbose=False)
        try:
            frame, _ = capture_fresh(cap)
        finally:
            cap.release()
        return frame

    def overlay(self, frame, name):
        """<name>_overlay.png: the photo with the paper (green), the artwork box (yellow) and the artwork's outline where the PLAN
        put it (magenta). Paint outside the magenta outline = the placement / calibration is off; inside but missing = the painting."""
        try:
            out = frame.copy()
            if self._art_outline is None:
                self._art_outline = []
                if self.art is not None and self.H_inv is not None:
                    placed = _art_from_plan(self.art, frame, self.cfg, np.linalg.inv(self.H_inv))
                    if placed is not None:
                        art_r, map_x, map_y = placed
                        ahsv = cv2.cvtColor(art_r, cv2.COLOR_BGR2HSV)
                        ink = (((ahsv[..., 1] > 90) & (ahsv[..., 2] > 40)) | (ahsv[..., 2] < 100)).astype(np.float32)
                        ink_f = cv2.remap(ink, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0.5
                        self._art_outline, _ = cv2.findContours(ink_f.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            if self.paper_poly is not None:
                cv2.polylines(out, [np.round(np.array(self.paper_poly)).astype(np.int32)], True, (0, 255, 0), 2, cv2.LINE_AA)
            pl = self.cfg.get("placement")
            if pl and self.H_inv is not None:
                from vision import placement_corners_mm
                box = np.round(mm_to_px_array(self.H_inv, placement_corners_mm(pl))).astype(np.int32)
                cv2.polylines(out, [box], True, (0, 220, 255), 2, cv2.LINE_AA)
            if len(self._art_outline):
                cv2.drawContours(out, self._art_outline, -1, (255, 0, 255), 1, cv2.LINE_AA)
            stamp_picture(out, f"{name}: green = paper, yellow = artwork box, magenta = artwork as planned")
            cv2.imwrite(name[:-4] + "_overlay.png", out)
        except Exception as e:                               # a fault here must not stop the painting
            print(f"   (overlay of {name} not made: {e})")

    def snap(self, machine):
        """A stroke photo (STROKE_PHOTOS): head out of view, the sheet cropped to the paper saved as strokes_<date>_<time>/00001.jpg,
        then back to where it stopped. Nothing is judged; a fault only skips this photo."""
        try:
            if getattr(self, "snap_dir", None) is None:
                self.snap_dir = f"strokes_{time.strftime('%Y%m%d_%H%M%S')}"
                os.makedirs(self.snap_dir, exist_ok=True)
                self.snap_n = 0
            go_to_snapshot(machine, self.cfg, settle=float(getattr(S, "STROKE_PHOTO_SETTLE_S", 0.4)))
            frame = self._frame(settle=0.15)
            if self.paper_poly is not None:
                q = np.array(self.paper_poly, float)
                x0, y0 = np.maximum(q.min(axis=0) - 30, 0).astype(int)
                x1, y1 = (q.max(axis=0) + 30).astype(int)
                frame = frame[y0:y1, x0:x1]
            self.snap_n += 1
            cv2.imwrite(os.path.join(self.snap_dir, f"{self.snap_n:05d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        except Exception as e:
            print(f"   (stroke photo not taken: {e})")
        finally:
            if self.xy is not None:
                machine.send_line(f"G0 X{self.xy[0]:.3f} Y{self.xy[1]:.3f} F{self.cfg['feed']}", timeout=60)

    def make_stroke_video(self):
        """strokes_<...>.mp4 from the stroke photos (STROKE_PHOTO_FPS frames per second, the last one held 3 s)."""
        d = getattr(self, "snap_dir", None)
        if not d or not getattr(self, "snap_n", 0):
            return
        try:
            files = sorted(f for f in os.listdir(d) if f.endswith(".jpg"))
            first = cv2.imread(os.path.join(d, files[0]))
            h, w = first.shape[0] // 2 * 2, first.shape[1] // 2 * 2
            fps = float(getattr(S, "STROKE_PHOTO_FPS", 12))
            out = cv2.VideoWriter(d + ".mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            for f in files:
                out.write(cv2.resize(cv2.imread(os.path.join(d, f)), (w, h)))
            last = cv2.resize(cv2.imread(os.path.join(d, files[-1])), (w, h))
            for _ in range(int(3 * fps)):
                out.write(last)
            out.release()
            print(f"stroke time-lapse: {d}.mp4 ({len(files)} photos, folder {d})")
        except Exception as e:
            print(f"(stroke video not made: {e}; the photos are in {d})")

    def take(self, machine, dips_done):
        self.n += 1
        name = f"paint{self.n}.png"                       # the same name every run: overwritten
        print(f"\nPHOTO {self.n} after {dips_done - 1} dips -> {name} (toolhead to the snapshot position, then back)")
        try:
            go_to_snapshot(machine, self.cfg)
            frame = self._frame()
            cv2.imwrite(name, frame)
            self.overlay(frame, name)
            self.gif_frames.append((f"{self.n}", frame.copy()))
            vprint(f"   saved {name}")
            try:
                self._check(frame, dips_done, name)
            except Exception as e:                       # a fault in the check must not stop the painting
                print(f"   (check of {name} not done: {e}; the painting goes on)")
        except Exception as e:
            print(f"   (picture {name} not taken: {e}; the painting goes on)")
        finally:
            if self.xy is not None:
                machine.send_line(f"G0 X{self.xy[0]:.3f} Y{self.xy[1]:.3f} F{self.cfg['feed']}", timeout=60)
                machine.flush_motion()


def stream_gcode(machine, code, cfg, preview=None, checker=None, photos=None):
    """Sends the G-code line by line and waits for the machine's 'ok' each time. Ctrl+C lifts the brush and stops.
    With a checker (check.py): every line passes through it (the Z of the strokes follows the learned depth map), at every dip the camera looks at the
    new strokes, and at the end of every layer the missing strokes are painted again. Returns True if everything was sent."""
    table, cum = command_table(code)
    cmds = [c[0] for c in table]
    total, t0, last = len(cmds), time.time(), 0
    vprint(f"sending {total} G-code lines to the machine. Ctrl+C stops it (the brush is lifted first).")
    prev_layer, prev_dips = "", 0
    prev_photo_dips = 0
    if checker is not None:
        try:
            checker.begin(machine)
            checker.start_message()
        except Exception as e:
            print(f"(touch check not available: {e}; the strokes are painted as planned)")
            checker = None
    # stroke photos (STROKE_PHOTOS): "leaf" = after the last stroke of every leaf, "stroke" = after every stroke, "off". Thin lines
    # (no leaf) get one every STROKE_PHOTO_LINES strokes. The head parks out of view for each, so the frames show only the paper
    mode = str(getattr(S, "STROKE_PHOTOS", "off")).lower() if photos is not None else "off"
    snap_at = set()
    if mode in ("leaf", "stroke"):
        ends = [(i, t[3]) for i, t in enumerate(table, 1) if t[3] is not None]
        every_lines, n_lines = max(1, int(getattr(S, "STROKE_PHOTO_LINES", 10))), 0
        for k, (i, g) in enumerate(ends):
            nxt = ends[k + 1][1] if k + 1 < len(ends) else None
            if g == "line":
                n_lines += 1
                if n_lines % every_lines == 0 or nxt != "line":
                    snap_at.add(i)
            elif mode == "stroke" or nxt != g:
                snap_at.add(i)
        print(f"stroke photos: {len(snap_at)} (one per {'stroke' if mode == 'stroke' else 'leaf'}; about "
              f"{len(snap_at) * float(getattr(S, 'STROKE_PHOTO_SECONDS', 5.0)) / 60:.0f} min more)")
    try:
        for i, line in enumerate(cmds, 1):
            if photos is not None and table[i - 1][2] != prev_photo_dips:
                prev_photo_dips = table[i - 1][2]
                if photos.due(prev_photo_dips):
                    photos.take(machine, prev_photo_dips)
            if checker is not None:
                layer, dips = table[i - 1][1], table[i - 1][2]
                if layer != prev_layer and prev_layer:
                    checker.layer_end(machine, prev_layer)
                if dips != prev_dips:
                    checker.checkpoint(machine, dips)
                prev_layer, prev_dips = layer, dips
                checker.layer = layer
                line = checker.transform(line)
            machine.last_error = None
            machine.send_line(line)
            if photos is not None:
                photos.seen(line)
            if preview is not None:
                preview.update(i, table[i - 1][1], table[i - 1][2], cum[i - 1])
            if i in snap_at:
                photos.snap(machine)
            if getattr(machine, "last_error", None) and line.startswith(("G0", "G1")):
                print(f"\nthe machine refused a move ({machine.last_error}) at line {i}: {line} - stopping")
                machine.send_line(f"G0 Z{cfg['z_high']:.3f} F3000")
                return False
            if (i * 10) // total != (last * 10) // total or i == total:
                last = i
                elapsed = time.time() - t0
                est_done = max(cum[i - 1], 1e-6)
                left = elapsed * (cum[-1] - est_done) / est_done if est_done > 0 else 0
                vprint(f"   {i}/{total} lines ({100 * i / total:.0f}%), about {left / 60:.0f} min left")
    except KeyboardInterrupt:
        print("\nstopped by you - lifting the brush")
        try:
            machine.send_line(f"G0 Z{cfg['z_high']:.3f} F3000")
            machine.flush_motion()
        except Exception:
            pass
        if checker is not None:
            try:
                checker.save_log()
            except Exception:
                pass
        return False
    machine.flush_motion()
    if checker is not None:
        checker.layer_end(machine, prev_layer, final=True)
        checker.finish(machine)
    return True


def place_on_paper(args, cfg):
    """Fills cfg['placement'] from a photo of the paper (or a saved paper.json). Returns (frame, detection)."""
    import sys
    import vision as PP
    from vision import load_calibration
    cal = load_calibration()
    H, H_inv = cal["H"], cal["H_inv"]
    frame = cv2.imread(args.frame) if args.frame else None
    if args.frame and frame is None:
        sys.exit(f"cannot read {args.frame}")
    if args.paper == "auto":
        if frame is None:
            from vision import open_camera
            cap = open_camera()
            frame, live = capture_fresh(cap)
            cap.release()
            cfg["photo_time"], cfg["photo_live"] = time.strftime("%H:%M:%S"), live
            cv2.imwrite("paper_frame.png", frame)
            vprint(f"camera picture taken {cfg['photo_time']}" + ("" if live else
                  "  - WARNING: the camera delivered identical frames (frozen or old picture)! Close other programs using the camera, "
                  "unplug/replug it and run again"))
        det = PP.detect_paper(frame, H, H_inv, calib_mm=cal["mm_pts"])
        if det is None:
            sys.exit("no paper found in the picture (is the toolhead in view, or the dots not on the paper?)")
        if min(det["size_mm"]) < 50.0:
            sys.exit(f"the paper found is only {det['size_mm'][0]:.0f} x {det['size_mm'][1]:.0f} mm: that is not the sheet "
                     f"(look at paper_frame.png - is the paper in view and lit, with nothing white and bigger beside it?)")
        det = PP.clip_to_reach(det)
        pf = PP.paper_frame(det, H)
    else:
        det, pf = PP.load_paper(args.paper)
    iw, ih = Image.open(args.input).size
    bottom = None
    if not bool(getattr(S, "SIGN_INSIDE", False)) and signature_file():   # room for the signature under the artwork
        import signature as SG
        _, sh = SG.block_size_mm(S.SIGNATURE, float(S.SIGN_WIDTH_MM), bool(getattr(S, "SIGN_YEAR", True)),
                                 year_h_mm=float(getattr(S, "SIGN_YEAR_MM", 4.0)))
        bottom = float(S.SIGN_GAP_MM) + sh + float(S.SIGN_EDGE_MM)
    pl = PP.fit_artwork(pf, iw, ih, margin_mm=args.margin, rotate=args.rotate, width_mm=args.width, bottom_mm=bottom)
    cfg["placement"], cfg["target_width"], cfg["rotate"], cfg["paper_frame"] = pl, pl["width_mm"], pl["rotate"], pf
    vprint(f"paper: {det['size_mm'][0]:.0f} x {det['size_mm'][1]:.0f} mm (usable {pf['paper_w']:.0f} x {pf['paper_h']:.0f}), "
          f"tilt {pf['angle_deg']:.1f} deg")
    vprint(f"artwork: {pl['width_mm']:.0f} x {pl['height_mm']:.0f} mm, rotated {pl['rotate']} deg, {args.margin:.0f} mm margin, centred"
           + (f" (bottom margin {max(bottom, args.margin):.0f} mm: the signature goes there)" if bottom else ""))
    if not pl["fits"]:
        print("WARNING: with this --width the artwork does not fit inside the paper margins")
    if str(args.rotate) == "0" and pl["scale_if_other_rotation"] > pl["scale_mm_per_px"] * 1.3:
        vprint(f"hint: turned by 90 degrees it would fit {100 * (pl['scale_if_other_rotation'] / pl['scale_mm_per_px'] - 1):.0f}% larger (--rotate 90)")
    return frame, det


def unused_settings():
    """Names set in settings.py that no script of the project reads: a typo or an old name, which would silently do nothing
    (2026-11-01: PROBE_COUNT = 2 still painted 4 lines; PROBE_LENGTH_MM was never read, the length is PROBE_LINE_MM)."""
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        names = re.findall(r"^([A-Z][A-Z0-9_]*)\s*=", open(os.path.join(here, "settings.py"), encoding="utf-8").read(), re.M)
        code = "\n".join(open(os.path.join(here, f), encoding="utf-8", errors="ignore").read()
                         for f in os.listdir(here) if f.endswith(".py") and f != "settings.py")
    except OSError:
        return []
    return [n for n in dict.fromkeys(names) if n != "BASE_DIR" and not re.search(r"\b" + n + r"\b", code)]


def main(argv=None):
    bad = unused_settings()
    if bad:
        print(f"\n*** settings.py: {', '.join(bad)} {'is' if len(bad) == 1 else 'are'} not used by any script "
              f"(a typo or an old name?): changing {'it' if len(bad) == 1 else 'them'} does NOTHING ***\n")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", nargs="?", help="the artwork (or ARTWORK in settings.py)")
    ap.add_argument("output", nargs="?", help="G-code file to write (default: artwork name + .gcode, or OUTPUT in settings.py)")
    ap.add_argument("--width", type=float, help="artwork width in mm (default: as large as fits on the paper)")
    ap.add_argument("--brush", type=float, help="brush width in mm (default 1.0)")
    ap.add_argument("--overlap", type=float, help="0..0.9 overlap between neighbouring lines (default 0)")
    ap.add_argument("--infill", choices=["auto", "lines", "rings"], help="auto = chosen per shape (default)")
    ap.add_argument("--angle", type=float, help="line angle in degrees (default: along the long axis of each shape)")
    ap.add_argument("--no_outline", action="store_true", help="no outline ring before line infill")
    ap.add_argument("--simplify", type=float, help="merge straight stretches within this many mm (default 0.1)")
    ap.add_argument("--z_offset", type=float, help="general Z offset in mm, added to EVERY Z value in the G-code (z_paint, travel height, dip depth, wipe...). "
                    "Use it when the brush is mounted lower: with 3, 'Z 0' becomes machine Z 3 and the machine has 3 mm of room below it. "
                    "Default 0 (or Z_OFFSET in settings.py)")
    ap.add_argument("--max_stroke_width", type=float, help="marks narrower than this (mm) are painted as ONE centre-line stroke whose depth follows the width "
                    "(thin = shallow, wide = deep); wider shapes are filled. Default 4.0 when the depth follows the width, else the brush width. "
                    "0 = old behaviour (only parts narrower than the brush are centre-lines)")
    ap.add_argument("--ladder", help="the brush profile (BRUSH_PROFILE in settings.py): a ladder file from 'closed_loop_trainer9.py ladder', e.g. S4Z3 (= ladder_dataset/S4Z3.json). "
                    "Takes the contact height (--z_offset), --brush, --line_depth, --max_stroke_width and --paint_feed from it, and paints every mark "
                    "at the depth where the REAL brush makes that width (anything you type yourself wins; --width_min/--width_ref switch to a straight-line mapping)")
    ap.add_argument("--dip_min", type=float, help="shortest painted path in mm before the next dip (default 200)")
    ap.add_argument("--dip_max", type=float, help="longest painted path in mm before the next dip (default 240); each dip picks a random distance between the two")
    ap.add_argument("--width_ref", type=float, help="a centre-line mark this wide (mm) or wider is painted at the FULL depth; thinner marks in proportion. "
                    "Default: the widest mark of the artwork. A smaller number spreads the depth over the typical marks (the run prints their widths)")
    ap.add_argument("--width_min", type=float, help="a mark this narrow (mm) or narrower is painted at the touch height (zero depth). Default 0")
    ap.add_argument("--point_step", type=float, help="distance between points along a centre-line stroke in mm (default 0.4): each point has its own Z")
    ap.add_argument("--paint_feed", type=int, help="painting speed in mm/min (default 200; your W2D drawings use 600)")
    ap.add_argument("--paint_accel", type=int, help="acceleration while painting in mm/s2 (default 200; W2D uses 100)")
    ap.add_argument("--wipe_feed", type=int, help="speed of the wipe move after the dip in mm/min (default 3000; the trip to the next stroke always uses the travel speed)")
    ap.add_argument("--plunge_feed", type=int, help="speed of the brush going down onto the paper in mm/min (default 2500; W2D uses half the painting speed)")
    ap.add_argument("--line_depth", type=float, help="how much DEEPER than the fill the thin lines (skeleton) may go (needs room below: see --z_offset). "
                    "The depth follows the line width: the thinnest lines stay at the fill Z, the widest thin line goes the full depth. "
                    "Default: ON with the whole room below (z_offset - 0.2 mm) when there is a Z offset, else 0. --line_depth 0 switches it off "
                    "(or LINE_DEPTH in settings.py)")
    ap.add_argument("--seed", type=int, help="fixed random seed (dip distance / position)")
    ap.add_argument("--plan_only", action="store_true", help="do not touch the machine: only look with the camera, plan, write the G-code")
    ap.add_argument("--yes", action="store_true", help="do not ask before painting")
    ap.add_argument("--probe", action="store_true", help="first paint ramps in the white margin and measure where THIS paint touches THIS paper "
                    "(contact height, tilt of the paper, width against depth); used instead of the profile's numbers. See probe.py. PROBES = True in settings.py does the same")
    ap.add_argument("--no_probe", action="store_true", help="do NOT measure where the paint touches the paper before this job (the probes are ON by default: "
                    "the contact depends on the paint, the paper, the speed and the brush, so it is measured every time). PROBES = False in settings.py switches them off for good")
    ap.add_argument("--check", action="store_true", help="touch check while painting (EXPERIMENTAL, off by default; CHECK = True in settings.py makes it the default): at every dip the head "
                    "parks, the camera looks at the new strokes, a depth map is learned and missing strokes are painted again at the end of a layer. See check.py")
    ap.add_argument("--no_check", action="store_true", help="no touch check (the default; only needed when CHECK = True in settings.py)")
    ap.add_argument("--probe_file", help="use the result of an earlier  python probe.py  run (probe_result.json) instead of painting probes now")
    ap.add_argument("--force_probe_file", action="store_true", help="use a --probe_file even if it does not fit this sheet (paper moved, too old). Not recommended")
    ap.add_argument("--unmeasured", action="store_true", help="if the probes were painted but gave no usable result, paint the job anyway with the settings / profile contact. "
                    "Not recommended: the depth can be far off. Without it the job STOPS and paints nothing more")
    ap.add_argument("--no_preview", action="store_true", help="no live camera window while painting (it is there for screen recording)")
    ap.add_argument("--paper", default="auto", help="auto (default) = photograph the paper now; paper.json = saved earlier; none = old mama.py offsets, no camera")
    ap.add_argument("--frame", help="use this saved camera picture instead of the camera (implies --plan_only)")
    ap.add_argument("--margin", type=float, default=10.0, help="empty paper all round, mm (default 10)")
    ap.add_argument("--rotate", default=None,
                    help="auto (default, ROTATE in settings.py: turned by 90 degrees when it then fits larger on the paper), 0 (always upright) or 90")
    ap.add_argument("--color", default=getattr(__import__("settings"), "PAINT_COLOR", "auto"),   # main() imports S itself further down
                    help="black (PAINT_COLOR in settings.py) = the whole artwork in black; auto = every colour in the artwork becomes a layer; "
                         "or paint just one: blue, red, ... or r,g,b")
    ap.add_argument("--tol", type=float, default=40.0, help="with r,g,b: how close a colour must be (Lab distance, default 40)")
    ap.add_argument("--list_colors", action="store_true", help="only list the colours in the artwork and exit")
    ap.add_argument("--h4xx0r", action="store_true", help="the h4xx0r theme of ../brushograph's WebUI: a green phosphor terminal "
                    "and a live preview behind dirty glass (or H4XX0R = True in settings.py)")
    a = ap.parse_intermixed_args(argv)                                   # file names and options in any order
    if a.h4xx0r or getattr(__import__("settings"), "H4XX0R", False):
        import h4xx0r
        h4xx0r.enable()
    print(f"paint.py build {BUILD}")
    S = load_settings()                                                  # settings.py: fills in what was not typed
    used = []
    for opt, key in (("input", "ARTWORK"), ("output", "OUTPUT"), ("width", "ARTWORK_WIDTH_MM"), ("ladder", "BRUSH_PROFILE"), ("z_offset", "Z_OFFSET"),
                     ("line_depth", "LINE_DEPTH"), ("width_min", "WIDTH_MIN"), ("width_ref", "WIDTH_REF"), ("brush", "BRUSH_WIDTH"),
                     ("max_stroke_width", "MAX_STROKE_WIDTH"), ("point_step", "POINT_STEP"), ("paint_feed", "PAINT_FEED"), ("paint_accel", "PAINT_ACCEL"),
                     ("plunge_feed", "PLUNGE_FEED"), ("wipe_feed", "WIPE_FEED"), ("dip_min", "DIP_MIN_MM"), ("dip_max", "DIP_MAX_MM"),
                     ("rotate", "ROTATE")):
        if getattr(a, opt, None) is None and S.get(key) is not None:
            val = S[key]
            if opt in ("paint_feed", "paint_accel", "plunge_feed", "wipe_feed"):
                val = int(val)
            setattr(a, opt, val)
            used.append(f"{key} = {val}")
    if used:
        vprint("from settings.py: " + ", ".join(used))
    a.rotate = str(a.rotate) if a.rotate is not None else "auto"
    if not a.input:
        raise SystemExit("no artwork: give it on the command line or set ARTWORK in settings.py")
    if not os.path.isfile(a.input):                                      # before the camera is opened and before the machine moves
        raise SystemExit(f"artwork not found: {a.input!r}  (a picture in this folder, e.g. PoletnaModra.png)")
    z_typed = a.z_offset                                                 # what YOU asked for (command line or settings.py)
    curve = None
    if a.ladder:
        path = BP.find(a.ladder)
        L = BP.load(path)
        taken = []
        a.from_profile = set()                                           # which options came from the profile and not from you (a probe may replace those)
        typed_linear = a.width_min is not None or a.width_ref is not None
        for opt, key in (("z_offset", "z_offset"), ("brush", "brush"), ("paint_feed", "paint_feed"), ("line_depth", "max_depth")):
            if getattr(a, opt) is None and L.get(key) is not None:
                setattr(a, opt, L[key]); taken.append(f"--{opt} {L[key]}"); a.from_profile.add(opt)
        if typed_linear:                                                 # you gave widths yourself: the old straight-line mapping
            for opt, key in (("width_min", "width_min"), ("width_ref", "width_ref"), ("max_stroke_width", "thin_max")):
                if getattr(a, opt) is None:
                    setattr(a, opt, L[key]); taken.append(f"--{opt} {L[key]}")
        else:
            curve = L["curve"]
        vprint(f"brush profile {path}  ({L['strokes']} strokes, {L['levels']} depths):")
        vprint(f"   the tip starts to mark at machine Z {L['contact']:.2f}; thinnest reliable line {L['curve'][1][0]:.2f} mm; "
              f"{L['w_curve_max']:.2f} mm at {L['max_depth']:.1f} mm deeper; measured at F{L['feed']:.0f}")
        vprint("   used: " + (" ".join(taken) if taken else "(everything was given by you)"))
        vprint("   mapping: " + ("each mark is pressed to the depth where the REAL brush makes that width (the profile's curve)" if curve is not None
                               else "straight line between --width_min and --width_ref (you gave widths yourself)"))
        if z_typed is not None and abs(z_typed - L["contact"]) > 0.3:
            print(f"   WARNING: your offset {z_typed:.2f} differs from the contact height of the profile ({L['contact']:.2f}) by {abs(z_typed - L['contact']):.2f} mm. "
                  f"Right if you moved the brush since the ladder (the curve is then used from your offset); otherwise the lines are painted that much too deep or too shallow.")
        if L["top_is_contact"]:
            print("   WARNING: even the TOP depth of that ladder marked reliably: the real contact may be higher - run the ladder again with a larger --z_top")
        if a.paint_feed and L["feed"] and abs(a.paint_feed - L["feed"]) > 1:
            print(f"   WARNING: you paint at F{a.paint_feed} but the widths were measured at F{L['feed']:.0f}: the width depends on the speed")
    elif z_typed is None and S.get("TOUCH_Z") is not None:
        a.z_offset = float(S["TOUCH_Z"])
        vprint(f"no brush profile: the touch height is TOUCH_Z = {a.z_offset:.2f} from settings.py")
    if a.list_colors:
        describe_colors(a.input)
        return
    out = a.output or (a.input.rsplit(".", 1)[0] + ".gcode")
    cfg = default_config()
    cfg["color"], cfg["tol"] = a.color, a.tol
    for k, v in (("target_width", a.width), ("brush_w", a.brush), ("overlap", a.overlap), ("infill_type", a.infill),
                 ("infill_angle", a.angle), ("simplify_mm", a.simplify), ("min_dist", a.dip_min), ("max_dist", a.dip_max)):
        if v is not None:
            cfg[k] = v
    if cfg["min_dist"] > cfg["max_dist"]:
        raise SystemExit(f"--dip_min ({cfg['min_dist']}) is larger than --dip_max ({cfg['max_dist']})")
    if a.no_outline:
        cfg["outline"] = False
    cfg["curve"] = curve
    if a.line_depth is not None:
        cfg["line_depth"] = a.line_depth
    if a.z_offset is not None:
        cfg["z_offset"] = a.z_offset
    if cfg["line_depth"] is None:                              # not set anywhere: ON when the Z offset gives room below, else off
        cfg["line_depth"] = round(cfg["z_offset"] - 0.2, 2) if cfg["z_offset"] >= 0.5 else 0.0     # the whole room below the touch height (down to Z 0.2)
        cfg["line_depth_auto"] = True
    if cfg["line_depth"] < 0:
        raise SystemExit("--line_depth cannot be negative")
    room = cfg["z_offset"] - 0.2                          # how far below the touch height the machine can go (Z 0.2)
    if curve is not None and cfg["line_depth"] > room > 0:
        vprint(f"depth limited to {room:.2f} mm (machine Z 0.2) by the room below the touch height {cfg['z_offset']:.2f}  (the profile goes to {cfg['line_depth']:.1f} mm)")
        cfg["line_depth"] = round(room, 2)
    for key, val in (("feed_paint", a.paint_feed), ("accel_paint", a.paint_accel), ("plunge_feed", a.plunge_feed), ("wipe_feed", a.wipe_feed), ("point_step", a.point_step)):
        if val is not None:
            cfg[key] = val
    cfg["width_ref"], cfg["width_min"] = a.width_ref, (a.width_min or 0.0)
    if a.width_ref is not None and a.width_ref <= cfg["width_min"]:
        raise SystemExit("--width_ref must be larger than --width_min")
    if a.max_stroke_width is not None:
        cfg["thin_max"] = a.max_stroke_width
    if not cfg["thin_max"]:                                    # not set (or 0): the widest line the brush makes, else 4 mm / the brush width
        if curve is not None:
            cfg["thin_max"] = round(float(np.interp(cfg["line_depth"], curve[0], curve[1])), 2)
        else:
            cfg["thin_max"] = 4.0 if cfg["line_depth"] > 0 else cfg["brush_w"] * 1.1
    cfg["thin_max"] = max(cfg["thin_max"], cfg["brush_w"] * 1.1)
    if a.max_stroke_width == 0:
        cfg["thin_max"] = cfg["brush_w"] * 1.1
    for key in ("z_paint", "z_low", "z_high", "z_wipe_exit", "dip_z"):          # machine Z = logical Z + offset
        cfg[key] += cfg["z_offset"]
    lowest = cfg["z_paint"] - cfg["line_depth"]
    if lowest < -1e-9 or min(cfg["z_paint"], cfg["dip_z"]) < -1e-9:
        raise SystemExit(f"the lowest Z would be {min(lowest, cfg['dip_z']):.2f} mm - below the machine's Z 0.\n"
                         f"  The brush needs more room below: mount it lower and use --z_offset {max(cfg['line_depth'], 0) + cfg['z_offset']:.1f} "
                         f"(or Z_OFFSET in settings.py), or use a smaller --line_depth.")
    if cfg["z_paint"] >= cfg["z_low"]:
        raise SystemExit(f"z_paint {cfg['z_paint']:.1f} is not below the travel height z_low {cfg['z_low']:.1f}")
    if a.seed is not None:
        random.seed(a.seed)
    machine = None
    try:
        if not a.plan_only and not a.frame:                    # live run: home, then step aside for the camera
            machine = connect_and_home(cfg)
            if a.paper != "none":
                go_to_snapshot(machine, cfg)
        frame = det = None
        if a.paper != "none":
            frame, det = place_on_paper(a, cfg)
        elif a.frame:
            frame = cv2.imread(a.frame)
        want_probe = (not a.no_probe) and bool(a.probe or a.probe_file or S.get("PROBES", True))        # ON unless you say no
        cfg["probe_status"] = ("not run: --no_probe" if a.no_probe else "not run: PROBES = False in settings.py" if not want_probe
                               else "not run: no paper was found in the camera picture" if det is None else "")
        if det is not None and want_probe:
            import probe as PB
            same_sheet = None if (a.probe_file or a.probe) else PB.reusable_result(det, frame)
            if a.probe_file:
                res = PB.load_result(a.probe_file)
                PB.check_result_fresh(res, det, force=a.force_probe_file)
            elif same_sheet:
                # this sheet was probed already (same place, today): new lines would only repaint the old marks
                res = same_sheet
                print(f"this sheet was probed at {res.get('created', '?')[11:16]}: that result is used, no new probe lines. "
                      f"(New brush or paint? Run with --probe: new lines go beside the old ones.)")
            elif machine is None:
                vprint("probes: skipped in this plan-only run (they need the machine to paint the ramps; --probe_file uses an earlier result)")
                res = None
            else:
                res = PB.run_probes(machine, cfg, frame, det, yes=a.yes, color=first_layer_color(a.input, cfg),
                                    ask=bool(getattr(__import__("settings"), "PROBE_ASK", False)),   # PROBE_ASK in settings.py (S here is the dict)
                                    question="Paint the probes, then the artwork if the probes work (otherwise nothing more is painted)?  [y/N] ")
                # why the job stops is ALWAYS printed (print, not vprint): with VERBOSE = False the run ended with only
                # "Disconnected." and no reason (2026-11-01)
                if res is None and PB.LAST_STATUS.startswith("you answered N"):
                    print("not started: you did not answer y. The machine stays homed. (--no_probe paints without probes.)")
                    machine.send_line(f"G0 Z{cfg['z_high']:.3f} F3000")
                    return
                if res is None and not a.unmeasured:                      # the probes were painted, but gave nothing usable: STOP
                    print(f"\nSTOPPED: {PB.LAST_STATUS or 'the probes gave no usable result'}.")
                    print("Nothing more is painted. The brush is lifted. Use a fresh sheet and run again.")
                    print("(--unmeasured paints anyway with the settings contact: not recommended.)")
                    machine.send_line(f"G0 Z{cfg['z_high']:.3f} F3000")
                    return
                cfg["confirmed_once"] = not a.yes                         # you said yes once: no second question, whatever the probes gave
            if res:
                apply_probe(cfg, res, a)
                cfg["probe_status"] = "USED"
                cfg["probe_text"] = (f"contact Z {res['contact_center']:.2f}, paper tilt {res.get('plane_spread_mm', 0):.2f} mm, widest {res['curve'][1][-1]:.2f} mm "
                                     f"at {res['curve'][0][-1]:.1f} mm deeper, measured {res.get('created', '?')}")
            else:
                cfg["probe_status"] = (PB.LAST_STATUS or "no probe result") if machine is not None else "not run: plan-only run (no machine)"
        code, overlay, plan = run(a.input, cfg, out, frame, det)   # stops here with a message if a paint dish is missing
        if machine is None:
            vprint("\nplanned only (nothing was sent to the machine).")
            if overlay and not a.frame:                            # a fresh camera picture: show what was planned on it
                show_picture(overlay)
            return
        if cfg.get("probe_status") == "USED":
            vprint(f"\nPROBES USED: {cfg['probe_text']}")
        else:
            print(f"\n*** PROBES NOT USED ({cfg.get('probe_status') or 'not run'}). The contact is Z {cfg['z_paint']:.2f} from your settings / profile, "
                  f"the widths are the profile's: the lines may be too high (in the air) or too deep. ***")
        if overlay:
            show_picture(overlay)
        ask = bool(getattr(__import__("settings"), "PROBE_ASK", False))     # PROBE_ASK = False: no question at all, only the countdown
        if cfg.get("confirmed_once") or (not ask and not a.yes):
            print(f"\nLook at {overlay or out}. " + ("You said yes once" if ask else "PROBE_ASK = False")
                  + ": painting starts in 10 seconds (Ctrl+C = stop).")
            try:
                for i in range(10, 0, -1):
                    print(f"   {i} ...", end="\r", flush=True)
                    time.sleep(1.0)
                print("            ")
            except KeyboardInterrupt:
                machine.send_line(f"G0 Z{cfg['z_high']:.3f} F3000")
                print("\nnot started (Ctrl+C). The machine stays homed.")
                return
        elif not a.yes:
            print(f"\nLook at {overlay or out}: strokes inside the yellow box, a circle on every paint dish.")
            if input("Start painting now? [y/N] ").strip().lower() not in ("y", "yes"):
                machine.send_line(f"G0 Z{cfg['z_high']:.3f} F3000")
                print("not started: you did not answer y. The machine stays homed.")
                return
        preview, checker, grabber = None, None, None
        if det is not None and machine is not None and not a.no_check and (a.check or bool(S.get("CHECK", False))):
            import check as CK
            grabber = CK.FrameGrabber()
            grabber.start()                                    # the camera is free again: the photo was taken before painting
            checker = CK.TouchCheck(cfg, load_calibration(), det, plan, grabber)
        photos_on = int(S.get("PHOTO_EVERY_DIPS", 5)) > 0                 # (S is the settings TABLE here: .get, not getattr)
        timelapse = bool(S.get("TIMELAPSE", True)) and checker is None
        rec, crop = None, None
        if timelapse:
            rec = f"timelapse_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
            if det is not None:                                             # the video shows the paper (and a little round it)
                q = np.array(det["corners_px"], float)
                x0, y0 = np.maximum(q.min(axis=0) - 30, 0).astype(int)
                x1, y1 = (q.max(axis=0) + 30).astype(int)
                crop = (int(x0), int(y0), int(x1), int(y1))
        if photos_on and not a.no_preview and not timelapse:
            print("live window: off while the photos are on (the photos need the camera for themselves)")
        if (not a.no_preview and not photos_on) or timelapse:
            # the photos take their pictures from this camera reader when it runs (never a second camera open); with the photos on the
            # window stays hidden, only the time-lapse is recorded
            table, cum = command_table(code)
            preview = LivePreview(plan, len(table), cum[-1] if cum else 1.0, grabber=grabber,
                                  show=not a.no_preview and not photos_on, record=rec, crop=crop)
            preview.start()
            if rec:
                print(f"time-lapse: recording to {rec} (a frame every {S.get('TIMELAPSE_EVERY_S', 2.0)} s while something changes)")
        try:
            every = int(S.get("PHOTO_EVERY_DIPS", 5))
            photos = (Photos(cfg, every, preview, code=code, H_inv=load_calibration()["H_inv"],
                             paper_poly=(det or {}).get("corners_px"), first_frame=frame,
                             art_bgr=cv2.imread(a.input))
                      if checker is None else None)
            if photos is not None:
                print(f"photos: a picture of the sheet after every {every} dips (paint1.jpg, paint2.jpg, ...)")
            elif checker is not None:
                print("photos: not taken together with the touch check (--check)")
            if photos is not None:
                photos.baseline(machine)
            try:
                done = stream_gcode(machine, code, cfg, preview, checker, photos=photos)
                if done and photos is not None:
                    photos.finish(machine)
            finally:
                # the GIF is made even when the run is stopped (Ctrl+C) or fails, from the pictures taken so far
                if photos is not None:
                    photos.make_gif()
                    photos.make_stroke_video()
            if done:
                print("\nfinished.")
                if preview is not None:
                    print("(live window stays for a few seconds - Q closes it)")
                    preview.finish()
        finally:
            if preview is not None:
                preview.stop()
            if grabber is not None:
                grabber.stop()
    finally:
        if machine is not None:
            machine.disconnect()


if __name__ == "__main__":
    main()
