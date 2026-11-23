"""
cups.py - the 12-cup palette (hardware/palette.py): where the cups are, what paint is in them, how the brush dips and wipes.

    python cups.py detect          # a picture now: the 12 cups found, their colours read, saved to cups.json (cups_detect.png)
    python cups.py empty           # ONCE, with every cup EMPTY: how an empty cup looks (cups_empty.json), so a cup with paint
                                   # (also a pale one) is told from an empty one
    python cups.py show            # cups.json: the cups, their paint, dips so far and the depth of the next dip
    python cups.py refill [n ...]  # cups refilled: their dip count back to 0 (every cup when no number is given)
    python cups.py test [n]        # one dip in cup n to watch it: the brush first WAITS over the cup (centred?), then dips

The palette: two holders end to end, 6 cups along it, 2 across. The column nearer the paper is OPAQUE paint, the other one
TRANSPARENT (CUP_OPAQUE_SIDE). Cups are numbered 1-6 (opaque) and 7-12 (transparent), along the palette in the direction of
the machine's X (or Y).

A dip in a cup (paint.py, instead of the old round dishes):
    straight up to CUP_CLEAR_MM over the rims -> over the cup -> slowly down into the paint -> a small stir -> up to just under
    the rim -> across the rim on the side facing the paper (the wipe) -> up -> out of the palette, over the rims -> the stroke.
    The depth: CUP_DIP_START_MM over the cup's floor for its first dip, CUP_DIP_STEP_MM lower for every dip after it, never lower
    than CUP_DIP_END_MM (the paint level sinks as it is used). Each cup counts its own dips (kept in cups.json across jobs, back
    to 0 with  python cups.py refill).
Heights are kept relative to the board the paper lies on: the probe's touch (the paper's top) minus RINSE_PAPER_MM.
"""
import json
import math
import os
import random
import sys
import time

import numpy as np

import settings as S

HERE = os.path.dirname(os.path.abspath(__file__))


def _set(name, default):
    return getattr(S, name, default)


def cup_file():
    f = _set("CUP_FILE", "cups.json")
    return None if f is None else os.path.join(HERE, f)


EMPTY_FILE = os.path.join(HERE, "cups_empty.json")


# ------------------------------------------------------------------------------------------------- the palette's geometry
def layout():
    """Numbers of the palette (hardware/palette.py), mm: one holder piece, its pockets, the cups' insides."""
    L, Wd = float(_set("CUP_HOLDER_MM", (140.0, 60.0))[0]), float(_set("CUP_HOLDER_MM", (140.0, 60.0))[1])
    outer, div, cols, rows = 2.5, 2.0, 3, 2
    px = (L - 2 * outer - (cols - 1) * div) / cols                    # pocket along the palette (43.7)
    py = (Wd - 2 * outer - (rows - 1) * div) / rows                   # pocket across (26.5)
    inside = tuple(float(v) for v in _set("CUP_INSIDE_MM", (39.9, 22.7)))
    along = [(c - 1) * (px + div) for c in range(cols)]               # cup centres along one piece, from its middle
    across = [-(py + div) / 2, (py + div) / 2]
    return dict(L=L, W=Wd, px=px, py=py, along=along, across=across, inside=inside, wall=1.6)


def heights(touch):
    """Machine Z of the board, the cups' floor, their rims and the safe height over them, for a brush touching the paper at `touch`."""
    board = float(touch) - float(_set("RINSE_PAPER_MM", 0.3))
    rim = board + float(_set("CUP_RIM_MM", 27.0))
    import machine as MC
    return {"board": board, "floor": board + float(_set("CUP_FLOOR_MM", 3.6)), "rim": rim,
            "safe": rim + float(_set("CUP_CLEAR_MM", 6.0)),           # over the cups (the way to them does not cross the rinse
            "safe_all": max(rim + float(_set("CUP_CLEAR_MM", 6.0)), MC.safe_z(touch))}   # container; rinse.check_moves warns)


def seen_bigger(height_mm):
    """How much bigger (away from the middle of the picture) the camera, calibrated on the paper, sees things this high: the rinse
    container's measured factor scaled to this height."""
    k_r = float(_set("RINSE_SEEN_BIGGER", 1.037))
    h_r = float(_set("RINSE_BOX_MM", (180.0, 20.0, 29.0))[2])
    return 1.0 + (k_r - 1.0) * float(height_mm) / h_r


# ------------------------------------------------------------------------------------------------- found by the camera
def detect(frame, H, H_inv, paper_poly_px=None, debug_png=None):
    """The two holder pieces in the picture (light boxes about CUP_HOLDER_MM), the 12 cup centres from the palette's geometry,
    in machine mm. Returns {"source": "camera", "cups": [...], "pieces": [...], "date"} or None (reason printed)."""
    import cv2
    import vision as V
    lay = layout()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    light = ((hsv[..., 2] > 150) & (hsv[..., 1] < 70)).astype(np.uint8) * 255
    if paper_poly_px is not None:
        ex = np.zeros(light.shape, np.uint8)
        cv2.fillPoly(ex, [np.asarray(paper_poly_px, np.int32)], 255)
        light[cv2.dilate(ex, np.ones((41, 41), np.uint8)) > 0] = 0
    light = cv2.morphologyEx(light, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    # the cups' insides hold paint of any colour: each light area (the white walls) is taken WITH its holes filled, so a holder
    # is a solid box whether its cups are empty or full
    cnts, _ = cv2.findContours(light, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h_img, w_img = frame.shape[:2]
    N = np.array(V.pixel_to_mm(H, w_img / 2.0, h_img / 2.0))            # straight below the camera, about
    k_mid = seen_bigger(float(_set("CUP_RIM_MM", 27.0)) / 2.0)           # a piece's light area: from its floor up to the rims
    pieces = []

    def accept(c, u_, Lr, Wr):
        pieces.append({"c": N + (np.array(c) - N) / k_mid, "u": u_, "Lr": Lr, "Wr": Wr})
    for cnt in cnts:
        if cv2.contourArea(cnt) < 2000:
            continue
        solid = np.zeros(light.shape, np.uint8)
        cv2.drawContours(solid, [cnt], -1, 255, -1)
        ys, xs = np.nonzero(solid)
        mm = V.px_to_mm_array(H, np.c_[xs, ys]).astype(np.float32)
        (cx, cy), (w, h), ang = cv2.minAreaRect(mm)
        Lr, Wr = max(w, h), min(w, h)
        a = math.radians(ang if w >= h else ang + 90.0)
        u_ = np.array([math.cos(a), math.sin(a)])
        if not (0.95 * lay["W"] < Wr < 1.35 * lay["W"]):
            continue
        if 0.9 * lay["L"] < Lr < 1.15 * lay["L"]:                      # one holder
            accept((cx, cy), u_, Lr, Wr)
        elif 1.85 * lay["L"] < Lr < 2.25 * lay["L"]:                   # both holders as one area (end to end): split in halves
            for sgn in (-1, 1):
                accept((cx + sgn * u_[0] * Lr / 4, cy + sgn * u_[1] * Lr / 4), u_, Lr / 2, Wr)
    if len(pieces) == 1:
        # the other holder's walls can be in shadow or broken up by paint: the holders are glued END TO END, so the second one
        # is one holder length further along. Which end: the one where the camera sees more white wall where it should be
        p = pieces[0]
        best = None
        for sgn in (-1.0, 1.0):
            c2 = p["c"] + sgn * p["u"] * lay["L"]
            v_ = np.array([-p["u"][1], p["u"][0]])
            q = [N + (c2 + su * lay["L"] / 2 * p["u"] + sv * lay["W"] / 2 * v_ - N) * k_mid for su in np.linspace(-0.95, 0.95, 40)
                 for sv in np.linspace(-0.95, 0.95, 14)]
            px = V.mm_to_px_array(H_inv, q).astype(int)
            ok = (px[:, 0] >= 0) & (px[:, 0] < w_img) & (px[:, 1] >= 0) & (px[:, 1] < h_img)
            share = float(light[px[ok, 1], px[ok, 0]].astype(bool).mean()) if ok.any() else 0.0
            if best is None or share > best[0]:
                best = (share, c2)
        if best[0] < 0.08:
            print(f"cups: one holder found, the other not (no white walls beside it: {100 * best[0]:.0f}%)")
            return None
        print(f"cups: one holder found whole, the other placed end to end beside it ({100 * best[0]:.0f}% white wall seen there)")
        pieces.append({"c": best[1], "u": p["u"], "Lr": lay["L"], "Wr": lay["W"]})
    if len(pieces) != 2:
        print(f"cups: the camera sees {len(pieces)} holder piece(s) of about {lay['L']:g} x {lay['W']:g} mm, not 2 "
              f"(are both holders in the picture, the toolhead out of the way?)")
        return None
    # fine fit: each holder moved (up to CUP_FIT_MM, both ways) until its 6 cups' rims lie on the white walls the camera sees.
    # A holder's light area is a rough guide only (shadowed walls, paint, the other holder not exactly in line)
    k_rim = seen_bigger(float(_set("CUP_RIM_MM", 27.0)))
    # the inner edge of every cup's rim: a sharp brightness step whether the cup is empty (white rim, shaded inside) or full
    a_w, b_w = lay["inside"][0] / 2, lay["inside"][1] / 2
    per = np.array([(x, -b_w) for x in np.linspace(-a_w * 0.9, a_w * 0.9, 24)] + [(x, b_w) for x in np.linspace(-a_w * 0.9, a_w * 0.9, 24)] +
                   [(-a_w, y) for y in np.linspace(-b_w * 0.85, b_w * 0.85, 12)] + [(a_w, y) for y in np.linspace(-b_w * 0.85, b_w * 0.85, 12)])
    grey = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 1.5)
    grad = cv2.magnitude(cv2.Sobel(grey, cv2.CV_32F, 1, 0), cv2.Sobel(grey, cv2.CV_32F, 0, 1))
    lightf = grad / (float(np.percentile(grad, 99.5)) or 1.0)
    R = float(_set("CUP_FIT_MM", 8.0))
    for p in pieces:
        v_ = np.array([-p["u"][1], p["u"][0]])
        rel = np.array([(a_, b_) for a_ in lay["along"] for b_ in lay["across"]])          # the 6 cup centres in the piece
        pts = np.array([r_[0] * p["u"] + r_[1] * v_ + q[0] * p["u"] + q[1] * v_ for r_ in rel for q in per])
        best = (-1.0, 0.0, 0.0)
        for da in np.arange(-R, R + 0.01, 0.5):
            for db in np.arange(-R, R + 0.01, 0.5):
                P = p["c"] + da * p["u"] + db * v_ + pts
                px = V.mm_to_px_array(H_inv, N + (P - N) * k_rim)
                xi = np.clip(px[:, 0].astype(int), 0, w_img - 1); yi = np.clip(px[:, 1].astype(int), 0, h_img - 1)
                sc = float(lightf[yi, xi].mean())
                if sc > best[0]:
                    best = (sc, da, db)
        p["c"] = p["c"] + best[1] * p["u"] + best[2] * v_
        p["fit"] = (round(best[0], 2), round(best[1], 1), round(best[2], 1))
    if any(abs(p["fit"][1]) >= R or abs(p["fit"][2]) >= R for p in pieces):
        print(f"cups: a holder fitted at the edge of the search ({[p['fit'] for p in pieces]}): check cups_detect.png")
    if any(p["fit"][0] < 0.15 for p in pieces):
        print(f"cups: the cups' rims are not clearly seen (fit {[p['fit'][0] for p in pieces]}): check cups_detect.png")
    # one axis for both: from the first piece's middle to the second's (end to end)
    p0, p1 = sorted(pieces, key=lambda p: (p["c"][0], p["c"][1]))
    u = p1["c"] - p0["c"]
    gap = float(np.linalg.norm(u))
    u = u / (gap or 1.0)
    if abs(gap - lay["L"]) > 12.0:
        print(f"cups: the two holders' middles are {gap:.0f} mm apart, {lay['L']:g} expected (end to end): check cups_detect.png")
    v = np.array([-u[1], u[0]])
    mid = (p0["c"] + p1["c"]) / 2.0
    # the opaque column is the one nearer the paper (or the middle of the picture)
    target = N
    if paper_poly_px is not None:
        target = V.px_to_mm_array(H, np.asarray(paper_poly_px, float)).mean(axis=0)
    side = _set("CUP_OPAQUE_SIDE", "paper")
    to_paper = 1.0 if float(np.dot(target - mid, v)) > 0 else -1.0
    vp = v * to_paper                                                    # across, towards the paper
    opaque_sign = 1.0 if side == "paper" else -1.0
    cups = []
    for k_col, kind in ((0, "opaque"), (1, "transparent")):
        off = opaque_sign * (1 if k_col == 0 else -1) * abs(lay["across"][0])
        for k_piece, pc in enumerate((p0, p1)):
            for k_c, a_ in enumerate(lay["along"]):
                xy = pc["c"] + u * a_ + vp * off
                cups.append({"kind": kind, "xy": [round(float(xy[0]), 2), round(float(xy[1]), 2)],
                             "along": k_piece * 3 + k_c})
    sh_along, sh_paper = (float(v) for v in _set("CUP_SHIFT_MM", (0.0, 0.0)))
    if sh_along or sh_paper:                                             # a correction by hand (seen the brush stand off the middle)
        for c in cups:
            xy = np.array(c["xy"]) + u * sh_along + vp * sh_paper
            c["xy"] = [round(float(xy[0]), 2), round(float(xy[1]), 2)]
    cups.sort(key=lambda c: (c["kind"] != "opaque", c["along"]))
    for k, c in enumerate(cups):
        c["n"] = k + 1
    out = {"source": "camera", "date": time.strftime("%Y-%m-%d %H:%M"), "shift": [sh_along, sh_paper],
           "u": [round(float(u[0]), 4), round(float(u[1]), 4)], "to_paper": [round(float(vp[0]), 4), round(float(vp[1]), 4)],
           "pieces": [{"c": [round(float(p["c"][0]), 2), round(float(p["c"][1]), 2)], "fit": p.get("fit")} for p in (p0, p1)],
           "cups": cups}
    if debug_png:
        draw(frame, H_inv, out, debug_png)
    return out


def footprint(cs, margin=None):
    """The palette as corners (true mm), widened by CUP_KEEP_OUT_MM: no move outside a cup dip may pass lower than the rims here."""
    lay = layout()
    margin = float(_set("CUP_KEEP_OUT_MM", 8.0)) if margin is None else margin
    u, vp = np.array(cs["u"]), np.array(cs["to_paper"])
    c0, c1 = np.array(cs["pieces"][0]["c"]), np.array(cs["pieces"][1]["c"])
    mid = (c0 + c1) / 2
    hl = np.linalg.norm(c1 - c0) / 2 + lay["L"] / 2 + margin
    hw = lay["W"] / 2 + margin
    return [mid + su * hl * u + sv * hw * vp for su, sv in ((-1, -1), (1, -1), (1, 1), (-1, 1))]


def _inside(p, poly):
    x, y = p
    ins = False
    for i in range(len(poly)):
        x1, y1 = poly[i]; x2, y2 = poly[(i + 1) % len(poly)]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / ((y2 - y1) or 1e-12) + x1:
            ins = not ins
    return ins


def read_colours(frame, H_inv, cs, empty=None):
    """What is in every cup: the median colour of its middle (seen at about half its height). A cup that looks like the empty
    reference (cups_empty.json) within CUP_EMPTY_DE (Lab), or without a reference is pale grey-white, is EMPTY."""
    import cv2
    import vision as V
    lay = layout()
    hsv_all = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lab_all = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(float)
    h_img, w_img = frame.shape[:2]
    cal_H = np.linalg.inv(np.asarray(H_inv, float))
    N = np.array(V.pixel_to_mm(cal_H, w_img / 2.0, h_img / 2.0))
    k = seen_bigger(float(_set("CUP_FLOOR_MM", 3.6)) + 6.0)                # the paint's surface, about
    u, vp = np.array(cs["u"]), np.array(cs["to_paper"])
    a_in, b_in = lay["inside"][0] / 2 - 6.0, lay["inside"][1] / 2 - 4.0
    thr = float(_set("CUP_EMPTY_DE", 12.0))
    for c in cs["cups"]:
        ctr = np.array(c["xy"])
        pts = [ctr + su * a_in * u + sv * b_in * vp for su in np.linspace(-1, 1, 9) for sv in np.linspace(-1, 1, 5)]
        seen = [N + (p - N) * k for p in pts]
        px = V.mm_to_px_array(H_inv, seen).astype(int)
        px[:, 0] = np.clip(px[:, 0], 0, w_img - 1); px[:, 1] = np.clip(px[:, 1], 0, h_img - 1)
        bgr = np.median(frame[px[:, 1], px[:, 0]].astype(float), axis=0)
        lab = np.median(lab_all[px[:, 1], px[:, 0]], axis=0)
        hsv = np.median(hsv_all[px[:, 1], px[:, 0]].astype(float), axis=0)
        ref = (empty or {}).get(str(c["n"]))
        if ref is not None:
            de = float(np.linalg.norm(lab - np.array(ref)))
            filled = de > thr
        else:
            de = None
            filled = hsv[1] > 45 or hsv[2] < 110
        c.update({"bgr": [round(float(x), 1) for x in bgr], "lab": [round(float(x), 1) for x in lab],
                  "hue": round(float(hsv[0]), 1), "sat": round(float(hsv[1]), 1), "val": round(float(hsv[2]), 1),
                  "filled": bool(filled), "dark": bool(filled and hsv[2] < 90 and (hsv[1] < 120 or (    # (2026-11-21: the
                      c.get("kind") == "opaque" and hsv[2] < 60))),     # new camera view read the black at sat 121: an OPAQUE cup this dark is black)
                  "empty_de": None if de is None else round(de, 1)})
        if c["n"] in tuple(_set("CUP_WHITE_CUPS", ())):       # white paint in a white cup: the camera cannot tell, declared
            c.update({"filled": True, "white": True, "dark": False})
    # front and back of one row hold the SAME colour (opaque in front, transparent behind): a cup of transparent paint looks
    # nearly black in its deep cup, so it takes the colour of its opaque partner, which the camera reads well
    if _set("CUP_ROW_SAME_COLOUR", True):
        front = {c["along"]: c for c in cs["cups"] if c["kind"] == "opaque"}
        for c in cs["cups"]:
            p = front.get(c["along"])
            if c["kind"] == "transparent" and c.get("filled") and p is not None and p.get("filled"):
                c.update({"hue_seen": c["hue"], "sat_seen": c["sat"], "partner": p["n"], "hue": p["hue"], "sat": p["sat"],
                          "dark": p.get("dark", False)})
    return cs


def save_empty(frame, H_inv, cs):
    cs = read_colours(frame, H_inv, json.loads(json.dumps(cs)), empty=None)
    ref = {str(c["n"]): c["lab"] for c in cs["cups"]}
    json.dump(ref, open(EMPTY_FILE, "w"), indent=1)
    return ref


def load():
    f = cup_file()
    if f is None or not os.path.isfile(f):
        return None
    try:
        return json.load(open(f))
    except Exception as e:
        print(f"cups: {f} could not be read ({e})")
        return None


def save(cs):
    f = cup_file()
    if f:
        json.dump(cs, open(f, "w"), indent=1)


def choose(frame, H, H_inv, paper_poly_px, saved=None):
    """For a job: the cups where the camera sees them now (the saved places kept when the camera reads them within CUP_LOCK_MM:
    the palette is glued down), their colours read now. The dip counts are carried over. None = no palette."""
    if cup_file() is None:
        return None
    saved = saved or load()
    seen = None
    try:
        seen = detect(frame, H, H_inv, paper_poly_px, debug_png=os.path.join(HERE, "cups_detect.png"))
    except Exception as e:
        print(f"cups: camera detection failed ({e})")
    cs = None
    if seen and saved and saved.get("cups"):
        d = max(math.dist(a["xy"], b["xy"]) for a, b in zip(seen["cups"], saved["cups"]))
        same_shift = list(saved.get("shift", [0.0, 0.0])) == list(seen.get("shift", [0.0, 0.0]))
        if d <= float(_set("CUP_LOCK_MM", 3.0)) and same_shift:
            print(f"cups: in their saved places (the camera reads them {d:.1f} mm from there; CUP_LOCK_MM)")
            cs = saved
        else:
            print(f"cups: the palette MOVED by {d:.0f} mm, the new places are used" if same_shift else
                  f"cups: CUP_SHIFT_MM changed to {seen.get('shift')}: the corrected places are used")
            seen["dips"] = saved.get("dips", {})
            seen["deeper"] = saved.get("deeper", {})
            if saved.get("lower") is not None:                  # the brush check's "too high" stays (2026-11-23: it was lost)
                seen["lower"] = saved["lower"]
            cs = seen
    elif seen:
        cs = seen
    elif saved:
        print("cups: the camera does not find the palette now: its last known place is used (cups.json)")
        cs = saved
    if cs is None:
        return None
    empty = json.load(open(EMPTY_FILE)) if os.path.isfile(EMPTY_FILE) else None
    if empty is None:
        print("cups: no picture of the empty cups yet (python cups.py empty): an empty cup is told by its pale colour only")
    read_colours(frame, H_inv, cs, empty)
    cs.setdefault("dips", {})
    save(cs)
    draw(frame, H_inv, cs, os.path.join(HERE, "cups_detect.png"))
    return cs


def pools(cs, kind=None):
    """The filled cups as paint pools for paint.assign_dishes (center_mm, radius_mm, area_mm2, hue, sat, dark ...)."""
    kind = kind or _set("CUP_KIND", "any")
    lay = layout()
    out = []
    for c in cs["cups"]:
        if not c.get("filled") or (kind != "any" and c["kind"] != kind):
            continue
        out.append({"center_mm": list(c["xy"]), "radius_mm": lay["inside"][1] / 2, "area_mm2": lay["inside"][0] * lay["inside"][1],
                    "hue": c["hue"], "sat": c["sat"], "bgr": c["bgr"], "dark": c.get("dark", False), "roundness": 1.0,
                    "cup": c["n"], "kind": c["kind"]})
    return out


def pick(cs, color="auto"):
    """The filled cup to probe or paint calibration dots with: the darkest ("auto"/"black"), or the one nearest in colour family
    to `color`. Returns (x, y, radius) like a dish, or None."""
    import paint as PT
    filled = [c for c in cs["cups"] if c.get("filled")]
    if not filled:
        return None
    opaque = [c for c in filled if c["kind"] == "opaque"]
    filled = opaque or filled                       # opaque paint: transparent marks are too faint to measure
    if color in ("auto", "black", None):
        c = min(filled, key=lambda c: c.get("val", 255))
    else:
        same = [c for c in filled if PT.hue_name(c.get("hue", 0)) == color]
        c = min(same or filled, key=lambda c: c.get("val", 255))
    print(f"paint from cup {c['n']} ({c['kind']}, hue {c.get('hue', 0):.0f}, value {c.get('val', 0):.0f}) at "
          f"X {c['xy'][0]:.0f} Y {c['xy'][1]:.0f}")
    return c["xy"][0], c["xy"][1], 11.0


def row_of(c):
    """Which colour a cup holds, by its row: 'white', 'black', 'yellow', 'magenta', 'cyan', 'green' (or None)."""
    if c.get("white"):
        return "white"
    if not c.get("filled"):
        return None
    if c.get("dark"):
        return "black"
    h = float(c.get("hue", 0))
    if 9 <= h <= 35:
        return "yellow"
    if 36 <= h <= 85:
        return "green"
    if 86 <= h <= 135:
        return "cyan"
    return "magenta"                                  # 136..180 and 0..8: magenta, pink, purple, red


def cup_for(cs, row, kind="transparent"):
    """The filled cup of this colour row and kind (the transparent cup takes its row's colour from the opaque one in front)."""
    hits = [c for c in cs["cups"] if c["kind"] == kind and c.get("filled") and row_of(c) == row]
    return hits[0] if hits else None


def cup_at(cs, x, y, tol=1.0):
    for c in (cs or {}).get("cups", []):
        if math.dist(c["xy"], (x, y)) <= tol:
            return c
    return None


# ------------------------------------------------------------------------------------------------- the moves
def dip_depth(cs, n, dips_now):
    """Height over the cup floor for this cup's next dip: start, a step lower per dip (those of earlier jobs too), never under end."""
    done = int((cs or {}).get("dips", {}).get(str(n), 0)) + int(dips_now)
    start, end = float(_set("CUP_DIP_START_MM", 10.0)), float(_set("CUP_DIP_END_MM", 5.0))
    deeper = float((cs or {}).get("deeper", {}).get(str(n), 0.0))          # the paint check found the brush too dry
    return max(end, start - float(_set("CUP_DIP_STEP_MM", 0.05)) * done - deeper)


def refill(ns=None):
    """The cups are full again (all, or those in ns): their dips start again at CUP_DIP_START_MM, the paint check's deeper dips
    are forgotten. -> the cup numbers."""
    cs = load()
    if not cs:
        return []
    ns = list(ns) if ns else [c["n"] for c in cs["cups"]]
    for n in ns:
        cs.setdefault("dips", {})[str(n)] = 0
        cs.setdefault("deeper", {}).pop(str(n), None)
    if (not ns or len(ns) == len(cs["cups"])) and not bool(_set("CUP_KEEP_LOWER", True)):
        cs.pop("lower", None)                                # the brush found too high in the last run: a new run starts plain
    # (2026-11-23, user: the corrections found while painting must carry on to the next picture - the brush's height has
    # nothing to do with how full the cups are; CUP_KEEP_LOWER = False forgets it at every refill as before)
    save(cs)
    return ns


def go_deeper(n):
    """The paint check (color_paint) saw too little paint from cup n: its dips go CUP_DEEPER_STEP_MM deeper from now on (kept
    in cups.json; python cups.py refill N sets it back). -> what was done, in words."""
    cs = load()
    if not cs:
        return "no cups.json"
    end = float(_set("CUP_DIP_END_MM", 5.0))
    now = dip_depth(cs, n, 0)
    if now <= end + 1e-6:
        return (f"its dips are already at the lowest ({end:g} mm over the floor): the cup is nearly EMPTY - refill it, then "
                f"python cups.py refill {n}")
    d = cs.setdefault("deeper", {})
    d[str(n)] = round(float(d.get(str(n), 0.0)) + float(_set("CUP_DEEPER_STEP_MM", 1.0)), 2)
    save(cs)
    return f"its next dips go {now - dip_depth(cs, n, 0):.1f} mm deeper ({dip_depth(cs, n, 0):.2f} mm over the floor)"


def dip_gcode(cs, cup, touch, dips_now, feed_travel, wipe_feed, start_xy=None, z_low=None):
    """One dip in `cup` as G-code lines, from wherever the brush is to the exit point beside the palette (at the safe height).
    The block keeps the shape paint.py and the repair expect: '; --- DIP CYCLE ---' first, the first 'G0 X' over the cup.
    Returns (lines, exit_xy)."""
    lay = layout()
    h = heights(touch)
    u, vp = np.array(cs["u"]), np.array(cs["to_paper"])
    ctr = np.array(cup["xy"], float)
    a_in, b_in = lay["inside"][0] / 2, lay["inside"][1] / 2
    stir = float(_set("CUP_STIR_MM", 4.0))
    ja = max(0.0, a_in - stir - 5.0) * float(_set("CUP_JITTER_SHARE", 0.6))
    jb = max(0.0, b_in - stir - 3.5) * float(_set("CUP_JITTER_SHARE", 0.6))
    p = ctr + u * random.uniform(-ja, ja) + vp * random.uniform(-jb, jb)
    over = dip_depth(cs, cup["n"], dips_now)
    z_dip = h["floor"] + over
    z_wipe = h["rim"] - float(_set("CUP_WIPE_BELOW_RIM_MM", 1.5))
    ft = int(feed_travel)
    # the point beside the palette (paper side, outside its keep-out) in line with the dip: the brush comes up DIAGONALLY on the way
    # there and is at the safe height before it reaches the first row of cups; then straight across to the cup
    pal_mid = (np.array(cs["pieces"][0]["c"]) + np.array(cs["pieces"][1]["c"])) / 2
    beyond = pal_mid.dot(vp) + lay["W"] / 2 + float(_set("CUP_KEEP_OUT_MM", 8.0)) + 2.0
    entry = p + vp * (beyond - p.dot(vp))
    head = ["\n; --- DIP CYCLE ---",
            f"; CUP {cup['n']} ({cup['kind']}) dip {int(cs.get('dips', {}).get(str(cup['n']), 0)) + dips_now + 1}: "
            f"{over:.2f} mm over its floor (Z {z_dip:.3f}), rim Z {h['rim']:.3f}"]
    acc_t0 = int(_set("ACCEL_TRAVEL", 12000))
    head += ["M400", f"M204 P{acc_t0} T{acc_t0}"]                         # travel acceleration (a stroke left the painting one on)
    if start_xy is not None and z_low is not None and float(np.dot(np.asarray(start_xy, float), vp)) > beyond - 0.5:
        head += [f"G0 Z{max(float(z_low), 0.0):.3f} F3000",                 # off the paper (a dip in the middle of a line)
                 f"G0 X{entry[0]:.3f} Y{entry[1]:.3f} Z{h['safe']:.3f} F{ft}",   # up on the way: safe before the first row
                 f"G0 X{p[0]:.3f} Y{p[1]:.3f} F{ft}"]
    else:                                                                   # already at / over the palette (after a rinse)
        head += [f"G0 Z{h['safe']:.3f} F3000", f"G0 X{p[0]:.3f} Y{p[1]:.3f} F{ft}"]
    out = head + [
           f"G1 Z{h['rim'] + 1.0:.3f} F3000",
           f"G1 Z{z_dip:.3f} F{int(_set('CUP_DIP_FEED', 600))}"]           # slowly into the paint
    for i in range(int(_set("CUP_STIR_LOOPS", 1)) * 4 + 1):              # a small stir in the paint
        ang = i * math.pi / 2
        q = p + stir * (math.cos(ang) * u + math.sin(ang) * vp)
        out.append(f"G1 X{q[0]:.3f} Y{q[1]:.3f} F{int(_set('CUP_STIR_FEED', 3000))}")
    if int(_set("CUP_SOAK_MS", 0)) > 0:                                   # the bristles take up paint
        out.append(f"G4 P{int(_set('CUP_SOAK_MS', 0))}")
    # removing the drops (as remove_drops in the earlier painters): out of the paint SLOWLY, so the drops run back into the cup,
    # a pause over the paint, then a slow wipe across the rim and a slow lift - a fast move flings the drops off
    slide = (p - ctr).dot(u) + random.uniform(-1.0, 1.0) * float(_set("CUP_WIPE_SPREAD_MM", 6.0))   # not always the same spot
    slide = max(-(a_in - 4.0), min(a_in - 4.0, slide))
    w0 = ctr + u * slide + vp * (b_in - 4.0)                              # inside, near the wall facing the paper ...
    w1 = ctr + u * slide + vp * (b_in + lay["wall"] + 1.5)                # ... across its top, the bristles bending over the rim
    lift_f, wipe_f = int(_set("CUP_LIFT_FEED", 400)), int(_set("CUP_WIPE_FEED", 600))
    acc_w, acc_t = int(_set("CUP_WIPE_ACCEL", 300)), int(_set("ACCEL_TRAVEL", 12000))
    out += ["; wipe on the rim",                                         # (the live preview shows WIPE)
            "M400", f"M204 P{acc_w} T{acc_w}",                            # gentle starts and stops: a jerk shakes drops off
            f"G1 X{p[0]:.3f} Y{p[1]:.3f} Z{min(z_dip + float(_set('CUP_DRIP_RISE_MM', 6.0)), z_wipe):.3f} F{lift_f}",
            f"G1 X{w0[0]:.3f} Y{w0[1]:.3f} Z{z_wipe:.3f} F{lift_f}"]
    if int(_set("CUP_DRIP_MS", 800)) > 0:                                 # drops fall back into the cup
        out.append(f"G4 P{int(_set('CUP_DRIP_MS', 800))}")
    out += [f"G1 X{w1[0]:.3f} Y{w1[1]:.3f} F{wipe_f}",
            f"G1 Z{h['rim'] + 3.0:.3f} F{lift_f}",
            "M400", f"M204 P{acc_t} T{acc_t}",
            f"G0 Z{h['safe']:.3f} F3000"]
    # out of the palette over the rims, beside it on the paper's side (the repair aims its wipe: this block keeps its own, ; CUP)
    edge = ctr.dot(vp)
    pal_mid = (np.array(cs["pieces"][0]["c"]) + np.array(cs["pieces"][1]["c"])) / 2
    beyond = pal_mid.dot(vp) + lay["W"] / 2 + float(_set("CUP_KEEP_OUT_MM", 8.0)) + 2.0
    e = ctr + vp * (beyond - edge) + u * (p - ctr).dot(u)
    out.append(f"G0 X{e[0]:.3f} Y{e[1]:.3f} F{ft}")
    return out, (float(e[0]), float(e[1]))


def check_moves(code, cs, touch, start=(0.0, 0.0, 99.0)):
    """Moves (outside the cup dips) that pass over the palette lower than the safe height over its rims."""
    import re
    if isinstance(code, str):
        code = code.split("\n")
    poly = footprint(cs)
    z_ok = heights(touch)["safe"] - 1.0
    pos, bad, in_cup = list(start), [], False
    for n, raw in enumerate(code):
        l = raw.strip()
        if l.startswith("; --- DIP CYCLE"):
            in_cup = False
        if l.startswith("; CUP "):
            in_cup = True
        if l.startswith("; travel to the next stroke") or l.startswith("; --- RINSE") or l.startswith("; ===== LAYER"):
            in_cup = False
        if not l.startswith(("G0", "G1")):
            continue
        tok = {k: float(v) for k, v in re.findall(r"([XYZ])(-?[\d.]+)", l.split(";")[0])}
        new = [tok.get("X", pos[0]), tok.get("Y", pos[1]), tok.get("Z", pos[2])]
        if not in_cup:
            d = math.hypot(new[0] - pos[0], new[1] - pos[1])
            for i in range(int(d // 2.0) + 2):
                t = i / (int(d // 2.0) + 1)
                q = [pos[k] + t * (new[k] - pos[k]) for k in range(3)]
                if q[2] < z_ok and _inside(q[:2], poly):
                    bad.append((n + 1, round(q[0], 1), round(q[1], 1), round(q[2], 1)))
                    break
        pos = new
    return bad


def warn_moves(code, cs, touch, what="the G-code"):
    bad = check_moves(code, cs, touch)
    if bad:
        print(f"\n   *** WARNING: {len(bad)} move(s) of {what} pass over the PALETTE lower than its rims "
              f"(first: line {bad[0][0]}, X {bad[0][1]} Y {bad[0][2]} Z {bad[0][3]}). The brush would hit a cup. "
              f"Answer N. ***\n")
    return bad


def add_dips(counts):
    """After a job painted to its end: the dips it made in each cup are added to cups.json (the next dips go deeper)."""
    cs = load()
    if not cs or not counts:
        return
    d = cs.setdefault("dips", {})
    for n, k in counts.items():
        d[str(n)] = int(d.get(str(n), 0)) + int(k)
    save(cs)


# ------------------------------------------------------------------------------------------------- pictures and commands
def rim_px(cs, cup, H, H_inv, size):
    """A cup's rim as the camera sees it (4 corners, camera px): its inside rectangle lifted to the rim's height (CUP_RIM_MM over
    the paper), so it sits where the cup is in the picture, not where it would be at paper height (parallax). size = (w, h) of
    the camera picture: things higher up are seen bigger away from its middle."""
    import vision as V
    lay = layout()
    u, vp = np.array(cs["u"]), np.array(cs["to_paper"])
    a, b = lay["inside"][0] / 2, lay["inside"][1] / 2
    k = seen_bigger(float(_set("CUP_RIM_MM", 27.0)))
    N = np.array(V.pixel_to_mm(np.asarray(H, float), size[0] / 2.0, size[1] / 2.0))
    ctr = np.array(cup["xy"], float)
    q = [N + (ctr + su * a * u + sv * b * vp - N) * k for su, sv in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    return V.mm_to_px_array(H_inv, q)


def draw(frame, H_inv, cs, png):
    import cv2
    import vision as V
    out = frame.copy()
    lay = layout()
    u, vp = np.array(cs["u"]), np.array(cs["to_paper"])
    a, b = lay["inside"][0] / 2, lay["inside"][1] / 2
    h_img, w_img = frame.shape[:2]
    H = np.linalg.inv(np.asarray(H_inv, float))
    N = np.array(V.pixel_to_mm(H, w_img / 2.0, h_img / 2.0))
    k = seen_bigger(float(_set("CUP_RIM_MM", 27.0)))
    for c in cs["cups"]:
        ctr = np.array(c["xy"])
        q = [N + (ctr + su * a * u + sv * b * vp - N) * k for su, sv in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        col = (0, 200, 255) if c["kind"] == "opaque" else (255, 160, 0)
        cv2.polylines(out, [V.mm_to_px_array(H_inv, q).astype(np.int32)], True, col, 2)
        m = V.mm_to_px_array(H_inv, [N + (ctr - N) * k])[0].astype(int)
        txt = f"{c['n']}" + ("" if c.get("filled", True) else " -")
        cv2.putText(out, txt, (int(m[0]) - 8, int(m[1]) + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(out, txt, (int(m[0]) - 8, int(m[1]) + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
    fp = [N + (p - N) * k for p in footprint(cs)]
    cv2.polylines(out, [V.mm_to_px_array(H_inv, fp).astype(np.int32)], True, (0, 0, 255), 1)
    ys = V.mm_to_px_array(H_inv, [N + (np.array(c["xy"]) - N) * k for c in cs["cups"]])
    x0, x1 = int(max(0, ys[:, 0].min() - 220)), int(min(w_img, ys[:, 0].max() + 220))
    y0, y1 = int(max(0, ys[:, 1].min() - 120)), int(min(h_img, ys[:, 1].max() + 120))
    cv2.imwrite(png, out[y0:y1, x0:x1])


def show(cs=None):
    cs = cs or load()
    if not cs:
        print("cups: none saved (python cups.py detect)")
        return
    print(f"cups.json ({cs.get('source')}, {cs.get('date')}); along the palette {cs['u']}, towards the paper {cs['to_paper']}")
    t = float(S.TOUCH_Z)
    h = heights(t)
    print(f"   heights for touch Z {t:.2f}: floor {h['floor']:.2f}, rim {h['rim']:.2f}, safe {h['safe']:.2f}")
    for c in cs["cups"]:
        over = dip_depth(cs, c["n"], 0)
        paint = ("EMPTY" if not c.get("filled", True) else
                 f"hue {c.get('hue', 0):5.1f} sat {c.get('sat', 0):5.1f} val {c.get('val', 0):5.1f}" + ("  dark" if c.get("dark") else "")
                 + (f"  (colour of cup {c['partner']})" if c.get("partner") else ""))
        print(f"   {c['n']:2d} {c['kind']:11s} X {c['xy'][0]:7.1f} Y {c['xy'][1]:7.1f}   {paint:40s} "
              f"dips {cs.get('dips', {}).get(str(c['n']), 0):3d}, next {over:.2f} mm over the floor")


def test(n=None):
    """python cups.py test [n]: one dip in cup n (default: the first filled cup), to watch it. The machine homes, the brush goes
    over the cup at the safe height and WAITS (is it over the middle?), then dips, stirs, wipes and leaves as in a job."""
    import re
    import machine as M
    cs = load()
    if not cs:
        sys.exit("no cups.json: python cups.py detect  first (after calibrate_dots.py)")
    cup = next((c for c in cs["cups"] if (n is None and c.get("filled")) or c["n"] == n), None)
    if cup is None:
        sys.exit("no filled cup found (python cups.py detect) - or give its number:  python cups.py test 7")
    try:
        import probe as PB
        off = PB.applied_offset_from_last_result()
    except Exception:
        off = [0.0, 0.0]
    touch = float(S.TOUCH_Z)
    h = heights(touch)
    lines, ex = dip_gcode(cs, cup, touch, 0, int(_set("FEED_TRAVEL", 12000)), int(_set("WIPE_FEED", 2000)))
    print(f"cup {cup['n']} ({cup['kind']}) at X {cup['xy'][0]:.1f} Y {cup['xy'][1]:.1f}; touch Z {touch:.2f} (TOUCH_Z): floor Z "
          f"{h['floor']:.2f}, rim {h['rim']:.2f}, safe {h['safe']:.2f}; tip offset ({off[0]:+.2f}, {off[1]:+.2f})")

    def shifted(l):
        l = re.sub(r"(?<=\bX)(-?\d+\.\d+)", lambda k: f"{float(k.group(1)) - off[0]:.3f}", l)
        return re.sub(r"(?<=\bY)(-?\d+\.\d+)", lambda k: f"{float(k.group(1)) - off[1]:.3f}", l)
    input("the toolhead free to home, the palette in place? Enter = start, Ctrl+C = stop ")
    m = M.open_machine(home=True)
    try:
        m.send_line(f"G0 Z{h['safe']:.3f} F1500")
        m.send_line(shifted(f"G0 X{cup['xy'][0]:.3f} Y{cup['xy'][1]:.3f} F6000"))
        m.send_line("M400", timeout=120)
        if input(f"over cup {cup['n']}, {h['safe'] - h['rim']:.0f} mm over the rims. Over the MIDDLE of the cup? "
                 f"Enter = dip, n = stop ").strip().lower() == "n":
            return
        for l in lines:
            if l.strip() and not l.strip().startswith(";"):
                m.send_line(shifted(l.strip()), timeout=120)
        m.send_line("M400", timeout=120)
        m.send_line(f"G0 X{_set('SNAPSHOT_X', 0.0):.3f} Y{_set('SNAPSHOT_Y', 453.0):.3f} F6000")
        m.send_line("M400", timeout=120)
        print("done (no dip counted)")
    finally:
        m.disconnect()


def _frame():
    import vision as V
    cal = V.load_calibration()
    cap = V.open_camera()
    try:
        fr = V.capture_frame(cap, 5)
    finally:
        cap.release()
    det = None
    try:
        det = V.detect_paper(fr, cal["H"], cal["H_inv"])
    except Exception:
        pass
    return fr, cal, (det or {}).get("corners_px")


def main(argv):
    cmd = argv[0] if argv else "show"
    if cmd == "detect":
        fr, cal, poly = _frame()
        cs = choose(fr, cal["H"], cal["H_inv"], poly)
        if cs:
            show(cs)
            print(f"check the picture: {os.path.join(HERE, 'cups_detect.png')} (orange = opaque, blue = transparent, "
                  f"red = where the brush stays high, '-' = empty)")
    elif cmd == "empty":
        fr, cal, poly = _frame()
        cs = choose(fr, cal["H"], cal["H_inv"], poly)
        if cs:
            save_empty(fr, cal["H_inv"], cs)
            print(f"the empty cups are remembered ({EMPTY_FILE})")
    elif cmd == "test":
        test(int(argv[1]) if len(argv) > 1 else None)
    elif cmd == "refill":
        cs = load()
        if not cs:
            raise SystemExit("cups: none saved (python cups.py detect)")
        ns = refill([int(x) for x in argv[1:]] or None)
        print(f"cups {', '.join(map(str, ns))}: dip count back to 0 (the next dip goes to {_set('CUP_DIP_START_MM', 10.0)} mm over the floor)")
    else:
        show()


if __name__ == "__main__":
    main(sys.argv[1:])
