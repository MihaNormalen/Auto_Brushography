"""
rinse.py - the long rinse container (hardware/rinse_container.stl): where it is, the rinse moves, and a safety check.

    python rinse.py calibrate      # jog the brush to the container once (again whenever the container is moved)
    python rinse.py test           # one rinse, to watch it (the machine homes first)
    python rinse.py show           # what rinse.json holds, and the heights the next job would use

CALIBRATE (a clean, dry brush; a sheet on the board): the brush is jogged by keys to five places, Enter records each:
    1. PAPER  - the tip just touching the paper (anywhere): the heights below are kept RELATIVE to this, so they follow the
                brush when it is mounted longer or shorter (the job uses the probe's measured touch instead)
    2. RIM    - over a wall of the container, the tip level with its top edge
    3. SLOT START - in the rinse slot at the blot end, the tip down where it should scrub (in the stuff you put in)
    4. SLOT END   - the same at the far end of the slot (keep a few mm from the end walls)
    5. BLOT   - the middle of the blot pad, the tip just touching it
After 2, 3, 4 and 5 the brush lifts by itself to RINSE_CLEAR_MM over the rim, so the next jog is safe.
Keys:  w / s = X - / +  (up / down in the camera picture)    a / d = Y - / +  (left / right in the picture)
       r / f = Z up / down        1 2 3 4 = step 0.1 / 1 / 5 / 10 mm        Enter = record        q = stop, nothing saved

The rinse (paint.py, before each colour of a job with several colours, and once at the end of every job):
    over the rim -> down into the slot at the blot end -> RINSE_PASSES times along the slot and back -> RINSE_BOBS times up and
    down at spots along it -> up over the rim -> BLOT_TIMES presses on the pad -> up over the rim -> on to the next dish.
The positions are saved like the dishes (where the TIP should be, the probe's tip offset removed), so the job's offset shift
puts the brush in the same place as during the calibration.
"""
import json
import math
import os
import sys
import time

import numpy as np

import settings as S

HERE = os.path.dirname(os.path.abspath(__file__))
RINSE_FILE = os.path.join(HERE, getattr(S, "RINSE_FILE", None) or "rinse.json")


def _set(name, default):
    return getattr(S, name, default)


# ------------------------------------------------------------------------------------------------- the saved container
def load():
    """rinse.json, or None (no container calibrated / RINSE_FILE = None in settings.py)."""
    if getattr(S, "RINSE_FILE", "rinse.json") is None or not os.path.isfile(RINSE_FILE):
        return None
    try:
        return json.load(open(RINSE_FILE))
    except Exception as e:
        print(f"rinse: {RINSE_FILE} could not be read ({e}): no rinsing")
        return None


def heights(r, touch):
    """Machine Z of the rinse heights for a brush that touches the paper at machine Z `touch`."""
    d = float(touch) - float(r["touch_z"])
    return {"rim": r["rim_z"] + d, "safe": r["rim_z"] + d + float(_set("RINSE_CLEAR_MM", 6.0)),
            "slot": (r["slot_z"][0] + d, r["slot_z"][1] + d), "blot": r["blot_z"] + d}


def footprint(r, margin=None):
    """The container as a segment and a half width (mm): every point closer than that to the segment is over the container."""
    margin = float(_set("RINSE_KEEP_OUT_MM", 8.0)) if margin is None else margin
    (x1, y1), (x2, y2), (bx, by) = r["slot"][0], r["slot"][1], r["blot"]
    L = math.hypot(x2 - x1, y2 - y1) or 1.0
    ux, uy = (x2 - x1) / L, (y2 - y1) / L
    # the container runs from beyond the blot pad (its end wall: about 17 mm past the pad's middle) to beyond the slot end
    tb = (bx - x1) * ux + (by - y1) * uy
    a = (x1 + ux * (min(tb, 0.0) - 17.0), y1 + uy * (min(tb, 0.0) - 17.0))
    b = (x2 + ux * 12.0, y2 + uy * 12.0)
    return a, b, 10.0 + margin


def _dist_seg(p, a, b):
    ax, ay, bx, by = a[0], a[1], b[0], b[1]
    L2 = (bx - ax) ** 2 + (by - ay) ** 2 or 1e-9
    t = max(0.0, min(1.0, ((p[0] - ax) * (bx - ax) + (p[1] - ay) * (by - ay)) / L2))
    return math.hypot(p[0] - ax - t * (bx - ax), p[1] - ay - t * (by - ay))


# ------------------------------------------------------------------------------------------------- found by the camera
def _runs(mask):
    """[(start, end)] index runs where mask is True (end exclusive)."""
    m = np.r_[False, np.asarray(mask, bool), False]
    d = np.diff(m.astype(int))
    return list(zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]))


def detect(frame, H, H_inv, paper_poly_px=None, debug_png=None, prior=None):
    """The rinse container in a camera picture: its BLACK RIM frames two light openings (the blot pad and the long slot).
    The rim stands RINSE_BOX_MM[2] above the paper, so the camera (calibrated on the paper) sees it a few % too big and pushed
    away from the middle of the picture: the known inside length (pad + divider + slot) says by how much, and that is taken out.
    Returns the rinse dict (like rinse.json, "source": "camera") or None, with the reason printed."""
    import cv2
    import vision as V
    L_box, W_box, H_box = (float(v) for v in _set("RINSE_BOX_MM", (180.0, 20.0, 30.0)))
    wall, dry = float(_set("RINSE_WALL_MM", 2.0)), float(_set("RINSE_DRY_INSIDE_MM", 30.0))
    slot_in = L_box - 3 * wall - dry                                     # 144
    inside_all = dry + wall + slot_in                                    # pad + divider + slot = 176
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    light = ((hsv[..., 2] > 120) & (hsv[..., 1] < 90)).astype(np.uint8) * 255
    if paper_poly_px is not None:                                        # never the paper
        ex = np.zeros(light.shape, np.uint8)
        cv2.fillPoly(ex, [np.asarray(paper_poly_px, np.int32)], 255)
        light[cv2.dilate(ex, np.ones((41, 41), np.uint8)) > 0] = 0
    light = cv2.morphologyEx(light, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(light, 8)
    boxes = []
    x_max = float(_set("MACHINE_X_MAX", 1e9)) + 15.0                     # the brush must reach it
    for i in range(1, n):
        if st[i, 4] < 400:
            continue
        ys, xs = np.nonzero(lab == i)
        mm = V.px_to_mm_array(H, np.c_[xs, ys]).astype(np.float32)
        (cx, cy), (w, h), ang = cv2.minAreaRect(mm)
        Lr, Wr = max(w, h), min(w, h)
        if not (0.85 * L_box < Lr < 1.3 * L_box and 0.7 * W_box < Wr < 1.8 * W_box):
            continue
        if cx > x_max:
            continue
        a = math.radians(ang if w >= h else ang + 90.0)
        boxes.append((abs(Lr / L_box - 1.05) + abs(Wr / W_box - 1.1), np.array([cx, cy]),
                      np.array([math.cos(a), math.sin(a)]), Lr, Wr))
    if prior and prior.get("slot") and prior.get("blot"):
        # the last known place is a candidate too (its light area can look nothing like a box once the water is coloured):
        # its box middle and direction, as the camera sees them at rim height; fit() searches around it
        k0 = float(_set("RINSE_SEEN_BIGGER", None) or 1.035)
        N0 = np.array(V.pixel_to_mm(H, frame.shape[1] / 2.0, frame.shape[0] / 2.0))
        pa_, pb_, pbl = (np.array(prior["slot"][0], float), np.array(prior["slot"][1], float), np.array(prior["blot"], float))
        ud = (pb_ - pa_) / (np.linalg.norm(pb_ - pa_) or 1.0)
        mid_true = pbl + ud * (inside_all / 2 - dry / 2)                     # pad middle -> box middle
        boxes.append((9.0, N0 + (mid_true - N0) * k0, ud, L_box * k0 * 1.05, W_box * k0 * 1.15))
    if not boxes:
        print("rinse: the camera sees no light box of about "
              f"{L_box:g} x {W_box:g} mm outside the paper (RINSE_BOX_MM in settings.py)")
        return None
    why = []

    def fit(c, u, Lr, Wr):
        v = np.array([-u[1], u[0]])
        R = 0.25                                                             # mm per sample of the straightened strip
        us = np.arange(-Lr / 2 - 8, Lr / 2 + 8, R)
        vs = np.arange(-Wr / 2 - 6, Wr / 2 + 6, R)
        UU, VV = np.meshgrid(us, vs)
        P = c + UU[..., None] * u + VV[..., None] * v
        pxy = V.mm_to_px_array(H_inv, P.reshape(-1, 2)).reshape(P.shape).astype(np.float32)
        g = cv2.cvtColor(cv2.remap(frame, pxy[..., 0], pxy[..., 1], cv2.INTER_LINEAR), cv2.COLOR_BGR2GRAY).astype(float)
        mid_u = np.abs(us) < 0.3 * Lr
        # The BLACK parts are found, not the light openings: coloured rinse water made the slot as dark as the rim and the pad and
        # slot ran together (2026-11-12). The rim's height factor is fixed geometry (RINSE_SEEN_BIGGER), so the black rim bands,
        # end walls and divider must sit at their known spacing: that pattern is searched for
        k_h = float(_set("RINSE_SEEN_BIGGER", None) or 1.035)
        hw = wall / 2.0 * k_h                                                 # half a wall, as seen

        def at(prof, xs, x):                                                  # mean brightness within half a wall of x
            m = np.abs(xs - x) <= hw
            return float(prof[m].mean()) if m.any() else 1e9
        # ACROSS: the two long rim bands, (W - wall) apart
        pa = g[:, mid_u].mean(axis=1)
        half_w = (W_box - wall) / 2.0 * k_h
        best_v = min((at(pa, vs, x - half_w) + at(pa, vs, x + half_w), x) for x in np.arange(-6.0, 6.0 + R, R))
        vc = best_v[1]
        if best_v[0] / 2 > 0.8 * float(np.median(pa)):
            why.append("box found, but no dark rim bands along it (is the rim still black?)")
            return None
        v0, v1 = vc - (W_box / 2 - wall) * k_h, vc + (W_box / 2 - wall) * k_h
        # ALONG: end wall, divider, end wall (the pad at either end)
        rows = np.abs(vs - vc) < 0.25 * (W_box - 2 * wall) * k_h
        pl = g[rows].mean(axis=0)
        e = (inside_all / 2 + wall / 2) * k_h                                 # end walls' middles from the box middle
        dv = (-inside_all / 2 + dry + wall / 2) * k_h                         # the divider's middle (pad on the - side)
        cands = []
        for u0 in np.arange(-20.0, 20.0 + R, R):
            for sg in (1.0, -1.0):
                sc = (at(pl, us, u0 - e) + at(pl, us, u0 + e) + at(pl, us, u0 + sg * dv)) / 3.0
                cands.append((sc, u0, sg))
        sc, u0, sign = min(cands)
        if sc > 0.8 * float(np.median(pl)):
            why.append("box found, but no dark end walls and divider at the right spacing")
            return None
        k_meas = k_h
        p_mid = u0 + sign * (-inside_all / 2 + dry / 2) * k_h                 # the blot pad's middle
        s0 = u0 + sign * (-inside_all / 2 + dry + wall) * k_h                 # the slot, from the divider ...
        s1 = u0 + sign * (inside_all / 2) * k_h                               # ... to the far end wall
        pad = tuple(sorted((u0 + sign * (-inside_all / 2) * k_h, u0 + sign * (-inside_all / 2 + dry) * k_h)))
        slot = tuple(sorted((s0, s1)))
        h_img, w_img = frame.shape[:2]
        N = np.array(V.pixel_to_mm(H, w_img / 2.0, h_img / 2.0))            # straight below the camera, about

        def true(uu):                                                         # a point of the centre line, seen at rim height -> real
            p = c + uu * u + vc * v
            return N + (p - N) / k_h
        end_m = float(_set("RINSE_END_MARGIN_MM", 6.0)) * k_h
        s_start, s_end = s0 + sign * end_m, s1 - sign * end_m
        a, b, bl = true(s_start), true(s_end), true(p_mid)
        quality = best_v[0] / 2 / float(np.median(pa)) + sc / float(np.median(pl))   # lower = blacker rim pattern
        return quality, dict(a=a, b=b, bl=bl, pad=pad, slot=slot, v0=v0, v1=v1, c=c, u=u, v=v, Lr=Lr, Wr=Wr, N=N,
                             k_h=k_h, k_meas=k_meas)
    fits = [f for f in (fit(bx[1], bx[2], bx[3], bx[4]) for bx in sorted(boxes, key=lambda t: t[0])) if f]
    if not fits:
        print("rinse: " + (why[0] if why else "no container found"))
        return None
    F = min(fits, key=lambda f: f[0])[1]
    a, b, bl, pad, slot, v0, v1, c, u, v, Lr, Wr, N, k_h, k_meas = (F[k] for k in (
        "a", "b", "bl", "pad", "slot", "v0", "v1", "c", "u", "v", "Lr", "Wr", "N", "k_h", "k_meas"))
    touch = float(S.TOUCH_Z)
    rim = touch + H_box - float(_set("RINSE_PAPER_MM", 0.3))
    r = {"source": "camera", "touch_z": round(touch, 3), "rim_z": round(rim, 3),
         "slot": [[round(float(a[0]), 2), round(float(a[1]), 2)], [round(float(b[0]), 2), round(float(b[1]), 2)]],
         "slot_z": [round(rim - float(_set("RINSE_SCRUB_BELOW_RIM_MM", 6.0)), 3)] * 2,
         "blot": [round(float(bl[0]), 2), round(float(bl[1]), 2)], "blot_z": round(rim - float(_set("BLOT_BELOW_RIM_MM", 4.0)), 3),
         "seen_bigger_by": round(k_h, 3), "seen_bigger_measured": round(k_meas, 3), "date": time.strftime("%Y-%m-%d %H:%M")}
    if debug_png:
        dbg = frame.copy()
        box = [c + su * (Lr / 2) * u + sv * (Wr / 2) * v for su, sv in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        cv2.polylines(dbg, [V.mm_to_px_array(H_inv, box).astype(np.int32)], True, (0, 200, 255), 1)
        for (p0, p1), col in (((pad[0], pad[1]), (255, 128, 0)), ((slot[0], slot[1]), (0, 255, 0))):
            q = [c + p0 * u + v0 * v, c + p1 * u + v0 * v, c + p1 * u + v1 * v, c + p0 * u + v1 * v]
            cv2.polylines(dbg, [V.mm_to_px_array(H_inv, q).astype(np.int32)], True, col, 2)
        # where the brush goes, drawn as the camera will see the tip at rim height (it should sit in the middle of the openings)
        for p, col in ((a, (0, 0, 255)), (b, (0, 0, 255)), (bl, (255, 0, 255))):
            q = V.mm_to_px_array(H_inv, [N + (p - N) * k_h])[0]
            cv2.circle(dbg, (int(q[0]), int(q[1])), 5, col, 2)
        x0, y0 = V.mm_to_px_array(H_inv, [c])[0].astype(int)
        crop = dbg[max(0, y0 - 140):y0 + 140, max(0, x0 - 420):x0 + 420]
        cv2.imwrite(debug_png, cv2.resize(crop, None, fx=2, fy=2))
    return r


def choose(frame, H, H_inv, paper_poly_px, saved):
    """For a job: the container where the camera sees it now. A hand calibration (python rinse.py calibrate) is kept, but a
    warning is printed when the camera puts the container somewhere else; without the camera the saved one is used."""
    seen = None
    try:
        seen = detect(frame, H, H_inv, paper_poly_px, debug_png=os.path.join(HERE, "rinse_detect.png"), prior=saved or load())
    except Exception as e:
        print(f"rinse: camera detection failed ({e})")
    if saved and saved.get("source") != "camera":
        if seen:
            d = max(math.dist(seen["slot"][0], saved["slot"][0]), math.dist(seen["slot"][1], saved["slot"][1]))
            if d > 4.0:
                print(f"\n   *** WARNING: the camera sees the rinse container {d:.0f} mm away from your calibration (rinse.json). "
                      f"Moved? Run  python rinse.py calibrate  again, or delete rinse.json to use the camera. ***\n")
        return saved
    if seen and saved and saved.get("slot"):
        d = max(math.dist(seen["slot"][0], saved["slot"][0]), math.dist(seen["slot"][1], saved["slot"][1]))
        if d <= float(_set("RINSE_LOCK_MM", 5.0)):
            print(f"rinse container: in its saved place (the camera reads it {d:.1f} mm from there; RINSE_LOCK_MM)")
            return saved
        # THE SAME BOX READ THE OTHER WAY ROUND (2026-11-21: with the new camera view the camera put the blot pad at the wrong
        # end - the brush scrubbed into the divider): compare the box's two outer ends, whichever part is where
        def ends(r):
            pts = [tuple(r["slot"][0]), tuple(r["slot"][1]), tuple(r["blot"])]
            a_, b_ = max(((p, q) for p in pts for q in pts), key=lambda pq: math.dist(*pq))
            return sorted([a_, b_])
        ea, eb = ends(seen), ends(saved)
        d_box = max(math.dist(ea[0], eb[0]), math.dist(ea[1], eb[1]))
        if d_box <= float(_set("RINSE_LOCK_MM", 5.0)):
            print(f"\n   *** rinse container: in its saved place, but the camera read it TURNED ROUND (blot pad at the other end). "
                  f"The saved positions are kept (rinse.json). Look at rinse_detect.png. ***\n")
            return saved
        print(f"rinse container: MOVED by {d:.0f} mm, the new place is used")
    if seen:
        print(f"rinse container found by the camera: slot X {seen['slot'][0][0]:.1f} Y {seen['slot'][0][1]:.1f} -> "
              f"X {seen['slot'][1][0]:.1f} Y {seen['slot'][1][1]:.1f}, blot X {seen['blot'][0]:.1f} Y {seen['blot'][1]:.1f} (rinse_detect.png)")
        try:
            json.dump(seen, open(RINSE_FILE, "w"), indent=1)
        except Exception:
            pass
        return seen
    if saved:
        print("rinse: the camera does not see the container now: its last known place is used (rinse.json)")
    return saved


def detect_now():
    """python rinse.py detect: a picture now, the container found in it, saved to rinse.json, rinse_detect.png to check."""
    import cv2
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
    saved = load()
    r = choose(fr, cal["H"], cal["H_inv"], (det or {}).get("corners_px"), None if (saved or {}).get("source") == "camera" else saved)
    if r:
        show(r)
        print(f"check the picture: {os.path.join(HERE, 'rinse_detect.png')} (green = slot, blue = blot pad, red / pink = where the brush goes)")
    cv2.imwrite(os.path.join(HERE, "rinse_frame.png"), fr)


# ------------------------------------------------------------------------------------------------- the moves
def gcode(r, touch, to_xy, feed_travel=None):
    """The rinse as G-code lines (positions where the tip should be: the job's tool-offset shift applies to them like to every
    move). Starts wherever the brush is (it first goes straight UP to the safe height), ends at the safe height above to_xy."""
    h = heights(r, touch)
    ft = int(feed_travel or _set("FEED_TRAVEL", 12000))
    fr = int(_set("RINSE_FEED", 2000))
    press = float(_set("RINSE_PRESS_MM", 1.5))
    (x1, y1), (x2, y2) = r["slot"]
    z1, z2 = h["slot"][0] - press, h["slot"][1] - press
    out = ["; --- RINSE ---", "M400", f"M204 P{_set('ACCEL_TRAVEL', 12000)} T{_set('ACCEL_TRAVEL', 12000)}",
           f"G0 Z{h['safe']:.3f} F3000",
           f"G0 X{x1:.3f} Y{y1:.3f} F{ft}",
           f"G1 Z{z1:.3f} F1500"]
    for _ in range(int(_set("RINSE_PASSES", 3))):                       # along the slot and back
        out += [f"G1 X{x2:.3f} Y{y2:.3f} Z{z2:.3f} F{fr}", f"G1 X{x1:.3f} Y{y1:.3f} Z{z1:.3f} F{fr}"]
    nb, bob = int(_set("RINSE_BOBS", 4)), float(_set("RINSE_BOB_MM", 4.0))
    for k in range(nb):                                                 # up and down at spots along the slot
        t = (k + 0.5) / max(nb, 1)
        x, y, z = x1 + t * (x2 - x1), y1 + t * (y2 - y1), z1 + t * (z2 - z1)
        out += [f"G1 X{x:.3f} Y{y:.3f} Z{z:.3f} F{fr}", f"G1 Z{min(z + bob, h['safe']):.3f} F1500", f"G1 Z{z:.3f} F1500"]
    out.append(f"G0 Z{h['safe']:.3f} F3000")
    bx, by = r["blot"]
    L = math.hypot(x2 - x1, y2 - y1) or 1.0
    ux, uy = (x2 - x1) / L, (y2 - y1) / L
    n_bl, bp, step = int(_set("BLOT_TIMES", 3)), float(_set("BLOT_PRESS_MM", 1.5)), float(_set("BLOT_STEP_MM", 3.0))
    for k in range(n_bl):                                               # pressed on the pad, a little further along each time
        o = (k - (n_bl - 1) / 2.0) * step
        px, py = bx + ux * o, by + uy * o
        out += [f"G0 X{px:.3f} Y{py:.3f} F{ft}", f"G1 Z{h['blot'] - bp:.3f} F1500", f"G1 Z{h['blot'] + 3.0:.3f} F1500"]
    out += [f"G0 Z{h['safe']:.3f} F3000", f"G0 X{to_xy[0]:.3f} Y{to_xy[1]:.3f} F{ft}", "; --- RINSE END ---"]
    return out


def check_moves(code, r, touch, start=(0.0, 0.0, 99.0)):
    """Moves of the G-code (lines or text) that pass over the container LOWER than over its rim (rinse moves excluded).
    Returns [(line number, x, y, z)] of the first offending point of each such move."""
    if isinstance(code, str):
        code = code.split("\n")
    import re
    a, b, half = footprint(r)
    z_ok = heights(r, touch)["safe"] - 1.0
    pos, bad, in_rinse = list(start), [], False
    for n, raw in enumerate(code):
        l = raw.strip()
        if l.startswith("; --- RINSE ---"):
            in_rinse = True
        elif l.startswith("; --- RINSE END"):
            in_rinse = False
        if not l.startswith(("G0", "G1")):
            continue
        tok = {k: float(v) for k, v in re.findall(r"([XYZ])(-?[\d.]+)", l.split(";")[0])}
        new = [tok.get("X", pos[0]), tok.get("Y", pos[1]), tok.get("Z", pos[2])]
        if not in_rinse:
            d = math.hypot(new[0] - pos[0], new[1] - pos[1])
            for i in range(int(d // 2.0) + 2):
                t = i / (int(d // 2.0) + 1)
                p = [pos[k] + t * (new[k] - pos[k]) for k in range(3)]
                if p[2] < z_ok and _dist_seg(p, a, b) < half:
                    bad.append((n + 1, round(p[0], 1), round(p[1], 1), round(p[2], 1)))
                    break
        pos = new
    return bad


def warn_moves(code, r, touch, what="the G-code"):
    bad = check_moves(code, r, touch)
    if bad:
        print(f"\n   *** WARNING: {len(bad)} move(s) of {what} pass over the RINSE CONTAINER lower than its rim "
              f"(first: line {bad[0][0]}, X {bad[0][1]} Y {bad[0][2]} Z {bad[0][3]}). The brush would hit it. "
              f"Move the paper or the container, or answer N. ***\n")
    return bad


# ------------------------------------------------------------------------------------------------- calibration (by hand)
def _key():
    import msvcrt
    c = msvcrt.getwch()
    if c in ("\x00", "\xe0"):
        msvcrt.getwch()
        return ""
    return c


def _jog(m, pos, title, lines):
    """Jog with the keys until Enter (returns the position) or q (returns None)."""
    steps = {"1": 0.1, "2": 1.0, "3": 5.0, "4": 10.0}
    step = 1.0
    print(f"\n=== {title} ===")
    for l in lines:
        print("   " + l)
    print("   w/s = X-/+   a/d = Y-/+   r/f = Z up/down   1/2/3/4 = step 0.1/1/5/10 mm   Enter = record   q = stop")
    while True:
        print(f"\r   X {pos[0]:7.2f}  Y {pos[1]:7.2f}  Z {pos[2]:6.2f}   step {step:g} mm     ", end="", flush=True)
        k = _key().lower()
        if k in steps:
            step = steps[k]
            continue
        if k in ("\r", "\n"):
            print()
            return list(pos)
        if k == "q":
            print()
            return None
        d = {"w": (0, -1), "s": (0, 1), "a": (1, -1), "d": (1, 1), "r": (2, 1), "f": (2, -1)}.get(k)
        if d is None:
            continue
        pos[d[0]] += d[1] * step
        if d[0] == 2:
            pos[2] = max(float(getattr(S, "Z_MIN_MM", 0.3)), pos[2])
            m.send_line(f"G0 Z{pos[2]:.3f} F1500")
        else:
            m.send_line(f"G0 X{pos[0]:.3f} Y{pos[1]:.3f} F3000")
        m.send_line("M400", timeout=60)


def _offset_now():
    """The tip offset the next job would apply (the last usable probe result), so the saved points are where the TIP is."""
    try:
        import probe as PB
        return PB.applied_offset_from_last_result()
    except Exception:
        return [0.0, 0.0]


def calibrate():
    import machine as M
    print(__doc__.split("CALIBRATE")[1].split("The rinse (")[0])
    input("A clean, dry brush, a sheet on the board, the toolhead free to home? Enter = start, Ctrl+C = stop ")
    m = M.open_machine(home=True)
    # start at a known point: over the middle of the calibration dots (on the sheet, far from the container), the brush
    # TOUCH_Z + 5 high
    cx = sum(c[0] for c in S.CANVAS_CORNERS_MM) / len(S.CANVAS_CORNERS_MM)
    cy = sum(c[1] for c in S.CANVAS_CORNERS_MM) / len(S.CANVAS_CORNERS_MM)
    pos = [cx, cy, max(float(_set("HOME_LIFT_Z", 8.0)), float(S.TOUCH_Z) + 5.0)]
    m.send_line("M204 P1000 T1000")
    m.send_line(f"G0 Z{pos[2]:.3f} F1500")
    m.send_line(f"G0 X{pos[0]:.3f} Y{pos[1]:.3f} F6000")
    m.send_line("M400", timeout=60)
    try:
        p = _jog(m, pos, "1 of 5: PAPER", ["move over the sheet, lower (f) until the tip JUST touches the paper"])
        if p is None:
            return
        touch = p[2]
        m.send_line(f"G0 Z{touch + 5:.3f} F1500"); pos = p; pos[2] = touch + 5
        p = _jog(m, pos, "2 of 5: RIM", ["raise (r) WELL above the container first, then move over one of its walls,",
                                          "lower until the tip is level with the top edge (the black rim)"])
        if p is None:
            return
        rim = p[2]
        safe = rim + float(_set("RINSE_CLEAR_MM", 6.0))

        def lift(p):
            m.send_line(f"G0 Z{safe:.3f} F1500"); m.send_line("M400", timeout=60)
            return [p[0], p[1], safe]
        pos = lift(p)
        p3 = _jog(m, pos, "3 of 5: SLOT START", ["the rinse slot at the BLOT end (keep ~5 mm from the divider),",
                                                  "lower until the tip is where it should scrub (in the stuff you put in)"])
        if p3 is None:
            return
        pos = lift(p3)
        p4 = _jog(m, pos, "4 of 5: SLOT END", ["the far end of the slot (keep ~5 mm from the end wall), the same depth"])
        if p4 is None:
            return
        pos = lift(p4)
        p5 = _jog(m, pos, "5 of 5: BLOT", ["the middle of the blot pad, the tip just touching it"])
        if p5 is None:
            return
        pos = lift(p5)
        off = _offset_now()
        r = {"touch_z": round(touch, 3), "rim_z": round(rim, 3),
             "slot": [[round(p3[0] + off[0], 3), round(p3[1] + off[1], 3)], [round(p4[0] + off[0], 3), round(p4[1] + off[1], 3)]],
             "slot_z": [round(p3[2], 3), round(p4[2], 3)],
             "blot": [round(p5[0] + off[0], 3), round(p5[1] + off[1], 3)], "blot_z": round(p5[2], 3),
             "tip_offset_at_calibration": off, "date": time.strftime("%Y-%m-%d %H:%M")}
        json.dump(r, open(RINSE_FILE, "w"), indent=1)
        print(f"\nsaved {RINSE_FILE}:")
        show(r)
        if input("\none rinse now, to watch it? [Y/n] ").strip().lower() != "n":
            _run(m, r, touch, off, pos[:2])
        m.send_line(f"G0 X{_set('SNAPSHOT_X', 0.0):.3f} Y{_set('SNAPSHOT_Y', 453.0):.3f} F6000")
        m.send_line("M400", timeout=120)
    finally:
        try:
            m.disconnect()
        except Exception:
            pass


def _run(m, r, touch, off, to_xy):
    """Sends one rinse to the machine (shifted by the tip offset, like a job)."""
    for l in gcode(r, touch, (to_xy[0] + off[0], to_xy[1] + off[1])):
        if l.startswith(("G0", "G1")):
            import re
            l = re.sub(r"(?<=\bX)(-?\d+\.\d+)", lambda k: f"{float(k.group(1)) - off[0]:.3f}", l)
            l = re.sub(r"(?<=\bY)(-?\d+\.\d+)", lambda k: f"{float(k.group(1)) - off[1]:.3f}", l)
        if not l.startswith(";"):
            m.send_line(l, timeout=120)
    m.send_line("M400", timeout=120)


def test():
    import machine as M
    r = load()
    if r is None:
        sys.exit(f"no {RINSE_FILE}: run  python rinse.py calibrate  first")
    touch = float(r["touch_z"])
    off = _offset_now()
    print(f"one rinse, brush touching the paper at Z {touch:.2f} as at the calibration, tip offset ({off[0]:+.2f}, {off[1]:+.2f})")
    m = M.open_machine(home=True)
    try:
        m.send_line(f"G0 Z{heights(r, touch)['safe']:.3f} F1500")
        sx, sy = _set("SNAPSHOT_X", 0.0), _set("SNAPSHOT_Y", 453.0)
        _run(m, r, touch, off, (sx, sy))
    finally:
        m.disconnect()


def show(r=None):
    r = r or load()
    if r is None:
        print(f"no {RINSE_FILE}")
        return
    h = heights(r, r["touch_z"])
    print(f"   paper touch Z {r['touch_z']:.2f}; rim Z {h['rim']:.2f} ({h['rim'] - r['touch_z']:+.1f} mm), moves over it at Z {h['safe']:.2f}")
    print(f"   slot  X {r['slot'][0][0]:.1f} Y {r['slot'][0][1]:.1f} Z {h['slot'][0]:.2f}  ->  X {r['slot'][1][0]:.1f} Y {r['slot'][1][1]:.1f} "
          f"Z {h['slot'][1]:.2f}  (scrubs {_set('RINSE_PRESS_MM', 1.5):g} mm deeper)")
    print(f"   blot  X {r['blot'][0]:.1f} Y {r['blot'][1]:.1f} Z {h['blot']:.2f}  (pressed {_set('BLOT_PRESS_MM', 1.5):g} mm, {_set('BLOT_TIMES', 3)} times)")
    print(f"   all heights follow the brush: the job adds (its measured touch - {r['touch_z']:.2f})")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "show"
    if cmd == "detect" and len(sys.argv) > 2:                           # python rinse.py detect saved_picture.png
        import cv2
        import vision as V
        cal = V.load_calibration()
        fr = cv2.imread(sys.argv[2])
        det = V.detect_paper(fr, cal["H"], cal["H_inv"])
        r = detect(fr, cal["H"], cal["H_inv"], (det or {}).get("corners_px"), debug_png=os.path.join(HERE, "rinse_detect.png"), prior=load())
        show(r) if r else None
    else:
        {"calibrate": calibrate, "test": test, "show": show, "detect": detect_now}.get(cmd, show)()
