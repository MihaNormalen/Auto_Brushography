"""
probe.py - finds where THIS brush with THIS paint touches THIS paper, on the white margin around the artwork.

    python probe.py               homes, photographs, asks, paints four long lines, measures, saves probe_result.json
    python probe.py --plan_only   only the plan picture (the machine is not touched)
    python probe.py --analyse     measure again from the saved pictures (probe_before.png / probe_after.png / probe_plan.json)

The long lines. Four lines run along the four sides of the paper, inside the white margin, each covering most of its side
(PROBE_END_MM of paper is left at both ends). Along each line the brush goes DOWN slowly and steadily, from a height that is
surely in the air (PROBE_ABOVE_MM above the expected contact) to a height that is surely deep enough (PROBE_BELOW_MM below).
The camera measures how wide the mark is at every point along the line:
  - touch point = where the mark first reaches PROBE_CONTACT_WIDTH_MM; its position along the line gives the height there;
  - paint reference = touch height minus PROBE_MARGIN_MM (deeper): the lines are painted from there, so they sit on the
    reliable part of the width curve, not on the edge where a small height error changes the width a lot;
  - the four touch heights around the sheet give the tilt of the paper (a plane through four points, with a check).
"""
import argparse
import json
import math
import os
import time

import cv2
import numpy as np
from scipy.ndimage import map_coordinates

import brush_profile as BP
import paint as PT
import settings as S
import vision as V

BEFORE, AFTER, PLAN = "probe_before.png", "probe_after.png", "probe_plan.json"
HERE = os.path.dirname(os.path.abspath(__file__))
LAST_STATUS = ""            # why the last run_probes gave no result (paint.py writes it into the console and the G-code header)

# settings of this tool, with defaults: an older settings.py without them keeps working
PROBE_END_MM = float(getattr(S, "PROBE_END_MM", 8.0))           # paper left free at each end of a line
PROBE_SAMPLE_MM = float(getattr(S, "PROBE_SAMPLE_MM", 1.0))     # one point of the line every this many mm
PROBE_ABOVE_MM = float(getattr(S, "PROBE_ABOVE_MM", 1.0))       # the line starts this far ABOVE the expected contact (in the air)
PROBE_BELOW_MM = float(getattr(S, "PROBE_BELOW_MM", 3.0))
PROBE_COUNT = int(getattr(S, "PROBE_COUNT", 4))                   # 4 = one line per edge (touch + tilt); 2 = left + right only
PROBE_SIDES = (("left", "right", "top", "bottom") if PROBE_COUNT >= 4                      # 2: PROBE_PAIR, left + right by default:
               else tuple(getattr(S, "PROBE_PAIR", ("left", "right"))))                      # the paper lies flattest there (user, 2026-11-01)
PROBE_LINE_MM = float(getattr(S, "PROBE_LINE_MM", 80.0))          # ONE line per side, this long at most, centred on its edge: one dip covers it
PROBE_RETRY_MM = float(getattr(S, "PROBE_RETRY_MM", 1.0))         # when lines touch at their start, the start is raised by this much and all lines are painted again
PROBE_RETRIES = int(getattr(S, "PROBE_RETRIES", 0))               # 0 = one probe only (retries repaint the same lines on top of the old marks)               # at most this many such retries          # each side is split into pieces of at most this length; every piece gets its own dip and its own ramp       # and ends this far BELOW it
PROBE_MARGIN_MM = float(getattr(S, "PROBE_MARGIN_MM", 0.25))    # paint reference = touch height minus this (deeper)
PROBE_COLOR = getattr(S, "PROBE_COLOR", "blue")
PROBE_CONTACT_WIDTH_MM = float(getattr(S, "PROBE_CONTACT_WIDTH_MM", 0.9))
PROBE_INSET_MM = float(getattr(S, "PROBE_INSET_MM", 5.0))       # the lines are this far inside the paper's edge
PROBE_STEP_STROKES = bool(getattr(S, "PROBE_STEP_STROKES", True))  # short strokes landed like artwork lines give height + width curve
PROBE_STEP_COUNT = int(getattr(S, "PROBE_STEP_COUNT", 10))
PROBE_STEP_LEN_MM = float(getattr(S, "PROBE_STEP_LEN_MM", 5.0))
PROBE_STEP_GAP_MM = float(getattr(S, "PROBE_STEP_GAP_MM", 3.5))
PROBE_STEP_ABOVE_MM = float(getattr(S, "PROBE_STEP_ABOVE_MM", 0.75))
PROBE_STEP_BELOW_MM = float(getattr(S, "PROBE_STEP_BELOW_MM", 2.75))
PROBE_FILE = os.path.join(HERE, getattr(S, "PROBE_FILE", "probe_result.json"))   # a plain name = next to the scripts
SHEET_FILE = os.path.join(HERE, "probe_sheet.json")             # where the last sheet was and how many probes it has had
PROBE_OFFSET_MAX_MM = float(getattr(S, "PROBE_OFFSET_MAX_MM", 6.0))          # a measured tip offset larger than this is not used
PROBE_OFFSET_SCATTER_MM = float(getattr(S, "PROBE_OFFSET_SCATTER_MM", 0.5))  # ... nor one where the lines disagree with the fit by more
Z_MIN = 0.3                                                      # the machine is never sent below this Z


# ------------------------------------------------------------------------------------------------ planning
def zpath(p, s):
    """The commanded Z at distance s (mm) along line p: a straight slope from z_start to z_end."""
    return p["z_start"] + (p["z_end"] - p["z_start"]) * (np.asarray(s, float) / p["length"])


def plan_probes(pf, contact_guess):
    """Four long lines inside the paper's sides (inset PROBE_INSET_MM from the edge, PROBE_END_MM short of each corner).
    Each line: start, direction, its length and Z at its two ends, and one sample point per PROBE_SAMPLE_MM."""
    o, u, v = np.array(pf["origin_mm"]), np.array(pf["u"]), np.array(pf["v"])
    inset, end, ds = PROBE_INSET_MM, PROBE_END_MM, PROBE_SAMPLE_MM
    z_start = contact_guess + PROBE_ABOVE_MM
    z_end = max(Z_MIN, contact_guess - PROBE_BELOW_MM)
    # (name, axis the line runs along, fixed coordinate, from, to), coordinates in the paper's u / v axes (mm)
    spec = [("left", "v", pf["a_left"] + inset, pf["b_bot"] + end, pf["b_top"] - end),
            ("right", "v", pf["a_right"] - inset, pf["b_bot"] + end, pf["b_top"] - end),
            ("top", "u", pf["b_top"] - inset, pf["a_left"] + end, pf["a_right"] - end),
            ("bottom", "u", pf["b_bot"] + inset, pf["a_left"] + end, pf["a_right"] - end)]
    probes = []
    for name, axis, fixed, t0, t1 in spec:
        n = max(2, int(round((t1 - t0) / ds)))
        if axis == "v":
            start, d = o + fixed * u + t0 * v, v
        else:
            start, d = o + t0 * u + fixed * v, u
        L = n * ds
        p = {"name": name, "start": start.tolist(), "dir": d.tolist(), "normal": [-float(d[1]), float(d[0])],
             "length": float(L), "seg": ds, "z_start": float(z_start), "z_end": float(z_end)}
        p["z_steps"] = [float(zpath(p, (i + 0.5) * ds)) for i in range(n)]    # Z at the middle of every sample cell
        probes.append(p)
    return probes


def plan_probes_on_sheet(corners_mm, contact_guess, slot=0):
    """Probe lines inside each PROBE_INSET_MM from its OWN sheet edge (corners tl, tr, br, bl in machine mm). One line per side,
    at most PROBE_LINE_MM long, with its own dip and its own ramp from PROBE_ABOVE_MM above the expected contact down to
    PROBE_BELOW_MM below it (so the ramp reaches the wide end).
    slot = how many probes this sheet already had: the lines move along their edge (centre, then one line length to one side,
    then to the other) so they never land on the old marks - the before/after photo would see no new paint there."""
    C = np.array(corners_mm, float)
    centre = C.mean(axis=0)
    inset, end, ds = PROBE_INSET_MM, PROBE_END_MM, PROBE_SAMPLE_MM
    z_start = contact_guess + PROBE_ABOVE_MM
    z_end = max(Z_MIN, contact_guess - PROBE_BELOW_MM)
    spec = [s for s in [("left", 3, 0), ("right", 2, 1), ("top", 0, 1), ("bottom", 3, 2)] if s[0] in PROBE_SIDES]
    probes = []
    for name, ia, ib in spec:
        start0, d, nrm, L_total = _edge_line(C, centre, ia, ib, PROBE_LINE_MM, slot)
        npieces = 1
        plen = L_total
        for k in range(npieces):
            nsamp = max(2, int(round(plen / ds)))
            L = nsamp * ds
            start = start0 + k * plen * d
            p = {"name": name if npieces == 1 else f"{name} {k + 1}", "start": start.tolist(), "dir": d.tolist(),
                 "normal": nrm.tolist(), "length": float(L), "seg": ds, "z_start": float(z_start), "z_end": float(z_end)}
            p["z_steps"] = [float(zpath(p, (i + 0.5) * ds)) for i in range(nsamp)]
            probes.append(p)
    return probes


def _edge_line(C, centre, ia, ib, L_want, slot):
    """A line inside the sheet edge from corner ia to ib: PROBE_INSET_MM in, PROBE_END_MM short of the corners, at most L_want
    long, centred on the edge and moved along it by the slot. Returns (start, direction, inside normal, length)."""
    A, B = C[ia], C[ib]
    e = B - A
    L_edge = float(np.linalg.norm(e))
    d = e / L_edge
    nrm = np.array([-d[1], d[0]])
    if np.dot(nrm, centre - A) < 0:
        nrm = -nrm
    start0 = A + PROBE_INSET_MM * nrm + PROBE_END_MM * d
    L_full = L_edge - 2.0 * PROBE_END_MM
    L_total = min(L_full, L_want)
    start0 = start0 + 0.5 * (L_full - L_total) * d                      # centred on the edge
    room = 0.5 * (L_full - L_total)                                     # how far the line can move either way along its edge
    step = min(L_total + 10.0, room)                                    # one line length (+10 mm) aside: clear of the old line
    shifts = (0.0, step, -step, 0.5 * step, -0.5 * step)
    return start0 + shifts[slot % len(shifts)] * d, d, nrm, L_total


def plan_step_row(corners_mm, contact_guess, slot=0):
    """PROBE_STEP_STROKES: a row of short strokes along the top edge (free with PROBE_COUNT = 2), each painted like an artwork
    stroke (glide in, a constant depth, glide out) at depths from PROBE_STEP_ABOVE_MM above the expected contact down to
    PROBE_STEP_BELOW_MM below it. Their widths give the width curve AS STROKES ARE PAINTED (2026-11-01: on ko/14 the
    artwork came out ~2.5 mm wide whatever the ramp curve planned - the ramp descends gently, a stroke lands steeply).
    None when off, or when the top edge carries a ramp line (PROBE_COUNT = 4)."""
    if not PROBE_STEP_STROKES or "top" in PROBE_SIDES:
        return None
    C = np.array(corners_mm, float)
    n, sl, gap = PROBE_STEP_COUNT, PROBE_STEP_LEN_MM, PROBE_STEP_GAP_MM
    start, d, nrm, L = _edge_line(C, C.mean(axis=0), 0, 1, n * sl + (n - 1) * gap, slot)
    n = max(3, int((L + gap) // (sl + gap)))                             # a short edge takes fewer strokes
    zs = np.linspace(contact_guess + PROBE_STEP_ABOVE_MM, max(Z_MIN, contact_guess - PROBE_STEP_BELOW_MM), n)
    strokes = [{"s0": k * (sl + gap), "s1": k * (sl + gap) + sl, "z": float(z)} for k, z in enumerate(zs)]
    return {"name": "steps", "start": start.tolist(), "dir": d.tolist(), "normal": nrm.tolist(),
            "length": float(strokes[-1]["s1"]), "strokes": strokes}


def paper_check_text(probes, corners_mm):
    """One line for the console: how far each probe line sits inside its own sheet edge (start and end)."""
    C = np.array(corners_mm, float)
    area = 0.5 * sum(C[i, 0] * C[(i + 1) % 4, 1] - C[(i + 1) % 4, 0] * C[i, 1] for i in range(4))
    sign = 1.0 if area > 0 else -1.0
    own_edge = {"top": 0, "right": 1, "bottom": 2, "left": 3}
    parts = []
    for p in probes:
        i = own_edge[p["name"].split()[0]]
        a, b = C[i], C[(i + 1) % 4]
        e = b - a
        n = max(np.linalg.norm(e), 1e-9)
        s_ = np.array(p["start"], float)
        end = s_ + np.array(p["dir"], float) * float(p["length"])
        m = [sign * (e[0] * (q[1] - a[1]) - e[1] * (q[0] - a[0])) / n for q in (s_, end)]
        parts.append(f"{p['name']} {min(m):.1f}-{max(m):.1f} mm")
    w = float(np.linalg.norm(C[1] - C[0])); h = float(np.linalg.norm(C[2] - C[1]))
    return f"paper check: sheet {w:.0f} x {h:.0f} mm; probe lines inside their own edges: " + ", ".join(parts)


def build_gcode(probes, dish, cfg, contact_guess, steps=None):
    """The painter's own dip, then for every line: the brush goes to its start in the air and descends steadily along it.
    steps (plan_step_row): after one more dip, each short stroke is landed and lifted exactly like an artwork line
    (Painter._land / _lift), with its glide height taken from the expected contact as the painting takes it from the probe's."""
    P = PT.Painter(cfg)
    ds = PROBE_SAMPLE_MM
    for k, p in enumerate(probes, 1):
        sx, sy = p["start"]
        d = np.array(p["dir"])
        n = len(p["z_steps"])
        P.gcode.append(f"; ===== PROBE LINE {k} {p['name']}: {p['length']:.0f} mm, Z {p['z_start']:.2f} -> {p['z_end']:.2f} =====")
        P._dip_and_travel(sx, sy, dish)
        P._pen_down(p["z_start"])                                       # the start is in the air
        for i in range(1, n + 1):
            s = i * ds
            P.gcode.append(f"G1 X{sx + s * d[0]:.3f} Y{sy + s * d[1]:.3f} Z{float(zpath(p, s)):.3f}")
        P.gcode.append(f"G0 Z{cfg['z_low']:.3f} F3000")
    if steps:
        o, d = np.array(steps["start"]), np.array(steps["dir"])
        P.cfg = dict(P.cfg, z_paint=float(contact_guess), tilt=None)    # the glide's air height, as for the artwork
        P.gcode.append(f"; ===== PROBE STEP STROKES: {len(steps['strokes'])} strokes, Z {steps['strokes'][0]['z']:.2f} -> "
                       f"{steps['strokes'][-1]['z']:.2f}, landed like artwork lines =====")
        P._dip_and_travel(*P._lead_point(o + steps["strokes"][0]["s0"] * d, -d), dish)
        for st in steps["strokes"]:
            a, b = o + st["s0"] * d, o + st["s1"] * d
            P._set_speed("travel")
            P._land(a, (-d[0], -d[1]), st["z"])
            P.gcode.append(f"G1 X{b[0]:.3f} Y{b[1]:.3f} Z{st['z']:.3f}")
            P._lift(b, (d[0], d[1]))
    P.gcode.append(f"G0 Z{cfg['z_high']:.3f} F3000")
    return "\n".join(P.gcode)


# ------------------------------------------------------------------------------------------------ measuring
def ink_difference(frame1, frame2, paper_poly_px):
    """What became darker between the two pictures (inside the paper only), in grey levels."""
    g1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY).astype(np.float32)
    poly = np.zeros(g1.shape, np.uint8)
    cv2.fillPoly(poly, [np.round(np.array(paper_poly_px)).astype(np.int32)], 255)
    inside = cv2.erode(poly, np.ones((5, 5), np.uint8)) > 0
    g2 = g2 * (float(np.median(g1[inside])) / max(float(np.median(g2[inside])), 1.0))        # the same exposure
    d = np.clip(cv2.GaussianBlur(g1 - g2, (0, 0), 0.7), 0, None)
    d[~inside] = 0
    return d


def _profiles(D, H_inv, p, s_vals, t_vals):
    S_, T_ = np.meshgrid(s_vals, t_vals, indexing="ij")
    pts = np.array(p["start"])[None, None, :] + S_[..., None] * np.array(p["dir"]) + T_[..., None] * np.array(p["normal"])
    px = V.mm_to_px_array(H_inv, pts.reshape(-1, 2)).reshape(S_.shape + (2,))
    return map_coordinates(D, [px[..., 1], px[..., 0]], order=1, mode="nearest")


def _runs_width(row, thr, dt):
    """Width (mm) of the contiguous run above thr that contains the maximum, with linear interpolation at its two ends."""
    i = int(np.argmax(row))
    if row[i] < thr:
        return 0.0, 0.0
    a = i
    while a > 0 and row[a - 1] >= thr:
        a -= 1
    b = i
    while b < len(row) - 1 and row[b + 1] >= thr:
        b += 1
    left = a - ((row[a] - thr) / max(row[a] - row[a - 1], 1e-6) if a > 0 else 0.0)
    right = b + ((row[b] - thr) / max(row[b] - row[b + 1], 1e-6) if b < len(row) - 1 else 0.0)
    cen = float(np.sum(np.arange(a, b + 1) * row[a:b + 1]) / max(row[a:b + 1].sum(), 1e-6))
    return (right - left) * dt, cen


def measure_line(D, H_inv, p, w_ref=PROBE_CONTACT_WIDTH_MM):
    """The width of the mark at every sample of one line, then the touch point: where the widths first reach w_ref."""
    seg, zs, L = p["seg"], p["z_steps"], p["length"]
    n = len(zs)
    out = {"name": p["name"], "ok": False, "flags": [], "zs": list(zs)}
    ds, dt = 0.25, 0.2
    s_vals = np.arange(-2.0, L + 2.0 + 1e-9, ds)
    # across the line +-8 mm: near the sheet edges, outside the calibrated area, the paint was seen 3.5 mm beside the planned line
    # (2026-10-28, big sheet) and fell into the background band of the old +-5 mm window
    t_vals = np.arange(-8.0, 8.0 + 1e-9, dt)
    I = _profiles(D, H_inv, p, s_vals, t_vals)
    edge = np.abs(t_vals) > 6.8
    I = I - np.median(I[:, edge], axis=1, keepdims=True)
    peak = I.max(axis=1)
    noise = 1.4826 * float(np.median(np.abs(I[:, edge]))) + 1e-3
    full = float(np.percentile(peak, 92)) if peak.max() > 0 else 0.0
    if full < max(15.0, 8 * noise):
        out["flags"].append("no paint seen on this line: no paint on the brush, or the brush never reached the paper")
        return out
    vis = max(6.0, 5 * noise)
    widths = np.zeros(n)
    for i in range(n):
        lo, hi = (i + 0.25) * seg, (i + 0.75) * seg                     # the middle of the sample cell, away from its ends
        ws = []
        for j in np.nonzero((s_vals >= lo) & (s_vals <= hi))[0]:
            if peak[j] >= vis:
                wdt, _ = _runs_width(I[j], 0.5 * peak[j], dt)
                ws.append(wdt)
        if ws:
            widths[i] = float(np.median(ws))
    widths[widths < 0.5] = 0.0
    # drops from the freshly dipped brush or splashes from the dish beside the sheet can mark the line before the real touch, even
    # at its very start (2026-10-29 left line at 9.21; 2026-10-30 left line: specks at 0-3, 10 and 20 mm, the mark from 39 mm):
    # the real mark keeps going for at least 5 mm, everything before the first such run is ignored
    marked = np.r_[widths > 0, False]
    run_min = max(2, int(round(5.0 / seg)))
    k_run, i = None, 0
    while i < n:
        if marked[i]:
            j = i
            while marked[j]:
                j += 1
            if j - i >= run_min:
                k_run = i
                break
            i = j
        i += 1
    if k_run is not None and (widths[:k_run] > 0).any():
        widths[:k_run] = 0.0
        out["flags"].append(f"specks before {(k_run + 0.5) * seg:.0f} mm (not a continuing mark): ignored")
    # the brush runs out of paint towards the deep end: the mark gets clearly narrower again, at two points in a row
    runmax = np.maximum.accumulate(widths)
    stop = n
    for i in range(1, n - 1):
        if runmax[i] >= w_ref and runmax[i] - widths[i] >= 0.5 and runmax[i + 1] - widths[i + 1] >= 0.5:
            stop = i
            out["starved_at_mm"] = float((i + 0.5) * seg)
            break
    env = np.maximum.accumulate(widths)
    over = np.nonzero(env >= w_ref)[0]
    out.update({"widths": widths.tolist(), "stop": int(stop)})
    if len(over) == 0:
        out["flags"].append(f"never reached {w_ref:g} mm: the lowest height ({zs[-1]:.2f}) is not deep enough, or the paint ran out")
        out["contact_below_range"] = True
        return out
    k = int(over[0])
    if k == 0:
        out["flags"].append(f"already marks at the start ({zs[0]:.2f}): the contact is ABOVE the start of the line (raise PROBE_ABOVE_MM)")
        out["contact_above_range"] = True
        return out
    w0, w1 = env[k - 1], env[k]
    f = (w_ref - w0) / max(w1 - w0, 1e-6)
    out.update({"ok": True,
                "z_contact": float(zs[k - 1] + f * (zs[k] - zs[k - 1])),
                "s_contact": float((k - 0.5 + f) * seg)})                # distance along the line from its start
    # a hanging drop or the tip hairs can draw a flat ~1 mm trail long before the brush presses on the paper (2026-11-01: the top
    # line marked from Z 6.25 while it only started to widen at 4.60, the right line had no trail and widened at once at 4.37):
    # the touch is where the mark starts to WIDEN - a fit of "flat trail, then growing" over the mark up to 2.5 mm wide
    zt = _widening_start(np.asarray(zs, float), widths, k, stop)
    if zt is not None and zt < out["z_contact"] - 0.05:
        s_of = np.interp(zt, np.asarray(zs, float)[::-1], ((np.arange(n) + 0.5) * seg)[::-1])
        out["flags"].append(f"flat trail from {out['z_contact']:.2f} (drop or tip hairs, not pressing): touch where it widens, {zt:.2f}")
        out.update({"z_contact": float(zt), "s_contact": float(s_of)})
    return out


def _widening_start(zs, widths, k, stop, w_top=2.5):
    """Z where the mark starts to grow: least-squares fit of w = w0 + slope * max(0, z_t - z) over the samples from the first mark
    (k) until the mark is w_top wide; None when there is too little of it or it never grows."""
    i1 = min(stop, len(widths))
    env = np.maximum.accumulate(widths[k:i1])
    over = np.nonzero(env >= w_top)[0]
    i1 = k + (int(over[0]) + 1 if len(over) else len(env))
    z, w = zs[k:i1], widths[k:i1]
    sel = w > 0
    z, w = z[sel], w[sel]
    if len(z) < 6:
        return None
    best = None
    for zt in np.arange(z.min() + 0.3, z.max() + 1e-9, 0.01):
        x = np.maximum(0.0, zt - z)
        A = np.stack([np.ones_like(x), x], 1)
        coef, *_ = np.linalg.lstsq(A, w, rcond=None)
        sse = float(((A @ coef - w) ** 2).sum())
        if coef[1] > 0.1 and (best is None or sse < best[0]):
            best = (sse, float(zt))
    return best[1] if best else None


def measure_steps(D, H_inv, steps):
    """Width in the middle 2 mm of every short stroke (where its depth is constant), measured across it like the sloping lines.
    0 = no mark (the stroke stayed in the air)."""
    dt = 0.2
    t_vals = np.arange(-8.0, 8.0 + 1e-9, dt)
    edge = np.abs(t_vals) > 6.8
    o, d = np.array(steps["start"]), np.array(steps["dir"])
    out = []
    for st in steps["strokes"]:
        mid = 0.5 * (st["s0"] + st["s1"])
        s_vals = np.arange(mid - 1.0, mid + 1.0 + 1e-9, 0.25)
        I = _profiles(D, H_inv, steps, s_vals, t_vals)
        I = I - np.median(I[:, edge], axis=1, keepdims=True)
        noise = 1.4826 * float(np.median(np.abs(I[:, edge]))) + 1e-3
        peak = I.max(axis=1)
        ws = [_runs_width(I[j], 0.5 * peak[j], dt)[0] for j in range(len(s_vals)) if peak[j] >= max(10.0, 6 * noise)]
        w = float(np.median(ws)) if len(ws) >= 0.5 * len(s_vals) else 0.0
        xy = o + mid * d
        out.append({"x": float(xy[0]), "y": float(xy[1]), "z": float(st["z"]), "w": w if w >= 0.5 else 0.0})
    return out


def probes_needed():
    """How many good lines a result needs: 4 for the tilt plane (with a check), or both lines with PROBE_COUNT = 2."""
    return 4 if PROBE_COUNT >= 4 else len(PROBE_SIDES)


def solve(results, probes, pf, w_ref=PROBE_CONTACT_WIDTH_MM, margin=PROBE_MARGIN_MM, steps=None):
    """Tilt of the paper from the touch points of all lines (a plane), then the width curve from every sample.
    steps (measure_steps): when the short strokes painted, THEY set the touch height (the plane is moved to where a landed stroke
    starts to mark) and the width curve; the sloping lines then only give the tilt."""
    good = [(r, p) for r, p in zip(results, probes) if r["ok"]]
    if len(good) < probes_needed():                                      # a plane has 3 numbers: with fewer there is no check
        return None
    o = np.array(pf["origin_mm"])
    xc, yc = float(o[0]), float(o[1])
    pos = np.array([np.array(p["start"]) + r["s_contact"] * np.array(p["dir"]) for r, p in good])
    z = np.array([r["z_contact"] for r, _ in good])
    use = np.ones(len(z), bool)
    if PROBE_COUNT >= 4:
        A = np.stack([np.ones(len(z)), pos[:, 0] - xc, pos[:, 1] - yc], axis=1)
        for _ in range(3):
            sol, *_ = np.linalg.lstsq(A[use], z[use], rcond=None)
            res = z - A @ sol
            mad = 1.4826 * np.median(np.abs(res[use])) + 1e-3
            new = np.abs(res) < max(3 * mad, 0.12)
            if new.sum() < 3 or (new == use).all():
                break
            use = new
        b, c = float(sol[1]), float(sol[2])
    else:                                                                # PROBE_COUNT = 2 (left + right): a straight slope from one to the other
        d = pos[1] - pos[0]                                              # (along the lines' own direction the paper counts as level)
        L2 = float(d @ d)
        g = (z[1] - z[0]) / L2 if L2 > 1.0 else 0.0
        b, c = float(g * d[0]), float(g * d[1])
        res = np.zeros(len(z))
    # the contact = the MEDIAN of the lines that passed, corrected for the tilt to the centre of the paper (not the fitted intercept)
    a = float(np.median(z[use] - b * (pos[use, 0] - xc) - c * (pos[use, 1] - yc)))
    plane = lambda x, y: a + b * (x - xc) + c * (y - yc)
    rms = float(np.sqrt((res[use] ** 2).mean()))
    # the width curve against depth below the TOUCH plane, from every sample above the stop
    dep, wid = [], []
    for r, p in good:
        for i in range(r["stop"]):
            xy = np.array(p["start"]) + (i + 0.5) * p["seg"] * np.array(p["dir"])
            depth = plane(xy[0], xy[1]) - r["zs"][i]
            if depth >= -0.05 and r["widths"][i] > 0:
                dep.append(depth)
                wid.append(r["widths"][i])
    dep, wid = np.array(dep), np.array(wid)
    d_list, w_list = [0.0], [float(w_ref)]
    binned = []
    for cdep in np.arange(0.25, (dep.max() if len(dep) else 0.0) + 1e-9, 0.25):
        sel = np.abs(dep - cdep) <= 0.125
        if sel.any():
            binned.append((float(cdep), float(np.median(wid[sel]))))
    if binned and len(dep) >= 5:
        env_w = np.maximum.accumulate([w for _, w in binned])            # deeper is never narrower
        for (dd, _), ww in zip(binned, env_w):
            if ww >= w_ref * 0.98 and ww > w_list[-1] + 1e-3:
                d_list.append(dd)
                w_list.append(float(ww))
    curve_from, step_rows = "the sloping lines", None
    if steps:
        sd = np.array([plane(s_["x"], s_["y"]) - s_["z"] for s_ in steps])   # depth of each stroke below the sloping lines' touch
        sw = np.array([s_["w"] for s_ in steps])
        painted = np.nonzero(sw >= w_ref)[0]
        if len(painted) >= 3:
            k0 = int(painted[0])
            # the touch AS A LANDED STROKE SEES IT: between the last stroke without a mark and the first one with it
            d_touch = 0.5 * (sd[k0 - 1] + sd[k0]) if k0 > 0 else sd[0] - 0.2
            sd_l, sw_l = [0.0], [float(w_ref)]
            for dd, ww in zip(sd[k0:] - d_touch, np.maximum.accumulate(sw[k0:])):
                if dd > sd_l[-1] + 1e-3 and ww > sw_l[-1] + 1e-3:
                    sd_l.append(float(dd))
                    sw_l.append(float(ww))
            if len(sd_l) >= 3:
                a -= d_touch                                             # the whole plane moves (plane() reads a when called)
                d_list, w_list, curve_from = sd_l, sw_l, "the short strokes"
                step_rows = [{"depth": round(float(dd - d_touch), 2), "width": round(float(ww), 2), "z": s_["z"]}
                             for dd, ww, s_ in zip(sd, sw, steps)]
                if k0 == 0:
                    print("NOTE: the first short stroke already marked - the touch may be higher still (raise PROBE_STEP_ABOVE_MM)")
        if curve_from != "the short strokes":
            print(f"NOTE: the short strokes did not give a curve ({len(painted)} of {len(sw)} marked): curve from the sloping lines")
    if len(d_list) < 3:
        return None
    # move the curve to the PAINT REFERENCE (touch minus the margin): the thinnest line is then the width at that depth
    w_m = float(np.interp(margin, d_list, w_list))
    nd, nw = [0.0], [w_m]
    for dd, ww in zip(d_list, w_list):
        if dd > margin + 1e-9:
            nd.append(dd - margin)
            nw.append(ww)
    if len(nd) < 3:
        return None
    # direction-dependent width: the probe lines measure the mark ACROSS the line. left/right lines (along v) give the brush's
    # footprint along u, top/bottom lines (along u) the footprint along v. Relative to the average width, so 1.0 = round brush.
    def width_at(names, d_ref=0.5):
        ws = []
        for r, p in good:
            if p["name"] not in names:
                continue
            for i in range(r["stop"]):
                xy = np.array(p["start"]) + (i + 0.5) * p["seg"] * np.array(p["dir"])
                depth = plane(xy[0], xy[1]) - r["zs"][i]
                if abs(depth - d_ref) <= 0.25 and r["widths"][i] > 0:
                    ws.append(r["widths"][i])
        return float(np.median(ws)) if len(ws) >= 3 else None
    w_avg = float(np.interp(0.5, d_list, w_list))
    ext_u, ext_v = width_at({"left", "right"}), width_at({"top", "bottom"})
    aniso = None
    if ext_u and ext_v and w_avg > 0:
        aniso = [round(float(np.clip(ext_u / w_avg, 0.6, 1.6)), 3), round(float(np.clip(ext_v / w_avg, 0.6, 1.6)), 3)]
    return {"contact_center": a - margin, "touch_center": a, "margin_mm": margin, "aniso": aniso,
            "tilt_x": b, "tilt_y": c, "center_mm": [xc, yc], "rms_mm": rms,
            "n_probes": len(good), "n_used": int(use.sum()), "curve": [nd, nw],
            "brush": round(w_m, 1), "w_max": float(nw[-1]), "max_depth": float(nd[-1]),
            "plane_spread_mm": float(np.ptp([plane(q[0], q[1]) for q in pos])),
            "curve_from": curve_from, "step_strokes": step_rows}


# ------------------------------------------------------------------------------------------------ pictures, reports, files
def plan_picture(frame, probes, dish, cal, path, det=None, steps=None):
    """The camera picture: the sheet's edges in GREEN, the planned probe lines in RED (numbered), the paint dish as a blue circle.
    Errors are printed, not hidden, and the full path of the file is printed when it is written."""
    import os, time
    base, ext = os.path.splitext(path)
    path = f"{base}_{time.strftime('%H%M%S')}{ext}"              # a new name: a picture that is open in a viewer cannot be overwritten
    try:
        out = frame.copy()
        if det is not None and det.get("corners_px"):
            q = np.round(np.array(det["corners_px"])).astype(np.int32)
            cv2.polylines(out, [q], True, (0, 255, 0), 3, cv2.LINE_AA)
        for k, p in enumerate(probes, 1):
            s = np.array(p["start"])
            e = s + p["length"] * np.array(p["dir"])
            pa, pb = V.mm_to_px_array(cal["H_inv"], [tuple(s), tuple(e)])
            cv2.line(out, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])), (0, 0, 255), 3, cv2.LINE_AA)
            cv2.putText(out, f"{k} {p['name']}", (int(pa[0]), int(pa[1]) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
        if steps:                                                       # the short strokes in MAGENTA
            o, d = np.array(steps["start"]), np.array(steps["dir"])
            for st in steps["strokes"]:
                pa, pb = V.mm_to_px_array(cal["H_inv"], [tuple(o + st["s0"] * d), tuple(o + st["s1"] * d)])
                cv2.line(out, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])), (255, 0, 255), 3, cv2.LINE_AA)
            pa = V.mm_to_px_array(cal["H_inv"], [tuple(o)])[0]
            cv2.putText(out, "short strokes", (int(pa[0]), int(pa[1]) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2, cv2.LINE_AA)
        c = V.mm_to_px_array(cal["H_inv"], [(dish[0], dish[1])])[0]
        cv2.circle(out, (int(c[0]), int(c[1])), max(6, int(dish[2] * 2.9)), (255, 128, 0), 3, cv2.LINE_AA)
        cv2.putText(out, "paint dish", (int(c[0]) + 10, int(c[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 128, 0), 2, cv2.LINE_AA)
        ok = cv2.imwrite(path, out)
        if not ok:
            raise OSError("cv2.imwrite returned False")
        print(f"plan picture written: {os.path.abspath(path)}  (green = sheet edges, red = probe lines)")
        return path
    except Exception as e:
        print(f"PLAN PICTURE FAILED ({type(e).__name__}: {e}) - the lines are still in probe_plan.json")
        return None


def report_picture(frame, probes, results, cal, path, sol=None, steps=None):
    out = frame.copy()
    rows = (sol or {}).get("step_strokes") or []
    if steps:                                                           # the short strokes: their Z, and the width measured (green)
        o, d = np.array(steps["start"]), np.array(steps["dir"])
        for k, st in enumerate(steps["strokes"]):
            pa, pb = V.mm_to_px_array(cal["H_inv"], [tuple(o + st["s0"] * d), tuple(o + st["s1"] * d)])
            cv2.line(out, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])), (0, 200, 255), 1, cv2.LINE_AA)
            label = f"{st['z']:.2f}" + (f"/{rows[k]['width']:.1f}" if k < len(rows) else "")
            cv2.putText(out, label, (int(pa[0]), int(pa[1]) - 8 - 12 * (k % 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 160, 0), 1, cv2.LINE_AA)
    for p, r in zip(probes, results):
        a = np.array(p["start"])
        b = a + p["length"] * np.array(p["dir"])
        pa, pb = V.mm_to_px_array(cal["H_inv"], [tuple(a), tuple(b)])
        cv2.line(out, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])), (0, 200, 255), 1, cv2.LINE_AA)
        if r["ok"]:
            xy = a + r["s_contact"] * np.array(p["dir"])
            q = V.mm_to_px_array(cal["H_inv"], [tuple(xy)])[0]
            cv2.drawMarker(out, (int(q[0]), int(q[1])), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 16, 2)
            cv2.putText(out, f"{r['z_contact']:.2f}", (int(q[0]) + 6, int(q[1]) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        else:
            cv2.putText(out, "?", (int(pa[0]), int(pa[1]) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imwrite(path, out)


def print_report(probes, results, sol, contact_guess):
    print(f"\n{'line':9s} {'length':>7s} {'start Z':>8s} {'touch Z':>8s} {'touch at':>10s}   notes")
    for p, r in zip(probes, results):
        if r["ok"]:
            note = f"; the brush ran out of paint at {r['starved_at_mm']:.0f} mm" if "starved_at_mm" in r else ""
            print(f"{r['name']:9s} {p['length']:6.0f}mm {p['z_start']:8.2f} {r['z_contact']:8.2f} {r['s_contact']:6.1f} mm in{note}")
        else:
            print(f"{r['name']:9s} {p['length']:6.0f}mm {p['z_start']:8.2f} {'-':>8s}   " + "; ".join(r["flags"]))
    if sol is None and sum(1 for r in results if r["ok"]) >= probes_needed():
        print("\nthe touch was found on every line, but not how the width grows with depth: below the touch the lines were too short "
              "(the ramp ends at machine Z 0.3) or the brush ran out of paint. Mount the brush lower, so it touches higher up.")
    elif sol is None:
        print(f"\nnot enough good lines (at least {probes_needed()}) to say where the brush touches.")
        return
    print(f"\ntouch height at the middle of the paper: machine Z {sol['touch_center']:.2f}  (your setting / profile says {contact_guess:.2f}: "
          f"{sol['touch_center'] - contact_guess:+.2f} mm)")
    print(f"paint reference (touch minus {sol['margin_mm']:.2f} mm, deeper): machine Z {sol['contact_center']:.2f}  <- the paint is made from here")
    print(f"tilt of the paper: {sol['plane_spread_mm']:.2f} mm between the lines' places  "
          f"({sol['tilt_x'] * 100:+.2f} / {sol['tilt_y'] * 100:+.2f} mm per 100 mm in X / Y)")
    print(f"lines agree to {sol['rms_mm']:.2f} mm (RMS), {sol['n_used']} of {sol['n_probes']} used")
    print(f"touch height and width curve from {sol.get('curve_from', 'the sloping lines')}")
    d, w = sol["curve"]
    pick = [0.0, 0.3, 0.6, 1.0, 1.5, 2.0]
    print("width of the line against depth below the paint reference:  " +
          "  ".join(f"{x:.1f}mm:{np.interp(x, d, w):.2f}" for x in pick if x <= d[-1] + 1e-9))
    print(f"thinnest line {sol['brush']:.1f} mm; the widest reached {sol['w_max']:.2f} mm at {sol['max_depth']:.1f} mm deeper")


def save_result(sol, probes, results, det, path=PROBE_FILE):
    d = dict(sol)
    d["created"] = time.strftime("%Y-%m-%d %H:%M:%S")
    d["created_ts"] = time.time()
    d["paper_corners_mm"] = det["corners_mm"] if det else None
    d["lines"] = [{"name": r["name"], "ok": r["ok"], "z_contact": r.get("z_contact"),
                   "touch_at_mm": r.get("s_contact"), "start": [float(v) for v in p["start"]], "dir": [float(v) for v in p["dir"]], "length": float(p["length"])}
                  for p, r in zip(probes, results)]
    json.dump(d, open(path, "w"), indent=1)
    return d


def load_result(path=PROBE_FILE):
    if not os.path.isfile(path):
        raise SystemExit(f"no probe result at {path}: run  python probe.py  first")
    return json.load(open(path))


def result_problems(res, det, max_age_h=6.0, max_move_mm=3.0):
    """Why a saved result (or sheet record) does not belong to this sheet: [] = it does."""
    problems = []
    age = (time.time() - res.get("created_ts", 0)) / 3600.0
    if age > max_age_h:
        problems.append(f"the probe result is {age:.0f} hours old (limit {max_age_h:.0f})")
    old = res.get("paper_corners_mm")
    if not old or not det:
        problems.append("the result does not say where its paper was")
    else:
        move = max(np.hypot(a[0] - b[0], a[1] - b[1]) for a, b in zip(old, det["corners_mm"]))
        if move > max_move_mm:
            problems.append(f"the paper is {move:.0f} mm from where the probes were measured (limit {max_move_mm:.0f} mm)")
    w0 = touch_width_problem(res)
    if w0:
        problems.append(w0)
    return problems


def touch_width_problem(res):
    """'' or why the measured touch height is not believable: a brush that already makes a wide mark 'at the touch' was
    pressed in there, so the paper is really higher (2026-10-29: 1.76 mm at the touch, the starts left dots)."""
    import settings as _S
    curve = res.get("curve")
    lim = float(getattr(_S, "PROBE_TOUCH_W_MAX", 1.5))
    if not curve or lim <= 0:
        return ""
    w0 = float(curve[1][0])
    if w0 > lim:
        return (f"the brush already made {w0:.2f} mm 'at the touch' (limit PROBE_TOUCH_W_MAX {lim:.2f}): "
                f"the touch height is too low, the paper is really higher")
    return ""


def marks_visible(frame, res):
    """True when the result's own probe lines can be seen on the paper in this photo (it is the same sheet, not a fresh one put
    in the same place). None = the result does not say where its lines were (older results)."""
    lines = [ln for ln in res.get("lines", []) if ln.get("start") and ln.get("ok")]
    if frame is None or not lines:
        return None
    cal = V.load_calibration()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    seen = 0
    for ln in lines:
        a, d = np.array(ln["start"], float), np.array(ln["dir"], float)
        n = np.array([-d[1], d[0]])
        s0 = float(ln.get("touch_at_mm") or 0.0)                   # the paint starts at the touch point
        dark, vals = 0, []
        ss = np.linspace(s0, ln["length"], 30)
        for s in ss:
            pts = [tuple(a + s * d + t * n) for t in np.linspace(-2.0, 2.0, 9)]
            px = np.round(V.mm_to_px_array(cal["H_inv"], pts)).astype(int)
            ok = (px[:, 0] >= 0) & (px[:, 0] < w) & (px[:, 1] >= 0) & (px[:, 1] < h)
            if not ok.any():
                continue
            g = gray[px[ok, 1], px[ok, 0]].astype(float)
            vals.append(float(np.max(g)))
            dark += float(np.min(g)) < 0.6 * float(np.max(g))           # paint well darker than the paper beside it
        if vals and dark >= 0.5 * len(ss):
            seen += 1
    return seen >= max(1, len(lines) // 2)


def reusable_result(det, frame=None):
    """The saved probe result, when it was measured on THIS sheet (same place, today, its probe marks visible in the photo):
    painting new probes would put the lines on top of the old ones, where the photos cannot see new paint. None = no such result."""
    if not det or not os.path.isfile(PROBE_FILE):
        return None
    try:
        res = load_result(PROBE_FILE)
    except Exception:
        return None
    if result_problems(res, det) or marks_visible(frame, res) is False:
        return None
    return res


def sheet_probe_count(det):
    """How many probes were already painted on this sheet (also the failed ones): the next lines go beside them."""
    try:
        rec = json.load(open(SHEET_FILE))
    except Exception:
        return 0
    return 0 if result_problems(rec, det) else int(rec.get("rounds", 0))


def note_sheet_probe(det, rounds):
    json.dump({"paper_corners_mm": det["corners_mm"], "created_ts": time.time(), "rounds": int(rounds)}, open(SHEET_FILE, "w"))


def check_result_fresh(res, det, max_age_h=6.0, max_move_mm=3.0, force=False):
    """A saved result is used only on the sheet it was measured on. Refuses (SystemExit) when the paper has moved,
    the result is too old, or the result does not say where the paper was. force=True prints the warnings and goes on."""
    problems = result_problems(res, det, max_age_h, max_move_mm)
    if not problems:
        return True
    msg = "this probe result does not fit this sheet: " + "; ".join(problems) + \
          ". Run  python probe.py  on this sheet (each sheet gets one set of probes)."
    if not force:
        raise SystemExit(msg + "\n  --force_probe_file uses it anyway (not recommended).")
    print("WARNING (--force_probe_file): " + msg)
    return False


def lateral_offset(D, H_inv, p):
    """Where the mark sits ACROSS its line: the centre of the difference across the line, in mm (+ = towards the inside normal).
    None if the mark is too faint on this line."""
    L = float(p["length"])
    s_vals = np.linspace(0.2 * L, 0.8 * L, 40)
    t_vals = np.arange(-12.0, 12.0 + 1e-9, 0.1)                     # wide: a mark 4-5 mm off its line must not reach the background band
    I = _profiles(D, H_inv, p, s_vals, t_vals)
    I = I - np.median(I[:, np.abs(t_vals) > 9.0], axis=1, keepdims=True)
    peak = I.max(axis=1)
    strong = peak > max(15.0, 0.5 * float(np.percentile(peak, 90)))
    if int(strong.sum()) < 5:
        return None
    w = np.clip(I[strong], 0.0, None)
    tc = (w * t_vals[None, :]).sum(axis=1) / np.maximum(w.sum(axis=1), 1e-9)
    return float(np.median(tc))


def tool_offset_from_lines(D, H_inv, probes):
    """The shift of the whole pattern, from the four lines: each line sees the shift along its own normal (t_i = shift . n_i).
    One shift is fitted to all lines (least squares; with 4 lines also a common 'all marks inward' term, which is not applied).
    The scatter is how far the lines disagree with that fit (RMS), NOT the spread of the t_i (those differ by design).
    Returns ([dx, dy] mm, scatter mm) or (None, None)."""
    vals = []
    for p in probes:
        t = lateral_offset(D, H_inv, p)
        if t is not None:
            vals.append((t, np.array(p["normal"], float)))
    t = np.array([t for t, _ in vals])
    A = np.array([n for _, n in vals])
    if len(vals) < 3:
        # PROBE_COUNT = 2: one vertical and one horizontal line give the shift exactly, but nothing checks it (scatter 0)
        if len(vals) == 2 == len(probes) and abs(float(np.cross(A[0], A[1]))) > 0.7:
            sol = np.linalg.solve(A, t)
            return [round(float(sol[0]), 3), round(float(sol[1]), 3)], 0.0
        # left + right (parallel, opposite normals): t_i = s * (n_i . u) + c gives the shift s across the lines (c = the common
        # 'all marks inward' term, not applied); along the lines nothing is seen: no extra shift there, so the offset the probe
        # moves were already corrected by (the last probe's, added by the caller) stays
        if len(vals) == 2 == len(probes) and float(A[0] @ A[1]) < -0.7:
            u = A[0] / np.linalg.norm(A[0])
            s, _c = np.linalg.solve(np.array([[float(A[0] @ u), 1.0], [float(A[1] @ u), 1.0]]), t)
            sol = s * u
            return [round(float(sol[0]), 3), round(float(sol[1]), 3)], 0.0
        return None, None
    if len(vals) >= 4:
        A = np.hstack([A, np.ones((len(vals), 1))])
    sol, *_ = np.linalg.lstsq(A, t, rcond=None)
    scatter = float(np.sqrt(np.mean((t - A @ sol) ** 2)))
    return [round(float(sol[0]), 3), round(float(sol[1]), 3)], round(scatter, 3)


def offset_usable(off, scatter):
    """True when a measured tip offset is a real, consistent shift: not too large, and the lines agree with one shift."""
    return (off is not None and scatter is not None and float(np.hypot(off[0], off[1])) <= PROBE_OFFSET_MAX_MM
            and float(scatter) <= PROBE_OFFSET_SCATTER_MM)


def applied_offset_from_last_result():
    """The offset the last probe measured (the probe moves are corrected by it), or [0, 0]."""
    try:
        if os.path.isfile(PROBE_FILE):
            r = json.load(open(PROBE_FILE))
            o = r.get("tool_offset_mm")
            if o and offset_usable(o, r.get("tool_offset_scatter_mm")):
                return [float(o[0]), float(o[1])]
    except Exception:
        pass
    return [0.0, 0.0]


def analyse(frame1, frame2, plan, cal, det_poly, contact_guess):
    probes = plan["probes"]
    D = ink_difference(frame1, frame2, det_poly)
    results = [measure_line(D, cal["H_inv"], p) for p in probes]
    above = [r["name"] for r in results if r.get("contact_above_range")]
    below = [r["name"] for r in results if r.get("contact_below_range")]
    steps = measure_steps(D, cal["H_inv"], plan["steps"]) if plan.get("steps") else None
    sol = None if (above or below) else solve(results, probes, plan["pf"], steps=steps)
    if steps:
        print("short strokes (landed like artwork lines):  " + "  ".join(f"Z {s_['z']:.2f}:{s_['w']:.2f}mm" for s_ in steps))
    if sol is not None:
        off, scat = tool_offset_from_lines(D, cal["H_inv"], probes)
        if off is not None:
            prev = plan.get("applied_offset", [0.0, 0.0])
            off = [round(off[0] + prev[0], 3), round(off[1] + prev[1], 3)]
            sol["tool_offset_mm"] = off
            sol["tool_offset_scatter_mm"] = scat
            print(f"paint lands {off[0]:+.2f}, {off[1]:+.2f} mm from where the machine is told (measured from the photo; the lines "
                  f"disagree by {scat:.2f} mm RMS) -> " + ("USED" if offset_usable(off, scat) else
                  f"NOT used (limits: {PROBE_OFFSET_MAX_MM:g} mm, {PROBE_OFFSET_SCATTER_MM:g} mm RMS in settings.py)"))
    print_report(probes, results, sol, contact_guess)
    if sum("no paint seen" in " ".join(r["flags"]) for r in results) >= max(3, len(results) - 1):
        print("\nNO PAINT on (nearly) all lines. Two possible causes: (1) the brush has no paint: look at the dish, is there paint, did the brush take it up? "
              f"(2) the brush touches only below the lowest height (Z {probes[0]['z_end']:.2f}): then the paint is fine and the line must go "
              "deeper. Check probe_result.png before you change anything.")
    if above or below:
        # NOT used: a fit to the other lines would be biased if the touch lies outside the line on some sides
        msg = "\nRESULT NOT USED: "
        if above:
            msg += (f"on {len(above)} line(s) ({', '.join(above)}) the brush already marked at the start, so the touch is above the line. "
                    f"Set PROBE_ABOVE_MM = {PROBE_ABOVE_MM + 1.0:g} in settings.py and probe again. ")
        if below:
            msg += (f"On {len(below)} line(s) ({', '.join(below)}) the mark never reached {PROBE_CONTACT_WIDTH_MM:g} mm at the lowest height "
                    f"(Z {probes[0]['z_end']:.2f}). Is there paint on the brush? Or the brush is mounted too high for this paper.")
        print(msg)
    return probes, results, sol


def contact_guess_from(cfg):
    """Where the paint touches, to start the ramps from: the last good probe result if there is one, else the settings."""
    try:
        if os.path.exists(PROBE_FILE):
            with open(PROBE_FILE, encoding="utf-8") as f:
                c = json.load(f).get("contact_center")
            if c is not None:
                return float(c)
    except Exception:
        pass
    return float(cfg["z_paint"])


def _set_status(text):
    global LAST_STATUS
    LAST_STATUS = text


def pick_dish(pools, color, cfg):
    near = lambda p: np.hypot(p["center_mm"][0] - cfg["dip_x"], p["center_mm"][1] - cfg["dip_y"]) <= cfg["palette_radius"]
    ok = [p for p in pools if p["roundness"] >= cfg["min_roundness"] and p["area_mm2"] >= cfg["min_pool_area"] and near(p)]
    if color == "black":                                         # black: the dark pool (vision.find_pools marks it)
        cands = sorted([p for p in ok if p.get("dark")], key=lambda p: -p["area_mm2"])
    else:
        # the coloured pool NEAREST in colour to the named colour (not "its hue is in the blue range": a lilac at 128 passed as
        # blue and the probes were painted lilac); a dark pool never counts as a colour
        centre = {n: np.mean([0.5 * (lo + hi) for lo, hi in r]) if n != "red" else 0.0 for n, r in PT.COLOR_HUES.items()}.get(color)
        cands = [p for p in ok if not p.get("dark") and PT.nearest_hue_name(p["hue"]) == color]
        if centre is not None:
            cands.sort(key=lambda p: (V.hue_distance(p["hue"], centre) + 0.1 * abs(200.0 - p.get("sat", 200.0))))
    if not cands:
        seen = "\n".join(f"    {PT.hue_name(p['hue']):7s} X {p['center_mm'][0]:5.0f} Y {p['center_mm'][1]:5.0f}  area {p['area_mm2']:5.0f} mm2  "
                         f"roundness {p['roundness']:.2f}" for p in pools) or "    nothing"
        raise SystemExit(f"no {color} paint dish found for the probes. The camera sees:\n{seen}\n  Put paint in the dish or use --color / PROBE_COLOR.")
    p = cands[0]
    return p["center_mm"][0], p["center_mm"][1], p["radius_mm"]


def probe_problems(probes, corners_mm, lo_mm, hi_mm):
    """Each probe line must lie INSIDE the sheet, at least lo_mm from every edge, and its own edge must be no further than
    hi_mm away all along (so a line cannot sit in the artwork). corners_mm: tl, tr, br, bl in machine mm.
    Returns a list of problems (empty = all good)."""
    C = np.array(corners_mm, float)
    area = 0.5 * sum(C[i, 0] * C[(i + 1) % 4, 1] - C[(i + 1) % 4, 0] * C[i, 1] for i in range(4))
    sign = 1.0 if area > 0 else -1.0
    own_edge = {"top": 0, "right": 1, "bottom": 2, "left": 3}

    def dist_to_edge(q, i):
        a_, b_ = C[i], C[(i + 1) % 4]
        e = b_ - a_
        cross = e[0] * (q[1] - a_[1]) - e[1] * (q[0] - a_[0])
        return sign * cross / max(np.linalg.norm(e), 1e-9)

    problems = []
    for p in probes:
        s_ = np.array(p["start"], float); d = np.array(p["dir"], float); L = float(p["length"])
        pts = [s_ + d * t for t in np.linspace(0.0, L, int(L) + 1)]
        inside_min = min(min(dist_to_edge(q, i) for i in range(4)) for q in pts)
        own = [dist_to_edge(q, own_edge[p["name"].split()[0]]) for q in pts]
        if inside_min < lo_mm:
            what = "leaves the paper" if inside_min < 0 else "is too close to the paper edge"
            problems.append(f"'{p['name']}' {what} (closest margin {inside_min:+.1f} mm, needs at least {lo_mm:.1f})")
        elif max(own) > hi_mm:
            problems.append(f"'{p['name']}' is {max(own):.1f} mm inside its edge (at most {hi_mm:.1f}): it would be on the artwork")
    return problems


def run_probes(machine, cfg, frame, det, yes=False, color=None, ask=True, question="Paint the probes now?  [y/N] "):
    """Paints the four long lines (a dip before each), photographs, measures. Returns the result dict (saved to PROBE_FILE) or None.
    cfg heights are machine Z. The machine must be homed and parked."""
    _set_status("")
    cal = V.load_calibration()
    pf = cfg.get("paper_frame") or V.paper_frame(det, cal["H"])
    guess = contact_guess_from(cfg)
    pools = V.find_pools(frame, cal["H"], det["corners_px"])
    used = color or PROBE_COLOR
    try:
        dish = pick_dish(pools, used, cfg)
    except SystemExit:
        if used == PROBE_COLOR:
            raise
        print(f"probes: no {used} paint dish with paint in it; probing with {PROBE_COLOR} instead (PROBE_COLOR in settings.py)")
        used = PROBE_COLOR
        dish = pick_dish(pools, used, cfg)
    slot = max(sheet_probe_count(det), 1 if reusable_result(det, frame) else 0)
    if slot:
        print(f"this sheet already has {slot} set(s) of probe lines: the new lines go beside them along each edge")
    probes = plan_probes_on_sheet(det["corners_mm"], guess, slot)
    print(paper_check_text(probes, det["corners_mm"]))
    import settings as _S
    inset_s = float(getattr(_S, "PROBE_INSET_MM", 5.0))
    problems = probe_problems(probes, det["corners_mm"], 0.5 * inset_s, inset_s + 5.0)
    if problems:
        raise SystemExit("PROBE STOPPED before the machine moved: " + "; ".join(problems) +
                         ". The sheet and the probe plan disagree. Nothing was painted.")
    applied = applied_offset_from_last_result()
    steps = plan_step_row(det["corners_mm"], guess, slot)
    code = PT.shift_gcode_xy(build_gcode(probes, dish, cfg, guess, steps), -applied[0], -applied[1])
    if applied != [0.0, 0.0]:
        print(f"probe moves corrected by the last measured offset: ({-applied[0]:+.2f}, {-applied[1]:+.2f}) mm")
    cv2.imwrite(BEFORE, frame)
    plan = {"probes": probes, "pf": pf, "dish": list(dish), "contact_guess": guess, "applied_offset": applied, "steps": steps}
    json.dump(plan, open(PLAN, "w"), indent=1)
    lengths = ", ".join(f"{p['name']} {p['length']:.0f}" for p in probes)
    print(f"\nPROBES: {len(probes)} long lines inside the margin (mm: {lengths}), each going down from Z {probes[0]['z_start']:.2f} (air) "
          f"to Z {probes[0]['z_end']:.2f}; dish {used} at X {dish[0]:.0f} Y {dish[1]:.0f}; about {len(probes) * 1.0:.0f} min")
    if steps:
        print(f"        + {len(steps['strokes'])} short strokes along the top edge, landed like artwork lines, Z {steps['strokes'][0]['z']:.2f} "
              f"-> {steps['strokes'][-1]['z']:.2f} (they give the touch height and the width curve)")
    plan_path = plan_picture(frame, probes, dish, cal, "probe_plan.png", det, steps)
    if not yes:
        # shown also when not asked (PROBE_ASK = False): the probe then starts right away, the picture is to look at
        print("Look at probe_plan.png: red lines = the probe lines (inside the margin), blue circle = the paint dish.")
        if plan_path:
            PT.show_picture(plan_path)
    if ask and not yes and input(question).strip().lower() not in ("y", "yes", "j", "ja", "d", "da"):
        print("probes skipped: the profile / settings contact is used.")
        _set_status("you answered N to 'Paint the probes now?'")
        return None
    attempt = 0
    while True:
        if attempt > 0:
            guess = guess + PROBE_RETRY_MM
            probes = plan_probes_on_sheet(det["corners_mm"], guess, slot + attempt)
            steps = plan_step_row(det["corners_mm"], guess, slot + attempt)
            code = PT.shift_gcode_xy(build_gcode(probes, dish, cfg, guess, steps), -applied[0], -applied[1])
            plan = {"probes": probes, "pf": pf, "dish": list(dish), "contact_guess": guess, "applied_offset": applied, "steps": steps}
            json.dump(plan, open(PLAN, "w"), indent=1)
            cap = V.open_camera()
            frame, live = PT.capture_fresh(cap)                       # the paper now carries the earlier marks: they become the 'before'
            cap.release()
            cv2.imwrite(BEFORE, frame)
            print(f"\nRETRY {attempt}: some lines touched at their start, so the start is raised by {PROBE_RETRY_MM:.1f} mm "
                  f"(to Z {guess + PROBE_ABOVE_MM:.2f}) and all lines are painted again, beside the first ones")
        note_sheet_probe(det, slot + attempt + 1)
        if not PT.stream_gcode(machine, code, cfg):
            raise SystemExit("probe painting was stopped.")
        PT.go_to_snapshot(machine, cfg)
        time.sleep(1.5)
        cap = V.open_camera()
        frame2, live = PT.capture_fresh(cap)
        cap.release()
        cv2.imwrite(AFTER, frame2)
        probes, results, sol = analyse(frame, frame2, plan, cal, det["corners_px"], guess)
        report_picture(frame2, probes, results, cal, "probe_result.png", sol, plan.get("steps"))
        print("picture: probe_result.png (yellow = the lines, red cross = the touch point on each line, with its Z)")
        if sol is not None:
            res = save_result(sol, probes, results, det)
            bad = touch_width_problem(res)
            if bad:
                print(f"\nWARNING: {bad}. The lines are entered at a slant (LINE_LEAD_IN_MM), so the starts stay clean, "
                      f"but the marks may come out a bit wide.")
            return res
        touched = [p["name"] for p, r in zip(probes, results) if "already" in json.dumps(r).lower()]
        if not touched or attempt >= PROBE_RETRIES:
            print("no usable result: see the notes in the table.")
            _set_status("the probes were painted but gave no usable result (see the table above)")
            return None
        attempt += 1


def main(argv=None):
    import sys
    if argv is None and len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        print("probe.py does not take a picture or an output file.\n"
              "  To paint the artwork (with the probe first):   python paint.py men.png out.gcode\n"
              "  To run only the probe:                          python probe.py")
        return
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan_only", action="store_true", help="plan and show, never touch the machine")
    ap.add_argument("--yes", action="store_true", help="do not ask")
    ap.add_argument("--analyse", action="store_true", help="measure again from the saved pictures, no camera, no machine")
    ap.add_argument("--color", default=PROBE_COLOR, help=f"colour of the paint dish (default {PROBE_COLOR}; a dark colour shows best)")
    a = ap.parse_args(argv)
    cal = V.load_calibration()
    cfg = PT.default_config()
    if getattr(S, "Z_OFFSET", None) is not None:
        contact = float(S.Z_OFFSET)
    elif getattr(S, "BRUSH_PROFILE", None):
        contact = float(BP.load(BP.find(S.BRUSH_PROFILE))["z_offset"])
    else:
        contact = float(S.TOUCH_Z)
    cfg["z_offset"] = contact
    for key in ("z_paint", "z_low", "z_high", "z_wipe_exit", "dip_z"):
        cfg[key] += contact
    if a.analyse:
        plan = json.load(open(PLAN))
        f1, f2 = cv2.imread(BEFORE), cv2.imread(AFTER)
        det = V.clip_to_reach(V.detect_paper(f1, cal["H"], cal["H_inv"], calib_mm=cal["mm_pts"]))
        probes, results, sol = analyse(f1, f2, plan, cal, det["corners_px"], plan.get("contact_guess", contact))
        report_picture(f2, probes, results, cal, "probe_result.png", sol, plan.get("steps"))
        if sol is not None:
            save_result(sol, probes, results, det)
            print(f"saved {PROBE_FILE}")
        return
    machine = None
    if not a.plan_only:
        machine = PT.connect_and_home(cfg)
        PT.go_to_snapshot(machine, cfg)
    cap = V.open_camera()
    frame, live = PT.capture_fresh(cap)
    cap.release()
    det = V.clip_to_reach(V.detect_paper(frame, cal["H"], cal["H_inv"], calib_mm=cal["mm_pts"]))
    if det is None:
        raise SystemExit("no paper found in the picture (is the toolhead in view?)")
    cfg["paper_frame"] = V.paper_frame(det, cal["H"])
    if a.plan_only:
        pf = cfg["paper_frame"]
        slot = max(sheet_probe_count(det), 1 if reusable_result(det, frame) else 0)
        probes = plan_probes_on_sheet(det["corners_mm"], contact, slot)
        pools = V.find_pools(frame, cal["H"], det["corners_px"])
        dish = pick_dish(pools, a.color, cfg)
        plan_path = plan_picture(frame, probes, dish, cal, "probe_plan.png", det, plan_step_row(det["corners_mm"], contact, slot))
        print(f"plan picture: probe_plan.png ({len(probes)} lines, dish {a.color} at X {dish[0]:.0f} Y {dish[1]:.0f}); the machine was not touched.")
        return
    try:
        res = run_probes(machine, cfg, frame, det, yes=a.yes, color=a.color)
        if res is not None:
            print(f"\nresult saved: {PROBE_FILE}  (paint.py --probe_file {os.path.basename(PROBE_FILE)} uses it)")
    finally:
        if machine is not None:
            machine.disconnect()


if __name__ == "__main__":
    main()
