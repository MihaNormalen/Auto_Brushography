"""
checkup.py - after a photo of the sheet: which planned strokes of the last block of dips are NOT on the paper?

A stroke is the centre-line stroke the G-code marks with "; skeleton stroke". Its path is taken from the G-code (XY only).
The ink that the last block of dips added (the difference between the photo before and the photo now) is sampled along the path,
across it as well (the stroke has a width). A stroke whose ink is much weaker than the healthy strokes of the same block is MISSING.
Strokes that lie on paint that was already there are not judged (they are 'over old paint'), so no blob is ever repaired.
Only report: nothing on the machine changes here. The report is check_<n>.json and check_<n>.png (red = missing, green = ok,
grey = over old paint or not judged).
"""
import json
import re

import cv2
import numpy as np
from scipy.ndimage import map_coordinates

import vision as V

STEP_MM = 0.5                       # sampling step along a stroke
NOT_JUDGED_LOW = 0.35               # a stroke below this fraction of the healthy level is missing
MIN_GREY = 4.0                      # never judge below this grey level, however quiet the picture is


def parse_strokes(code):
    """Centre-line strokes of the job: dict(dip=<dip number, 1 = the first dip>, width=<mm>, pts=[(x, y), ...])."""
    strokes, dips, cur = [], 0, None
    for raw in code.split("\n"):
        l = raw.strip()
        if l.startswith("; --- DIP CYCLE"):
            dips += 1
            cur = None
        elif l.startswith("; skeleton stroke"):
            m = re.search(r"width ([\d.]+)-([\d.]+) mm", l)
            cur = {"dip": dips, "width": float(m.group(2)) if m else 0.5, "pts": []}
            strokes.append(cur)
        elif l.startswith(";"):
            cur = None
        elif l.startswith(("G0", "G1")) and cur is not None:
            tok = dict(re.findall(r"([XY])(-?[\d.]+)", l))
            if "X" not in tok or "Y" not in tok:
                continue
            if l.startswith("G0") and cur["pts"]:
                cur = None                                   # a travel move ends the stroke
                continue
            cur["pts"].append((float(tok["X"]), float(tok["Y"])))
    return [s for s in strokes if len(s["pts"]) >= 2]


def _resample(pts):
    """Points every STEP_MM along the polyline, with the unit tangent at each point."""
    p = np.asarray(pts, float)
    seg = np.diff(p, axis=0)
    L = np.hypot(seg[:, 0], seg[:, 1])
    cum = np.concatenate([[0.0], np.cumsum(L)])
    total = cum[-1]
    if total < 1e-6:
        return None, None
    s = np.arange(0.0, total + 1e-9, STEP_MM)
    if len(s) < 3:                                       # a dot or a very short stroke: too short to judge
        return None, None
    x = np.interp(s, cum, p[:, 0])
    y = np.interp(s, cum, p[:, 1])
    tx, ty = np.gradient(x), np.gradient(y)
    n = np.hypot(tx, ty) + 1e-9
    return np.stack([x, y], 1), np.stack([tx / n, ty / n], 1)


def ink_difference(frame1, frame2, paper_poly_px):
    """(ink, grey1, noise, paper): the darkening of frame2 against frame1 (paper only), the grey of frame1, the typical
    spread of the SIGNED difference over the whole paper (the noise of this pair of pictures), and the paper mask."""
    g1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY).astype(np.float32)
    poly = np.zeros(g1.shape, np.uint8)
    cv2.fillPoly(poly, [np.round(np.array(paper_poly_px)).astype(np.int32)], 255)
    inside = cv2.erode(poly, np.ones((5, 5), np.uint8)) > 0
    g2 = g2 * (float(np.median(g1[inside])) / max(float(np.median(g2[inside])), 1.0))
    signed = cv2.GaussianBlur(g1 - g2, (0, 0), 0.7)
    s = signed[inside]
    noise = 1.4826 * float(np.median(np.abs(s - np.median(s))))
    d = np.clip(signed, 0, None)
    d[~inside] = 0
    return d, g1, noise, inside


def _sample(img, H_inv, xy):
    px = V.mm_to_px_array(H_inv, xy)
    return map_coordinates(img, [px[:, 1], px[:, 0]], order=1, mode="nearest")


def stroke_score(D, old_dark, H_inv, stroke):
    """(score, over_old_paint): the ink along the stroke (mean over its length of the strongest value across its width),
    and whether the stroke lies on paint that was already on the paper before this block."""
    centre, tan = _resample(stroke["pts"])
    if centre is None:
        return 0.0, False
    hw = max(0.5, stroke["width"] / 2.0 + 0.3)
    offs = np.linspace(-hw, hw, 5)
    nrm = np.stack([-tan[:, 1], tan[:, 0]], 1)
    best = np.full(len(centre), -np.inf)
    for t in offs:
        xy = centre + t * nrm
        best = np.maximum(best, _sample(D, H_inv, xy))
    score = float(best.mean())
    old = float(_sample(old_dark, H_inv, centre).mean())
    return score, old > 25.0


def check_block(before_frame, now_frame, strokes, H_inv, paper_poly_px, stem):
    """Judges the strokes of one block of dips. Returns the list of results and writes stem.json and stem.png."""
    D, g_before, noise, inside = ink_difference(before_frame, now_frame, paper_poly_px)
    med = float(np.median(g_before[inside])) if inside.any() else 255.0
    old_dark = np.clip(med - g_before, 0, None)
    floor = max(MIN_GREY, 4.0 * noise)
    rows = []
    for k, st in enumerate(strokes):
        score, over = stroke_score(D, old_dark, H_inv, st)
        rows.append({"stroke": k, "dip": st["dip"], "score": round(score, 2), "over_old_paint": bool(over)})
    judged = [r["score"] for r in rows if not r["over_old_paint"]]
    ref = float(np.percentile(judged, 75)) if judged else 0.0
    reliable = ref >= 3.0 * floor
    for r in rows:
        if r["over_old_paint"]:
            r["status"] = "over old paint"
        elif not reliable:
            r["status"] = "not judged (block too faint to say)"
        elif r["score"] < max(NOT_JUDGED_LOW * ref, floor):
            r["status"] = "MISSING"
        else:
            r["status"] = "ok"
    summary = {"strokes": len(rows), "missing": sum(r["status"] == "MISSING" for r in rows),
               "ok": sum(r["status"] == "ok" for r in rows), "healthy_level": round(ref, 2), "floor": round(floor, 2),
               "reliable": bool(reliable), "rows": rows}
    json.dump(summary, open(stem + ".json", "w"), indent=1)
    _overlay(now_frame, strokes, rows, H_inv, stem + ".png")
    return summary


def _overlay(frame, strokes, rows, H_inv, path):
    out = frame.copy()
    color = {"ok": (0, 200, 0), "MISSING": (0, 0, 255)}
    for st, r in zip(strokes, rows):
        c = color.get(r["status"], (160, 160, 160))
        centre, _ = _resample(st["pts"])
        if centre is None:
            continue
        px = V.mm_to_px_array(H_inv, centre).astype(np.int32)
        cv2.polylines(out, [px.reshape(-1, 1, 2)], False, c, 2, cv2.LINE_AA)
    cv2.imwrite(path, out)


# ------------------------------------------------------------------------------------------------ end-of-job repair
def parse_strokes_full(code):
    """Centre-line strokes of the job, each with its LAYER (colour), and the dip cycle of EACH layer (its own dish).
    Returns (strokes, dip_blocks): dip_blocks maps layer name -> the lines of that layer's first dip cycle."""
    strokes, dip_blocks = [], {}
    layer, dips_in_layer, cur, collecting = None, 0, None, False
    for raw in code.split("\n"):
        l = raw.strip()
        if l.startswith("; ===== LAYER"):
            layer = l.split("LAYER", 1)[1].strip(" =")
            dips_in_layer, cur, collecting = 0, None, False
            continue
        if l.startswith("; --- DIP CYCLE"):
            dips_in_layer += 1
            cur = None
            collecting = dips_in_layer == 1 and layer is not None
            if collecting:
                dip_blocks.setdefault(layer, [])
            continue
        if l.startswith("; travel to the next stroke"):          # the dip cycle ends here: the trip after it goes to one stroke only
            collecting = False
            continue
        if l.startswith("; skeleton stroke"):
            collecting = False
            m = re.search(r"width ([\d.]+)-([\d.]+) mm", l)
            cur = {"layer": layer, "dip": dips_in_layer, "width": float(m.group(2)) if m else 0.5,
                   "pts": [], "lines": [l]}
            strokes.append(cur)
            continue
        if l.startswith(";") or not l:
            if l.startswith(";"):
                cur = None
            continue
        if collecting:
            dip_blocks[layer].append(l)
            continue
        if cur is None:
            continue
        if l.startswith(("G0", "G1")):
            tok = dict(re.findall(r"([XY])(-?[\d.]+)", l))
            if l.startswith("G0") and cur["pts"]:
                cur = None                                   # a travel move ends the stroke
                continue
            if "X" in tok and "Y" in tok:
                cur["pts"].append((float(tok["X"]), float(tok["Y"])))
        cur["lines"].append(l)
    return [s for s in strokes if len(s["pts"]) >= 2], dip_blocks


def shift_z(line, delta, z_min=0.3):
    """The same G-code line with its Z lowered by delta mm (never below z_min)."""
    m = re.search(r"Z(-?[\d.]+)", line)
    if not m:
        return line
    z = max(z_min, float(m.group(1)) - delta)
    return line[:m.start()] + f"Z{z:.3f}" + line[m.end():]


def repair_code(dip_block, missing, batch=8, delta=0.2, z_high=20.0):
    """G-code that paints the missing strokes again, deeper by delta, with a fresh dip before every batch of `batch` strokes."""
    out = ["; ===== REPAIR PASS: strokes still missing at the end of the job =====", "M400"]
    for k in range(0, len(missing), batch):
        out.append(f"; repair batch {k // batch + 1}")
        out.extend(dip_block)
        for st in missing[k:k + batch]:
            out.extend(shift_z(l, delta) for l in st["lines"])
        out.append(f"G0 Z{z_high:.3f} F3000")
    return "\n".join(out)


# ------------------------------------------------------------------------------------------------ judgement from ONE picture
def local_contrast(frame, strokes, H_inv):
    """For every stroke: how much darker its path is than the paper right beside it, in the same picture (grey levels).
    No start picture is needed. None = too short to judge."""
    g = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 0.8)
    vals = []
    for st in strokes:
        centre, tan = _resample(st["pts"])
        if centre is None:
            vals.append(None)
            continue
        nrm = np.stack([-tan[:, 1], tan[:, 0]], 1)
        hw = max(0.5, st["width"] / 2.0)
        on = np.min([_sample(g, H_inv, centre + t * nrm) for t in np.linspace(-hw, hw, 3)], axis=0)
        side = np.mean([_sample(g, H_inv, centre + s * nrm) for s in (-(hw + 1.5), hw + 1.5)], axis=0)
        vals.append(float(np.mean(side - on)))
    return vals


def judge_local(frame, strokes, H_inv, paper_poly_px, stem, min_healthy=15.0):
    """Judges every stroke from the finished picture alone. MISSING = its path is far less dark than the paper beside it.
    Not reliable when the picture shows too little paint in general (then nothing is judged)."""
    vals = local_contrast(frame, strokes, H_inv)
    judged = [v for v in vals if v is not None]
    ref = float(np.percentile(judged, 75)) if judged else 0.0
    reliable = ref >= min_healthy
    rows = []
    for k, v in enumerate(vals):
        if v is None:
            status = "too short to judge"
        elif not reliable:
            status = "not judged (block too faint to say)"
        elif v < max(8.0, 0.25 * ref):
            status = "MISSING"
        else:
            status = "ok"
        rows.append({"stroke": k, "contrast": None if v is None else round(v, 2), "status": status})
    summary = {"strokes": len(rows), "ok": sum(r["status"] == "ok" for r in rows),
               "missing": sum(r["status"] == "MISSING" for r in rows), "healthy_level": round(ref, 2),
               "reliable": bool(reliable), "rows": rows}
    json.dump(summary, open(stem + ".json", "w"), indent=1)
    _overlay(frame, strokes, [{"status": r["status"]} for r in rows], H_inv, stem + ".png")
    return summary


def repair_code_steps(dip_blocks, items, batch=8, z_high=20.0):
    """G-code that paints each (stroke, extra_depth_mm) again. Strokes are grouped by their own LAYER, and every batch starts
    with a dip into THAT layer's dish, never another colour's."""
    out = ["; ===== REPAIR PASS =====", "M400"]
    by_layer = {}
    for st, delta in items:
        by_layer.setdefault(st.get("layer"), []).append((st, delta))
    for layer, group in by_layer.items():
        dip = dip_blocks.get(layer)
        if not dip:
            out.append(f"; layer {layer}: no dip cycle found, {len(group)} strokes NOT repainted")
            continue
        for k in range(0, len(group), batch):
            out.append(f"; repair {layer} batch {k // batch + 1}")
            out.extend(dip)
            for st, delta in group[k:k + batch]:
                out.append(f"G0 Z{z_high:.3f} F3000")              # lift BEFORE the travel: no dragging across the paper
                for l in st["lines"]:
                    # only the paint-down moves are lowered; travel moves (G0) keep their planned height
                    out.append(shift_z(l, delta) if l.startswith("G1") else l)
            out.append(f"G0 Z{z_high:.3f} F3000")
    return "\n".join(out)


# ------------------------------------------------------------------------------------------------ what is painted, from the difference to the blank start picture
def paint_mask(frame, baseline, paper_poly_px):
    """Where the sheet now carries paint: the colour difference from the blank start picture, inside the paper only.
    The exposure of the two pictures is matched on the paper first, so lighting changes do not count as paint."""
    poly = np.zeros(frame.shape[:2], np.uint8)
    cv2.fillPoly(poly, [np.round(np.array(paper_poly_px)).astype(np.int32)], 255)
    inside = cv2.erode(poly, np.ones((5, 5), np.uint8)) > 0
    gf = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gb = cv2.cvtColor(baseline, cv2.COLOR_BGR2GRAY).astype(np.float32)
    k = float(np.median(gb[inside])) / max(float(np.median(gf[inside])), 1.0)
    f = frame.astype(np.float32) * k
    d = np.sqrt(((f - baseline.astype(np.float32)) ** 2).sum(-1))
    d = cv2.GaussianBlur(d, (0, 0), 1.0)
    thr = max(25.0, 0.5 * float(np.percentile(d[inside], 99)))
    mask = ((d > thr) & inside).astype(np.float32)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return mask


def judge_by_difference(frame, baseline, strokes, H_inv, paper_poly_px, stem, full=0.5, empty=0.35):
    """A stroke is ok when most of its path is painted in the photo, MISSING when little of it is. Only MISSING strokes are
    repainted, so painted areas are left alone."""
    mask = paint_mask(frame, baseline, paper_poly_px)
    rows = []
    for k, st in enumerate(strokes):
        centre, tan = _resample(st["pts"])
        if centre is None:
            rows.append({"stroke": k, "fraction": None, "status": "too short to judge"})
            continue
        nrm = np.stack([-tan[:, 1], tan[:, 0]], 1)
        hw = max(0.5, st["width"] / 2.0) * 0.6
        hits = np.mean([_sample(mask, H_inv, centre + t * nrm) for t in np.linspace(-hw, hw, 3)], axis=0)
        frac = float(np.mean(hits > 0.5))
        status = "ok" if frac >= full else ("MISSING" if frac < empty else "partly painted")
        rows.append({"stroke": k, "fraction": round(frac, 2), "status": status})
    judged = [r["fraction"] for r in rows if r["fraction"] is not None]
    summary = {"strokes": len(rows), "ok": sum(r["status"] == "ok" for r in rows),
               "missing": sum(r["status"] == "MISSING" for r in rows), "partly": sum(r["status"] == "partly painted" for r in rows),
               "reliable": bool(len(judged) > 0), "rows": rows}
    json.dump(summary, open(stem + ".json", "w"), indent=1)
    _overlay(frame, strokes, [{"status": r["status"]} for r in rows], H_inv, stem + ".png")
    return summary
