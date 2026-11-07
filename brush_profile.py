"""
brush_profile.py - the brush profile: what a ladder run measured, and what it means for painting.
ONE place for it (train.py writes the report, paint.py reads the numbers; before, each had its own copy).

  python brush_profile.py S4Z32        print the table and the numbers of a profile in ladder_dataset
"""
import json
import os
import sys

import numpy as np

RELIABLE = 0.8          # a depth counts as "marking" when at least this share of the stroke has a mark


def find(name):
    """'S4Z32', 'S4Z32.json' or a path -> the file. Also looks in ladder_dataset and next to this script."""
    here = os.path.dirname(os.path.abspath(__file__))
    names = [name] if name.lower().endswith(".json") else [name, name + ".json"]
    cands = []
    for n in names:
        cands += [n, os.path.join("ladder_dataset", n), os.path.join(here, n), os.path.join(here, "ladder_dataset", n)]
    for c in cands:
        if os.path.isfile(c):
            return c
    raise SystemExit(f"brush profile '{name}' not found. Looked for: " + ", ".join(dict.fromkeys(cands)))


def rows_from_samples(samples):
    """One row per depth, from the highest (lightest) to the lowest: strokes, share of the length with a mark, width median/p10/p90."""
    rows = []
    for z in sorted({round(s["z"], 3) for s in samples}, reverse=True):
        S = [s for s in samples if abs(s["z"] - z) < 1e-6]
        w = np.array([s["width_mm"] for s in S if s["visible"]], float)
        rows.append({"z": z, "machine_z": S[0]["z_machine"], "strokes": len({s["stroke"] for s in S}),
                     "vis": sum(1 for s in S if s["visible"]) / max(len(S), 1),
                     "w_med": float(np.median(w)) if len(w) else 0.0,
                     "w_p10": float(np.percentile(w, 10)) if len(w) else 0.0, "w_p90": float(np.percentile(w, 90)) if len(w) else 0.0})
    return rows


def analyse(rows):
    """The contact height (the highest depth that marks reliably), the width there, the widest line and the depth that gives it,
    and the CURVE: depth below contact -> width of the real brush (never decreasing). None if no depth marks reliably."""
    ok = [r for r in rows if r["vis"] >= RELIABLE]
    if not ok:
        return None
    contact = max(ok, key=lambda r: r["z"])
    deeper = [r for r in ok if r["z"] <= contact["z"]]
    wmax = max(r["w_med"] for r in deeper)
    wide = max([r for r in deeper if r["w_med"] >= 0.95 * wmax], key=lambda r: r["z"])
    pts = sorted((round(contact["machine_z"] - r["machine_z"], 3), r["w_med"]) for r in deeper)
    dd = np.array([p[0] for p in pts], float)
    ww = np.maximum.accumulate(np.array([p[1] for p in pts], float))        # a deeper press never makes a thinner line
    keep = np.r_[True, np.diff(ww) > 1e-6]                                  # the inverse needs strictly rising widths
    dd, ww = dd[keep], ww[keep]
    return {"z_contact": contact["z"], "machine_contact": contact["machine_z"], "w_min": contact["w_med"], "w_max": wmax,
            "z_wide": wide["z"], "depth": contact["z"] - wide["z"], "contact_is_top": contact["z"] >= rows[0]["z"] - 1e-6,
            "curve": (dd.tolist(), ww.tolist()), "max_depth": round(float(dd[-1]), 2), "w_curve_max": round(float(ww[-1]), 2)}


def report_lines(rows, rec):
    """The table and the numbers, as text (what train.py prints and saves)."""
    lines = ["", "  Z (logical) | machine Z | below contact | strokes | marked along | width median  (p10 - p90) mm",
             "  ------------+-----------+---------------+---------+--------------+------------------------------"]
    for r in rows:
        below = f"{rec['z_contact'] - r['z']:+.2f}" if rec and r["z"] <= rec["z_contact"] + 1e-6 else "  -  "
        w = f"{r['w_med']:.2f}  ({r['w_p10']:.2f} - {r['w_p90']:.2f})" if r["vis"] > 0 else "no mark"
        lines.append(f"  {r['z']:11.2f} | {r['machine_z']:9.2f} | {below:>13s} | {r['strokes']:7d} | {100 * r['vis']:10.0f} % | {w}")
    lines.append("")
    if rec is None:
        lines += ["  No level marked along 80% of the stroke: lower --z_bottom, or check the dip (is the brush wet?) and the paper."]
        return lines
    lines += [f"  Contact height: logical Z = {rec['z_contact']:.2f} (machine Z = {rec['machine_contact']:.2f}): the highest level that marks along >= 80% of the stroke.",
              f"  Thinnest reliable line: {rec['w_min']:.2f} mm.  Widest: {rec['w_max']:.2f} mm, reached {rec['depth']:.2f} mm below contact (logical Z = {rec['z_wide']:.2f})."]
    if rec["contact_is_top"]:
        lines.append("  WARNING: even the TOP level marks reliably - the real contact is higher: run again with a larger --z_top.")
    if rec["depth"] < 0.2:
        lines.append("  NOTE: the width hardly changes with depth for this brush/paper: lines will come out about the same thickness.")
    lines += ["", "  Use it in paint.py:  BRUSH_PROFILE = \"<this profile's name>\" in settings.py  (contact, widths and speed are taken from the file)."]
    return lines


def load(path):
    """Reads a ladder file for paint.py: the contact height (machine Z), the thinnest / widest line, the depth that gives it,
    the speed it was measured at, and the curve."""
    try:
        d = json.load(open(path, encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise SystemExit(f"cannot read the ladder file {path}: {e}")
    S = d.get("samples", [])
    if not S:
        raise SystemExit(f"{path} has no samples")
    rows = rows_from_samples(S)
    rec = analyse(rows)
    if rec is None:
        raise SystemExit(f"{path}: no depth marked along 80% of the stroke - the ladder has to be run again (see its report)")
    feed = d.get("setup", {}).get("feed")
    return {"z_offset": round(float(rec["machine_contact"]), 2), "line_depth": max(round(rec["depth"], 1), 0.1),
            "width_min": round(rec["w_min"], 1), "width_ref": round(rec["w_max"], 1), "brush": round(rec["w_min"], 1), "thin_max": round(rec["w_max"], 1),
            "paint_feed": int(feed) if feed else None, "feed": feed, "levels": len(rows), "strokes": len({s["stroke"] for s in S}),
            "top_is_contact": rec["contact_is_top"], "curve": rec["curve"], "max_depth": rec["max_depth"], "w_curve_max": rec["w_curve_max"],
            "contact": round(float(rec["machine_contact"]), 2)}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    path = find(sys.argv[1])
    d = json.load(open(path, encoding="utf-8"))
    rows = rows_from_samples(d["samples"])
    print(f"profile {path}  (measured at F{d.get('setup', {}).get('feed')})")
    print("\n".join(report_lines(rows, analyse(rows))))
