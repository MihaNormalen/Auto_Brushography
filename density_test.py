"""density_test.py - THE 12-PAINT TEST SHEET (2026-11-23): what every paint really does, alone and over every other one.

    python density_test.py make            # ko/dt/: the pass pictures, ko/dt/layout.json, ko/dt/simulation.png (the model's guess)
    python density_test.py paint           # all 48 jobs, drying between the rounds (one night)
    python density_test.py paint 17        # from job 17 on (after a stop; the sheet must not move)

THE SHEET (the paper of the colour runs, 363 x 246 mm; the picture 330 x 212 mm):
  left, THE MATRIX OF OVERLAPS, 12 x 12:
    round 1  every paint as a VERTICAL band (10 mm), 16 mm apart
    round 2  every paint as a HORIZONTAL band over them (after drying)
    round 3  every paint VERTICAL again, half a band to the right (after drying)
    -> in every column: its left third = the column's paint UNDER the row's, the middle = two coats of it, the right third
       = the column's paint OVER the row's. Every pair of the 12 paints in both orders, and two coats of each.
  right, FOR EVERY PAINT ONE ROW from ONE DIP (round 1): dabs of 2 / 3.5 / 5 mm, two short strokes (10 mm), and a zig-zag
    of 3 x 70 mm until the brush runs dry - is a dab darker than a stroke, how far does one dip carry.
Questions it answers (PDF "Kaj čopič res naredi" / "What the brush really does", and the separation PDF): the order of
coats, white under a glaze, the opaques over glazes, dab or stroke, when to dip again.
Every job's photo: ko/dt/after_NN.png (the blank sheet first: ko/dt/blank.png) - stroke by stroke and cell by cell they are
measured afterwards (stroke_density, the cells of the layout).
"""
import json
import os
import shutil
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import color_chart as CC                                    # noqa: E402

DIR = os.path.join(HERE, "ko", "dt")
LAYOUT = os.path.join(DIR, "layout.json")
PX = 10                                  # picture pixels per mm
W_MM, H_MM = 330.0, 212.0                # the picture (the colour runs' sheet minus a 16 mm margin)
MARGIN_MM = 16.0
BAND, PERIOD, ROW_H, ROW_GAP = 10.0, 16.0, 10.0, 4.4
DRY_GLAZE, DRY_OPAQUE = 25.0, 40.0
ORDER = ["orange", "blue", "green", "red", "black", "orange_o", "green_o", "blue_o", "red_o", "black_o", "white", "white_o"]


def paints_show():
    import color_paint as CP
    return {p: CP.PAINTS[p]["show"] for p in ORDER}


def layout():
    """The jobs: [{"n", "round", "paint", "kind", "shapes"}], shapes in picture mm (x right, y down)."""
    x0, y0 = 4.0, 6.0
    cols = [x0 + k * PERIOD for k in range(12)]
    rows = [y0 + 4.0 + k * (ROW_H + ROW_GAP) for k in range(12)]
    y_top, y_bot = y0, rows[-1] + ROW_H + 4.0
    x_end = cols[-1] + PERIOD
    rx = x_end + 8.0                                        # the single-coat rows
    jobs = []
    for p, x in zip(ORDER, cols):
        jobs.append({"round": 1, "paint": p, "kind": "band_v", "shapes": [["rect", x, y_top, x + BAND, y_bot]]})
    for k, p in enumerate(ORDER):                           # one row per paint, one dip each
        y = rows[k]
        sh = []
        for j, d in enumerate((2.0, 3.5, 5.0)):
            for n in range(2):
                sh.append(["dab", rx + 3 + j * 13 + n * 6.5, y + 2.5, d])
        for n in range(2):
            sh.append(["line", [[rx + 42 + n * 14, y + 2.5], [rx + 52 + n * 14, y + 2.5]], 3.0])
        legs = [[rx + 2, y + 7.0], [rx + 72, y + 7.0], [rx + 72, y + 10.5], [rx + 2, y + 10.5]]
        sh.append(["drain", legs, 2.5])
        jobs.append({"round": 1, "paint": p, "kind": "single", "shapes": sh})
    for p, y in zip(ORDER, rows):
        jobs.append({"round": 2, "paint": p, "kind": "band_h", "shapes": [["rect", x0 - 2, y, x_end - 2, y + ROW_H]]})
    for p, x in zip(ORDER, cols):
        jobs.append({"round": 3, "paint": p, "kind": "band_v2", "shapes": [["rect", x + BAND / 2, y_top, x + 1.5 * BAND, y_bot]]})
    for n, j in enumerate(jobs, 1):
        j["n"] = n
    return {"jobs": jobs, "cols": cols, "rows": rows, "y_top": y_top, "y_bot": y_bot, "rx": rx, "picture_mm": [W_MM, H_MM],
            "px_per_mm": PX, "margin_mm": MARGIN_MM, "order": ORDER}


def draw(shapes, img, val=1, scale=PX):
    for sh in shapes:
        if sh[0] == "rect":
            _, a, b, c, d = sh
            img[int(round(b * scale)):int(round(d * scale)), int(round(a * scale)):int(round(c * scale))] = val
        elif sh[0] == "dab":
            cv2.circle(img, (int(round(sh[1] * scale)), int(round(sh[2] * scale))), max(1, int(round(sh[3] / 2 * scale))), val, -1)
        else:
            pts = np.array([[int(round(x * scale)), int(round(y * scale))] for x, y in sh[1]], np.int32)
            cv2.polylines(img, [pts], False, val, max(1, int(round(sh[2] * scale))))
    return img


def colour_of(p):
    base = p[:-2] if p.endswith("_o") else p
    return (60, 60, 60) if base not in CC.HUE_OF else tuple(int(v) for v in CC.name_rgb(base, 235)[::-1])


def make():
    os.makedirs(DIR, exist_ok=True)
    L = layout()
    w, h = int(W_MM * PX), int(H_MM * PX)
    for j in L["jobs"]:
        img = np.full((h, w, 3), 255, np.uint8)
        m = draw(j["shapes"], np.zeros((h, w), np.uint8))
        img[m > 0] = colour_of(j["paint"])
        j["picture"] = f"job{j['n']:02d}.png"
        cv2.imwrite(os.path.join(DIR, j["picture"]), img)
    json.dump(L, open(LAYOUT, "w"), indent=1)
    print(f"density test: {len(L['jobs'])} jobs (round 1: {sum(1 for j in L['jobs'] if j['round'] == 1)}, "
          f"round 2: {sum(1 for j in L['jobs'] if j['round'] == 2)}, round 3: {sum(1 for j in L['jobs'] if j['round'] == 3)}) -> ko/dt/")
    simulate(L)
    print("next:  python density_test.py paint      (a fresh sheet, the palette filled, about 4-5 hours)")


def coat_values():
    """The model's guess of every paint: the glazes as the strokes measured them (2026-11-23), the others as run K learned."""
    import color_paint as CP
    out = {}
    st = {}
    p_k = os.path.join(HERE, "ko", "cp_runs", "2026-11-22_1648", "cp_state.json")
    if os.path.isfile(p_k):
        st = json.load(open(p_k))
    meas = {"orange": [0.93, 0.92, 0.44], "red": [1.0, 0.28, 0.57], "blue": [0.45, 0.91, 0.98], "green": [0.33, 0.71, 0.71],
            "black": [0.82, 0.81, 0.82]}
    for p in ORDER:
        if p in meas:
            out[p] = (np.array(meas[p], float), np.zeros(3))
        else:
            out[p] = CP.coat_factor(p, st)
    return out


def simulate(L, out=None):
    fac = coat_values()
    s = 4
    w, h = int(W_MM * s), int(H_MM * s)
    sheet = np.ones((h, w, 3)) * np.array([0.93, 0.94, 0.92])
    for j in L["jobs"]:
        t, r = fac[j["paint"]]
        if j["kind"] == "single":
            for sh in j["shapes"]:
                if sh[0] != "drain":
                    m = draw([sh], np.zeros((h, w), np.uint8), 1, s) > 0
                    sheet[m] = sheet[m] * t + r
                    continue
                done = 0.0
                for a, b in zip(sh[1], sh[1][1:]):
                    seg = float(np.hypot(b[0] - a[0], b[1] - a[1]))
                    for k in range(int(seg)):
                        u0, u1 = k / seg, min(1.0, (k + 1) / seg)
                        f = max(0.0, 1.0 - 0.0026 * (done + k))
                        m = draw([["line", [[a[0] + (b[0] - a[0]) * u0, a[1] + (b[1] - a[1]) * u0],
                                            [a[0] + (b[0] - a[0]) * u1, a[1] + (b[1] - a[1]) * u1]], sh[2]]],
                                 np.zeros((h, w), np.uint8), 1, s) > 0
                        sheet[m] = sheet[m] * t ** f + r * f
                    done += seg
            continue
        m = draw(j["shapes"], np.zeros((h, w), np.uint8), 1, s) > 0
        sheet[m] = sheet[m] * t + r
    lin = np.clip(sheet, 0, 1)
    rgb = np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * lin ** (1 / 2.4) - 0.055)
    img = (rgb[..., ::-1] * 255).astype(np.uint8)
    cv2.imwrite(out or os.path.join(DIR, "simulation.png"), img)
    return img


# --------------------------------------------------------------------------------------------------------- paint
def run_job(j, L, first):
    import settings
    import paint as PT
    import probe as PB
    import color_paint as CP
    settings.SIGNATURE = None                                 # no signature, no room for it
    settings.SIGN_PAINT = False
    settings.REPAIR_AT_END = False                            # nothing painted again: we measure what one go does
    settings.RINSE_BEFORE_JOB = first
    settings.INFILL_STYLE = "bands" if j["kind"].startswith("band") else "skeleton"
    single = j["kind"] == "single"
    for k_, v_ in (("DIP_MIN_MM", 10000.0 if single else None), ("DIP_MAX_MM", 10000.0 if single else None),
                   ("DIP_BY_WIDTH", False if single else None)):
        key = "_plain_" + k_
        if not hasattr(settings, key):
            setattr(settings, key, getattr(settings, k_))
        setattr(settings, k_, getattr(settings, key) if v_ is None else v_)   # one dip for the whole row (single coats)
    PB.PROBE_COLOR = CP.PROBE_PAINT
    PT.first_layer_color = lambda *a_, **k_: CP.PROBE_PAINT
    PT.JOB_INFO, PT.TEST_STROKES = None, []
    argv = [os.path.join("ko", "dt", j["picture"]), "--rotate", "0", "--margin", f"{L['margin_mm']:g}", "--color", "auto",
            "--yes"] + CP.cup_arg(j["paint"])
    probe_file = os.path.join(HERE, "probe_result.json")
    if not first and os.path.isfile(probe_file):
        argv += ["--probe_file", probe_file, "--probe_max_age_h", "24"]
    argv += ["--paper_keep", os.path.join(DIR, "paper.json"), "--scene_keep", os.path.join(DIR, "scene.json")]
    print(f"\n{'=' * 100}\ndensity test: JOB {j['n']} of {len(L['jobs'])} - round {j['round']}, {paints_show()[j['paint']]}, "
          f"{j['kind']}   ({time.strftime('%H:%M')})\n   paint.py " + " ".join(argv))
    PT.LAST_JOB_DONE = False
    PT.main(argv)
    if not PT.LAST_JOB_DONE:
        raise SystemExit(f"\njob {j['n']} was NOT painted to the end: stopped. Go on with:  python density_test.py paint {j['n']}")
    final = os.path.join(HERE, "paint_final.png")
    if os.path.isfile(final):
        shutil.copyfile(final, os.path.join(DIR, f"after_{j['n']:02d}.png"))
        if first:
            pf = os.path.join(HERE, "paper_frame.png")       # the photo before the probes and the paint: the blank sheet
            if os.path.isfile(pf):
                shutil.copyfile(pf, os.path.join(DIR, "blank.png"))


def paint(start=1):
    if not os.path.isfile(LAYOUT):
        raise SystemExit("no ko/dt/layout.json: run  python density_test.py make  first")
    L = json.load(open(LAYOUT))
    jobs = L["jobs"]
    if start == 1:
        for f in ("paper.json", "scene.json"):
            if os.path.isfile(os.path.join(DIR, f)):
                os.remove(os.path.join(DIR, f))
    t0 = time.time()
    for j in jobs[start - 1:]:
        prev = jobs[j["n"] - 2] if j["n"] > 1 else None
        if prev is not None and prev["round"] != j["round"] and j["n"] > start:
            opaque = any(x["paint"].endswith("_o") or x["paint"].startswith("white") for x in jobs if x["round"] == prev["round"])
            CC.wait_dry(DRY_OPAQUE if opaque else DRY_GLAZE, f"round {j['round']}")
        run_job(j, L, first=(j["n"] == start))
    print(f"\ndensity test: done, {len(jobs) - start + 1} jobs in {(time.time() - t0) / 3600:.1f} h. Photos: ko/dt/after_NN.png")


def main(argv):
    if not argv or argv[0] not in ("make", "paint"):
        raise SystemExit(__doc__)
    if argv[0] == "make":
        make()
    else:
        paint(int(argv[1]) if len(argv) > 1 else 1)


if __name__ == "__main__":
    main(sys.argv[1:])
