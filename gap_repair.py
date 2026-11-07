"""
gap_repair.py - paints only the GAPS: where the artwork has colour and the finished sheet has none.

  python gap_repair.py --art men.png --photo paint_final.png --calib calibration.npz --job_gcode out.gcode
                       --contact 3.7 [--high 19] [--out repair_gaps.gcode] [--overlay gap_overlay.png]

1. Finds the paper in the photo, and the paint on it.
2. Places the artwork on the paper (rough placement from the margin, then refined by matching the ink).
3. Gap = artwork has colour AND the photo has no paint there. Thin rims along the outlines are dropped
   (opened with the brush width), so only real holes remain.
4. Each gap takes the colour of the artwork under it (red or blue), and is filled with stroke cells.
5. Writes G-code: each colour's strokes are dipped into THAT colour's dish (the dip cycle of its own layer in the job G-code).
Nothing goes to the machine: you read the overlay and the summary first.
"""
import argparse, json, math, os, sys
import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import paint as PT
import checkup as CU
from vision import pixel_to_mm, px_to_mm_array, load_calibration
from paint import RES


def paper_box(hsv):
    pap = ((hsv[..., 1] < 70) & (hsv[..., 2] > 160)).astype(np.uint8)
    pap = cv2.morphologyEx(pap, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(pap)
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    return st[k, :4]


def align(art, photo_bgr):
    """Places the artwork on the paper: scale and position from the paper, refined by matching the ink (ECC)."""
    H_, W_ = photo_bgr.shape[:2]
    hsv = cv2.cvtColor(photo_bgr, cv2.COLOR_BGR2HSV)
    px, py, pw, ph_ = paper_box(hsv)
    ahsv = cv2.cvtColor(art, cv2.COLOR_BGR2HSV)
    ink_a = ((ahsv[..., 1] > 90) & (ahsv[..., 2] > 40)).astype(np.float32)
    xs = np.arange(W_)[None, :]; ys = np.arange(H_)[:, None]
    roi = (xs > px) & (xs < px + pw) & (ys > py) & (ys < py + ph_)
    # paint = clearly more coloured OR clearly darker than the bare paper (pale paint is still paint)
    paper_v = float(np.median(hsv[..., 2][roi & (hsv[..., 1] < 70)]))
    paper_s = float(np.median(hsv[..., 1][roi & (hsv[..., 1] < 70)]))
    paint = (((hsv[..., 1] > paper_s + 35) | (hsv[..., 2] < paper_v - 35))) & roi
    d = 2
    P = cv2.resize(paint.astype(np.float32), (W_ // d, H_ // d), interpolation=cv2.INTER_AREA) > 0.3
    best = (-1, None)
    s0 = (pw * 274.0 / 295.0) / art.shape[1]                       # the job's artwork: 274 mm on the 295 mm sheet
    for fs in np.linspace(0.9, 1.1, 5):
        s = s0 * fs; aw, ah = art.shape[1] * s, art.shape[0] * s
        for dx in range(-40, 41, 8):
            for dy in range(-40, 41, 8):
                x0 = px + (pw - aw) / 2.0 + dx; y0 = py + (ph_ - ah) / 2.0 + dy
                M = np.array([[s / d, 0, x0 / d], [0, s / d, y0 / d]], np.float32)
                A = cv2.warpAffine(ink_a, M, (W_ // d, H_ // d)) > 0.5
                iou = (A & P).sum() / max((A | P).sum(), 1)
                if iou > best[0]:
                    best = (iou, (s, x0, y0))
    s, x0, y0 = best[1]
    M = np.array([[s, 0, x0], [0, s, y0]], np.float32)
    warp0 = cv2.warpAffine(ink_a, M, (W_, H_))
    warp = np.eye(2, 3, dtype=np.float32)
    try:
        cc, warp = cv2.findTransformECC(paint.astype(np.float32), warp0, warp, cv2.MOTION_AFFINE,
                                        (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-5), None, 5)
        A3 = np.vstack([M, [0, 0, 1]]); W3 = np.vstack([warp, [0, 0, 1]])
        M = (np.linalg.inv(W3) @ A3)[:2].astype(np.float32)
        print(f"alignment: overlap {best[0]:.2f} (coarse), refined correlation {cc:.3f}")
    except cv2.error:
        print(f"alignment: coarse only, overlap {best[0]:.2f} (refinement did not converge)")
    return M, roi, paint, (px, py, pw, ph_)


def colour_of(hue, sat, val):
    """'red' or 'blue' for the paint colour; None for anything else (not repainted)."""
    if sat < 90 or val < 40:
        return None
    if hue < 20 or hue > 160:
        return "red"
    if 95 <= hue <= 140:
        return "blue"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--art", required=True); ap.add_argument("--photo", required=True)
    ap.add_argument("--calib", required=True); ap.add_argument("--job_gcode", required=True)
    ap.add_argument("--contact", type=float, required=True, help="machine Z of the paint contact (from the probe)")
    ap.add_argument("--high", type=float, default=19.0, help="machine Z for travel (lift), as in the job header")
    ap.add_argument("--brush_min", type=float, default=1.9); ap.add_argument("--brush_max", type=float, default=4.2)
    ap.add_argument("--probe", required=True, help="probe_result.json of this run (its curve gives the depth for each width)")
    ap.add_argument("--out", default="repair_gaps.gcode"); ap.add_argument("--overlay", default="gap_overlay.png")
    a = ap.parse_args()

    art = cv2.imread(a.art); photo = cv2.imread(a.photo)
    if art is None or photo is None:
        sys.exit("cannot read the artwork or the photo")
    cal = load_calibration(a.calib)                         # keeps the lens
    H = cal["H"]
    # millimetres per pixel at the paper centre (from the calibration)
    cx, cy = photo.shape[1] / 2.0, photo.shape[0] / 2.0
    p0 = np.array(pixel_to_mm(H, cx, cy)); p1 = np.array(pixel_to_mm(H, cx + 1, cy))
    px_per_mm = 1.0 / float(np.linalg.norm(p1 - p0))
    brush_px = a.brush_min * px_per_mm

    M, roi, paint, box = align(art, photo)
    gap = (cv2.warpAffine(((cv2.cvtColor(art, cv2.COLOR_BGR2HSV)[..., 1] > 90) &
                           (cv2.cvtColor(art, cv2.COLOR_BGR2HSV)[..., 2] > 40)).astype(np.float32),
                          M, (photo.shape[1], photo.shape[0])) > 0.5) & roi & ~paint
    # drop the thin rims along the outlines: only holes at least one brush wide survive
    k = max(3, int(round(brush_px)) | 1)
    gap = cv2.morphologyEx(gap.astype(np.uint8), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) > 0
    print(f"scale {px_per_mm:.2f} px/mm, brush {a.brush_min} mm = {brush_px:.1f} px, gap pixels after dropping rims: {int(gap.sum())}")

    # colour of each gap pixel = the colour of the artwork under it
    ahsv = cv2.cvtColor(art, cv2.COLOR_BGR2HSV)
    Minv = cv2.invertAffineTransform(M)
    ys, xs = np.nonzero(gap)
    ax = Minv[0, 0] * xs + Minv[0, 1] * ys + Minv[0, 2]; ay = Minv[1, 0] * xs + Minv[1, 1] * ys + Minv[1, 2]
    axi = np.clip(np.round(ax).astype(int), 0, art.shape[1] - 1); ayi = np.clip(np.round(ay).astype(int), 0, art.shape[0] - 1)
    hsv_s = ahsv[ayi, axi]
    colour = np.array([colour_of(int(h), int(s), int(v)) or "" for h, s, v in hsv_s])

    # gap pixels -> machine mm
    pts_px = np.c_[xs, ys].astype(np.float32).reshape(-1, 1, 2)
    mm = px_to_mm_array(H, np.c_[xs, ys])                   # lens-aware

    pr = json.load(open(a.probe))
    cd, cw = pr["curve"]
    cfg = PT.default_config()
    cfg.update({"brush_w": a.brush_min, "thin_max": a.brush_max, "cell_len_mm": (10.0, 25.0), "seed": 3,
                "outline": False, "curve": (list(cd), list(cw)), "z_paint": a.contact,
                "line_depth": round(max(min(cd[-1], a.contact - 0.2), 0.0), 2)})
    wmax = float(cw[-1])

    def depth_z(w):
        """Machine Z for a centre-line mark of width w: the depth at which the real brush makes that width (as in the job)."""
        w = min(float(w), wmax)
        return max(0.3, a.contact - float(np.interp(w, cw, cd)))      # never below the machine's Z limit (0.3)
    dip_blocks = CU.parse_strokes_full(open(a.job_gcode, encoding="utf-8", errors="ignore").read())[1]
    out = ["; ===== GAP REPAIR: only the gaps, each colour from its own dish =====",
           "G90", "G21", "M400"]
    summary = {}
    for layer in ("red", "blue"):
        sel = colour == layer
        if not sel.any():
            continue
        m = mm[sel]
        origin = m.min(axis=0) - 2.0
        W_g = int(math.ceil((m[:, 0].max() - origin[0]) * RES)) + 4
        H_g = int(math.ceil((m[:, 1].max() - origin[1]) * RES)) + 4
        grid = np.zeros((H_g, W_g), bool)
        gx = np.clip(((m[:, 0] - origin[0]) * RES).astype(int), 0, W_g - 1)
        gy = np.clip(((m[:, 1] - origin[1]) * RES).astype(int), 0, H_g - 1)
        grid[gy, gx] = True
        grid = cv2.morphologyEx(grid.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)) > 0
        # the SAME planner as the job: skeleton centre lines (thin parts) with the real width at every point
        strokes, _ = PT.plan_strokes(grid, cfg, res=RES)
        dip = dip_blocks.get(layer)
        if not dip:
            print(f"{layer}: no dip cycle in the job G-code, {len(strokes)} strokes NOT written")
            continue
        out.append(f"; ----- {layer}: {len(strokes)} gap strokes -----")
        for k2 in range(0, len(strokes), 8):
            out.append(f"; {layer} batch {k2 // 8 + 1}")
            out.extend(dip)
            for st in strokes[k2:k2 + 8]:
                mm_pts = [(origin[0] + x / RES, origin[1] + y / RES) for x, y in st]
                if len(mm_pts) < 2:
                    continue
                if getattr(st, "widths", None) is not None and st.kind == "line":
                    zs = [depth_z(w) for w in st.widths]
                else:
                    zs = [a.contact] * len(mm_pts)
                out.append(f"G0 Z{a.high:.3f} F3000")
                out.append(f"G0 X{mm_pts[0][0]:.3f} Y{mm_pts[0][1]:.3f} F12000")
                out.append(f"G1 Z{zs[0]:.3f} F2000")
                out.append("G1 F3000")
                for (x, y), z in zip(mm_pts[1:], zs[1:]):
                    out.append(f"G1 X{x:.3f} Y{y:.3f} Z{z:.3f}")
                out.append(f"G0 Z{a.high:.3f} F3000")
        summary[layer] = len(strokes)
    out.append("G0 X0 Y0 F12000 ; park out of camera view")
    open(a.out, "w").write("\n".join(out) + "\n")
    print("strokes per colour:", summary)
    if colour.size:
        print("gap pixels by colour:", {c: int((colour == c).sum()) for c in ("red", "blue", "")})
    ov = photo.copy()
    ov[gap] = (0, 255, 255)
    cv2.imwrite(a.overlay, ov)
    print(f"written {a.out} and {a.overlay}")


if __name__ == "__main__":
    main()
