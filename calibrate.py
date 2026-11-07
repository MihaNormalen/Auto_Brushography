"""
calibrate.py
============
  python calibrate.py                     machine draws dots; camera photographs BLANK paper first and the
                                          fresh (wet) dots right after; solves + saves calibration.npz
  python calibrate.py --dry_run           only write registration_dots.gcode
  python calibrate.py --from_images before.png after.png    solve from two saved photos
  python calibrate.py --no_draw           dots are on the paper right now (still wet!): photograph + solve
  --h4xx0r                                the h4xx0r theme (green phosphor behind dirty glass; the guided window holds still)

Magic-water paper: the dots fade within seconds, so photos are taken automatically.
"""
import argparse, math, time
import cv2
import numpy as np

import h4xx0r as HX

import machine as mc
from settings import (TOUCH_Z, HOME_LIFT_Z, CAL_HOVER_MM, Z_SAFE_DIP, CANVAS_CORNERS_MM, DOT_DWELL_MS, DOT_FEED,
                      FEED_TRAVEL, X_OFFSET, Y_OFFSET, DIP_X, DIP_Y, DIP_Z, CALIB_MARKER, MARKER_OFFSET_MM)
from vision import (open_camera, capture_frame, compute_homography, mm_to_pixel,
                    save_calibration, calibration_is_sane, solve_calibration, describe_orientation)

# heights (machine Z): everything is relative to the touch height now (before: Z_SAFE 2.8 was used both after homing and for hovering)
CAL_HOVER_Z = TOUCH_Z + CAL_HOVER_MM         # over a corner / a dot: just above the paper (low = little parallax), never touching
DOT_Z_DOWN = TOUCH_Z                         # the dot is pressed to the paper at the touch height
Z_DIP_SAFE = TOUCH_Z + Z_SAFE_DIP            # clearance around the water dish
Z_DIP = TOUCH_Z + DIP_Z                      # depth into the water


def make_dot_gcode(corners_mm):
    """Returns (pre_lines, draw_lines). pre = homing; draw = dip, dots, park."""
    pre = ["G90", "G21", "G28 Z", f"G0 Z{HOME_LIFT_Z:.3f} F3000", "G28 XY", "M400"]
    draw = [f"G0 Z{Z_DIP_SAFE:.3f} F3000",
            f"G0 X{DIP_X:.3f} Y{DIP_Y:.3f} Z{Z_DIP_SAFE:.3f} F{FEED_TRAVEL}",
            f"G1 Z{Z_DIP:.3f} F{DOT_FEED} ; lower into water", "G4 P500",
            f"G0 Z{Z_DIP_SAFE:.3f} F3000"]
    dots = [(cx + X_OFFSET, cy + Y_OFFSET, f"corner {i+1}") for i, (cx, cy) in enumerate(corners_mm)]
    if CALIB_MARKER:      # origin marker: BL dot + one more dot MARKER_OFFSET_MM towards +X
        bx, by = dots[0][0], dots[0][1]
        dots.insert(1, (bx + MARKER_OFFSET_MM, by, "origin marker"))
    for ax, ay, name in dots:
        draw += [f"; --- {name} at ({ax:.1f}, {ay:.1f}) ---",
                 f"G0 X{ax:.3f} Y{ay:.3f} Z{CAL_HOVER_Z:.3f} F{FEED_TRAVEL}",
                 f"G1 Z{DOT_Z_DOWN:.3f} F{DOT_FEED} ; press to paper",
                 f"G4 P{DOT_DWELL_MS}", f"G0 Z{CAL_HOVER_Z:.3f} F3000"]
    draw += [f"G0 Z{Z_DIP_SAFE:.3f} F3000", "G0 X0 Y0 F12000 ; park out of camera view", "M400"]
    return pre, draw


def _blobs(mask, w, h, open_k=3):
    m = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k)))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        a = cv2.contourArea(c)
        if not 30 < a < w*h*0.002: continue
        per = cv2.arcLength(c, True)
        if per < 1 or 4*math.pi*a/per**2 < 0.45: continue
        (_, _), (rw, rh), _ = cv2.minAreaRect(c)
        if min(rw, rh) < 1 or max(rw, rh)/min(rw, rh) > 1.8: continue
        M = cv2.moments(c)
        if M['m00']:
            out.append((M['m10']/M['m00'], M['m01']/M['m00'], a))
    return out


def find_marks(after, before, n_expected, debug=True):
    h, w = after.shape[:2]
    ga = cv2.GaussianBlur(cv2.cvtColor(after, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    if before is not None:
        gb = cv2.GaussianBlur(cv2.cvtColor(before, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        d32 = gb.astype(np.float32) - ga.astype(np.float32)
        d32 -= np.median(d32)                             # ignore global brightness change (auto-exposure)
        diff = np.clip(d32, 0, 255).astype(np.uint8)      # >0 where the paper got darker
        t, _ = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask = (diff > max(t, 12)).astype(np.uint8) * 255
        print(f"  Before/after difference: strongest darkening {int(diff.max())} levels")
    else:
        hsv = cv2.cvtColor(after, cv2.COLOR_BGR2HSV)
        mask = ((hsv[..., 2] < 90) & (hsv[..., 1] < 100)).astype(np.uint8) * 255
        print("  (no 'before' photo: looking for dark dots directly - less reliable)")
    cands = _blobs(mask, w, h)
    print(f"  {len(cands)} dot candidate(s), need {n_expected}")
    cands.sort(key=lambda c: c[2], reverse=True)
    dbg = after.copy()
    for cx, cy, a in cands:
        cv2.circle(dbg, (int(cx), int(cy)), 22, (0, 255, 255), 2)
    for cx, cy, a in cands[:n_expected]:
        cv2.circle(dbg, (int(cx), int(cy)), 22, (0, 255, 0), 3)
    if debug:
        cv2.imwrite("debug_dot_detection.png", dbg)
    if len(cands) < n_expected:
        print("  [ERROR] not enough dots -> see debug_dot_detection.png (dots may have dried before the photo)")
        return None
    return [(c[0], c[1]) for c in cands[:n_expected]]


def solve_and_save(after, before=None):
    n = len(CANVAS_CORNERS_MM) + (1 if CALIB_MARKER else 0)
    blobs = find_marks(after, before, n)
    if blobs is None:
        return False
    ordered, H, err, merr = solve_calibration(blobs, CANVAS_CORNERS_MM, CALIB_MARKER, MARKER_OFFSET_MM)
    if not calibration_is_sane(ordered):
        print("[ERROR] dots form a crossed shape - not saved."); return False
    H_inv = np.linalg.inv(H)
    info = describe_orientation(H, H_inv, CANVAS_CORNERS_MM)
    print("\nRESULT")
    for nm, (px, py), mm in zip(["BL", "BR", "TR", "TL"], ordered, CANVAS_CORNERS_MM):
        print(f"  {nm}: pixel ({px:.0f},{py:.0f})  = machine {mm}")
    print(f"  In the camera image: machine +X points {info['x_dir'].upper()}, +Y points {info['y_dir'].upper()}")
    print(f"  Corner order BL->BR->TR->TL looks {info['corner_order']} on screen"
          f"{'  (image is MIRRORED relative to a top-down view)' if info['mirrored'] else ''}")
    print(f"  Scale: {info['px_per_mm'][0]:.2f} px/mm along X, {info['px_per_mm'][1]:.2f} px/mm along Y")
    if merr is not None:
        print(f"  Origin-marker check: {merr:.2f} mm off" + ("   <-- BAD, do not trust this calibration" if merr > 2.0 else "  (good)"))
        if merr > 2.0:
            return False
    save_calibration(H, ordered, CANVAS_CORNERS_MM, image_size=(after.shape[1], after.shape[0]))
    dbg = after.copy()
    for nm, (px, py) in zip(["BL", "BR", "TR", "TL"], ordered):
        cv2.circle(dbg, (int(px), int(py)), 25, (0, 255, 0), 4)
        cv2.putText(dbg, nm, (int(px)+30, int(py)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    for mx, my in CANVAS_CORNERS_MM:
        px, py = mm_to_pixel(H_inv, mx, my)
        cv2.circle(dbg, (int(px), int(py)), 34, (0, 0, 255), 3)
    cv2.imwrite("calibration_result.png", dbg)
    print("  Overlay -> calibration_result.png (green = detected, red = projected)")
    return True


def run_calibration(dry_run=False, gcode_out="registration_dots.gcode", no_draw=False):
    pre, draw = make_dot_gcode(CANVAS_CORNERS_MM)
    open(gcode_out, "w").write("\n".join(pre + [""] + draw) + "\n")
    print(f"G-code written -> {gcode_out}")
    if dry_run:
        return
    cap = open_camera()
    if no_draw:
        after = capture_frame(cap, flush=5); cap.release()
        cv2.imwrite("calibration_capture.png", after)
        solve_and_save(after, None); return
    from machine import Machine
    with Machine() as m:
        for l in pre: m.send_line(l)
        m.wait_motion_complete()
        print("Photographing blank paper...")
        before = capture_frame(cap, flush=8)
        cv2.imwrite("calibration_before.png", before)
        print("Drawing dots...")
        for l in draw: m.send_line(l)
        m.wait_motion_complete()
        print("Photographing wet dots NOW...")
        after = capture_frame(cap, flush=6)
    cap.release()
    cv2.imwrite("calibration_capture.png", after)
    solve_and_save(after, before)



# ══ GUIDED CALIBRATION: the MACHINE shows which corner is which, you click what you see ══════════
class GuidedCal:
    NAMES = ["BL", "BR", "TR", "TL"]

    def __init__(self, machine=None, corners=CANVAS_CORNERS_MM):
        self.m, self.corners = machine, corners
        self.clicks = [None] * 4
        self.cur = None
        self.msg = "Press 1 to send the brush to corner BL"
        self.hx = HX if HX.ON and HX.has_font() else None
        self.glass = HX.Glass(still=True) if self.hx else None        # the picture holds still: you click on it
        self.t0 = time.time()

    def goto(self, i):
        x, y = self.corners[i]
        self.cur, self.clicks[i] = i, None
        if self.m is not None:
            self.msg = f"Moving brush over {self.NAMES[i]} (X{x:.0f} Y{y:.0f}) ..."
            self.m.send_line(f"G0 Z{HOME_LIFT_Z:.3f} F3000")                       # travel high ...
            self.m.send_line(f"G0 X{x + X_OFFSET:.3f} Y{y + Y_OFFSET:.3f} F{FEED_TRAVEL}", timeout=30)
            self.m.send_line(f"G0 Z{CAL_HOVER_Z:.3f} F3000")                       # ... and come down to just above the paper
            self.m.send_line("M400")
        self.msg = f"CLICK the brush tip (or the dot under it) for {self.NAMES[i]}"

    def click(self, px, py):
        if self.cur is None:
            self.msg = "First press 1, 2, 3 or 4"; return
        self.clicks[self.cur] = (float(px), float(py))
        nxt = [i for i in range(4) if self.clicks[i] is None]
        self.msg = (f"{self.NAMES[self.cur]} saved. Press {nxt[0]+1} for {self.NAMES[nxt[0]]}" if nxt
                    else "All 4 set. Check the grid, then press ENTER to save (R = redo)")
        self.cur = None

    def solve(self):
        if any(c is None for c in self.clicks):
            return None
        return compute_homography(self.clicks, self.corners)

    def render(self, frame, scale, plain=False):
        if self.hx and not plain:
            return self._render_h4xx0r(frame, scale)
        d = cv2.resize(frame, None, fx=scale, fy=scale)
        H = self.solve()
        if H is not None:
            Hi = np.linalg.inv(H)
            mm = np.array(self.corners)
            for x in range(int(mm[:, 0].min()) - 20, int(mm[:, 0].max()) + 21, 10):
                a, b = mm_to_pixel(Hi, x, mm[:, 1].min() - 20), mm_to_pixel(Hi, x, mm[:, 1].max() + 20)
                cv2.line(d, (int(a[0]*scale), int(a[1]*scale)), (int(b[0]*scale), int(b[1]*scale)), (255, 255, 0), 1)
            for y in range(int(mm[:, 1].min()) - 20, int(mm[:, 1].max()) + 21, 10):
                a, b = mm_to_pixel(Hi, mm[:, 0].min() - 20, y), mm_to_pixel(Hi, mm[:, 0].max() + 20, y)
                cv2.line(d, (int(a[0]*scale), int(a[1]*scale)), (int(b[0]*scale), int(b[1]*scale)), (255, 255, 0), 1)
            cx, cy = mm[:, 0].mean(), mm[:, 1].mean()
            o = mm_to_pixel(Hi, cx, cy)
            for (dx, dy), col, lab in (((30, 0), (0, 0, 255), "+X"), ((0, 30), (0, 220, 0), "+Y")):
                e = mm_to_pixel(Hi, cx + dx, cy + dy)
                cv2.arrowedLine(d, (int(o[0]*scale), int(o[1]*scale)), (int(e[0]*scale), int(e[1]*scale)), col, 4, cv2.LINE_AA, tipLength=0.15)
                cv2.putText(d, lab, (int(e[0]*scale)+8, int(e[1]*scale)+8), cv2.FONT_HERSHEY_SIMPLEX, 1.0, col, 3)
            i = describe_orientation(H, Hi, self.corners)
            info = f"+X->{i['x_dir']}  +Y->{i['y_dir']}  {i['px_per_mm'][0]:.1f} px/mm  {i['corner_order']}" + ("  MIRRORED" if i['mirrored'] else "")
            cv2.putText(d, info, (12, d.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        for k, p in enumerate(self.clicks):
            if p is not None:
                q = (int(p[0]*scale), int(p[1]*scale))
                cv2.circle(d, q, 14, (0, 255, 0), 2)
                cv2.putText(d, self.NAMES[k], (q[0] + 18, q[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        bar = d.copy(); cv2.rectangle(bar, (0, 0), (d.shape[1], 62), (0, 0, 0), -1)
        cv2.rectangle(bar, (0, d.shape[0]-28), (d.shape[1], d.shape[0]), (0, 0, 0), -1)
        d = cv2.addWeighted(bar, 0.75, d, 0.25, 0)
        cv2.putText(d, self.msg, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        state = "   ".join(f"{n}:{'OK' if c else '..'}" for n, c in zip(self.NAMES, self.clicks))
        cv2.putText(d, f"keys 1 2 3 4 = brush to BL BR TR TL   click = mark it   {state}", (12, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(d, "ENTER = save    R = redo all    ESC = quit", (12, d.shape[0]-9), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        return d

    def _render_h4xx0r(self, frame, scale):
        """The same window in the h4xx0r theme: the grid in dim green, +X bright, +Y hot, the marks bright, the lettering in VT323
        on see-through strips, and the glass over it - still, so a click lands where the picture shows it."""
        C = self.hx.BGR
        d = cv2.resize(frame, None, fx=scale, fy=scale)
        H = self.solve()
        lines = [(self.msg, "error" if ("NOT reached" in self.msg or "first" in self.msg) else "bright"),
                 ("   ".join(f"{n}:{'OK' if c else '..'}" for n, c in zip(self.NAMES, self.clicks)), "dim")]
        labels = []
        if H is not None:
            Hi = np.linalg.inv(H)
            mm = np.array(self.corners)
            px = lambda x, y: tuple(int(v * scale) for v in mm_to_pixel(Hi, x, y))
            for x in range(int(mm[:, 0].min()) - 20, int(mm[:, 0].max()) + 21, 10):
                cv2.line(d, px(x, mm[:, 1].min() - 20), px(x, mm[:, 1].max() + 20), C["dim"], 1)
            for y in range(int(mm[:, 1].min()) - 20, int(mm[:, 1].max()) + 21, 10):
                cv2.line(d, px(mm[:, 0].min() - 20, y), px(mm[:, 0].max() + 20, y), C["dim"], 1)
            cx, cy = mm[:, 0].mean(), mm[:, 1].mean()
            for (dx, dy), kind, lab in (((30, 0), "bright", "+X"), ((0, 30), "hot", "+Y")):
                e = px(cx + dx, cy + dy)
                cv2.arrowedLine(d, px(cx, cy), e, C[kind], 4, cv2.LINE_AA, tipLength=0.15)
                labels.append((lab, (e[0] + 8, e[1] - 12), kind))
            i = describe_orientation(H, Hi, self.corners)
            lines.append((f"+X->{i['x_dir']}  +Y->{i['y_dir']}  {i['px_per_mm'][0]:.1f} px/mm  {i['corner_order']}"
                          + ("  MIRRORED" if i['mirrored'] else ""), "warn" if i['mirrored'] else "ink"))
        for k, p in enumerate(self.clicks):
            if p is not None:
                q = (int(p[0] * scale), int(p[1] * scale))
                cv2.circle(d, q, 14, C["bright"], 2)
                labels.append((self.NAMES[k], (q[0] + 18, q[1] - 24), "bright"))
        out = self.hx.screen(d, "calibrate::guided", lines, clock=time.strftime("%H:%M:%S"),
                             bottom=[("[1-4] brush to BL BR TR TL   [click] mark it   [ENTER] save   [R] redo all   [ESC] quit", "dim")],
                             labels=labels, strip=0.6, t=time.time() - self.t0)
        return self.glass.apply(out)


def guided_calibration(use_machine=True):
    cap = open_camera()
    machine = None
    if use_machine:
        machine = mc.open_machine(home=True, quiet=False)
    g = GuidedCal(machine)
    ok, fr = cap.read()
    scale = 1280.0 / fr.shape[1]
    win = "brushography://calibrate  [1-4] [click] [ENTER] [ESC]" if g.hx else "Guided calibration"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win, lambda ev, x, y, fl, pa: g.click(x / scale, y / scale) if ev == cv2.EVENT_LBUTTONDOWN else None)
    saved = False
    while True:
        ok, fr = cap.read()
        if not ok:
            continue
        cv2.imshow(win, g.render(fr, scale))
        k = cv2.waitKey(30) & 0xFF
        if k in (ord('1'), ord('2'), ord('3'), ord('4')):
            g.goto(k - ord('1'))
        elif k in (ord('r'), ord('R')):
            g.clicks, g.cur, g.msg = [None] * 4, None, "Press 1 to send the brush to corner BL"
        elif k in (13, 10):
            H = g.solve()
            if H is None:
                g.msg = "Mark all 4 corners first"; continue
            save_calibration(H, g.clicks, CANVAS_CORNERS_MM, image_size=(fr.shape[1], fr.shape[0]))
            cv2.imwrite("calibration_result.png", g.render(fr, scale, plain=True)); saved = True     # the record: plain
            break
        elif k == 27:
            break
    cv2.destroyAllWindows(); cap.release()
    if machine is not None:
        machine.send_line(f"G0 Z{Z_DIP_SAFE:.3f} F3000"); machine.send_line("G0 X0 Y0 F12000", timeout=30)
        machine.send_line("M400"); machine.disconnect()
    print("Calibration saved. Now run calibration_verify.py" if saved else "Cancelled - nothing saved.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--guided", action="store_true", help="RECOMMENDED: machine moves over each corner, you click it")
    ap.add_argument("--no_machine", action="store_true", help="with --guided: do not move the machine, just click the 4 corners yourself")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--no_draw", action="store_true")
    ap.add_argument("--gcode_out", default="registration_dots.gcode")
    ap.add_argument("--from_images", nargs=2, metavar=("BEFORE", "AFTER"))
    ap.add_argument("--detect_only", action="store_true", help="re-solve calibration_capture.png (+ calibration_before.png)")
    HX.add_flag(ap)
    a = ap.parse_args()
    HX.from_args(a)
    if a.guided:
        guided_calibration(use_machine=not a.no_machine)
    elif a.from_images or a.detect_only:
        bp, ap_ = a.from_images if a.from_images else ("calibration_before.png", "calibration_capture.png")
        b, af = cv2.imread(bp), cv2.imread(ap_)
        if af is None: print(f"Cannot read {ap_}")
        else: solve_and_save(af, b)
    else:
        run_calibration(a.dry_run, a.gcode_out, a.no_draw)
