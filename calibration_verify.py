"""
calibration_verify.py — live camera + machine-mm grid overlay.

  Move mouse   live mm coordinate under cursor
  Click        pin a coordinate
  G grid   C corners   R reload calibration   S snapshot   Q/ESC quit
"""
import math
import cv2

from settings import CALIBRATION_FILE, VERSION
from vision import open_camera
from vision import load_calibration, pixel_to_mm, mm_to_pixel, calibration_is_sane, describe_orientation

GRID_STEP_MM = 10
GRID_MARGIN  = 40          # mm drawn beyond the calibrated corners
GRID_COLOR   = (255, 255, 0)
AXIS_COLOR   = (0, 180, 255)
CORNER_COLOR = (0, 255, 0)
CLICK_COLOR  = (0, 60, 255)


class State:
    cal = H = H_inv = None
    msg = ""
    click_pt = hover = None
    show_grid = show_corners = True
    frame_size = None

    def load(self):
        try:
            self.cal = load_calibration(CALIBRATION_FILE)
            self.H, self.H_inv = self.cal['H'], self.cal['H_inv']
            self.msg = ""
            if not calibration_is_sane(self.cal['pixel_pts']):
                self.msg = "BAD CALIBRATION (crossed corners) - run calibrate.py --no_draw"
            print(f"Calibration loaded: {CALIBRATION_FILE}")
            print(f"  saved image size: {self.cal['image_size']}")
            return True
        except FileNotFoundError:
            self.cal = self.H = self.H_inv = None
            self.msg = "NO calibration.npz - run: python calibrate.py"
            print(self.msg); return False


st = State()


def mouse_cb(event, px, py, flags, param):
    if st.H is None:
        return
    st.hover = (px, py) + pixel_to_mm(st.H, float(px), float(py))
    if event == cv2.EVENT_LBUTTONDOWN:
        st.click_pt = st.hover
        print(f"  pixel ({px},{py}) → machine ({st.hover[2]:.2f}, {st.hover[3]:.2f}) mm")


def _line(img, a, b, color, t=1):
    cv2.line(img, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), color, t, cv2.LINE_AA)


def draw_grid(frame):
    mm = st.cal['mm_pts']
    x0 = math.floor((mm[:, 0].min()-GRID_MARGIN)/GRID_STEP_MM)*GRID_STEP_MM
    x1 = math.ceil((mm[:, 0].max()+GRID_MARGIN)/GRID_STEP_MM)*GRID_STEP_MM
    y0 = math.floor((mm[:, 1].min()-GRID_MARGIN)/GRID_STEP_MM)*GRID_STEP_MM
    y1 = math.ceil((mm[:, 1].max()+GRID_MARGIN)/GRID_STEP_MM)*GRID_STEP_MM
    layer = frame.copy()
    for x in range(int(x0), int(x1)+1, GRID_STEP_MM):
        major = x % 50 == 0
        _line(layer, mm_to_pixel(st.H_inv, x, y0), mm_to_pixel(st.H_inv, x, y1),
              AXIS_COLOR if major else GRID_COLOR, 2 if major else 1)
        if major:
            px, py = mm_to_pixel(st.H_inv, x, y0)
            cv2.putText(layer, f"X{x}", (int(px)+4, int(py)+18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, AXIS_COLOR, 2)
    for y in range(int(y0), int(y1)+1, GRID_STEP_MM):
        major = y % 50 == 0
        _line(layer, mm_to_pixel(st.H_inv, x0, y), mm_to_pixel(st.H_inv, x1, y),
              AXIS_COLOR if major else GRID_COLOR, 2 if major else 1)
        if major:
            px, py = mm_to_pixel(st.H_inv, x0, y)
            cv2.putText(layer, f"Y{y}", (int(px)+4, int(py)-6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, AXIS_COLOR, 2)
    return cv2.addWeighted(layer, 0.55, frame, 0.45, 0)


def draw_corners(frame):
    pts = []
    for lab, (mx, my) in zip(["BL", "BR", "TR", "TL"], st.cal['mm_pts']):
        px, py = mm_to_pixel(st.H_inv, float(mx), float(my))
        pts.append((int(px), int(py)))
        cv2.circle(frame, pts[-1], 20, CORNER_COLOR, 3)
        cv2.putText(frame, f"{lab} ({mx:.0f},{my:.0f})", (pts[-1][0]+24, pts[-1][1]+8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, CORNER_COLOR, 2)
    for i in range(4):
        cv2.line(frame, pts[i], pts[(i+1) % 4], CORNER_COLOR, 1, cv2.LINE_AA)
    for px, py in st.cal['pixel_pts']:               # where dots were seen at calibration
        cv2.circle(frame, (int(px), int(py)), 8, (0, 0, 255), 2)


def draw_axes(frame):
    """Arrows showing where machine +X (red) and +Y (green) point in the camera image."""
    mm = st.cal['mm_pts']
    cx, cy = float(mm[:, 0].mean()), float(mm[:, 1].mean())
    o = mm_to_pixel(st.H_inv, cx, cy)
    for (dx, dy), col, lab in (((30, 0), (0, 0, 255), "+X"), ((0, 30), (0, 220, 0), "+Y")):
        e = mm_to_pixel(st.H_inv, cx + dx, cy + dy)
        cv2.arrowedLine(frame, (int(o[0]), int(o[1])), (int(e[0]), int(e[1])), col, 4, cv2.LINE_AA, tipLength=0.15)
        cv2.putText(frame, lab, (int(e[0]) + 8, int(e[1]) + 8), cv2.FONT_HERSHEY_SIMPLEX, 1.0, col, 3)


def draw_marker(frame, pt, color, big=True):
    px, py, mx, my = pt
    if big:
        cv2.line(frame, (px-20, py), (px+20, py), color, 2)
        cv2.line(frame, (px, py-20), (px, py+20), color, 2)
    label = f"X {mx:.1f}  Y {my:.1f} mm"
    lx = px + 18 if px < frame.shape[1]-320 else px - 320
    ly = py - 18 if py > 40 else py + 40
    cv2.rectangle(frame, (lx-6, ly-26), (lx+300, ly+8), (0, 0, 0), -1)
    cv2.putText(frame, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)


def draw_hud(frame):
    h, w = frame.shape[:2]
    bar = frame.copy()
    cv2.rectangle(bar, (0, h-80), (w, h), (0, 0, 0), -1)
    frame[:] = cv2.addWeighted(bar, 0.6, frame, 0.4, 0)
    ok = st.H is not None and not st.msg
    cv2.putText(frame, "CAL: OK" if ok else st.msg, (12, h-48), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (0, 220, 0) if ok else (0, 0, 255), 2)
    cv2.putText(frame, "Move=coord Click=pin G=grid C=corners R=reload S=save Q=quit",
                (12, h-14), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
    cv2.putText(frame, f"{w}x{h}", (w-160, h-48), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
    cv2.putText(frame, f"brushograph {VERSION}", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    if st.H is not None:
        i = describe_orientation(st.H, st.H_inv, st.cal['mm_pts'])
        cv2.putText(frame, f"+X->{i['x_dir']}  +Y->{i['y_dir']}  {i['px_per_mm'][0]:.1f} px/mm  {i['corner_order']}"
                    + ("  MIRRORED" if i['mirrored'] else ""), (12, h-88), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    sz = st.cal['image_size'] if st.cal else None
    if sz and (w, h) != sz:
        cv2.putText(frame, f"RESOLUTION MISMATCH (cal {sz[0]}x{sz[1]})", (12, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)


def main():
    st.load()
    cap = open_camera()
    win = "Calibration Verify"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1280, 720)
    cv2.setMouseCallback(win, mouse_cb)
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Camera read failed"); break
        d = frame.copy()
        if st.H is not None:
            if st.show_grid:    d = draw_grid(d)
            if st.show_corners: draw_corners(d); draw_axes(d)
            if st.click_pt:     draw_marker(d, st.click_pt, CLICK_COLOR)
            if st.hover:        draw_marker(d, st.hover, (255, 255, 255), big=False)
        draw_hud(d)
        cv2.imshow(win, d)
        k = cv2.waitKey(1) & 0xFF
        if k in (ord('q'), ord('Q'), 27): break
        elif k in (ord('g'), ord('G')): st.show_grid = not st.show_grid
        elif k in (ord('c'), ord('C')): st.show_corners = not st.show_corners
        elif k in (ord('r'), ord('R')): st.load(); st.click_pt = None
        elif k in (ord('s'), ord('S')):
            cv2.imwrite("verify_snapshot.png", d); print("saved verify_snapshot.png")
    cap.release(); cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
