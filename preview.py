"""
brushograph/preview.py  —  plain live camera preview (no calibration needed)

  python preview.py            open CAMERA_URL (or, if None, CAMERA_INDEX) from settings.py
  python preview.py --url http://host:8080/stream   try a network camera stream
  python preview.py --list     find which camera indices work
  python preview.py --index 1  try another USB camera (ignores CAMERA_URL)

  python preview.py --h4xx0r  the h4xx0r theme (green phosphor behind dirty glass)

Keys:  Q/ESC quit   S save frame   X toggle crosshair   F toggle FPS/info
"""
import argparse, time
import cv2
import h4xx0r as HX
from vision import open_camera, list_cameras
from settings import CAMERA_INDEX, CAMERA_URL

ap = argparse.ArgumentParser()
ap.add_argument("--index", type=int, default=None)
ap.add_argument("--url", default=None)
ap.add_argument("--list", action="store_true")
HX.add_flag(ap)
a = ap.parse_args()
themed = HX.from_args(a)

if a.list:
    cams = list_cameras()
    print("Working cameras:", cams or "none found")
    raise SystemExit

cap = open_camera(index=CAMERA_INDEX if a.index is None else a.index,
                  url=a.url or (CAMERA_URL if a.index is None else None))
source = a.url or (CAMERA_URL if a.index is None else None) or f"usb camera {CAMERA_INDEX if a.index is None else a.index}"
win = "brushography://preview  [Q] quit  [S] save  [X] crosshair" if themed else "Preview  (Q quit, S save, X crosshair)"
cv2.namedWindow(win, cv2.WINDOW_NORMAL)
cv2.resizeWindow(win, 1280, 720)
cross, info, n, t0, fps = False, True, 0, time.time(), 0.0
glass, start, saved_at = (HX.Glass() if themed else None), time.time(), -10.0
while True:
    ok, frame = cap.read()
    if not ok:
        print("Frame read failed"); break
    n += 1
    if time.time() - t0 >= 1:
        fps, n, t0 = n / (time.time() - t0), 0, time.time()
    h, w = frame.shape[:2]
    raw = frame.copy()                       # S saves the camera picture without the text / crosshair drawn below
    shown = None
    if themed:                               # the h4xx0r theme: 1280 wide, phosphor lettering, the glass over it
        d = cv2.resize(frame, (1280, int(round(1280 * h / w))), interpolation=cv2.INTER_AREA)
        dh, dw = d.shape[:2]
        if cross:
            cv2.line(d, (dw // 2, 0), (dw // 2, dh), HX.BGR["bright"], 1)
            cv2.line(d, (0, dh // 2), (dw, dh // 2), HX.BGR["bright"], 1)
        if info:
            banners = [("*** SNAPSHOT SAVED ***", "hot")] if time.time() - saved_at < 1.5 else []
            d = HX.screen(d, "preview::camera", [(f"{source}   {w}x{h}   {fps:.0f} fps", "ink")], clock=time.strftime("%H:%M:%S"),
                          bottom=[("[S] save snapshot   [X] crosshair   [F] lettering   [Q] quit", "dim")], banners=banners,
                          t=time.time() - start) if HX.has_font() else None
        if d is not None:
            shown = glass.apply(d)
    if shown is None:
        if cross:
            cv2.line(frame, (w//2, 0), (w//2, h), (0, 255, 0), 1)
            cv2.line(frame, (0, h//2), (w, h//2), (0, 255, 0), 1)
        if info:
            cv2.putText(frame, f"{w}x{h}  {fps:.0f} fps", (15, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        shown = frame
    cv2.imshow(win, shown)
    k = cv2.waitKey(1) & 0xFF
    if k in (ord('q'), 27): break
    if k == ord('s'): cv2.imwrite("preview_snapshot.png", raw); print("saved preview_snapshot.png"); saved_at = time.time()
    if k == ord('x'): cross = not cross
    if k == ord('f'): info = not info
cap.release(); cv2.destroyAllWindows()
