"""
brushograph/preview.py  —  plain live camera preview (no calibration needed)

  python preview.py            open CAMERA_INDEX from settings.py
  python preview.py --list     find which camera indices work
  python preview.py --index 1  try another camera

Keys:  Q/ESC quit   S save frame   X toggle crosshair   F toggle FPS/info
"""
import argparse, time
import cv2
from vision import open_camera, list_cameras
from settings import CAMERA_INDEX

ap = argparse.ArgumentParser()
ap.add_argument("--index", type=int, default=CAMERA_INDEX)
ap.add_argument("--list", action="store_true")
a = ap.parse_args()

if a.list:
    cams = list_cameras()
    print("Working cameras:", cams or "none found")
    raise SystemExit

cap = open_camera(index=a.index)
win = "Preview  (Q quit, S save, X crosshair)"
cv2.namedWindow(win, cv2.WINDOW_NORMAL)
cv2.resizeWindow(win, 1280, 720)
cross, info, n, t0, fps = False, True, 0, time.time(), 0.0
while True:
    ok, frame = cap.read()
    if not ok:
        print("Frame read failed"); break
    n += 1
    if time.time() - t0 >= 1:
        fps, n, t0 = n / (time.time() - t0), 0, time.time()
    h, w = frame.shape[:2]
    raw = frame.copy()                       # S saves the camera picture without the text / crosshair drawn below
    if cross:
        cv2.line(frame, (w//2, 0), (w//2, h), (0, 255, 0), 1)
        cv2.line(frame, (0, h//2), (w, h//2), (0, 255, 0), 1)
    if info:
        cv2.putText(frame, f"{w}x{h}  {fps:.0f} fps", (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.imshow(win, frame)
    k = cv2.waitKey(1) & 0xFF
    if k in (ord('q'), 27): break
    if k == ord('s'): cv2.imwrite("preview_snapshot.png", raw); print("saved preview_snapshot.png")
    if k == ord('x'): cross = not cross
    if k == ord('f'): info = not info
cap.release(); cv2.destroyAllWindows()
