#!/usr/bin/env python3
"""check_paper_frame.py - takes ONE picture and shows where the probe frame sits on the sheet. The machine is NOT moved.
Run from the program folder (with calibration.npz, vision.py, settings.py, probe.py next to it):
    python check_paper_frame.py
Writes check_paper_frame.png (sheet corners in green, probe frame in red) and prints the numbers.
Put the toolhead out of the picture first.
"""
import math
import cv2
import numpy as np

import vision as V

cal = V.load_calibration()
cap = V.open_camera()
ok, frame = False, None
for _ in range(15):                          # let the camera settle, take the last frame
    ok, frame = cap.read()
cap.release()
if not ok or frame is None:
    raise SystemExit("no picture from the camera")

det = V.detect_paper(frame, cal["H"], cal["H_inv"], calib_mm=cal["mm_pts"])
if det is None:
    raise SystemExit("no paper found in the picture")
pf = V.paper_frame(det, cal["H"])

cm = np.array(det["corners_mm"], float)
centre_corners_mm = cm.mean(axis=0)
print("sheet corners (machine mm, tl tr br bl):", np.round(cm, 1).tolist())
print(f"sheet centre (corners average): X {centre_corners_mm[0]:.1f}  Y {centre_corners_mm[1]:.1f}")
print(f"frame origin (probe frame):     X {pf['origin_mm'][0]:.1f}  Y {pf['origin_mm'][1]:.1f}")
off = np.array(pf["origin_mm"]) - centre_corners_mm
print(f"frame origin minus sheet centre: {off[0]:+.1f} mm in X, {off[1]:+.1f} mm in Y")
print(f"frame size: {pf['paper_w']:.1f} x {pf['paper_h']:.1f} mm   angle {pf['angle_deg']:+.2f} deg")
print(f"sheet size (detected): {det['size_mm'][0]:.1f} x {det['size_mm'][1]:.1f} mm")

# the frame's four corners, back in the picture
c, u, v = np.array(pf["origin_mm"]), np.array(pf["u"]), np.array(pf["v"])
a0, a1, b0, b1 = pf["a_left"], pf["a_right"], pf["b_bot"], pf["b_top"]
frame_mm = [c + a0 * u + b0 * v, c + a1 * u + b0 * v, c + a1 * u + b1 * v, c + a0 * u + b1 * v]
px = [V.mm_to_pixel(cal["H_inv"], float(p[0]), float(p[1])) for p in frame_mm]
out = frame.copy()
cv2.polylines(out, [np.round(np.array(det["corners_px"])).astype(np.int32)], True, (0, 255, 0), 3)
cv2.polylines(out, [np.round(np.array(px)).astype(np.int32)], True, (0, 0, 255), 3)
cv2.imwrite("check_paper_frame.png", out)
print("written check_paper_frame.png (green = sheet, red = probe frame)")
