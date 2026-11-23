"""focus_check.py - a live camera window with a SHARPNESS number, for focusing the camera by hand.

    python focus_check.py

Turn the focus ring slowly: the bigger the number, the sharper. "best" keeps the highest seen; come back to it.
Measured in the middle of the picture (where the paper is). Put something with fine detail there (printed text, a ruler).
Q or Esc closes the window. Nothing on the machine moves.
"""
import cv2
import numpy as np

import vision as V


def main():
    cap = V.open_camera()
    best = 0.0
    hist = []
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        h, w = frame.shape[:2]
        y0, y1, x0, x1 = int(h * 0.3), int(h * 0.7), int(w * 0.3), int(w * 0.7)
        g = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        sharp = float(cv2.Laplacian(g, cv2.CV_64F).var())
        hist = (hist + [sharp])[-8:]
        s = float(np.median(hist))
        best = max(best, s)
        view = frame.copy()
        cv2.rectangle(view, (x0, y0), (x1, y1), (0, 200, 255), 2)
        pct = 100 * s / best if best else 0
        cv2.rectangle(view, (20, 20), (20 + int(6 * min(pct, 100)), 70), (0, 200, 0) if pct > 95 else (0, 160, 255), -1)
        cv2.putText(view, f"sharpness {s:7.1f}   best {best:7.1f}   ({pct:3.0f}% of best)", (30, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)
        cv2.putText(view, "turn the focus ring slowly - Q / Esc closes", (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        small = cv2.resize(view, (w // 2, h // 2))
        cy, cx = h // 2, w // 2                                   # the middle at full size: every camera pixel shown
        zoom = frame[cy - 270:cy + 270, cx - 240:cx + 240].copy()
        cv2.putText(zoom, "1:1 middle", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow("focus check", np.hstack([small, zoom]))
        k = cv2.waitKey(30) & 0xFF
        if k in (ord("q"), 27):
            break
    cap.release()
    cv2.destroyAllWindows()
    print(f"best sharpness {best:.1f}")


if __name__ == "__main__":
    main()
