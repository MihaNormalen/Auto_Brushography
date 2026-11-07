"""The signature (SIGNATURE in settings.py) and the year, painted in the bottom margin under the artwork's right edge.

block_mask() gives the signature + year as a paint mask in the same convention as paint.load_mask (True = paint,
rows flipped: row 0 = bottom), so paint.py traces it into centre-line strokes like any other thin mark."""
import time

import cv2
import numpy as np


def _ink(path):
    """The signature picture -> bool ink mask, cropped to the ink."""
    g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if g is None:
        raise FileNotFoundError(f"cannot read the signature {path}")
    ink = g < 128
    ink = cv2.morphologyEx(ink.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0   # specks of scan noise
    ys, xs = np.nonzero(ink)
    if not len(xs):
        raise ValueError(f"no ink in the signature {path}")
    return ink[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def block_size_mm(path, width_mm, year=True, year_h_mm=4.0, gap_mm=3.0):
    """(width, height) of the signature + year block in mm, before planning (fit_artwork needs the height)."""
    ink = _ink(path)
    h = width_mm * ink.shape[0] / ink.shape[1]
    w = width_mm + ((gap_mm + _year_w_mm(year_h_mm)) if year else 0.0)
    return w, h


def _year_w_mm(h_mm):
    (tw, th), _ = cv2.getTextSize(time.strftime("%Y"), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1.0, 1)
    return h_mm * tw / th


def block_mask(path, width_mm, res, line_mm=0.8, year=True, year_h_mm=4.0, gap_mm=3.0):
    """Signature (width_mm wide) with the year to its right, bottoms aligned. Returns (mask, w_mm, h_mm);
    mask rows are flipped (row 0 = bottom), res px per mm. line_mm: how thick the year is drawn (only its centre line is used)."""
    ink = _ink(path)
    sw, sh = int(round(width_mm * res)), max(1, int(round(width_mm * res * ink.shape[0] / ink.shape[1])))
    sig = cv2.resize(ink.astype(np.uint8) * 255, (sw, sh), interpolation=cv2.INTER_AREA) > 100
    w_mm, h_mm = block_size_mm(path, width_mm, year, year_h_mm, gap_mm)
    W, Hh = int(round((w_mm + line_mm) * res)) + 4, sh + 4             # + the year's line thickness
    canvas = np.zeros((Hh, W), np.uint8)                       # image orientation here (row 0 = top), flipped at the end
    canvas[2:2 + sh, 2:2 + sw] = sig.astype(np.uint8) * 255
    if year:
        txt = time.strftime("%Y")
        (tw, th), base = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1.0, 1)
        scale = year_h_mm * res / th
        x = 2 + sw + int(round(gap_mm * res))
        y = Hh - 2 - int(round(base * scale * 0.3))                # baseline near the bottom of the signature
        cv2.putText(canvas, txt, (x, y), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, scale, 255,
                    max(1, int(round(line_mm * res))), cv2.LINE_AA)
    return np.flipud(canvas > 100), w_mm, h_mm
