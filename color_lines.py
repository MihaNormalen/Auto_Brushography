"""color_lines.py - THE PICTURE'S DRAWING for color_paint's finish (2026-11-23).

The colour passes bring colour and large areas, but never draw the form: the face, hands, legs and outlines stayed soft. At
the end (when no colour pass helps any more) color_paint FINISHES like a brushographer:

  * EDGES: thin passes along the picture's own lines (TEED on ComfyUI, the best line source we tried - figure outline, face,
    fence, leaves; few lines in the textured hills), ONLY where the sheet there is still lighter than the picture wants -
    so a line is drawn where the picture has a dark edge the sheet lost, never as a cartoon outline round everything. The
    paint is the one that brings those spots nearest (usually the black or a dark glaze), painted as centre lines with the
    probe's width curve - the black Čopičograf's own proven way.
  * LIGHTS: small spots where the picture is lighter than all round it (a local light: eyes, a fold, a lit edge) and the
    sheet is clearly too dark there: opaque white, small.

The line map is made once per picture and kept in ko/comfy/<name>_teed.png. Without ComfyUI: a local line drawing (a
difference of Gaussians on the lightness, the classic XDoG way) - coarser, but the run never stops for it.
"""
import json
import os
import sys
import time
import urllib.parse

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import settings as S                                        # noqa: E402

CACHE = os.path.join(HERE, "ko", "comfy")


def setting(name, default):
    return getattr(S, name, default)


def teed_path(image):
    return os.path.join(CACHE, os.path.splitext(os.path.basename(image))[0] + "_teed.png")


def _comfy_teed(image, p, res=1024, wait_s=300):
    import comfy_client as C
    if setting("COMFY_URL", None):
        C.URL = str(setting("COMFY_URL", C.URL)).rstrip("/")
    print(f"color_lines: the picture's drawing (TEED) on ComfyUI ({C.URL}) ...", flush=True)
    name = C.upload(image)
    wf = {"1": {"class_type": "LoadImage", "inputs": {"image": name}},
          "2": {"class_type": "TEEDPreprocessor", "inputs": {"image": ["1", 0], "resolution": int(res)}},
          "3": {"class_type": "SaveImage", "inputs": {"images": ["2", 0], "filename_prefix": "brush_teed"}}}
    pid = json.loads(C._post("/prompt", json.dumps({"prompt": wf}).encode()))["prompt_id"]
    t0 = time.time()
    while time.time() - t0 < wait_s:
        try:
            h_ = json.loads(C._get(f"/history/{pid}", timeout=30))
        except OSError:
            time.sleep(3)
            continue
        st = h_.get(pid, {}).get("status", {})
        if st.get("status_str") == "error":
            raise RuntimeError(f"ComfyUI: {json.dumps(st)[:300]}")
        if st.get("completed"):
            img_ = h_[pid]["outputs"]["3"]["images"][0]
            q = urllib.parse.urlencode({"filename": img_["filename"], "subfolder": img_.get("subfolder", ""),
                                        "type": img_.get("type", "output")})
            os.makedirs(CACHE, exist_ok=True)
            open(p, "wb").write(C._get(f"/view?{q}", timeout=120))
            print(f"  -> {os.path.relpath(p, HERE)} ({time.time() - t0:.0f} s)")
            return
        time.sleep(2)
    raise RuntimeError(f"no TEED map after {wait_s} s")


def _local_lines(img):
    """A line drawing without ComfyUI: difference of Gaussians on the lightness (dark lines), thresholded (XDoG-like)."""
    L = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
    s = max(1.0, max(img.shape[:2]) / 700.0)
    d = cv2.GaussianBlur(L, (0, 0), s) - cv2.GaussianBlur(L, (0, 0), 1.6 * s)
    v = np.clip(-d / max(float(np.percentile(-d, 99)), 1e-3), 0, 1)       # darker than round it = a line
    return (v > 0.35).astype(np.float32)


def line_map(image):
    """0..1 per pixel of the picture (its own size): 1 on a line. TEED from ComfyUI (cached), else local."""
    img = cv2.imread(image)
    if img is None:
        raise SystemExit(f"cannot read {image}")
    h, w = img.shape[:2]
    p = teed_path(image)
    if str(setting("COLOR_LINES_SOURCE", "teed")).lower() == "teed":
        if not (os.path.isfile(p) and os.path.getmtime(p) >= os.path.getmtime(image)):
            try:
                _comfy_teed(image, p)
            except Exception as e:
                print(f"color_lines: no TEED ({e}) - the local line drawing is used")
        if os.path.isfile(p):
            t = cv2.resize(cv2.imread(p, 0), (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
            return (t > float(setting("COLOR_LINES_TEED_MIN", 0.5))).astype(np.float32)
    return _local_lines(img)


def lines_on_grid(image, shape_hw, mm_px):
    """The line map on the work grid, every line COLOR_EDGE_LINE_MM wide (bool)."""
    H, W = shape_hw
    m = cv2.resize(line_map(image), (W, H), interpolation=cv2.INTER_AREA) > 0.2
    w_mm = float(setting("COLOR_EDGE_LINE_MM", 1.2))
    r = int(round(max(0.0, w_mm / 2.0 - mm_px / 2.0) / mm_px))
    if r >= 1:
        m = cv2.dilate(m.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))) > 0
    return m


def local_lights(target_L, mm_px):
    """Where the picture is clearly lighter than all round it (a light accent), on the work grid (bool). target_L: Lab L."""
    L = target_L.astype(np.float32)
    s = max(1.0, float(setting("COLOR_LIGHTS_AROUND_MM", 3.0)) / mm_px)
    return (L - cv2.GaussianBlur(L, (0, 0), s)) > float(setting("COLOR_LIGHTS_DL", 6.0))


if __name__ == "__main__":                                     # python color_lines.py <picture>: make and show the map
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    m = line_map(sys.argv[1])
    out = os.path.join(HERE, "ko", "color_lines_view.png")
    img = cv2.imread(sys.argv[1])
    v = img.copy()
    v[m > 0] = (0, 0, 0)
    cv2.imwrite(out, np.hstack([img, v]))
    print(f"{100 * m.mean():.1f}% of the picture on a line -> {os.path.relpath(out, HERE)}")
