"""
vision.py - everything about the camera:  1. open/capture   2. calibration (pixel <-> machine mm)   3. the paper   4. the paint pools.
(replaces camera_utils.py + homography.py + paper.py + the used part of palette.py)
"""
import json
import math
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from settings import (CAMERA_URL, CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_EXPOSURE, CAMERA_BACKEND, CAMERA_FOURCC, CAMERA_SET_EXPOSURE,
                      CALIBRATION_FILE, CAMERA_X_AXIS, CAMERA_Y_AXIS, PAPER_FILE)

# A native crash (the program ends without a message) is written to crash_log.txt in the folder you run from.
try:
    import faulthandler
    _CRASH_LOG = open(Path.cwd() / "crash_log.txt", "a", buffering=1)
    _CRASH_LOG.write(f"\n--- run started {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
    faulthandler.enable(file=_CRASH_LOG, all_threads=True)
except Exception:
    pass

# ======================================================================================================
# 1. THE CAMERA  - open it, capture a frame (several backends, MJPG, warm-up)
# ======================================================================================================

def _backend_list(requested: str) -> list:
    table = {"dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF,
             "v4l2": cv2.CAP_V4L2, "avfoundation": cv2.CAP_AVFOUNDATION}
    if requested != "auto":
        return [table.get(requested.lower(), cv2.CAP_ANY), cv2.CAP_ANY]
    if sys.platform == "win32":
        return [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    if sys.platform == "darwin":
        return [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]
    return [cv2.CAP_V4L2, cv2.CAP_ANY]


class StreamCamera:
    """A network camera (an MJPEG stream over HTTP, RTSP, ...) that behaves like cv2.VideoCapture for this program
    (read / get / set / isOpened / release).
    A thread reads the stream all the time and keeps only the newest picture: OpenCV queues network pictures, so a program
    that reads only now and then would otherwise get pictures from seconds ago. read() waits for a picture that arrived after
    the previous read(), like a real camera, so "read a few to flush, then take one" still gives a fresh picture.
    A dropped connection is opened again."""

    def __init__(self, url, timeout=5.0, verbose=True):
        self.url, self.timeout, self.verbose = url, timeout, verbose
        self._cond = threading.Condition()
        self._frame, self._seq, self._read_seq = None, 0, 0
        self._run = True
        self._cap = self._connect()
        if self._cap is None:
            raise RuntimeError(f"cannot open the camera stream {url}")
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        with self._cond:
            if not self._cond.wait_for(lambda: self._frame is not None, timeout=timeout):
                self.release()
                raise RuntimeError(f"the camera stream {url} opened but sent no picture in {timeout:.0f} s")

    def _connect(self):
        for be in (cv2.CAP_FFMPEG, cv2.CAP_ANY):
            cap = cv2.VideoCapture(self.url, be)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                return cap
            cap.release()
        return None

    def _loop(self):
        fails = 0
        while self._run:
            ok, f = self._cap.read() if self._cap is not None else (False, None)
            if ok and f is not None:
                fails = 0
                with self._cond:
                    self._frame, self._seq = f, self._seq + 1
                    self._cond.notify_all()
                continue
            fails += 1
            if fails < 20:
                time.sleep(0.05)
                continue
            if self.verbose:
                print(f"camera stream {self.url}: no pictures, connecting again")
            if self._cap is not None:
                self._cap.release()
            time.sleep(1.0)
            self._cap = self._connect() if self._run else None
            fails = 0

    def read(self):
        """The next picture (one that arrived after the previous read), or (False, None) after `timeout` seconds without one."""
        with self._cond:
            if not self._cond.wait_for(lambda: not self._run or self._seq > self._read_seq, timeout=self.timeout) or not self._run:
                return False, None
            self._read_seq = self._seq
            return True, self._frame.copy()

    def get(self, prop):
        f = self._frame
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return float(f.shape[1]) if f is not None else 0.0
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(f.shape[0]) if f is not None else 0.0
        cap = self._cap
        return cap.get(prop) if cap is not None else 0.0

    def set(self, prop, value):
        return False                                   # size, exposure, focus are set on the camera's own server, not from here

    def isOpened(self):
        return self._run

    def release(self):
        self._run = False
        with self._cond:
            self._cond.notify_all()
        t = getattr(self, "_thread", None)
        if t is not None and t is not threading.current_thread():
            t.join(timeout=3)
        if self._cap is not None:
            self._cap.release()
            self._cap = None


def _open_stream(url, verbose=True) -> StreamCamera:
    cap = StreamCamera(url, verbose=verbose)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if verbose:
        print(f"Camera stream {url} opened: {w}x{h}")
    try:
        cal_size = np.load(CALIBRATION_FILE)["image_size"] if Path(CALIBRATION_FILE).exists() else None
    except Exception:
        cal_size = None
    if cal_size is not None and tuple(int(v) for v in cal_size) != (w, h):
        print(f"  WARNING: the calibration ({CALIBRATION_FILE}) was made at {int(cal_size[0])}x{int(cal_size[1])}, the stream is {w}x{h}. "
              "Calibrate again (calibrate_grid.py, then calibrate_dots.py).")
    return cap


def open_camera(index=CAMERA_INDEX, width=CAMERA_WIDTH, height=CAMERA_HEIGHT,
                backend=CAMERA_BACKEND, verbose=True, tries=4, url=CAMERA_URL):
    """Open the camera, with retries. The camera is often still held for a moment after another part of the program let go of it,
    and OpenCV sometimes raises 'Unknown C++ exception' instead of returning a failure. Both are retried, with a longer pause each time.
    If all tries fail, the reason is raised as RuntimeError.
    With a `url` (CAMERA_URL in settings.py) the network stream is used instead of the USB camera `index`."""
    last = None
    for k in range(1, tries + 1):
        try:
            if url:
                return _open_stream(url, verbose)
            return _open_camera_once(index, width, height, backend, verbose)
        except Exception as e:                       # includes OpenCV's own exceptions
            last = e
            if verbose:
                print(f"camera: attempt {k} of {tries} failed ({e})" + ("; trying again" if k < tries else ""))
            time.sleep(1.5 * k)
    raise RuntimeError(f"the camera could not be opened after {tries} tries: {last}. Unplug the camera for a few seconds, "
                       "close other programs that use it, and run again.")


def _open_camera_once(index=CAMERA_INDEX, width=CAMERA_WIDTH, height=CAMERA_HEIGHT,
                      backend=CAMERA_BACKEND, verbose=True) -> cv2.VideoCapture:
    """Open a USB camera once. Raises RuntimeError with hints on failure."""
    last = None
    for be in _backend_list(backend):
        cap = cv2.VideoCapture(index, be)
        if not cap.isOpened():
            cap.release()
            continue
        if CAMERA_FOURCC:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*CAMERA_FOURCC))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        if CAMERA_SET_EXPOSURE:
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)   # manual (DSHOW)
            cap.set(cv2.CAP_PROP_EXPOSURE, CAMERA_EXPOSURE)
        # warm up: first frames are often black
        frame = None
        for _ in range(15):
            ok, f = cap.read()
            if ok and f is not None and f.mean() > 1:
                frame = f
            time.sleep(0.03)
        if frame is None:
            last = f"backend {be}: opened but no usable frames"
            cap.release()
            continue
        aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if verbose:
            print(f"Camera {index} opened (backend {be}): {aw}x{ah}")
        if (aw, ah) != (width, height):
            print(f"  WARNING: asked for {width}x{height}, got {aw}x{ah}. "
                  f"Calibration is resolution-specific — recalibrate or fix config.")
        return cap
    raise RuntimeError(
        f"Cannot open camera index {index} ({last or 'no backend worked'}).\n"
        "  - Close other apps using the camera (Zoom, browser, another preview)\n"
        "  - Try:  python preview.py --list   to find the right CAMERA_INDEX\n"
        "  - Try CAMERA_BACKEND = 'msmf' or 'dshow' in config.py")


def capture_frame(cap: cv2.VideoCapture, flush: int = 3) -> np.ndarray:
    for _ in range(flush):
        cap.read()
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Camera capture failed")
    return frame


def list_cameras(max_index: int = 6) -> list:
    """Probe indices 0..max_index-1, return [(index, w, h)] that deliver frames."""
    found = []
    for i in range(max_index):
        for be in _backend_list(CAMERA_BACKEND)[:1]:
            cap = cv2.VideoCapture(i, be)
            if cap.isOpened():
                ok, f = cap.read()
                if ok:
                    found.append((i, f.shape[1], f.shape[0]))
            cap.release()
    return found


# ======================================================================================================
# 2. CALIBRATION  - pixel <-> machine mm (the single source of truth)
# ======================================================================================================

def compute_homography(pixel_pts, machine_mm_pts):
    H, _ = cv2.findHomography(np.array(pixel_pts, np.float32),
                              np.array(machine_mm_pts, np.float32), 0)
    if H is None:
        raise RuntimeError("findHomography returned None — check point ordering")
    return H


def save_calibration(H, pixel_pts, mm_pts, path=CALIBRATION_FILE, image_size=None, lens=None, **more):
    """H maps (undistorted, if lens is given) camera pixels to machine mm. `more` = extra arrays kept for diagnostics."""
    extra = {"image_size": np.array(image_size, dtype=np.int32)} if image_size else {}
    if lens is not None:
        extra["lens"] = np.array(lens, float)
    extra.update(more)
    np.savez(path, H=np.asarray(H, float), H_inv=np.linalg.inv(np.asarray(H, float)),
             pixel_pts=np.array(pixel_pts, np.float32),
             mm_pts=np.array(mm_pts, np.float32), **extra)
    print(f"Calibration saved → {path}")


def load_calibration(path=CALIBRATION_FILE) -> dict:
    if not Path(path).exists():
        raise FileNotFoundError(f"No calibration file at {path} — run calibrate.py first")
    d = np.load(path)
    out = {k: d[k] for k in ('H', 'H_inv', 'pixel_pts', 'mm_pts')}
    out['lens'] = tuple(float(v) for v in d['lens']) if 'lens' in d.files else None
    if out['lens'] is not None:                                     # the matrices carry the lens, so every pixel_to_mm / mm_to_pixel uses it
        cal = {"lens": out['lens']}
        out['H'], out['H_inv'] = CalMat(out['H'], cal), CalMat(out['H_inv'], cal)
    out['image_size'] = tuple(int(v) for v in d['image_size']) if 'image_size' in d.files else None
    return out


class CalMat(np.ndarray):
    """A 3x3 homography that remembers the LENS model of its calibration (cal = {"lens": (cx, cy, R, a1, r2max)}).
    For cv2 / numpy it is an ordinary matrix; pixel_to_mm / mm_to_pixel / px_to_mm_array / mm_to_px_array use the lens as well.
    A calibration without a lens is a plain array and works exactly as before."""
    def __new__(cls, a, cal=None):
        o = np.asarray(a, float).view(cls)
        o.cal = cal
        return o

    def __array_finalize__(self, obj):
        self.cal = getattr(obj, "cal", None)


def _lens_of(M):
    cal = getattr(M, "cal", None)
    return None if cal is None else cal.get("lens")


def _lens_scale(r2, a1, r2max):
    return 1.0 + a1 * np.minimum(r2, r2max)           # one term; beyond r2max the correction stops growing (safe where nothing was measured)


def undistort_px(P, lens):
    """Camera pixels (as photographed) -> pixels of an ideal pinhole picture (straight lines are straight)."""
    cx, cy, R, a1, r2max = lens
    c = np.array([cx, cy], float)
    d = np.asarray(P, float) - c
    return c + d * _lens_scale((d ** 2).sum(-1, keepdims=True) / R ** 2, a1, r2max)


def distort_px(Pu, lens):
    """The inverse of undistort_px (fixed-point iterations)."""
    cx, cy, R, a1, r2max = lens
    c = np.array([cx, cy], float)
    d = np.asarray(Pu, float) - c
    P = c + d
    for _ in range(12):
        P = c + d / _lens_scale(((P - c) ** 2).sum(-1, keepdims=True) / R ** 2, a1, r2max)
    return P


def px_to_mm_array(H, pts):
    """Camera pixels (N, 2) -> machine mm (N, 2)."""
    P = np.asarray(pts, float).reshape(-1, 2)
    lens = _lens_of(H)
    if lens is None:
        return cv2.perspectiveTransform(P.reshape(1, -1, 2).astype(np.float32), np.asarray(H, float))[0]
    q = np.c_[undistort_px(P, lens), np.ones(len(P))] @ np.asarray(H, float).T
    return q[:, :2] / q[:, 2:3]


def mm_to_px_array(H_inv, pts):
    """Machine mm (N, 2) -> camera pixels (N, 2)."""
    M = np.asarray(pts, float).reshape(-1, 2)
    lens = _lens_of(H_inv)
    if lens is None:
        return cv2.perspectiveTransform(M.reshape(1, -1, 2).astype(np.float32), np.asarray(H_inv, float))[0]
    q = np.c_[M, np.ones(len(M))] @ np.asarray(H_inv, float).T
    return distort_px(q[:, :2] / q[:, 2:3], lens)


def pixel_to_mm(H, px, py):
    if _lens_of(H) is not None:
        r = px_to_mm_array(H, [(px, py)])[0]
        return float(r[0]), float(r[1])
    r = cv2.perspectiveTransform(np.array([[[px, py]]], np.float32), H)
    return float(r[0][0][0]), float(r[0][0][1])


def mm_to_pixel(H_inv, mx, my):
    if _lens_of(H_inv) is not None:
        r = mm_to_px_array(H_inv, [(mx, my)])[0]
        return float(r[0]), float(r[1])
    r = cv2.perspectiveTransform(np.array([[[mx, my]]], np.float32), H_inv)
    return float(r[0][0][0]), float(r[0][0][1])


def calibration_is_sane(pixel_pts) -> bool:
    """A valid 4-corner quad is convex (not a self-intersecting bowtie)."""
    p = np.array(pixel_pts, np.float64)
    s = []
    for i in range(4):
        a, b, c = p[i], p[(i+1) % 4], p[(i+2) % 4]
        s.append((b[0]-a[0])*(c[1]-b[1]) - (b[1]-a[1])*(c[0]-b[0]))
    return all(v > 0 for v in s) or all(v < 0 for v in s)


AXIS_VEC = {"right": (1, 0), "left": (-1, 0), "down": (0, 1), "up": (0, -1)}


def _reproj_err(H, src, dst):
    out = cv2.perspectiveTransform(np.array([src], np.float32), H)[0]
    return float(sum(math.hypot(o[0]-d[0], o[1]-d[1]) for o, d in zip(out, dst)))


def assign_by_axes(pixel_pts, x_dir=None, y_dir=None):
    """4 dots -> [BL, BR, TR, TL] using the known image direction of machine +X / +Y."""
    ex = np.array(AXIS_VEC[x_dir or CAMERA_X_AXIS], float)
    ey = np.array(AXIS_VEC[y_dir or CAMERA_Y_AXIS], float)
    P = np.array(pixel_pts, float)
    a, b = P @ ex, P @ ey
    slots = {}
    for p, ai, bi in zip(P, a > a.mean(), b > b.mean()):
        slots[(bool(ai), bool(bi))] = tuple(p)
    if len(slots) != 4:
        raise RuntimeError("Dots do not form 4 distinct corners for the configured axes")
    return [slots[(False, False)], slots[(True, False)], slots[(True, True)], slots[(False, True)]]


def assign_with_marker(pts, marker_offset_mm=None):
    """
    5 dots -> ([BL, BR, TR, TL], marker_px). The two closest dots are BL and the origin
    marker (12 mm to BL's +X side). The marker is the one closer to the other corners.
    Needs no assumption about camera rotation or mirroring.
    """
    P = np.array(pts, float)
    d = np.linalg.norm(P[:, None] - P[None], axis=2)
    np.fill_diagonal(d, 1e12)
    i, j = np.unravel_index(np.argmin(d), d.shape)
    rest = [k for k in range(len(P)) if k not in (i, j)]
    cen = P[rest].mean(axis=0)
    marker, bl = (i, j) if np.linalg.norm(P[i]-cen) < np.linalg.norm(P[j]-cen) else (j, i)
    u = P[marker] - P[bl]
    vecs = P[rest] - P[bl]
    cosang = (vecs @ u) / (np.linalg.norm(vecs, axis=1) * np.linalg.norm(u))
    br = rest[int(np.argmax(cosang))]
    rem = [k for k in rest if k != br]
    tl, tr = (rem[0], rem[1]) if np.linalg.norm(P[rem[0]]-P[bl]) < np.linalg.norm(P[rem[1]]-P[bl]) else (rem[1], rem[0])
    return [tuple(P[bl]), tuple(P[br]), tuple(P[tr]), tuple(P[tl])], tuple(P[marker])


def solve_calibration(blobs, corners_mm, use_marker=True, marker_offset_mm=12.0):
    """Returns (ordered_pixel_pts, H, reproj_err_mm, marker_err_mm_or_None)."""
    marker_err = None
    if use_marker and len(blobs) >= 5:
        ordered, marker_px = assign_with_marker(blobs[:5])
        H = compute_homography(ordered, corners_mm)
        mm = pixel_to_mm(H, *marker_px)
        want = (corners_mm[0][0] + marker_offset_mm, corners_mm[0][1])
        marker_err = math.hypot(mm[0]-want[0], mm[1]-want[1])
    else:
        ordered = assign_by_axes(blobs[:4])
        H = compute_homography(ordered, corners_mm)
    return ordered, H, _reproj_err(H, ordered, corners_mm), marker_err


def describe_orientation(H, H_inv, mm_pts):
    """How machine axes appear in the camera image (at canvas centre)."""
    cx, cy = float(np.mean(np.array(mm_pts)[:, 0])), float(np.mean(np.array(mm_pts)[:, 1]))
    o = np.array(mm_to_pixel(H_inv, cx, cy))
    vx = np.array(mm_to_pixel(H_inv, cx+10, cy)) - o
    vy = np.array(mm_to_pixel(H_inv, cx, cy+10)) - o
    name = lambda v: (("right" if v[0] > 0 else "left") if abs(v[0]) > abs(v[1]) else ("down" if v[1] > 0 else "up"))
    cross = vx[0]*vy[1] - vx[1]*vy[0]          # image y points down
    return {"x_dir": name(vx), "y_dir": name(vy),
            "px_per_mm": (float(np.linalg.norm(vx))/10, float(np.linalg.norm(vy))/10),
            "mirrored": bool(cross > 0),        # standard top-down view has cross < 0
            "corner_order": "ANTICLOCKWISE" if cross < 0 else "CLOCKWISE"}


def best_corner_assignment(pixel_pts, target_mm_pts):
    """Legacy 4-dot entry point: uses CAMERA_X_AXIS / CAMERA_Y_AXIS from settings.py."""
    ordered = assign_by_axes(pixel_pts)
    H = compute_homography(ordered, target_mm_pts)
    return [tuple(map(float, p)) for p in ordered], H, _reproj_err(H, ordered, target_mm_pts)


# ======================================================================================================
# 3. THE PAPER  - find the sheet in the picture, put an artwork on it
#    (the paper is the white, low-saturation area under the middle of the calibrated canvas; keep the toolhead out of the picture)
# ======================================================================================================

def _order_tl_tr_br_bl(pts):
    """Corners in camera-view order: top-left, top-right, bottom-right, bottom-left."""
    p = np.asarray(pts, float)
    c = p.mean(axis=0)
    ang = np.arctan2(p[:, 1] - c[1], p[:, 0] - c[0])          # image y points down: increasing angle = clockwise on screen
    p = p[np.argsort(ang)]                                     # starts at the corner left of straight-up ... order is clockwise
    s = p.sum(axis=1)
    start = int(np.argmin(s))                                  # top-left has the smallest x + y
    return np.roll(p, -start, axis=0)


def _fit_edges(comp, quad_px, H, H_inv, tol_mm=0.6, iters=400):
    """Sharper corners than the hull's: something light lying ALONG an edge (the dish tray, a rail by the head) bulges the hull
    and pulls its corner out. Each side is instead the straight line most of the sheet's outline points lie on (RANSAC), fitted
    in machine mm (the lens bends straight edges in the photo); the corners are where neighbouring lines cross.
    Falls back to the hull's corners when a side has too few points or the result moves a corner implausibly far."""
    cs, _ = cv2.findContours(comp.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        return quad_px
    P = px_to_mm_array(H, max(cs, key=len).reshape(-1, 2).astype(float))
    quad = px_to_mm_array(H, quad_px)
    rng = np.random.default_rng(0)
    lines = []
    for i in range(4):
        a, b = quad[i], quad[(i + 1) % 4]
        L = float(np.hypot(*(b - a)))
        d = (b - a) / max(L, 1e-9)
        n = np.array([-d[1], d[0]])
        t, off = (P - a) @ d, (P - a) @ n
        near = P[(t > 0.1 * L) & (t < 0.9 * L) & (np.abs(off) < 0.06 * L)]   # the middle of this side, corners left out
        if len(near) < 20:
            return quad_px
        best, best_n = None, -1
        for _ in range(iters):
            p, q = near[rng.choice(len(near), 2, replace=False)]
            e = q - p
            ne = float(np.hypot(*e))
            if ne < 0.2 * L:
                continue
            inl = np.abs((near - p) @ (np.array([-e[1], e[0]]) / ne)) < tol_mm
            if inl.sum() > best_n:
                best, best_n = inl, int(inl.sum())
        if best is None or best_n < 0.5 * len(near):
            return quad_px
        vx, vy, x0, y0 = cv2.fitLine(near[best].astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01).ravel()
        lines.append((np.array([x0, y0], float), np.array([vx, vy], float)))
    out = []
    for i in range(4):
        (p1, d1), (p2, d2) = lines[i - 1], lines[i]
        try:
            s = np.linalg.solve(np.array([d1, -d2]).T, p2 - p1)
        except np.linalg.LinAlgError:
            return quad_px
        out.append(p1 + s[0] * d1)
    out = mm_to_px_array(H_inv, np.array(out))
    if np.abs(out - quad_px).max() > 0.08 * float(np.hypot(*(quad_px[2] - quad_px[0]))):
        return quad_px
    return out


def detect_paper(frame, H, H_inv, seed_px=None, calib_mm=None):
    """Returns {'corners_px', 'corners_mm', 'size_mm', 'fill', 'mask'} or None.
    seed = middle of the calibrated canvas (the paper must be under it)."""
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    s, v = hsv[..., 1], hsv[..., 2]
    low_sat = s < 70
    if low_sat.sum() < 1000:
        return None
    t, _ = cv2.threshold(v[low_sat].reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr = float(np.clip(t, 110, 190))
    mask = ((s < 70) & (v > thr)).astype(np.uint8) * 255
    # thin overlay lines / grid / pen marks must not cut the paper apart
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    if seed_px is None and calib_mm is not None:
        c = np.mean(np.asarray(calib_mm, float), axis=0)
        seed_px = mm_to_pixel(H_inv, float(c[0]), float(c[1]))
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return None
    k = None
    if seed_px is not None:
        sx, sy = int(round(seed_px[0])), int(round(seed_px[1]))
        for r in (0, 25, 60, 120):                                  # the seed pixel may sit on a drawn overlay mark
            y0, y1, x0, x1 = max(0, sy - r), min(h, sy + r + 1), max(0, sx - r), min(w, sx + r + 1)
            vals = lab[y0:y1, x0:x1]; vals = vals[vals > 0]
            if len(vals):
                k = int(np.bincount(vals).argmax()); break
    biggest = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    if k is None or st[k, cv2.CC_STAT_AREA] < 0.5 * st[biggest, cv2.CC_STAT_AREA]:
        k = biggest                                                 # the paper is not under the canvas middle: a pale streak in the wood is, not the paper
    comp = (lab == k).astype(np.uint8)
    # something white touching the sheet (the toolhead parked at its corner, a white rail) joins it into one blob. The sheet is
    # a big rectangle and such a contact is narrow: an opening with a big SQUARE (it keeps a rectangle's corners sharp) cuts it,
    # and the largest piece left is the sheet
    ks = max(15, int(min(h, w) * 0.075) | 1)
    cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    solid = np.zeros_like(comp)
    cv2.drawContours(solid, cnts, -1, 1, cv2.FILLED)                  # paint on the sheet = holes in the white: filled first
    core = cv2.morphologyEx(solid, cv2.MORPH_OPEN, np.ones((ks, ks), np.uint8))
    nc, labc, stc, _ = cv2.connectedComponentsWithStats(core, connectivity=8)
    if nc > 1:
        comp = ((labc == 1 + int(np.argmax(stc[1:, cv2.CC_STAT_AREA]))) & (solid > 0)).astype(np.uint8)
    cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cnts, key=cv2.contourArea)
    hull = cv2.convexHull(c)
    per = cv2.arcLength(hull, True)
    quad = None
    for eps in (0.01, 0.02, 0.03, 0.05):
        ap = cv2.approxPolyDP(hull, eps * per, True)
        if len(ap) == 4:
            quad = ap.reshape(-1, 2).astype(float); break
    if quad is None:
        quad = cv2.boxPoints(cv2.minAreaRect(hull)).astype(float)
    quad = _order_tl_tr_br_bl(quad)
    quad = _fit_edges(comp, quad, H, H_inv)
    mm = np.array([pixel_to_mm(H, float(x), float(y)) for x, y in quad])
    sizes = [float(np.linalg.norm(mm[(i + 1) % 4] - mm[i])) for i in range(4)]
    fill = float(cv2.contourArea(hull) / max(cv2.contourArea(quad.astype(np.float32)), 1.0))
    return {"corners_px": quad.tolist(), "corners_mm": mm.tolist(),
            "size_mm": [0.5 * (sizes[0] + sizes[2]), 0.5 * (sizes[1] + sizes[3])],      # (top/bottom edge length, left/right edge length)
            "fill": fill, "mask": comp}


def clip_to_reach(det, margin_mm=2.0):
    """A sheet bigger than the machine can reach (2026-10-28: a 296 mm sheet, X travel ends at ~289): its corners in machine mm
    are pulled in to MACHINE_X/Y_MIN/MAX, so the probes and the artwork stay where the brush can go - each side only as far as
    needed (REACH_CENTRED = True: by the same amount on opposite sides). The picture corners stay."""
    import settings as S
    try:                                                          # the paint lands at head + tip offset: the reach of the PAINT
        import probe as PB
        off = PB.applied_offset_from_last_result()
    except Exception:
        off = [0.0, 0.0]
    lo = np.array([float(getattr(S, "MACHINE_X_MIN", -1e9)) + off[0] + margin_mm, float(getattr(S, "MACHINE_Y_MIN", -1e9)) + off[1] + margin_mm])
    hi = np.array([float(getattr(S, "MACHINE_X_MAX", 1e9)) + off[0] - margin_mm, float(getattr(S, "MACHINE_Y_MAX", 1e9)) + off[1] - margin_mm])
    mm = np.array(det["corners_mm"], float)                         # tl, tr, br, bl
    if ((mm >= lo - 1e-9) & (mm <= hi + 1e-9)).all():
        return det
    # the sheet is shrunk along its own edges (it may lie a little rotated) until every corner is in reach. Each side is cut
    # only as far as the reach needs (2026-11-08: a 180 mm sheet at X -6..174 with reach 0..151 was cut 22 mm on BOTH sides,
    # so the left probe sat at X 20 although the brush paints from X 2). REACH_CENTRED = True: the same cut on opposite
    # sides, so probes and artwork stay centred on the real sheet (but up to twice the width is lost).
    centred = bool(getattr(S, "REACH_CENTRED", True))
    u = (mm[1] - mm[0]) + (mm[2] - mm[3])
    v = (mm[3] - mm[0]) + (mm[2] - mm[1])
    u, v = u / np.linalg.norm(u), v / np.linalg.norm(v)
    c = mm.mean(axis=0)
    su, sv = np.sign((mm - c) @ u), np.sign((mm - c) @ v)            # which side of the middle each corner is on
    side = ((su > 0).astype(int), (sv > 0).astype(int))             # per paper axis: 0 = the minus end, 1 = the plus end
    cut = np.zeros((2, 2))                                           # cut[axis][end]: how much comes off that end, along u / v
    for _ in range(50):
        q = mm - np.outer(su * cut[0][side[0]], u) - np.outer(sv * cut[1][side[1]], v)
        over = np.maximum(q - hi, 0.0) + np.maximum(lo - q, 0.0)     # how far each corner is out of reach, in machine X / Y
        if over.max() < 0.01:
            break
        for k in range(2):                                           # machine axis k: cut along the paper axis closest to it
            j = 0 if abs(u[k]) >= abs(v[k]) else 1
            for e in range(2):
                on_end = side[j] == e
                worst = over[on_end, k].max() if on_end.any() else 0.0
                if worst > 0:
                    cut[j][e] += worst / max(abs((u, v)[j][k]), 0.5)
        if centred:
            cut[:] = cut.max(axis=1, keepdims=True)
    det = dict(det)
    det["corners_mm"] = q.tolist()
    print(f"the sheet is bigger than the machine can reach (MACHINE_X/Y_MIN/MAX in settings.py): {cut[0][0]:.0f} / {cut[0][1]:.0f} mm "
          f"left free at left / right, {cut[1][0]:.0f} / {cut[1][1]:.0f} mm at top / bottom"
          + (" (REACH_CENTRED: the margins stay even)" if centred else ""))
    return det


def paper_frame(det, H):
    """The paper's own coordinate frame in machine mm.
    origin = paper centre, u = camera 'right' along the paper edges, v = camera 'up' along the paper edges
    (both as machine-mm unit vectors), plus the largest rectangle that fits inside the paper."""
    cpx = np.array(det["corners_px"], float)                       # tl, tr, br, bl
    top, bot = cpx[1] - cpx[0], cpx[2] - cpx[3]
    e = top + bot
    th = math.atan2(e[1], e[0])                                      # direction of the top/bottom edge in the camera picture
    center_px = cpx.mean(axis=0)
    c_mm = np.array(pixel_to_mm(H, float(center_px[0]), float(center_px[1])))
    d = 10.0
    ex = (np.array(pixel_to_mm(H, center_px[0] + d, center_px[1])) - c_mm) / d       # mm per pixel step to the right
    ey = (np.array(pixel_to_mm(H, center_px[0], center_px[1] + d)) - c_mm) / d       # mm per pixel step down
    right_c, up_c = (math.cos(th), math.sin(th)), (math.sin(th), -math.cos(th))
    u = ex * right_c[0] + ey * right_c[1]
    v = ex * up_c[0] + ey * up_c[1]
    # the paper's own direction, taken from its corners in MACHINE mm (the camera-pixel angle pushed through a local
    # approximation of the mapping is a few degrees off where the camera is not square to the table)
    mc = np.array(det["corners_mm"], float)
    e_mm = (mc[1] - mc[0]) + (mc[2] - mc[3])
    if np.linalg.norm(e_mm) > 1e-6:
        u = e_mm / np.linalg.norm(e_mm)
    v = np.array([-u[1], u[0]])
    if v @ (ex * up_c[0] + ey * up_c[1]) < 0:                        # v points the same way as camera-up
        v = -v
    P = np.array(det["corners_mm"], float) - c_mm
    a, b = P @ u, P @ v
    a_left, a_right = max(a[0], a[3]), min(a[1], a[2])
    b_bot, b_top = max(b[2], b[3]), min(b[0], b[1])
    return {"origin_mm": c_mm.tolist(), "u": u.tolist(), "v": v.tolist(), "angle_deg": math.degrees(th),
            "a_left": float(a_left), "a_right": float(a_right), "b_bot": float(b_bot), "b_top": float(b_top),
            "paper_w": float(a_right - a_left), "paper_h": float(b_top - b_bot)}


def fit_artwork(pf, art_w_px, art_h_px, margin_mm=10.0, rotate="auto", width_mm=None, bottom_mm=None):
    """Largest placement of the artwork inside the paper (margin all round), centred. rotate: 'auto' | 0 | 90.
    width_mm: force the artwork width (after rotation) instead of 'as large as fits'.
    bottom_mm: a wider margin under the picture's own bottom (room for the signature); the artwork is centred in what is
    left. Turned by 90 degrees (counter-clockwise), the picture's bottom is the paper's right side."""
    bottom = max(margin_mm, bottom_mm or 0.0)

    def room(rot):                                                   # (avail_w, avail_h, right margin, bottom margin)
        right, bot = (bottom, margin_mm) if rot == 90 else (margin_mm, bottom)
        return pf["paper_w"] - margin_mm - right, pf["paper_h"] - margin_mm - bot, right, bot
    if min(room(0)[:2] + room(90)[:2]) <= 5:
        raise ValueError(f"paper too small for a {margin_mm:.0f} mm margin: {pf['paper_w']:.0f} x {pf['paper_h']:.0f} mm")
    opts = {}
    for rot, (w, h) in ((0, (art_w_px, art_h_px)), (90, (art_h_px, art_w_px))):
        aw, ah = room(rot)[:2]
        opts[rot] = (min(aw / w, ah / h), w, h)
    rot = (0 if opts[0][0] >= opts[90][0] else 90) if rotate == "auto" else int(rotate)
    scale, w, h = opts[rot]
    avail_w, avail_h, _, bottom = room(rot)
    if width_mm:
        scale = width_mm / w
    art_w, art_h = w * scale, h * scale
    fits = art_w <= avail_w + 1e-6 and art_h <= avail_h + 1e-6
    a0 = pf["a_left"] + margin_mm + (avail_w - art_w) / 2.0
    b0 = pf["b_bot"] + bottom + (avail_h - art_h) / 2.0
    c, u, v = np.array(pf["origin_mm"]), np.array(pf["u"]), np.array(pf["v"])
    return {"origin_mm": (c + a0 * u + b0 * v).tolist(), "u": pf["u"], "v": pf["v"], "width_mm": float(art_w),
            "height_mm": float(art_h), "rotate": rot, "scale_mm_per_px": float(scale), "margin_mm": margin_mm,
            "fits": bool(fits), "scale_if_other_rotation": float(opts[90 if rot == 0 else 0][0])}


def placement_corners_mm(pl):
    o, u, v = np.array(pl["origin_mm"]), np.array(pl["u"]), np.array(pl["v"])
    w, h = pl["width_mm"], pl["height_mm"]
    return [o, o + w * u, o + w * u + h * v, o + h * v]


def save_paper(det, pf, path=PAPER_FILE):
    d = {k: v for k, v in det.items() if k != "mask"}
    json.dump({"detected": d, "frame": pf}, open(path, "w"), indent=2)


def load_paper(path=PAPER_FILE):
    d = json.load(open(path))
    return d["detected"], d["frame"]


def draw_paper(frame, det, H_inv=None, pf=None, pl=None):
    out = frame.copy()
    q = np.array(det["corners_px"], np.int32)
    cv2.polylines(out, [q], True, (0, 255, 0), 3, cv2.LINE_AA)
    for i, nm in enumerate(("TL", "TR", "BR", "BL")):
        cv2.circle(out, tuple(q[i]), 8, (0, 255, 0), -1)
        cv2.putText(out, nm, (q[i][0] + 10, q[i][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    if pl is not None and H_inv is not None:
        pts = np.array([mm_to_pixel(H_inv, float(x), float(y)) for x, y in placement_corners_mm(pl)], np.int32)
        cv2.polylines(out, [pts], True, (0, 220, 255), 3, cv2.LINE_AA)
        cv2.putText(out, "artwork", (pts[3][0] + 6, pts[3][1] + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 255), 2, cv2.LINE_AA)
    t = f"paper {det['size_mm'][0]:.0f} x {det['size_mm'][1]:.0f} mm"
    cv2.putText(out, t, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(out, t, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
    return out


# ======================================================================================================
# 4. PAINT POOLS  - find the dishes with colour in the picture (saturated blobs, one colour family at a time)
# ======================================================================================================
def _mm_per_px(H, px, py, d=10.0):
    x0, y0 = pixel_to_mm(H, px, py)
    xa, ya = pixel_to_mm(H, px + d, py)
    xb, yb = pixel_to_mm(H, px, py + d)
    return 0.5 * (math.hypot(xa - x0, ya - y0) + math.hypot(xb - x0, yb - y0)) / d


def circular_mean_hue(h):
    """Mean of OpenCV hues (0..179, wraps around): red pools mix 172..179 and 0..8."""
    a = np.asarray(h, float) * 2.0 * math.pi / 180.0
    return (math.degrees(math.atan2(np.sin(a).mean(), np.cos(a).mean())) % 360.0) / 2.0


def hue_distance(a, b):
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


HUE_FAMILIES = [(0, 8), (8, 22), (22, 38), (38, 85), (85, 100), (100, 132), (132, 165)]   # red, orange, yellow, green, cyan, blue, violet/magenta


RED_WRAP = 165                                                                              # hue >= this is red again


def _fill_holes(mask):
    """Fills holes inside a boolean blob (outer contour only)."""
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros(mask.shape, np.uint8)
    cv2.drawContours(out, cnts, -1, 1, -1)
    return out.astype(bool)


def _roundness(filled):
    """area / area of the circle with the same extent: 1 = a disc, small = a streak or an irregular spill."""
    ys, xs = np.nonzero(filled)
    r = max(xs.max() - xs.min(), ys.max() - ys.min()) / 2.0 + 1.0
    return float(filled.sum() / (math.pi * r * r))


def _inner_radius_px(filled):
    """The radius of the biggest circle that fits inside the pool: how far the brush may move from the centre while dipping.
    For a round dish about its radius, for a square container half its side (the radius of a circle of the same area would
    be 0.56 of the side and reach the wall)."""
    return float(cv2.distanceTransform(filled.astype(np.uint8), cv2.DIST_L2, 5).max())


def find_pools(frame, H, exclude_poly_px=None, exclude_mm=None, min_area_mm2=100.0, open_mm=3.4):
    """Pools of paint (round dishes or square containers) anywhere in the picture OUTSIDE the paper (the polygon, widened by
    exclude_mm, default POOL_PAPER_GAP_MM). Saturated colour, plus very dark pools (marked "dark", for black).
    Wood (orange, medium saturation) and small specks / splatter are ignored. White and clear paints cannot be found this way.
    Returns [{'center_px','center_mm','radius_mm','area_mm2','bgr','hue','sat'}] biggest first; radius_mm = the biggest circle inside."""
    import settings as _S0
    if exclude_mm is None:
        exclude_mm = float(getattr(_S0, "POOL_PAPER_GAP_MM", 15.0))
    h, w = frame.shape[:2]
    mmpp = _mm_per_px(H, w / 2.0, h / 2.0)
    ppm = 1.0 / mmpp
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hh, ss, vv = hsv[..., 0].astype(int), hsv[..., 1].astype(int), hsv[..., 2].astype(int)
    paint = (ss >= 90) & (vv >= 50) & ~(((hh >= 9) & (hh <= 35)) & (ss < 170))
    # pale paints (lilac, light green, pink ...: much white in them) are only weakly saturated, but LIGHT; the wood is orange-brown
    # and darker, so a light pixel with some colour that is not orange-yellow is paint too. Kept apart from the strong colours:
    # only pale pools get their glare bridged (a strong dish must not grow into the splatter beside it)
    import settings as _S
    pastel_sat = int(getattr(_S, "POOL_MIN_SAT_PASTEL", 35))
    pastel = (ss >= pastel_sat) & (vv >= 140) & ~((hh >= 5) & (hh <= 40)) & ~paint
    mask = paint.astype(np.uint8) * 255
    pmask = pastel.astype(np.uint8) * 255
    if exclude_poly_px is not None:
        ex = np.zeros((h, w), np.uint8)
        cv2.fillPoly(ex, [np.asarray(exclude_poly_px, np.int32)], 255)
        k = max(3, int(round(2 * exclude_mm * ppm)) | 1)
        exd = cv2.dilate(ex, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) > 0
        mask[exd] = 0
        pmask[exd] = 0
    ko = max(3, int(round(open_mm * ppm)) | 1)
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ko, ko))
    kc = max(3, int(round(8.0 * ppm)) | 1)
    kclose = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kc, kc))
    out = []
    # one colour family at a time: two dishes with different paints that TOUCH (or are bridged by a spill) must not become one blob
    # the colour families are the hues REALLY present (peaks of the hue histogram of the paint pixels), every pixel joins the nearest:
    # a dish whose hue sits on a fixed border (a green at 84/85) would otherwise be split pixel by pixel into two speckled halves
    anyp = (mask > 0) | (pmask > 0)
    hist = np.bincount(hh[anyp].ravel(), minlength=180).astype(float)[:180]
    kg = np.exp(-0.5 * (np.arange(-6, 7) / 2.5) ** 2)
    sm = np.convolve(np.r_[hist[-6:], hist, hist[:6]], kg / kg.sum(), "same")[6:-6]
    min_px = 0.5 * min_area_mm2 * ppm * ppm                            # a peak needs at least half a small dish of pixels
    peaks = [i for i in range(180) if sm[i] >= sm[i - 1] and sm[i] > sm[(i + 1) % 180]
             and hist[[(i + d) % 180 for d in range(-6, 7)]].sum() >= min_px]
    peaks.sort(key=lambda p: -sm[p])
    centres = []
    for p in peaks:                                                     # peaks closer than 8 hue steps are one colour
        if all(min(abs(p - q), 180 - abs(p - q)) >= 8 for q in centres):
            centres.append(p)
    if not centres:
        return []
    dist = np.stack([np.minimum(np.abs(hh - c), 180 - np.abs(hh - c)) for c in centres])
    fam = np.argmin(dist, axis=0)
    fam[np.min(dist, axis=0) > 20] = -1                                 # far from every peak: not one of these paints
    for f in range(len(centres)):
        strong = cv2.morphologyEx(((mask > 0) & (fam == f)).astype(np.uint8) * 255, cv2.MORPH_OPEN, kern)
        pale = ((pmask > 0) & (fam == f)).astype(np.uint8) * 255
        pale = cv2.morphologyEx(cv2.morphologyEx(pale, cv2.MORPH_CLOSE, kclose), cv2.MORPH_OPEN, kern)   # bridge the glare of a glossy dish
        # one colour, one dish: a pale dish often has a more saturated rim. Strong pixels touching a pale area join it (the pale
        # area was bridged, the strong ones were not, so a strong dish does not grow into splatter beside it)
        if pale.any():
            n_p, lab_p = cv2.connectedComponents(pale, connectivity=8)
            touch = np.unique(lab_p[cv2.dilate(strong, kern) > 0])
            n_s, lab_s = cv2.connectedComponents(strong, connectivity=8)
            for t in touch[touch > 0]:
                near = cv2.dilate((lab_p == t).astype(np.uint8), kern) > 0
                for s_ in np.unique(lab_s[near]):
                    if s_ > 0 and (lab_s == s_).sum() < (lab_p == t).sum():   # only a smaller strong part (a rim) joins; a big strong dish stays alone
                        pale[lab_s == s_] = 255
                        strong[lab_s == s_] = 0
            pale = cv2.morphologyEx(pale, cv2.MORPH_CLOSE, kclose)
            pale[cv2.dilate(strong, kclose) > 0] = 0                      # pale pixels round a big strong dish (its glare) are not a dish
        for mf in (strong, pale):
            n, lab, st, cen = cv2.connectedComponentsWithStats(mf, connectivity=8)
            for k in range(1, n):
                m = lab == k
                filled = _fill_holes(m)                               # glare / reflections inside a dish are not "no paint"
                area_mm2 = float(filled.sum()) * mmpp * mmpp
                if area_mm2 < min_area_mm2:
                    continue
                ys, xs = np.nonzero(filled)
                cx, cy = float(xs.mean()), float(ys.mean())
                mx, my = pixel_to_mm(H, cx, cy)
                out.append({"center_px": (cx, cy), "center_mm": (mx, my), "radius_mm": _inner_radius_px(filled) * mmpp,
                            "area_mm2": area_mm2, "bgr": [int(v) for v in np.median(frame[m], axis=0)],
                            "hue": circular_mean_hue(hh[m]), "sat": float(np.median(ss[m])),
                            "roundness": _roundness(filled)})
    # black (or very dark) paint: no colour to find, but much darker than anything else round on the table. Its glare (bright
    # reflections) is bridged; the pool is marked "dark" so only a black layer can use it
    import settings as _S2
    # (a nearly black pixel's saturation is noise - (5, 2, 3) is 150+ - so below V 30 it is not asked)
    dark = ((vv < int(getattr(_S2, "POOL_DARK_V", 70))) & ((ss < 200) | (vv < 30))).astype(np.uint8) * 255
    if exclude_poly_px is not None:
        dark[exd] = 0
    # no bridging here (it would join the dish to dark rails and shadows next to it): a wide opening cuts thin dark connections
    # apart, and the glare INSIDE a dish is a hole that _fill_holes closes
    kd = max(3, int(round(float(getattr(_S2, "POOL_DARK_OPEN_MM", 8.0)) * ppm)) | 1)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kd, kd)))
    n, lab, st, cen = cv2.connectedComponentsWithStats(dark, connectivity=8)
    for k in range(1, n):
        m = lab == k
        filled = _fill_holes(m)
        area_mm2 = float(filled.sum()) * mmpp * mmpp
        if area_mm2 < min_area_mm2:
            continue
        ys, xs = np.nonzero(filled)
        cx, cy = float(xs.mean()), float(ys.mean())
        mx, my = pixel_to_mm(H, cx, cy)
        out.append({"center_px": (cx, cy), "center_mm": (mx, my), "radius_mm": _inner_radius_px(filled) * mmpp,
                    "area_mm2": area_mm2, "bgr": [int(v) for v in np.median(frame[m], axis=0)],
                    "hue": circular_mean_hue(hh[m]), "sat": float(np.median(ss[m])), "dark": True,
                    "roundness": _roundness(filled)})
    return _snap_to_cups(sorted(out, key=lambda d: -d["area_mm2"]))


def _snap_to_cups(pools):
    """With a fixed holder (CUPS in settings.py) the camera only tells WHICH paint is in which cup; where the cups are is known
    better from the holder than from a picture (the containers sit at the edge of the view, where a rough calibration is off by
    10 mm and more). Each pool within half a cup spacing (X) and a cup length (Y) of a colour cup is moved onto that cup, with
    the cup's free half-width as its radius; the biggest pool per cup is kept. Pools elsewhere - paint tubes, the holder's own
    pink plastic, the water cup - are dropped. Without CUPS the pools are returned unchanged."""
    import settings as _S
    cups = dict(getattr(_S, "CUPS", None) or {})
    if not cups:
        return pools
    xs = sorted(c[0] for c in cups.values())
    half = 0.5 * min(np.diff(xs)) if len(xs) > 1 else 16.0
    length = float(getattr(_S, "CUP_LENGTH_MM", 23.5))
    best = {}
    for p in pools:
        x, y = p["center_mm"]
        name, (cx, cy) = min(cups.items(), key=lambda kv: math.hypot(x - kv[1][0], y - kv[1][1]))
        if name == "water" or abs(x - cx) > half or abs(y - cy) > length:
            continue
        if bool(p.get("dark")) != (name == "black"):            # the black cup holds the dark pool, a colour cup never does
            continue                                            # (2026-11-07: the holder's pink rim beat the black paint by area)
        score = p["area_mm2"] * min(float(p.get("roundness", 1.0)), 1.0)    # a filled cup beats a ragged scrap of pink holder
        if name not in best or score > best[name]["_score"]:
            p = dict(p, _score=score)
            q = dict(p, camera_mm=p["center_mm"], center_mm=(float(cx), float(cy)), cup=name,
                     radius_mm=0.5 * float(getattr(_S, "CUP_WIDTH_MM", 16.2)))
            best[name] = q
    return sorted(best.values(), key=lambda d: -d["area_mm2"])
