"""
brushograph/monitor.py
======================
Live monitor window for the trainer:
  * CameraStream  - reads the camera continuously in a background thread, so the
                    picture never freezes while the machine moves. Also draws the window.
  * Monitor       - holds the current status (phase, progress, parameters, results,
                    what is being learned) and paints it on top of the live picture.

Keys in the window:   Q = skip the drying wait     ESC = stop after the current stroke
                      S = save a screenshot
"""
import collections, re, threading, time
import cv2
import numpy as np

from vision import open_camera
from settings import CALIBRATION_FILE
from vision import load_calibration, mm_to_pixel

WIN = "Brushograph Monitor"
OUT_W = 1280
PANEL_W = 360

PHASE_COLORS = {            # BGR
    "DIP":     (255, 170, 0),
    "PAINT":   (0, 200, 255),
    "PARK":    (200, 200, 200),
    "CAPTURE": (255, 255, 0),
    "MEASURE": (255, 0, 255),
    "DRY":     (0, 140, 255),
    "DONE":    (0, 220, 0),
    "IDLE":    (180, 180, 180),
}


def _ascii(s, n=60):
    return re.sub(r"[^\x20-\x7e]", "", s).strip()[:n]


def _txt(img, text, x, y, scale=0.5, color=(255, 255, 255), thick=1):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


class Monitor:
    def __init__(self):
        self.lock = threading.RLock()
        self.cam = None
        self.headless = False
        self.t0 = time.time()
        self.phase_key, self.phase, self.detail = "IDLE", "READY", ""
        self.phase_t0 = time.time()
        self.progress = None
        self.mode = self.title = ""
        self.learning = []
        self.attempt = self.total = 0
        self.params, self.result = {}, {}
        self.chart, self.chart_label = [], ""
        self.records = None
        self.dry = None
        self.roi = None
        self.trail, self.pos, self.paint_i, self.paint_n = [], None, 0, 38
        self.log = collections.deque(maxlen=6)
        self.hold_frame, self.hold_until, self.inset, self._pending = None, 0.0, None, None
        self.ladder_rows = None                      # live table of the ladder run (see set_ladder)
        self.skip = self.abort = False
        self.finished = False
        self.last_render = None
        self.warning = ""
        self.show_overlay = False        # O key: keep the orange path / box visible while drying

    # ---- state setters (called from the trainer thread) ---------------------
    def set_phase(self, key, name, detail=""):
        with self.lock:
            self.phase_key, self.phase, self.detail = key, name, detail
            self.phase_t0 = time.time()
            if key != "DRY":
                self.progress = None

    def set_mode(self, mode, title, learning, total, chart_label=""):
        with self.lock:
            self.mode, self.title, self.learning = mode, title, list(learning)
            self.total, self.chart_label = total, chart_label

    def new_attempt(self, n, total):
        with self.lock:
            self.attempt, self.total = n, total
            self.trail, self.pos, self.paint_i = [], None, 0
            self.result, self.dry, self.roi, self.inset, self.warning = {}, None, None, None, ""
            self.set_phase("DIP", "PREPARING", "starting stroke")

    def set_params(self, **kw):
        with self.lock:
            self.params = kw

    def set_result(self, **kw):
        with self.lock:
            self.result.update(kw)

    def set_ladder(self, rows, current_z=None):
        """rows = [{'z','machine_z','vis','w_med','strokes'}] - shown in the side panel instead of the explanation text."""
        with self.lock:
            self.ladder_rows, self.ladder_current = list(rows), current_z

    def set_chart(self, values):
        with self.lock:
            self.chart = list(values)

    def set_records(self, n):
        with self.lock:
            self.records = n

    def set_dry(self, stable, window, dry_frac, roi):
        """Called every drying check. dry_frac: 0 = stroke fully wet, 1 = back to blank paper.
        Returns True if the user asked to skip."""
        with self.lock:
            self.dry = (stable, window, dry_frac)
            self.roi = roi
            self.progress = min(max(dry_frac, 0.0), 1.0)
            self.detail = (f"stroke {dry_frac*100:.0f}% dry   "
                           f"(confirmed {stable}/{window} checks)" if dry_frac < 0.999 else "dry")
            s, self.skip = self.skip, False
            return s

    def min_wait(self, secs):
        end = time.time() + secs
        self.set_phase("DRY", "WAITING FOR PAPER TO DRY", "")
        while time.time() < end and not self.abort:
            with self.lock:
                self.detail = f"minimum soak time: {end - time.time():.0f}s left"
                self.progress = 1 - (end - time.time()) / max(secs, 1e-6)
                if self.skip:
                    self.skip = False
                    break
            time.sleep(0.1)

    def warn(self, text):
        with self.lock:
            self.warning = text
            self.add_log('WARNING: ' + text)

    def add_log(self, msg):
        m = _ascii(msg, 70)
        if m:
            with self.lock:
                self.log.append(m)

    # ---- hooks -----------------------------------------------------------------
    def feed(self, msg):
        """Called with every message the trainer prints; updates phase/log."""
        m = re.search(r"(?:TRAINING|LADDER) STROKE (\d+)/(\d+)", msg)
        if m:
            self.new_attempt(int(m[1]), int(m[2])); return
        m = re.search(r"(?:SHAPE-MATCH|TRACE) ATTEMPT (\d+)/(\d+)", msg)
        if m:
            self.new_attempt(int(m[1]), int(m[2])); return
        if "Dipping brush" in msg:
            self.set_phase("DIP", "DIPPING BRUSH IN WATER", "then moving to stroke start")
        elif "Parking" in msg:
            self.set_phase("PARK", "PARKING TOOLHEAD", "moving out of the camera's view")
        elif "Capturing" in msg:
            self.set_phase("CAPTURE", "PHOTOGRAPHING WET STROKE", "")
        elif "Trajectory Matched" in msg or "Stroke found" in msg:
            self.set_phase("MEASURE", "MEASURING STROKE", "skeleton + width along the stroke")
        elif "Monitoring evaporation" in msg:
            self.set_phase("DRY", "WAITING FOR PAPER TO DRY", "")
        m = re.search(r"Shape distance to target: ([\d.]+)", msg)
        if m:
            self.set_result(distance=float(m[1]))
        if "Spine Pt" not in msg:
            self.add_log(msg)

    def line_hook(self, line):
        """Called by sender for every G-code line, BEFORE it is sent. Never printed."""
        l = line.strip()
        if l.startswith("; --- HIGH-CLEARANCE PETRI DIP"):
            self.set_phase("DIP", "DIPPING BRUSH IN WATER", "dip, spiral, wipe")
            return
        if l.startswith(";") or not l:
            return
        tok = {}
        for p in l.split()[1:]:
            if p[0] in "XYZF":
                try: tok[p[0]] = float(p[1:])
                except ValueError: pass
        if not l.startswith(("G0", "G1")):
            return
        if "X" in tok and "Y" in tok:
            self.pos = (tok["X"], tok["Y"], tok.get("Z"))
            if l.startswith("G1") and "Z" in tok and self.phase_key in ("DIP", "PAINT"):
                self.paint_i += 1
                self.trail.append((tok["X"], tok["Y"]))
                if self.phase_key != "PAINT":
                    self.set_phase("PAINT", "PAINTING STROKE", "")
                with self.lock:
                    self.progress = min(1.0, self.paint_i / self.paint_n)
                    self.detail = f"brush Z (machine) = {tok['Z']:.2f} mm   feed {tok.get('F', 0):.0f}"

    # ---- cv2 drop-in replacements (so the trainer never opens its own windows) -----
    def imshow(self, name, img):
        if "Target vs" in name:
            self.inset = img
        else:
            self._pending = img

    def waitKey(self, ms=1):
        if ms >= 1500 and self._pending is not None:
            self.hold_frame, self.hold_until = self._pending, time.time() + ms / 1000
        self._pending = None
        end = time.time() + ms / 1000
        while time.time() < end:
            time.sleep(0.05)
        if self.skip:
            self.skip = False
            return ord("q")
        return -1

    def noop(self, *a, **k):
        pass

    def finish(self, msg="FINISHED"):
        self.set_phase("DONE", msg, "press any key in the window to close")
        self.finished = True
        t = time.time()
        while self.finished and not self.abort and not self.headless and time.time() - t < 3600:
            time.sleep(0.2)

    # ---- drawing -------------------------------------------------------------------
    def render(self, frame):
        h, w = frame.shape[:2]
        s = OUT_W / w
        H = int(h * s)
        with self.lock:
            holding = self.hold_frame is not None and time.time() < self.hold_until
            base = self.hold_frame if holding else frame
            d = cv2.resize(base, (OUT_W, H))
            key = self.phase_key
            col = PHASE_COLORS.get(key, (255, 255, 255))

            live = self.phase_key in ("DIP", "PAINT")
            if not holding and self.cam is not None and self.cam.H_inv is not None:
                if self.trail and (live or self.show_overlay):
                    pts = np.array([mm_to_pixel(self.cam.H_inv, x, y) for x, y in self.trail]) * s
                    cv2.polylines(d, [pts.astype(np.int32).reshape(-1, 1, 2)], False, (0, 200, 255), 3, cv2.LINE_AA)
                    cv2.circle(d, tuple(pts[-1].astype(int)), 10, (0, 0, 255), 2, cv2.LINE_AA)
                if self.roi is not None and self.show_overlay:
                    x1, y1, x2, y2 = [int(v * s) for v in self.roi]
                    frac = self.dry[2] if self.dry else 0
                    cv2.rectangle(d, (x1, y1), (x2, y2), (0, 255, 0) if frac > 0.9 else (0, 140, 255), 2)

            # top bar
            bar = d.copy()
            cv2.rectangle(bar, (0, 0), (OUT_W, 78), (0, 0, 0), -1)
            cv2.rectangle(bar, (OUT_W - PANEL_W, 78), (OUT_W, H), (0, 0, 0), -1)
            cv2.rectangle(bar, (0, H - 30), (OUT_W, H), (0, 0, 0), -1)
            d = cv2.addWeighted(bar, 0.82, d, 0.18, 0)
            cv2.circle(d, (28, 30), 12, col, -1)
            _txt(d, self.phase + ("   (analysis frame)" if holding else ""), 52, 40, 1.0, col, 2)
            _txt(d, self.detail, 52, 66, 0.6, (230, 230, 230))
            if self.progress is not None:
                cv2.rectangle(d, (0, 74), (OUT_W, 80), (60, 60, 60), -1)
                cv2.rectangle(d, (0, 74), (int(OUT_W * min(max(self.progress, 0), 1)), 80), col, -1)

            if self.warning:
                import textwrap
                for i, ln in enumerate(textwrap.wrap(_ascii(self.warning, 200), 58)[:3]):
                    _txt(d, ln, 14, 108 + 28 * i, 0.7, (0, 0, 255), 2)

            # right panel
            x, y = OUT_W - PANEL_W + 14, 104

            def head(t):
                nonlocal y
                _txt(d, t, x, y, 0.55, (0, 200, 255), 1); y += 22

            def line(t, c=(235, 235, 235), sc=0.46):
                nonlocal y
                _txt(d, _ascii(t, 42), x, y, sc, c); y += 20

            head("MODE")
            line(self.title or "-", (255, 255, 255))
            line(f"stroke {self.attempt}/{self.total}     elapsed {int(time.time()-self.t0)//60:02d}:{int(time.time()-self.t0)%60:02d}")
            y += 6; head("THIS STROKE")
            p = self.params
            if not p:
                line("-")
            for k, v in p.items():
                line(f"{k}: {v:.2f}" if isinstance(v, float) else f"{k}: {v}")
            y += 6; head("LAST RESULT")
            r = self.result
            if not r:
                line("(measured after painting)", (160, 160, 160))
            if "points" in r:
                line(f"samples kept {r['points']}   dab trimmed {r.get('dab', 0)}")
            if "marked_pct" in r:
                line(f"marked along {r['marked_pct']:.0f}% of the stroke", (0, 255, 0) if r["marked_pct"] >= 80 else (0, 165, 255))
            if "w_min" in r:
                line(f"width mm  {r['w_min']:.2f} / {r['w_med']:.2f} / {r['w_max']:.2f}  (min/med/max)")
            if "distance" in r:
                line(f"shape distance {r['distance']:.4f} (0 = perfect)")
            if "w_err" in r:
                line(f"width error {r['w_err']:.2f} mm   path error {r.get('path_err', 0):.2f} mm")
            if "calib_offset_mm" in r:
                bad = r["calib_offset_mm"] > 8 or r.get("calib_angle_deg", 0) > 20
                line(f"calib check: {r['calib_offset_mm']:.1f} mm, {r.get('calib_angle_deg', 0):.0f} deg off",
                     (0, 0, 255) if bad else (0, 255, 0))
            if "best" in r:
                line(f"best so far: {r['best']:.4f}", (0, 255, 0))
            if self.chart:
                cy0, ch, cw = y + 4, 60, PANEL_W - 28
                cv2.rectangle(d, (x, cy0), (x + cw, cy0 + ch), (70, 70, 70), 1)
                v = np.array(self.chart, float)
                lo, hi = v.min(), v.max()
                if len(v) > 1:
                    pts = [(x + int(i * cw / (len(v) - 1)), cy0 + ch - int((val - lo) / (hi - lo + 1e-9) * (ch - 6)) - 3)
                           for i, val in enumerate(v)]
                    cv2.polylines(d, [np.array(pts, np.int32).reshape(-1, 1, 2)], False, (0, 255, 255), 2, cv2.LINE_AA)
                _txt(d, self.chart_label, x + 4, cy0 + 14, 0.4, (200, 200, 200))
                y = cy0 + ch + 18
            if self.ladder_rows:
                y += 4; head("LADDER SO FAR   Z   machine   marked   width")
                for rw in self.ladder_rows:
                    cur = self.ladder_current is not None and abs(rw["z"] - self.ladder_current) < 1e-6
                    wtxt = f"{rw['w_med']:.2f} mm" if rw["vis"] > 0 else "no mark"
                    col = (0, 255, 255) if cur else ((0, 255, 0) if rw["vis"] >= 0.8 else (190, 190, 190))
                    _txt(d, f"{rw['z']:+6.2f}   {rw['machine_z']:5.2f}   {100 * rw['vis']:4.0f}%   {wtxt}" + ("   <" if cur else ""), x, y, 0.4, col)
                    y += 15
            else:
                y += 4; head("WHAT IT IS LEARNING")
                for t in self.learning:
                    line(t, (200, 255, 200), 0.42)
            if self.records is not None:
                line(f"samples saved this run: {self.records}", (0, 255, 0))
            y += 6; head("LOG")
            for t in list(self.log)[-(3 if self.ladder_rows else 5):]:
                line(t, (190, 190, 190), 0.38)

            if self.inset is not None and self.inset.size:
                ins = cv2.resize(self.inset, (300, 150))
                d[H - 30 - 150:H - 30, 0:300] = ins

            _txt(d, f"phase time {int(time.time()-self.phase_t0)}s     O = path overlay {'ON' if self.show_overlay else 'off'}   Q = skip drying   ESC = stop after stroke   S = screenshot",
                 10, H - 9, 0.5, (200, 200, 200))
            self.last_render = d
            return d


MON = Monitor()


class CameraStream:
    """Drop-in for CameraWatcher inside the trainer (has .H, .H_inv, .capture())."""

    def __init__(self, monitor=MON, headless=False):
        self.mon, self.headless = monitor, headless
        self.cap = None
        self.H = self.H_inv = None
        self.latest, self.seq = None, 0
        self._run = False
        self._thread = None

    def __enter__(self):
        self.cap = open_camera()
        cal = load_calibration(CALIBRATION_FILE)
        self.H, self.H_inv = cal["H"], cal["H_inv"]
        self.mon.cam, self.mon.headless = self, self.headless
        self._run = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        while self.latest is None:
            time.sleep(0.05)
        return self

    def __exit__(self, *a):
        if not self.headless and not self.mon.abort:
            self.mon.set_phase("DONE", "SESSION FINISHED", "closing in 4 seconds")
            time.sleep(4)
        self._run = False
        if self._thread:
            self._thread.join(timeout=3)
        if self.cap is not None:
            self.cap.release()

    def _loop(self):
        if not self.headless:
            cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WIN, OUT_W, 720)
        shown = False
        while self._run:
            ok, f = self.cap.read()
            if not ok:
                time.sleep(0.05); continue
            self.latest = f
            self.seq += 1
            if self.headless:
                self.mon.render(f); continue
            cv2.imshow(WIN, self.mon.render(f))
            shown = True
            k = cv2.waitKey(1) & 0xFF
            if self.mon.finished and k != 255:
                self.mon.finished = False
            elif k in (ord("q"), ord("Q")):
                self.mon.skip = True
            elif k == 27:
                self.mon.abort = True
                self.mon.finished = False
                self.mon.add_log("ESC: will stop after this stroke")
            elif k in (ord("o"), ord("O")):
                self.mon.show_overlay = not self.mon.show_overlay
            elif k in (ord("s"), ord("S")):
                cv2.imwrite("monitor_screenshot.png", self.mon.last_render)
            try:
                if shown and cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
                    self.mon.abort = True; self.mon.finished = False
            except cv2.error:
                pass
        if not self.headless:
            cv2.destroyAllWindows()

    def capture(self, flush=2):
        target = self.seq + flush + 1
        t0 = time.time()
        while self.seq < target and time.time() - t0 < 5:
            time.sleep(0.005)
        return self.latest.copy()
