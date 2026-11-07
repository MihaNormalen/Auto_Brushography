"""
check.py - the touch check: while the job paints, the camera looks at the paper at every dip and the program learns where the brush does not touch.

How it works
  * The head parks where the camera sees the whole sheet (the snapshot position), a picture is taken and compared with the one before:
    what became darker is what the last strokes painted.
  * Every planned stroke is looked for in that picture: painted, partly painted or MISSING. Marks painted at the threshold (the thinnest
    lines, 3/4 of this artwork) are free touch probes: a missing one says the plate is higher there than the plane of the probes says.
  * So a depth correction MAP is learned (regional: how much more or less than the rest of the sheet; never more than CHECK_MAX_CORR_MM) plus one slow offset for the whole sheet (never more than CHECK_GLOBAL_MAX_MM) and added to the Z of every stroke that
    follows. It is a bed-levelling mesh that the job builds by itself, on top of the plane the six probes measured.
  * At the end of every layer the program looks at the whole layer, finds the strokes that are still missing and paints them again, deeper.
Everything is logged (one line per check on the screen, check_log.json with the final map). Any failure switches the check off and the job goes on.
"""
import json
import math
import re
import threading
import time

import cv2
import numpy as np
from scipy.ndimage import map_coordinates

import settings as S
import vision as V

Z_MIN = 0.3                                                         # the machine is never sent below this Z


def _s(name, default):
    return getattr(S, name, default)


CHECK_EVERY_DIPS = int(_s("CHECK_EVERY_DIPS", 1))                   # a check every this many dips
CHECK_SETTLE_S = float(_s("CHECK_SETTLE_S", 1.0))                   # wait after parking, so the picture is sharp and the exposure settled
CHECK_MAX_CORR_MM = float(_s("CHECK_MAX_CORR_MM", 0.3))             # the correction never goes beyond this (deeper or shallower)
CHECK_REPAINT = bool(_s("CHECK_REPAINT", True))
CHECK_GAIN_MM = float(_s("CHECK_GAIN_MM", 0.12))                    # a region that misses much more than the rest of the sheet goes at most this much deeper per check ...
CHECK_TARGET_MISS = float(_s("CHECK_TARGET_MISS", 0.30))            # the aim for the WHOLE sheet: about this share of threshold marks missing (short strokes always lose some to the bristle lag; the rest is painted again)
CHECK_GLOBAL_MAX_MM = float(_s("CHECK_GLOBAL_MAX_MM", 0.15))        # the same extra depth everywhere (when most marks are missing everywhere) never goes beyond this
CHECK_LAG_MM = float(_s("CHECK_LAG_MM", 1.2))                       # the first mm of a stroke is not judged: the bristles need that to start marking
CHECK_WIDE = float(_s("CHECK_WIDE", 1.35))                          # a mark wider than this times the expected width says the brush is deeper than planned
CHECK_TRUST = float(_s("CHECK_TRUST", 6.0))                         # a node needs about this many marks near it before it takes the full step (few marks: a small step)
CHECK_LOG = _s("CHECK_LOG", "check_log.json")


# ------------------------------------------------------------------------------------------------ the camera
class FrameGrabber:
    """Keeps the camera open and the latest picture ready (the live window and the check both use it, because only one program can hold the camera)."""
    def __init__(self):
        self.latest = None
        self.stop_flag = False
        self.thread = None
        self.error = None

    def start(self):
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        cap = None
        try:
            cap = V.open_camera(verbose=False)
            while not self.stop_flag:
                ok, frame = cap.read()
                if ok:
                    self.latest = (time.time(), frame)
                else:
                    time.sleep(0.03)
        except Exception as e:
            self.error = str(e)
        finally:
            if cap is not None:
                cap.release()

    def read(self):
        """Like cap.read(): the newest picture (the same one again if no new one has arrived)."""
        if self.latest is None:
            time.sleep(0.03)
            return False, None
        return True, self.latest[1]

    def frame_after(self, t, timeout=8.0):
        """A picture taken after time t (None if the camera does not deliver one)."""
        end = time.time() + timeout
        while time.time() < end:
            if self.latest is not None and self.latest[0] >= t:
                return self.latest[1].copy()
            time.sleep(0.03)
        return None

    def stop(self):
        self.stop_flag = True
        if self.thread:
            self.thread.join(timeout=3)


# ------------------------------------------------------------------------------------------------ the learned map
class CorrMap:
    """Extra depth (mm, positive = deeper) on a coarse grid over the paper, bilinear in between."""
    def __init__(self, x0, x1, y0, y1, cell=50.0, limit=CHECK_MAX_CORR_MM):
        self.xs = np.linspace(x0, x1, max(2, int(math.ceil((x1 - x0) / cell)) + 1))
        self.ys = np.linspace(y0, y1, max(2, int(math.ceil((y1 - y0) / cell)) + 1))
        self.v = np.zeros((len(self.xs), len(self.ys)))
        self.limit = limit
        self.cell = cell

    def at(self, x, y):
        xs, ys = self.xs, self.ys
        fx = np.clip((x - xs[0]) / (xs[-1] - xs[0]) * (len(xs) - 1), 0, len(xs) - 1 - 1e-9)
        fy = np.clip((y - ys[0]) / (ys[-1] - ys[0]) * (len(ys) - 1), 0, len(ys) - 1 - 1e-9)
        i, j = int(fx), int(fy)
        a, b = fx - i, fy - j
        return float((1 - a) * (1 - b) * self.v[i, j] + a * (1 - b) * self.v[i + 1, j] + (1 - a) * b * self.v[i, j + 1] + a * b * self.v[i + 1, j + 1])

    def relax(self, obs, gain, min_weight=1.5):
        """obs = [(x, y, 1 if the mark is much wider than expected else 0)]. Where most marks are too wide the brush is deeper than planned: the map goes shallower."""
        if not obs:
            return
        gx, gy = np.meshgrid(self.xs, self.ys, indexing="ij")
        num, den = np.zeros_like(self.v), np.zeros_like(self.v)
        for x, y, sc in obs:
            w = np.exp(-((gx - x) ** 2 + (gy - y) ** 2) / (2.0 * (0.8 * self.cell) ** 2))
            num += w * sc
            den += w
        rate = num / np.maximum(den, 1e-9)
        trust = den / (den + CHECK_TRUST)
        self.v = np.clip(self.v - np.where(den >= min_weight, gain * np.maximum(rate - 0.25, 0.0) * trust, 0.0), -self.limit, self.limit)

    def learn(self, obs, gain, target, min_weight=1.5):
        """obs = [(x, y, score)] with score 1 = the mark is missing, 0.5 = partly there, 0 = painted. Every node looks at the marks near it (a Gaussian
        weight): the share of missing marks above the target makes it deeper, below the target slightly shallower. A node that has seen too few marks does not move."""
        if not obs:
            return
        gx, gy = np.meshgrid(self.xs, self.ys, indexing="ij")
        num, den = np.zeros_like(self.v), np.zeros_like(self.v)
        for x, y, sc in obs:
            w = np.exp(-((gx - x) ** 2 + (gy - y) ** 2) / (2.0 * (0.8 * self.cell) ** 2))
            num += w * sc
            den += w
        rate = num / np.maximum(den, 1e-9)
        trust = den / (den + CHECK_TRUST)                                    # five marks are not much to go on: 5 / (5 + 6) = 45% of the step
        step = np.where(den >= min_weight, gain * (rate - target) * trust, 0.0)
        self.v = np.clip(self.v + step, -self.limit, self.limit)


# ------------------------------------------------------------------------------------------------ the check
class TouchCheck:
    def __init__(self, cfg, cal, det, plan, grabber):
        self.cfg, self.cal, self.grabber = cfg, cal, grabber
        to = cfg.get("tool_offset") or (0.0, 0.0)
        self.dx, self.dy = float(to[0]), float(to[1])                 # the paint lands at (commanded + this)
        self.tilt = cfg.get("tilt")
        self.curve = cfg.get("curve")
        self.zmax_paint = cfg["z_paint"] + 1.0
        self.paper_poly = np.array(det["corners_px"], float)
        cx = [c[0] for c in det["corners_mm"]]
        cy = [c[1] for c in det["corners_mm"]]
        self.map = CorrMap(min(cx) - 10, max(cx) + 10, min(cy) - 10, max(cy) + 10)
        self.dishes = (plan or {}).get("dishes", {})
        self.park = tuple(cfg["snapshot"])
        self.pos = [0.0, 0.0, 12.0]
        self.rz = 12.0                                                   # the last Z as the planner wrote it
        self.pen = False
        self.cur = None
        self.chunk, self.layer_strokes, self.layer = [], {}, ""
        self.clean = self.prev = None
        self.mask = None
        self.log, self.fail, self.enabled = [], 0, True
        self.n_repainted = 0
        self.g = 0.0                                                      # the same extra depth everywhere
        self.g_num = self.g_den = 0.0                                     # all threshold marks judged so far: how many, how many missing

    # ---- geometry helpers
    def contact_at(self, x, y):
        z = self.cfg["z_paint"]
        if self.tilt:
            z += self.tilt[0] * (x - self.tilt[2]) + self.tilt[1] * (y - self.tilt[3])
        return z

    def width_at_depth(self, depth):
        if self.curve is None or depth < -0.02:
            return 0.0
        return float(np.interp(max(depth, 0.0), self.curve[0], self.curve[1]))

    # ---- the stream: every line passes here (the Z of painting moves gets the learned correction)
    def transform(self, line):
        l = line.strip()
        m = re.match(r"(G[01])\s", l)
        if not m:
            return line
        tok = {k: float(v) for k, v in re.findall(r"([XYZF])(-?\d+\.?\d*)", l)}
        g1 = m.group(1) == "G1"
        has_xy = "X" in tok or "Y" in tok
        x, y = tok.get("X", self.pos[0]), tok.get("Y", self.pos[1])
        z_in = tok.get("Z")                                              # the Z in the line, as the planner wrote it (before the correction)
        # a vertical press onto the paper: it comes down from the travel height, and not into a dish
        pen_down = (g1 and z_in is not None and not has_xy and z_in < self.rz - 0.2 and self.rz <= self.cfg["z_low"] + 0.1 and z_in <= self.zmax_paint)
        stroke_pt = g1 and has_xy and self.pen
        out, zc = line, (z_in if z_in is not None else self.pos[2])
        if pen_down or stroke_pt:
            base = z_in if z_in is not None else self.rz
            zc = max(Z_MIN, base - self.map.at(x + self.dx, y + self.dy) - self.g)
            if z_in is not None:
                out = re.sub(r"Z-?\d+\.?\d*", f"Z{zc:.3f}", l, count=1)
            else:                                                          # a constant-Z move (infill): it gets its Z so that it can follow the map
                out = re.sub(r"(Y-?\d+\.?\d*)", lambda k: f"{k.group(1)} Z{zc:.3f}", l, count=1)
            if pen_down:
                self.pen, self.cur = True, {"pts": [(x, y, zc)], "base": [base], "layer": self.layer}
            elif self.cur is not None:
                self.cur["pts"].append((x, y, zc))
                self.cur["base"].append(base)
            self.rz = base
        elif self.pen and (m.group(1) == "G0" or (g1 and z_in is not None and not has_xy and z_in > self.zmax_paint)):
            self._end_stroke()
            self.rz = z_in if z_in is not None else self.rz
        elif z_in is not None:
            self.rz = z_in
        self.pos = [x, y, zc]
        return out

    def _end_stroke(self):
        if self.cur is not None and len(self.cur["pts"]) >= 2:
            self.chunk.append(self.cur)
            self.layer_strokes.setdefault(self.cur["layer"], []).append(self.cur)
        self.cur, self.pen = None, False

    # ---- pictures
    def _grab(self, machine):
        machine.send_line(f"G0 Z{self.cfg['z_high']:.3f} F3000")
        machine.send_line(f"G0 X{self.park[0]:.3f} Y{self.park[1]:.3f} F{self.cfg['feed']}", timeout=90)
        machine.flush_motion()
        t = time.time() + CHECK_SETTLE_S
        time.sleep(CHECK_SETTLE_S)
        return self.grabber.frame_after(t)

    def _gray(self, frame, ref=None):
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if ref is not None and self.mask is not None:
            g = g * (float(np.median(ref[self.mask])) / max(float(np.median(g[self.mask])), 1.0))        # the same exposure
        return g

    def begin(self, machine=None):
        """The picture before the first stroke (the head is parked, as it will be at every check)."""
        frame = self.grabber.frame_after(time.time() - 5.0, timeout=8.0)
        if frame is None:
            raise RuntimeError("no camera picture")
        poly = np.zeros(frame.shape[:2], np.uint8)
        cv2.fillPoly(poly, [np.round(self.paper_poly).astype(np.int32)], 255)
        self.mask = cv2.erode(poly, np.ones((9, 9), np.uint8)) > 0
        self.clean = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 0.7)
        self.prev = self.clean.copy()

    def _diff(self, a, b):
        """How much darker b is than a (grey levels), inside the paper."""
        d = np.clip(cv2.GaussianBlur(a - b, (0, 0), 0.7), 0, None)
        d[~self.mask] = 0
        return d

    # ---- looking for the strokes in a picture
    def _samples(self, strokes, step=0.6):
        """Points every `step` mm along each stroke (target positions = commanded + tool offset), with the unit normal."""
        out = []
        for si, s in enumerate(strokes):
            P = np.array([(p[0] + self.dx, p[1] + self.dy) for p in s["pts"]], float)
            Z = np.array([p[2] for p in s["pts"]], float)
            seg = np.hypot(*np.diff(P, axis=0).T)
            L = float(seg.sum())
            if L < 2.0:
                continue
            cum = np.r_[0.0, np.cumsum(seg)]
            u = np.arange(0.0, L + 1e-9, step)
            px, py, pz = np.interp(u, cum, P[:, 0]), np.interp(u, cum, P[:, 1]), np.interp(u, cum, Z)
            tx, ty = np.gradient(px), np.gradient(py)
            n = np.hypot(tx, ty) + 1e-9
            out.append({"si": si, "x": px, "y": py, "z": pz, "u": u, "nx": -ty / n, "ny": tx / n, "len": L,
                        "base": np.interp(u, cum, np.array(s["base"], float))})
        return out

    def _profile(self, D, smp, shift=(0.0, 0.0), half=2.0, dt=0.25):
        t = np.arange(-half, half + 1e-9, dt)
        pts = np.stack([smp["x"][:, None] + shift[0] + t[None, :] * smp["nx"][:, None],
                        smp["y"][:, None] + shift[1] + t[None, :] * smp["ny"][:, None]], axis=-1)
        px = V.mm_to_px_array(self.cal["H_inv"], pts.reshape(-1, 2)).reshape(pts.shape)
        return map_coordinates(D, [px[..., 1], px[..., 0]], order=1, mode="nearest"), t

    def _evaluate(self, D, B, strokes, vis):
        """Per stroke: how much of it is there (coverage), how wide, and whether the evidence is usable at all."""
        from probe import _runs_width
        smps = self._samples(strokes)
        if not smps:
            return []
        # where the picture really is: a small shift of all strokes together (calibration, tool offset) found by the ink under them
        best, best_v = (0.0, 0.0), -1.0
        allc = [(s["x"], s["y"]) for s in smps]
        X = np.concatenate([a for a, _ in allc]); Y = np.concatenate([b for _, b in allc])
        for sx in np.arange(-1.0, 1.01, 0.25):
            for sy in np.arange(-1.0, 1.01, 0.25):
                px = V.mm_to_px_array(self.cal["H_inv"], np.c_[X + sx, Y + sy])
                v = float(map_coordinates(D, [px[:, 1], px[:, 0]], order=1, mode="nearest").mean())
                if v > best_v + 1e-6:
                    best, best_v = (sx, sy), v
        res = []
        for smp in smps:
            P, t = self._profile(D, smp, best)
            Bp, _ = self._profile(B, smp, best, half=1.0)
            peak = P.max(axis=1)
            marked = np.zeros(len(peak), bool)
            width = np.zeros(len(peak))
            for j in range(len(peak)):
                if peak[j] >= vis:
                    w, _ = _runs_width(P[j], 0.5 * peak[j], 0.25)
                    width[j] = w
                    marked[j] = w >= 0.5
            informative = (Bp.max(axis=1) < 3.0 * vis) & (smp["u"] >= CHECK_LAG_MM)   # old paint under a stroke says nothing about it; its first mm is the bristles starting
            n_inf = int(informative.sum())
            cov = float(marked[informative].mean()) if n_inf >= 3 else None
            usable = n_inf >= 0.5 * len(peak) and n_inf >= 3
            depth = np.array([self.contact_at(x, y) - z for x, y, z in zip(smp["x"], smp["y"], smp["base"])])      # how deep the planner meant it, before any correction
            dsent = float(np.median([self.contact_at(x, y) - z for x, y, z in zip(smp["x"], smp["y"], smp["z"])]))     # how deep it really went (with the correction)
            res.append({"si": smp["si"], "cov": cov if usable else None, "width": float(np.median(width[marked])) if marked.any() else 0.0,
                        "w_exp": self.width_at_depth(dsent),
                        "depth": float(np.median(depth)), "x": float(np.mean(smp["x"])), "y": float(np.mean(smp["y"])), "len": smp["len"]})
        self.last_shift = best
        return res

    def _vis(self, D):
        return max(7.0, 3.0 * float(np.percentile(D[self.mask], 90)))

    # ---- the check at a dip
    def checkpoint(self, machine, dips):
        if not self.enabled or self.clean is None or dips < 2 or not self.chunk or (dips - 1) % CHECK_EVERY_DIPS != 0:
            return
        strokes, self.chunk = self.chunk, []
        try:
            frame = self._grab(machine)
            if frame is None:
                raise RuntimeError("the camera gave no new picture")
            g = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 0.7)
            g = self._gray_norm(g)
            D = self._diff(self.prev, g)
            B = self._diff(self.clean, self.prev)
            vis = self._vis(D)
            ev = self._evaluate(D, B, strokes, vis)
            n_miss = n_part = n_ok = n_unk = 0
            obs, wide = [], []
            for e in ev:
                if e["cov"] is None:
                    n_unk += 1
                    continue
                near = e["depth"] < 0.35                                      # planned at the threshold: this is where a high spot means no mark
                sc = 1.0 if e["cov"] < 0.25 else 0.5 if e["cov"] < 0.6 else 0.0
                n_miss += sc == 1.0
                n_part += sc == 0.5
                n_ok += sc == 0.0
                if near:
                    obs.append((e["x"], e["y"], sc))
                if sc == 0.0 and e["w_exp"] > 0.5 and e["width"] > 0:
                    wide.append((e["x"], e["y"], 1.0 if e["width"] > CHECK_WIDE * e["w_exp"] else 0.0))
            if obs:
                self.g_num += sum(o[2] for o in obs)
                self.g_den += len(obs)
                ref = self.g_num / self.g_den                                    # the whole sheet so far: the regions are judged against it, not against zero
                self.map.learn(obs, CHECK_GAIN_MM, ref)
                chunk_rate = sum(o[2] for o in obs) / len(obs)
                trust = len(obs) / (len(obs) + 8.0)
                self.g = float(np.clip(self.g + 0.5 * CHECK_GAIN_MM * (chunk_rate - CHECK_TARGET_MISS) * trust, 0.0, CHECK_GLOBAL_MAX_MM))
            self.map.relax(wide, CHECK_GAIN_MM)
            self.prev = g
            rec = {"dips": dips, "strokes": len(strokes), "usable": n_miss + n_part + n_ok, "missing": n_miss, "partial": n_part, "ok": n_ok,
                   "shift_mm": [float(self.last_shift[0]), float(self.last_shift[1])], "map_max": float(self.map.v.max()), "map_min": float(self.map.v.min()), "global_mm": float(self.g)}
            self.log.append(rec)
            print(f"   check at dip {dips}: {rec['strokes']} strokes, {rec['usable']} usable: {n_ok} painted, {n_part} partly, {n_miss} missing;"
                  f" depth map {rec['map_min']:+.2f} .. {rec['map_max']:+.2f} mm, same everywhere {self.g:+.2f} mm")
            self.fail = 0
        except Exception as e:                                                # never stop the painting for the check
            self.fail += 1
            print(f"   check at dip {dips} failed ({e}); painting goes on" + ("; the check is switched off" if self.fail >= 3 else ""))
            if self.fail >= 3:
                self.enabled = False

    def _gray_norm(self, g):
        if self.mask is None:
            return g
        return g * (float(np.median(self.clean[self.mask])) / max(float(np.median(g[self.mask])), 1.0))

    # ---- the end of a layer: everything still missing is painted again
    def layer_end(self, machine, layer, final=False):
        strokes = list(self.layer_strokes.get(layer, []))
        if not self.enabled or self.clean is None or not strokes:
            return
        try:
            self._end_stroke() if self.pen else None
            frame = self._grab(machine)
            if frame is None:
                raise RuntimeError("the camera gave no new picture")
            g = self._gray_norm(cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 0.7))
            T = self._diff(self.clean, g)                                      # all the paint on the sheet
            self.prev = g
            self.chunk = []
            vis = self._vis(T)
            ev = self._evaluate(T, np.zeros_like(T), strokes, vis)             # the whole layer against everything that is on the paper
            todo = [strokes[e["si"]] for e in ev if e["cov"] is not None and e["cov"] < 0.4 and e["len"] >= 2.0]
            total = len([e for e in ev if e["cov"] is not None])
            print(f"   layer {layer or '-'}: {len(todo)} of {total} strokes are still missing or weak" + (" - painting them again" if todo and CHECK_REPAINT else ""))
            self.log.append({"layer": layer, "strokes": total, "missing_at_end": len(todo)})
            if todo and CHECK_REPAINT:
                self._repaint(machine, layer, todo)
                self.n_repainted += len(todo)
        except Exception as e:
            print(f"   end-of-layer check failed ({e}); painting goes on")

    def _repaint(self, machine, layer, strokes):
        import paint as PT
        dish = self.dishes.get(layer)
        cfg = self.cfg
        if dish is None:
            print("   (no dish known for this layer: the strokes are not painted again)")
            return
        dish = (dish[0] - self.dx, dish[1] - self.dy, dish[2])               # the same shift as every other X / Y of the job
        P = PT.Painter(cfg)
        order, pos = [], (self.park[0], self.park[1])
        left = list(strokes)
        while left:                                                           # the nearest stroke next
            k = min(range(len(left)), key=lambda i: math.hypot(left[i]["pts"][0][0] - pos[0], left[i]["pts"][0][1] - pos[1]))
            s = left.pop(k)
            order.append(s)
            pos = (s["pts"][-1][0], s["pts"][-1][1])
        run_len, interval = 1e9, float(cfg.get("min_dist", 90.0))
        for s in order:
            pts = s["pts"]
            if run_len >= interval:
                P._dip_and_travel(pts[0][0], pts[0][1], dish)
                run_len = 0.0
            else:
                P.gcode.append(f"G0 X{pts[0][0]:.3f} Y{pts[0][1]:.3f} Z{cfg['z_low']:.3f} F{cfg['feed']}")
            z0 = self._rz(s, 0)
            P._pen_down(z0)
            for i in range(1, len(pts)):
                P.gcode.append(f"G1 X{pts[i][0]:.3f} Y{pts[i][1]:.3f} Z{self._rz(s, i):.3f}")
                run_len += math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1])
            P.gcode.append(f"G0 Z{cfg['z_low']:.3f} F3000")
        P.gcode.append(f"G0 Z{cfg['z_high']:.3f} F3000")
        for l in "\n".join(P.gcode).split("\n"):
            l = l.split(";")[0].strip()
            if l:
                machine.send_line(l)
        machine.flush_motion()

    def _rz(self, s, i):
        """Z of a repainted stroke point: the planned Z with today's correction, and a little deeper still."""
        x, y = s["pts"][i][0] + self.dx, s["pts"][i][1] + self.dy
        return max(Z_MIN, s["base"][i] - self.map.at(x, y) - self.g - 0.08)

    # ---- the end
    def start_message(self):
        print(f"touch check ON: at every dip the head parks at X {self.park[0]:.0f} Y {self.park[1]:.0f}, the camera looks at the new strokes, a depth map is learned "
              f"(at most {CHECK_MAX_CORR_MM:.2f} mm) and missing strokes are painted again at the end of the layer.")

    def finish(self, machine):
        try:
            self.save_log()
            m = self.map
            checks = [r for r in self.log if "dips" in r]
            miss = sum(r["missing"] for r in checks)
            seen = sum(r["usable"] for r in checks)
            print(f"touch check done: {len(checks)} checks, {miss} of {seen} usable strokes were missing at the first look (about {100 * miss / max(seen, 1):.0f}%); "
                  f"{self.n_repainted} painted again; depth map {m.v.min():+.2f} .. {m.v.max():+.2f} mm, plus {self.g:.2f} mm everywhere ({CHECK_LOG})")
        except Exception as e:
            print(f"(check summary failed: {e})")

    def save_log(self):
        json.dump({"checks": self.log, "map_x": self.map.xs.tolist(), "map_y": self.map.ys.tolist(), "map": self.map.v.tolist(), "global_mm": self.g,
                   "tool_offset": [self.dx, self.dy], "created": time.strftime("%Y-%m-%d %H:%M:%S")}, open(CHECK_LOG, "w"), indent=1)
