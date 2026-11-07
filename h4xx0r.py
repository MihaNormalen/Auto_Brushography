"""
h4xx0r.py - the h4xx0r theme of ../brushograph's WebUI (webui/static/webui.css), for this program's terminal and live preview.
    python paint.py artwork.png --h4xx0r          (or H4XX0R = True in settings.py)

A green phosphor terminal behind dirty glass. The values are the WebUI's, which were read out of the original theme
(f1adang/ESP3D-WEBUI, h4xx0r-theme-v0.1): one green and nothing else. An error is the hot green in reverse video, not red,
and a warning is underlined. The paint keeps its real colours, as the trays and plates do in the WebUI: the camera picture
and the planned strokes are the only colour on the screen.

  * the terminal (enable): green text on the ground colour, errors reversed, warnings underlined, a power-on line.
  * the live preview (Glass, hud): the WebUI's four glass layers on their own clocks - scanlines and a vignette that
    flicker (3.7 s), green static that drifts and thickens in a glitch, dropout streaks (1.3 s), a band of interference that
    rolls up the screen - and the picture loses the signal on a 9 s cycle, jitters a pixel (2.3 s) and powers on from a line.
    The lettering is VT323 (fonts/VT323.ttf, from ../brushograph/webui/static) with a phosphor glow.
"""
import atexit
import math
import os
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ON = False                                                        # set by enable(); the live preview asks it

# ------------------------------------------------------------------------------------------------ the palette (webui.css)
HEX = {"ground": "#020604", "panel": "#061009", "strip": "#08170c", "lit": "#0e2e16", "ink": "#2ee62e",
       "bright": "#33ff33", "hot": "#66ff66", "dim": "#2e9e3a", "rule": "#1e5a2c", "on_green": "#010a03"}
RGB = {k: tuple(int(v[i:i + 2], 16) for i in (1, 3, 5)) for k, v in HEX.items()}
BGR = {k: v[::-1] for k, v in RGB.items()}
GREEN = np.array(BGR["bright"], np.float32)                      # rgba(51,255,51,.x) everywhere in the glass


# ================================================================================================ the terminal
_ERR = re.compile(r"(?i)\b(error|errors|stopped|traceback|exception|failed|cannot|refused)\b")
_WARN = re.compile(r"(?i)\bwarning\b|\*\*\*")


def _sgr(rgb, bg=False):
    return f"\x1b[{48 if bg else 38};2;{rgb[0]};{rgb[1]};{rgb[2]}m"


class _Phosphor:
    """A stream that letters every line in phosphor green on the ground colour. Line by line (a partial line is let out on
    flush, so an input() prompt shows), and a \\r progress line keeps working."""
    def __init__(self, stream):
        self.s, self.buf = stream, ""

    def _line(self, text):
        if not text:
            return f"{_sgr(RGB['ground'], True)}\x1b[K\x1b[0m"
        if _ERR.search(text):                                     # reverse video: ground lettering on the hot green
            style = f"{_sgr(RGB['ground'], True)}{_sgr(RGB['hot'])}\x1b[7m"
        elif _WARN.search(text):
            style = f"{_sgr(RGB['ground'], True)}{_sgr(RGB['bright'])}\x1b[4m"
        elif text.lstrip().startswith((">>", "<<", "(")):     # the echo, asides: the dim green
            style = f"{_sgr(RGB['ground'], True)}{_sgr(RGB['dim'])}"
        else:
            style = f"{_sgr(RGB['ground'], True)}{_sgr(RGB['ink'])}"
        return f"{style}{text}\x1b[27;24m\x1b[K\x1b[0m"            # the rest of the row in ground, not in the reversed green

    def _emit(self, text):
        self.s.write("\r".join(self._line(p) for p in text.split("\r")))

    def write(self, text):
        self.buf += text
        while "\n" in self.buf:
            line, self.buf = self.buf.split("\n", 1)
            self._emit(line)
            self.s.write("\n")
        if "\r" in self.buf:                                      # a progress line: out now, ready to be overwritten
            head, _, self.buf = self.buf.rpartition("\r")
            self._emit(head + "\r")
        return len(text)

    def flush(self):
        if self.buf:
            self._emit(self.buf)
            self.buf = ""
        self.s.flush()

    def isatty(self):
        return self.s.isatty()

    def __getattr__(self, name):
        return getattr(self.s, name)


def enable(power_on=True):
    """Switches the theme on: the terminal (when it is one - NO_COLOR or a pipe get plain text) and the live preview."""
    global ON
    ON = True
    if os.environ.get("NO_COLOR") or not sys.stdout.isatty() or isinstance(sys.stdout, _Phosphor):
        return
    sys.stdout.flush()
    sys.stderr.flush()
    sys.stdout, sys.stderr = _Phosphor(sys.stdout), _Phosphor(sys.stderr)
    atexit.register(lambda: (sys.stdout.flush(), sys.__stdout__.write("\x1b[0m"), sys.__stdout__.flush()))
    if power_on:
        _power_on()


def _power_on():
    """The screen powers on from a line (h4x-power-on): a bright rule grows out of the middle, then the banner."""
    out = sys.__stdout__
    try:
        w = max(40, min(100, os.get_terminal_size().columns))
    except OSError:
        w = 80
    for k in range(1, 9):
        n = int(w * k / 8)
        pad = (w - n) // 2
        out.write("\r" + _sgr(RGB["ground"], True) + " " * pad + _sgr(RGB["hot"]) + "━" * n + " " * (w - n - pad) + "\x1b[0m")
        out.flush()
        time.sleep(0.035)
    out.write("\r\x1b[K")
    banner = ["▓▒░ BRUSHOGRAPHY ░▒▓   phosphor link established",
              "> camera, machine, brush: online. the paint is the only colour here.", ""]
    for line in banner:
        sys.stdout.write(line + "\n")
    sys.stdout.flush()


# ================================================================================================ the lettering
_FONTS = {}


def _font(px):
    """VT323 at px (its x-height is 0.40 of its size, Helvetica's 0.52: the WebUI scales it by 0.52 / 0.40 the same way)."""
    if px not in _FONTS:
        try:
            from PIL import ImageFont
            here = Path(__file__).resolve().parent
            for p in (here / "fonts" / "VT323.ttf", here.parent / "brushograph" / "webui" / "static" / "VT323.ttf"):
                if p.exists():
                    _FONTS[px] = ImageFont.truetype(str(p), int(round(px * 0.52 / 0.40)))
                    break
            else:
                _FONTS[px] = None
        except Exception:
            _FONTS[px] = None
    return _FONTS[px]


def has_font():
    """Whether VT323 is there to letter with (without it the windows keep their plain lettering, the glass still runs)."""
    return _font(20) is not None


def _text_layer(size, items):
    """items = [(text, (x, y), px, rgb, reverse)] -> (BGR picture, alpha 0..1) of the lettering, glow added."""
    from PIL import Image, ImageDraw
    w, h = size
    img = Image.new("RGB", (w, h), (0, 0, 0))
    a = Image.new("L", (w, h), 0)
    d, da = ImageDraw.Draw(img), ImageDraw.Draw(a)
    for text, pos, px, rgb, rev in items:
        if text == "#block":                                      # a solid block (VT323 has no U+2588): px = (x0, y0, x1, y1)
            d.rectangle(px, fill=rgb)
            da.rectangle(px, fill=255)
            continue
        x, y = pos
        f = _font(px)
        if f is None:
            continue
        if rev:                                                   # an error: a block of hot green, the lettering in ground
            box = d.textbbox((x, y), text, font=f)
            box = (box[0] - 4, box[1] - 2, box[2] + 4, box[3] + 2)
            d.rectangle(box, fill=RGB["hot"])
            da.rectangle(box, fill=255)
            d.text((x, y), text, font=f, fill=RGB["on_green"])
        else:
            d.text((x, y), text, font=f, fill=rgb)
            da.text((x, y), text, font=f, fill=255)
    bgr = np.asarray(img)[:, :, ::-1].astype(np.float32)
    alpha = np.asarray(a).astype(np.float32) / 255.0
    glow = cv2.GaussianBlur(alpha, (0, 0), 4.0) * 0.7               # text-shadow: 0 0 6px rgba(51,255,51,.7)
    return bgr, alpha, glow


def screen(frame, title, lines=(), clock=None, bottom=(), frac=None, done=False, banners=(), t=0.0, top_h=None,
           labels=(), strip=0.85):
    """Any window's lettering in the theme. A panel strip at the top: "> TITLE" with a blinking block cursor, `lines` under
    it (each (text, "ink" / "dim" / "bright" / "hot" / "warn" / "error")), the clock at the right. A strip at the bottom: the
    `bottom` lines, or a progress bar of blocks when frac is given. `banners` = [(text, kind)] across the middle.
    labels = [(text, (x, y), kind)] anywhere on the picture. strip = how opaque the strips are (less where the picture
    under them matters, as in a window you click on).
    frame = BGR uint8 at display size; returns a copy, or None when VT323 is missing (the caller then letters it plainly)."""
    if _font(20) is None:
        return None
    out = frame.astype(np.float32)
    h, w = out.shape[:2]
    lines, bottom = list(lines), list(bottom)
    top = top_h or (44 + 24 * len(lines))
    bot = 30 if (frac is not None or len(bottom) <= 1) else 8 + 22 * len(bottom)
    for y0, y1 in ((0, top), (h - bot, h)):                       # the strips: panel at 85%, a rule along their inner edge
        out[y0:y1] = out[y0:y1] * (1 - strip) + np.array(BGR["panel"], np.float32) * strip
    out[top - 1] = BGR["rule"]
    out[h - bot] = BGR["rule"]

    def item(text, xy, px, kind):
        if kind == "error":
            return (text, xy, px, None, True)                     # reverse video
        return (text, xy, px, RGB[{"warn": "bright"}.get(kind, kind)], False)

    head = "> " + title.upper()
    items = [item(head, (16, 4), 26, "bright")]
    for i, (text, kind) in enumerate(lines):
        items.append(item(text, (16, 40 + 24 * i), 18, kind))
        if kind == "warn":                                        # a warning is underlined
            y = 40 + 24 * i + 21
            items.append(("#block", None, (16, y, 16 + int(_font(18).getlength(text)), y + 1), RGB["bright"], False))
    if clock:
        items.append(item(clock, (w - 16 - int(_font(24).getlength(clock)), 6), 24, "dim"))
    if (t % 1.0) < 0.5:                                           # the block cursor, blinking (h4x-blink, 1 s)
        x = 16 + int(_font(26).getlength(head + " "))
        items.append(("#block", None, (x, 12, x + 13, 36), RGB["bright"], False))
    if frac is not None:                                          # the progress bar, a terminal's row of blocks
        n, seg, gap, y0 = 48, 9, 2, h - bot + 9
        k = int(round(n * max(0.0, min(1.0, frac))))
        lit = RGB["hot"] if done else RGB["bright"]
        for i in range(n):
            x = 16 + i * (seg + gap)
            items.append(("#block", None, (x, y0, x + seg - 1, y0 + 12), lit if i < k else RGB["lit"], False))
        items.append((f"{100 * frac:3.0f}%", (16 + n * (seg + gap) + 10, h - bot + 2), 20, lit, False))
    else:
        for i, (text, kind) in enumerate(bottom):
            items.append(item(text, (16, h - bot + 2 + 22 * i), 18, kind))
    for text, xy, kind in labels:
        items.append(item(text, (int(xy[0]), int(xy[1])), 22, kind))
    for i, (text, kind) in enumerate(banners):
        px = 34 if kind != "error" else 30
        x = w // 2 - int(_font(px).getlength(text)) // 2
        items.append(item(text, (x, h // 2 - 60 + 56 * i), px, kind))
    bgr, alpha, glow = _text_layer((w, h), items)
    out += glow[..., None] * GREEN * 0.5
    out = out * (1 - alpha[..., None]) + bgr * alpha[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def hud(frame, title, info, clock, frac, done=False, frozen=False, t=0.0):
    """paint.py's live preview: the title, one line of progress, the clock and the progress bar."""
    banners = ([("*** JOB COMPLETE ***", "hot")] if done else []) + ([("!! CARRIER LOST - CAMERA PICTURE FROZEN !!", "error")] if frozen else [])
    return screen(frame, title, [(info, "ink")], clock=clock, frac=frac, done=done, banners=banners, t=t, top_h=74)


def window(h=720, w=1280):
    """The theme's empty screen (ground colour) - for a window with nothing to show yet."""
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = BGR["ground"]
    return img


# ================================================================================================ the glass
def _interp(p, keys):
    """keys = [(percent, (v0, v1, ...)), ...] -> the values at percent p, linear between keyframes (an animation's tween)."""
    ps = [k[0] for k in keys]
    return [float(np.interp(p, ps, [k[1][i] for k in keys])) for i in range(len(keys[0][1]))]


def _step(p, keys):
    """steps(1): the values of the last keyframe at or before p."""
    v = keys[0][1]
    for q, vals in keys:
        if q <= p:
            v = vals
    return v


Z = (0, 0, 0, 1, 1, 1, 0, 0)                                      # dx, dy, skewX, scaleX, scaleY, brightness, blur, ghost
SIGNAL = [(0, Z), (13.9, Z), (14.2, (-7, 0, -2.5, 1, 1, 1.3, 0, 0)), (14.6, (5, 0, 1.5, 1, 1, 1.2, 0, 0)),
          (15.0, (-3, 0, -.8, 1, 1, 1.1, 0, 0)), (15.6, (2, 0, .4, 1, 1, 1.05, 0, 0)), (16.4, (-1, 0, 0, 1, 1, 1, 0, 0)),
          (17.1, Z), (37.9, Z), (40, (2, 0, -.6, 1, 1, .7, .6, 9)), (42.5, (-2, 0, .7, 1, 1, .6, .8, 12)),
          (45, (0, 0, -.3, 1, 1, .85, .3, 5)), (46.5, (0, 0, 0, 1, 1, 1.15, 0, 0)), (47.1, Z), (62.9, Z),
          (63.2, (0, -12, 0, 1, 1.01, 1.4, 0, 0)), (63.8, (0, 7, 0, 1, 1, 1.15, 0, 0)), (64.4, (0, -4, 0, 1, 1, 1.07, 0, 0)),
          (65.2, (0, 2, 0, 1, 1, 1, 0, 0)), (66.1, Z), (85.9, Z), (86.4, (0, 0, 0, .985, .99, 1.5, .5, 0)),
          (87.2, (0, 0, 0, .992, .995, 1.25, .3, 0)), (88.1, Z), (100, Z)]
HIDE = (100, 0)                                                   # clip-path inset(100% 0 0 0): no band
BAND = [(0, HIDE), (13.9, HIDE), (14.2, (31, 66)), (14.6, (34, 62)), (15.0, (36, 62.5)), (15.6, HIDE), (17.1, HIDE),
        (37.9, HIDE), (39, (72, 26)), (40.5, (55, 41)), (42, (38, 57)), (43.5, (20, 77)), (45, (6, 92.5)), (46.5, HIDE),
        (47.1, HIDE), (62.9, HIDE), (63.2, (0, 94)), (63.8, (96, 0)), (64.4, HIDE), (66.1, HIDE), (100, HIDE)]
SNOW = [(0, (.1,)), (13.9, (.1,)), (14.5, (.24,)), (17.1, (.1,)), (37, (.1,)), (40, (.3,)), (42, (.5,)), (44, (.35,)),
        (48, (.1,)), (62.9, (.1,)), (63.5, (.22,)), (66.1, (.1,)), (78, (.14,)), (80, (.1,)), (100, (.1,))]
SNOW_FLICKER = [(0, (1,)), (13, (1.5,)), (37, (.8,)), (61, (1.7,)), (82, (1.2,))]
FLICKER = [(0, (1,)), (12, (.85,)), (13, (1,)), (41, (.9,)), (42, (.8,)), (43, (1,)), (77, (.9,)), (78, (1,))]
JITTER = [(0, (0, 0)), (7, (1, 0)), (8, (0, 0)), (23, (-1, 0)), (24, (1, 0)), (25, (0, 0)), (49, (0, 1)), (50, (0, 0)),
          (71, (-2, 0)), (72, (1, 0)), (73, (0, 0)), (88, (1, 0)), (89, (0, 0))]
OFF = 130                                                         # background-position 130%: out of the picture
DROPOUT = [(0, ((12, 23), (70, 61), 44, 82, 1)), (14, ((64, 8), (22, 71), OFF, 76, .6)), (29, ((35, 52), (90, 33), 18, 70, 0)),
           (43, ((3, 66), (38, 47), 71, 64, 1)), (57, ((91, 29), (76, 5), OFF, 58, .5)), (71, ((27, 95), (14, 58), 9, 52, 1)),
           (86, ((58, 17), (83, 79), OFF, 46, 0))]


class Glass:
    """The WebUI's glass over a picture, on the same clocks. One per window; apply() every frame."""
    def __init__(self, still=False):
        """still: the picture itself does not move (no signal loss, no jitter) - for a window you click on; the glass over it
        (scanlines, static, dropouts, the rolling band) still runs, as the WebUI's dialogs sit still under the same glass."""
        self.t0 = time.time()
        self.size = None
        self.still = still

    def _prepare(self, h, w):
        self.size = (h, w)
        y, x = np.mgrid[0:h, 0:w].astype(np.float32)
        scan = np.where((y % 3) < 1, 0.65, 1.0).astype(np.float32)                     # rgba(0,0,0,.35) 1px in 3
        r = np.hypot((x - w / 2) / (w / 2), (y - h / 2) / (h / 2)) / math.sqrt(2) * 1.25
        vign = 1 - 0.4 * np.clip((r - 0.65) / 0.35, 0, 1)                             # transparent 65% -> .4 at the rim
        vmin = min(h, w) / 100.0
        edge = np.minimum.reduce([x, w - 1 - x, y, h - 1 - y])
        shade = 1 - 0.7 * np.exp(-edge / (3 * vmin))                                  # inset 0 0 3vmin rgba(0,0,0,.7)
        rx, ry = 3 * vmin, 4 * vmin                                                   # border-radius 3vmin / 4vmin
        cx = np.clip(np.maximum(rx - x, x - (w - 1 - rx)), 0, None) / rx
        cy = np.clip(np.maximum(ry - y, y - (h - 1 - ry)), 0, None) / ry
        corner = (cx ** 2 + cy ** 2) <= 1.0
        self.mult = scan * vign * shade * corner
        self.glow = (np.exp(-edge / (1.5 * vmin)) * 0.15 * corner)[..., None] * GREEN  # inset 0 0 1.5vmin green .15
        rng = np.random.default_rng(7)
        self.static = []                                                              # h4x-static: 4 drifting frames
        for _ in range(4):
            streak = cv2.resize(rng.random((max(h // 2, 1), max(w // 40, 1))).astype(np.float32), (w, h),
                                interpolation=cv2.INTER_LINEAR)                       # baseFrequency .012 x .75: long streaks
            fine = rng.random((h, w)).astype(np.float32)
            n = np.clip(2.2 * (0.6 * streak + 0.4 * fine) - 1.1, 0, 1)                # the feColorMatrix alpha: 2.2 a - 1.1
            self.static.append(n)
        self.lines = np.zeros((h, w), np.float32)
        self.lines[0::2] = 0.14                                                       # the band's green lines, 1px in 2

    def apply(self, img):
        h, w = img.shape[:2]
        if self.size != (h, w):
            self._prepare(h, w)
        t = time.time() - self.t0
        page = img
        # ---- the page moves: power-on, the signal (9 s), the jitter (2.3 s)
        if t < 1.1 and not self.still:                                                # h4x-power-on: a line opens into the picture
            sy, b = _interp(100 * t / 1.1, [(0, (.004, 1.8)), (35, (.004, 1.8)), (65, (1.04, 1.4)), (100, (1, 1))])
            hh = max(1, int(round(h * sy)))
            body = cv2.resize(page, (w, min(hh, 2 * h)))
            canvas = np.zeros_like(page)
            canvas[:] = BGR["ground"]
            y0 = (h - body.shape[0]) // 2
            ys, ye = max(0, y0), min(h, y0 + body.shape[0])
            canvas[ys:ye] = body[ys - y0:ye - y0]
            page = cv2.convertScaleAbs(canvas, alpha=b)
        p = 100 * ((t % 9.0) / 9.0)
        dx, dy, skew, sx, sy, bright, blur, ghost = _interp(p, SIGNAL) if not self.still else Z
        jx, jy = _step(100 * ((t % 2.3) / 2.3), JITTER) if not self.still else (0, 0)
        dx, dy = dx + jx, dy + jy
        if (dx, dy, skew, sx, sy) != (0, 0, 0, 1, 1):
            k = math.tan(math.radians(skew))
            M = np.float32([[sx, k, dx + (1 - sx) * w / 2 - k * h / 2], [0, sy, dy + (1 - sy) * h / 2]])
            page = cv2.warpAffine(page, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=BGR["ground"])
        if blur > 0.05:
            page = cv2.GaussianBlur(page, (0, 0), blur)
        f = page.astype(np.float32) * bright
        if ghost > 0.5:                                                               # drop-shadow(Npx 0 0 rgba(51,255,51,.3))
            g = int(round(ghost))
            sh = np.zeros_like(f[..., 0])
            sh[:, g:] = f[:, :-g].mean(axis=2) / 255.0
            f += sh[..., None] * GREEN * 0.3
        # ---- the band of interference rolling up (backdrop: blur 2px, brightness 1.5, contrast 1.2, green lines)
        top, bottom = _interp(p, BAND) if not self.still else HIDE                   # (a blurred band would hide the point to click)
        y0, y1 = int(h * top / 100), int(h * (100 - bottom) / 100)
        if y1 > y0:
            band = cv2.GaussianBlur(f[y0:y1], (0, 0), 2.0) * 1.5
            band = (band - 128) * 1.2 + 128
            f[y0:y1] = band + self.lines[y0:y1, :, None] * GREEN
        # ---- the static: drifting green snow, thicker in a glitch
        op = _interp(p, SNOW)[0] * _step(100 * ((t % 0.9) / 0.9), SNOW_FLICKER)[0]
        f += self.static[int(t / 0.06) % 4][..., None] * GREEN * op
        # ---- the dropouts (1.3 s, steps)
        (ax, ay), (bx, by), black_y, green_y, dop = _step(100 * ((t % 1.3) / 1.3), DROPOUT)
        if dop > 0:
            for (px, py), sw, sh_, a in (((ax, ay), .09, 1, .65), ((bx, by), .04, 2, .5)):
                if py < 100:
                    x0 = int((w - w * sw) * px / 100)
                    yy = int((h - sh_) * py / 100)
                    ramp = np.sin(np.linspace(0, math.pi, max(int(w * sw), 1)))[None, :, None]
                    f[yy:yy + sh_, x0:x0 + ramp.shape[1]] += ramp * GREEN * a * dop
            if black_y < 100:
                yy = int((h - 1) * black_y / 100)
                f[yy:yy + 1] *= 1 - 0.55 * dop
            yy = int((h - 2) * green_y / 100)
            f[yy:yy + 2] += GREEN * 0.16 * dop
        # ---- the glass itself: scanlines + vignette + rounded dark rim, flickering (3.7 s, steps)
        fl = _step(100 * ((t % 3.7) / 3.7), FLICKER)[0]
        f = f * (1 - fl * (1 - self.mult[..., None])) + self.glow * fl
        return np.clip(f, 0, 255).astype(np.uint8)


# ================================================================================================ for every program
HELP = "the h4xx0r theme of ../brushograph's WebUI: a green phosphor terminal and windows behind dirty glass (or H4XX0R = True in settings.py)"


def add_flag(ap):
    """The --h4xx0r option, the same in every program."""
    ap.add_argument("--h4xx0r", action="store_true", help=HELP)


def from_args(a):
    """Switches the theme on when --h4xx0r was given or H4XX0R = True in settings.py. Returns whether it is on."""
    try:
        import settings
        wanted = bool(getattr(a, "h4xx0r", False) or getattr(settings, "H4XX0R", False))
    except Exception:
        wanted = bool(getattr(a, "h4xx0r", False))
    if wanted and not ON:
        enable()
    return ON
