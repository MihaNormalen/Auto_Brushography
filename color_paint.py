"""
color_paint.py - an image painted in TRANSPARENT GLAZES, one paint and one coat per pass, drying in between.

    python color_paint.py make picture.jpg            # ONE STEP: a forecast, then the painting, automatic (overnight):
                                                      # probe once, rinse, drying between passes, a photo before every pass
    python color_paint.py picture.jpg                 # the same
    python color_paint.py make picture.jpg --plan_only   # only the forecast: ko/cp_preview.png, nothing moves
    python color_paint.py paint                       # paint the last picture again (a new sheet)
    python color_paint.py paint 4                     # go on from pass 4 (after a stop; the sheet must not have moved)

NO FIXED NUMBER OF PASSES. Before every pass the dry sheet is photographed and compared with the picture (glazes multiply:
the light the sheet still reflects, times what one more coat lets through). For every spot the coats still to come that
bring it nearest to the picture are found (a few coats of every paint ahead, so green = yellow AND cyan); the next pass is
the paint that helps the picture most, painted where it is wanted. Of the passes that help nearly as much, the stage decides
which goes first (COLOR_PREFER: an opaque underpainting first, glazes on top later).
HONEST PROMISES: after every pass the camera measures what it really did against what it promised; a paint's promises count
as much as its passes really kept (COLOR_TRUST_*), so a paint that keeps failing loses to the others.
It ends when another pass would bring the painting less than COLOR_STOP_GAIN nearer, when the last COLOR_STOP_PATIENCE passes
did not make the sheet better than its best (the camera's end), or COLOR_END_NO_BEST passes after its best (no pass limit,
COLOR_MAX_PASSES = 0). THEN THE FINISH (2026-11-23, finish_pass, color_lines.py): edge passes along the picture's own lines
where the sheet is too light there, and light accents in white - then a small pass with the signature.
THE PAINT CHECK: a pass that left much of its area unchanged had too little paint: its cup is dipped deeper from then on.
DRYING: COLOR_DRY_MIN, then the camera checks the last pass's area has stopped changing (COLOR_DRY_CHECK_MIN).
BROAD FIRST, DETAIL LAST: the passes go through stages (COLOR_STAGE_FROM), each with its own smoothing and infill style.
EACH COLOUR FINISHED before the next: spots of a pass that the dry photo shows unchanged get the same paint again (fill-ins,
COLOR_FILL_TRIES). TEST STROKES in a strip over the picture (COLOR_TEST_*): a paint's first dip paints one stroke, fat to thin,
crossing the earlier ones; the camera measures what the paint does over each paint under it (ko/cp_tests.json).
DRYING keys: + (or P) = DRY_EXTEND_MIN more, - = less, Enter = go on now (terminal or drying window), Esc = close the window.
What every pass really did is measured on the sheet and used for the rest of the run, and kept in ko/cp_measured.json as
the start values of the next run. The pass expected to be the last one also paints the signature.

THE PASSES go through paint.py like any job, in YOUR style (INFILL_STYLE in settings.py), with the same probe (painted once,
before pass 1, in PROBE_PAINT; reused after), dips and rinse. The sheet, dishes and rinse container are found once, before
pass 1. If a pass does not finish (a stop, a failed probe) the run stops: go on with  python color_paint.py paint <that pass>.

Files: ko/cp_plan.json (the passes painted so far), ko/cp_passNN.png (what each pass paints), ko/cp_preview.png (the forecast:
original | expected result | pass by pass), ko/cp_before_passNN.png (photos), ko/cp_replan_passNN.png (seen | target | pass).
"""
import argparse
import itertools
import json
import os
import shutil
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import color_chart as CC                                    # noqa: E402  (paint names -> picture colours, drying, Lab maths)
import settings as S                                        # noqa: E402

DIR = os.path.join(HERE, "ko")
PLAN = os.path.join(DIR, "cp_plan.json")
BLANK = os.path.join(DIR, "cp_blank.png")    # the photo of the blank sheet before pass 1: the white reference
BLANK_OLD = os.path.join(DIR, "cp_blank_prev.png")   # the run before's blank photo (pass 1 resumed on the same sheet?)
R_PREV = os.path.join(DIR, "cp_now.npy")     # what the sheet reflected before the last pass (what that pass did)
STATE = os.path.join(DIR, "cp_state.json")    # this run: what each paint really did (measured), signature painted
MEASURED = os.path.join(DIR, "cp_measured.json")   # what each paint did on the last sheet: the start values of the next run
SCENE = os.path.join(DIR, "cp_scene.json")  # the dishes and the rinse container, found before pass 1 and used by every pass
PAPER = os.path.join(DIR, "cp_paper.json")  # the sheet, found before pass 1 and used by every pass (a painted sheet
                                            # is not found reliably: 2026-11-13 passes 5-8 were painted 37 mm small)
PLAN_H = 1000               # pass pictures: this many pixels high (the size on paper follows the sheet, like any artwork)
MIN_AREA_MM = 2.0           # areas smaller than this square (on an A4 painting) are left out of a pass: no specks
MIN_SHARE = 0.005           # a pass for less of the picture than this is not worth it
COAT_COST = 0.3             # every coat costs this much (dE) when choosing: no coat that hardly helps
ORDER_SLACK = 0.5           # the lighter paint goes first when it helps at least this share of what the best paint would
DRY_MIN = float(getattr(S, "COLOR_DRY_MIN", 20.0))   # drying between passes (then the camera checks it, COLOR_DRY_CHECK_MIN)
PROBE_PAINT = "black"       # the probe lines are painted with this paint: the darkest OPAQUE cup (2026-11-14: the 12-cup palette;
                            # transparent marks are too faint to measure)
CUPS_KIND = "transparent"   # without cups.json yet: the passes dip only into the TRANSPARENT cups (paint.py --cups)
MARGIN_MM = float(getattr(S, "COLOR_MARGIN_MM", 16.0))   # the picture this far from the sheet's edge (the probes need room)
ADAPTIVE = True             # a photo of the dry sheet before every pass, the pass chosen from it. False: chosen from the forecast
REPAIR_PASSES = False       # no gap repair after a glaze pass: on wet, glossy, pale paint it sees glare as gaps and puts
                            # extra coats there (blotches). Later: one smart repair at the very end, from the dry painting

# paint name (paint.py's hue family = which dish) -> the share of the light (linear R, G, B, the paper = 1) ONE coat lets
# through. Measured on the sheet of 2026-11-13 (the portrait, 8 passes). "orange" = the yellow dish, "red" = magenta,
# "blue" = cyan. ko/cp_measured.json (the last run's own measurements) goes before these.
# One coat does, per colour of light (linear R, G, B, the paper = 1):   after = r + t x before
#   a glaze (transparent colour): r = 0, it only lets through the share t (darkens);
#   a white: lightens - transparent white partly, opaque white covers what was there (t small, r near the paper).
# "row": the palette row (cups.py) that holds it, "kind": the cup in front (opaque) or behind (transparent, the default).
# 2026-11-14 the 12-cup palette: glazes + green + whites (green and whites guessed, measured from their first pass on).
PAINTS = {
    "orange":  {"t": (1.00, 0.88, 0.14), "show": "yellow",  "row": "yellow"},
    "red":     {"t": (1.00, 0.48, 0.75), "show": "magenta", "row": "magenta"},
    "blue":    {"t": (0.75, 0.97, 1.00), "show": "cyan",    "row": "cyan"},
    "green":   {"t": (0.40, 0.80, 0.55), "show": "green",   "row": "green"},
    "black":   {"t": (0.60, 0.58, 0.55), "show": "black",   "row": "black"},
    # the transparent white lightens a little where it is dark (a veil). Run of 20:38 measured it at t .92, r 0 (did nothing;
    # painted on wet paint, and the old guess t .62 r .35 promised far too much). The user mixes it stronger: start from a
    # modest guess; its first pass (or its test band) measures the real one, and the trust lets it lose if it fails again.
    "white":   {"t": (0.80, 0.80, 0.80), "r": (0.15, 0.15, 0.15), "show": "white", "row": "white"},
    # opaque colours (front row): they cover; r = their colour as the camera sees it in the cup (cup_colour), t small
    "orange_o": {"t": (0.15, 0.15, 0.15), "r": "cup", "show": "opaque yellow",  "row": "yellow",  "kind": "opaque"},
    "red_o":    {"t": (0.15, 0.15, 0.15), "r": "cup", "show": "opaque magenta", "row": "magenta", "kind": "opaque"},
    "blue_o":   {"t": (0.15, 0.15, 0.15), "r": "cup", "show": "opaque cyan",    "row": "cyan",    "kind": "opaque"},
    "green_o":  {"t": (0.15, 0.15, 0.15), "r": "cup", "show": "opaque green",   "row": "green",   "kind": "opaque"},
    "black_o":  {"t": (0.15, 0.15, 0.15), "r": "cup", "show": "opaque black",   "row": "black",   "kind": "opaque"},
    "white_o": {"t": (0.15, 0.15, 0.15), "r": (0.82, 0.82, 0.82), "show": "opaque white", "row": "white", "kind": "opaque"},
}
ORDER = ["orange", "blue", "green", "red", "black", "orange_o", "green_o", "blue_o", "red_o", "black_o", "white", "white_o"]
# all paints; the preference: glazes by lightness first, then the covering paints in this order (whites last)


def cup_colour(p):
    """r of a covering paint: its colour as the camera sees it in its cup, relative to the white of the white cup (or of an
    empty cup), in linear light. Without the palette: a middle grey."""
    try:
        import cups as CUPS
        cs = CUPS.load()
        c = CUPS.cup_for(cs, PAINTS[p]["row"], PAINTS[p].get("kind", "opaque"))
        ref = next((x for x in cs["cups"] if x.get("white") or not x.get("filled")), None)
        lin = CC.srgb_to_lin(np.array(c["bgr"][::-1], float))
        white = CC.srgb_to_lin(np.array(ref["bgr"][::-1], float)) if ref else np.ones(3)
        return np.clip(lin / np.maximum(white, 1e-3), 0.02, 1.0)
    except Exception:
        return np.full(3, 0.4)


def is_white(p):
    """A covering paint (whites and the opaque colours): after = r + t x before with r > 0. Glazes have r = 0."""
    r = PAINTS[p].get("r", (0, 0, 0))
    return r == "cup" or any(v > 0 for v in r)


def order_key(p, fac, prefer="glaze"):
    """Which of the passes that help nearly as much goes first. prefer "glaze": lighter glazes first (dark cannot be taken
    back), covering paints after them, whites last. prefer "opaque": the covering paints first (an opaque UNDERPAINTING, the
    glazes go on top of it later; 2026-11-16: glazes first left the figure light and thin while heavy opaque green went all
    round it), the glazes after them. COLOR_PREFER sets it per stage."""
    if is_white(p):
        k = 10.0 + ORDER.index(p)                       # covering paints in ORDER, whites last
        return k - 30.0 if prefer == "opaque" else k
    t = fac[p][0]
    return -float(0.2126 * t[0] + 0.7152 * t[1] + 0.0722 * t[2])


def setting(name, default):
    return getattr(S, name, default)


def max_passes():
    """COLOR_MAX_PASSES: 0 / None = no limit - the run ends when it is done (the camera: nothing helps any more), not after a
    number of passes (user, 2026-11-23)."""
    v = int(setting("COLOR_MAX_PASSES", 0) or 0)
    return v if v > 0 else 10 ** 9


# what choose() must know of the run beyond the sheet (set by paint() before every choice): recent_white = the spots a
# white was painted on in the last COLOR_DARK_AFTER_WHITE passes; late = the detail stage; figure = the figure's inside
_CTX = {"recent_white": None, "late": False, "figure": None}


def lum(t):
    return float(0.2126 * t[0] + 0.7152 * t[1] + 0.0722 * t[2])


def dark_glaze(p, fac):
    """A glaze that darkens much (the black, a dark blue): one coat lets through less than COLOR_DARK_GLAZE_LUM of the light."""
    return not is_white(p) and (p in tuple(setting("COLOR_DARK_GLAZES", ("black",)))
                                or lum(fac[p][0]) < float(setting("COLOR_DARK_GLAZE_LUM", 0.6)))


_DE_NOW = [None]          # the sheet's colour error now (set by paint() before every choice)


def stage_of(n):
    """Which stage (0 = broad, ..) the n-th pass (fill-ins not counted) is in: COLOR_STAGE_FROM = the first pass of each, and
    (COLOR_STAGE_DE) not before the sheet is near enough to the picture - the broad structure first, details only when it is
    there (2026-11-18: the detail stage began at pass 11, dE 21, and the details came out as blobs on a sheet still far off)."""
    starts = setting("COLOR_STAGE_FROM", (1,))
    starts = list(starts) if isinstance(starts, (list, tuple)) else [int(starts)]
    st = max(0, sum(1 for a in starts if n >= int(a)) - 1)
    thr = setting("COLOR_STAGE_DE", None)
    if thr and _DE_NOW[0] is not None:
        st = min(st, sum(1 for t_ in list(thr)[1:] if t_ is not None and _DE_NOW[0] <= float(t_)))
    return st


def focus_map(img, shape_hw):
    """WHERE THE EYE GOES (Butina: perception looks for meaning; in a face the eyes are the strongest points): 1 on the faces
    (OpenCV's face finder), up to 0.3 where the picture has strong local contrast, 0 elsewhere; on the work grid."""
    H, W = shape_hw
    small = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    F = np.zeros((H, W), np.float32)
    try:
        cas = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = cas.detectMultiScale(cv2.equalizeHist(gray), 1.1, 5, minSize=(max(12, W // 25), max(12, W // 25)))
    except Exception:
        faces = []
    for (x, y, w, h) in faces:
        cv2.ellipse(F, (int(x + w / 2), int(y + h * 0.55)), (int(w * 0.75), int(h * 0.95)), 0, 0, 360, 1.0, -1)
    L = gray.astype(np.float32)
    sal = cv2.GaussianBlur(np.abs(L - cv2.GaussianBlur(L, (0, 0), max(3, W / 60))), (0, 0), max(2, W / 120))
    sal = 0.3 * np.clip(sal / max(float(np.percentile(sal, 97)), 1e-3), 0, 1)
    out = cv2.GaussianBlur(np.maximum(F, sal), (0, 0), max(2, W / 150))
    return np.clip(out, 0, 1), len(faces)


def _say(what, *a, **k):
    """THE ČOPIČOGRAF SPEAKS (voice.py, 2026-11-22): a short Slovene sentence about what it does now, in the background
    (VOICE_ON). Never stops or slows the painting. PAINT_VOICE = False: the painting says nothing (user, 2026-11-22)."""
    if not bool(setting("PAINT_VOICE", False)):
        return
    try:
        import voice
        getattr(voice, what)(*a, **k)
    except Exception as e:
        print(f"   (voice: {e})")


_PLANES = {"L": None}


def set_planes(L, Hp, Wp, focus=None):
    """DEPTH PLANES (color_planes.py, 2026-11-22; docs/depth_planes_comfyui.pdf): the picture sorted by depth (sky + hills,
    fence, figure, front plants). With them choose() smooths a pass's area INSIDE each plane (nothing filled across a plane
    border), each plane by its own amount (COLOR_PLANE_SMOOTH; the plane with the face COLOR_PLANE_SMOOTH_FOCUS), and keeps
    it sharp along the depth breaks (like COLOR_CRISP_EDGES at lightness breaks). L = plane number per spot on the work
    grid, or None = no planes. Returns the figure's plane number (or None)."""
    _PLANES.clear()
    _PLANES["L"] = L
    if L is None:
        return None
    import color_planes as CPL
    mm_px = (190.0 / max(Hp, Wp)) * Hp / L.shape[0]
    _PLANES.update(fac=CPL.factors(L, focus), brk=CPL.breaks(L, mm_px), fig=CPL.figure_plane(L, focus))
    fg = _PLANES["fig"]
    print(f"color_paint: {int(L.max()) + 1} depth planes (areas between depth jumps); smoothing x{_PLANES['fac'][0]:.2g} far "
          f".. x{_PLANES['fac'][-1]:.2g} near" + (f"; the FIGURE {100 * (L == fg).mean():.0f}% of the picture, "
                                                  f"x{_PLANES['fac'][fg]:.2g}" if fg is not None else "; no face found"))
    return _PLANES["fig"]


def clean_pass(m, Hp, Wp, smooth):
    """clean() for a pass's area: by plane when there are depth planes (set_planes), else for the whole picture."""
    L = _PLANES.get("L")
    sm = float(smooth if smooth is not None else per_stage("COLOR_MASK_SMOOTH_MM", 4.0, 1))
    if L is None or L.shape != m.shape or sm <= 1.0:
        return clean(m, Hp, Wp, smooth)
    import color_planes as CPL
    return CPL.clean_by_plane(m, L, sm, clean, Hp, Wp, _PLANES["fac"])


def per_stage(name, default, n):
    """A setting that may hold one value per stage (a tuple) for the n-th pass: broad first, detail last."""
    v = setting(name, default)
    if isinstance(v, (list, tuple)):
        return v[min(stage_of(n), len(v) - 1)] if v else default
    return v


STAGE_NAMES = ("broad", "middle", "detail")


def paint_bgr(p, fac):
    """The paint's colour for the overlay (BGR): as the camera sees it in its cup; without a palette, one coat on the paper."""
    c = cup_of(p) if p in PAINTS else None
    if c and c.get("bgr"):
        return tuple(int(v) for v in c["bgr"])
    t, r = fac[p]
    return tuple(int(v) for v in to_bgr(np.clip(r + t, 0, 1)[None, None, :])[0, 0])


def label_of(p):
    """The paint as the overlay names it: "opaque magenta", "transparent white"."""
    return PAINTS[p]["show"] if p.endswith("_o") else f"transparent {PAINTS[p]['show']}"


def lab_array(refl):
    """Linear reflectance (..., 3), the paper = 1 -> CIELAB (..., 3) (color_chart.refl_to_lab for whole arrays)."""
    refl = np.asarray(refl)
    dt = np.float32 if refl.dtype == np.float32 else float            # float32 stays float32 (choose(): twice as fast)
    xyz = refl.astype(dt, copy=False) @ (np.asarray(CC.M_XYZ, dt).T / np.asarray(CC.WHITE, dt))
    f = np.where(xyz > dt((6 / 29) ** 3), np.cbrt(np.maximum(xyz, 0)), xyz * dt(1 / (3 * (6 / 29) ** 2)) + dt(4 / 29))
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], axis=-1)


def wdist(a, b, lw=1.0):
    """Colour distance (Lab) with the lightness difference counted lw times: LIGHT AND DARK FIRST (Butina, Mala likovna
    teorija: colours of the same lightness separate the picture poorly - the value structure holds it together)."""
    a, b = np.asarray(a), np.asarray(b)
    dt = np.float32 if (a.dtype == np.float32 and b.dtype == np.float32) else float
    d = a.astype(dt, copy=False) - b.astype(dt, copy=False)
    if lw != 1.0:
        d = d * np.array([lw, 1.0, 1.0], dt)
    return np.sqrt(np.einsum("...i,...i->...", d, d))


def reachable_dark(fac, paints):
    """The darkest L the paints can really make on this sheet (each paint's coats on paper, a covering paint once)."""
    cap = int(setting("COLOR_MAX_COATS", 6))
    best = 100.0
    for p in paints:
        if p not in fac:
            continue
        T, R = coats(fac, p, 1 if is_white(p) else min(cap, 4))
        best = min(best, float(lab_array(np.clip(T + R, 0, 1)[None])[0][0]))
    return best


def target_lab(img, dark, shape_hw):
    """What the painting should look like (Lab, the paper = L 100) on a grid of shape_hw. The picture's own tones (a photo
    of a painting has no real white) are stretched: its lightest -> paper (L 96), its darkest -> L `dark`. A higher `dark`
    = a lighter, watercolour-like version that needs fewer coats."""
    work = cv2.resize(img, (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_AREA)
    lab = lab_array(CC.srgb_to_lin(work[..., ::-1]))
    lo, hi = np.percentile(lab[..., 0], [2, 99.5])
    lab[..., 0] = dark + (np.clip(lab[..., 0], lo, hi) - lo) / max(hi - lo, 1e-6) * (96.0 - dark)
    mod = float(setting("COLOR_MODULATION", 0.0))
    if mod > 0:
        # MODULATION (Butina after Cezanne: warm = light / near, cool = dark / far): the lights a little warmer (+b, a bit
        # of +a), the shadows a little cooler (-b, a bit of -a), up to `mod` Lab units at the ends of the picture's own
        # lightness range; the lightest (the paper, L > 90) stays white. The form turns by colour, not only by darkening -
        # and the cool shadows give the transparent cyan its work.
        L_ = lab[..., 0]
        mid = float(np.median(L_))
        k = np.clip((L_ - mid) / max((96.0 - dark) / 2.0, 1e-6), -1.0, 1.0)
        k = np.where(k > 0, k * np.clip((96.0 - L_) / 6.0, 0.0, 1.0), k)
        lab[..., 2] += mod * k
        lab[..., 1] += 0.3 * mod * k
    return lab


def to_bgr(refl):
    return CC.lin_to_srgb(np.clip(refl, 0, 1))[..., ::-1].astype(np.uint8)


def _tr(v):
    """(t, r) from a stored value: [t, t, t] (a glaze, older files), {"t": .., "r": ..}, or an older per-coat dict."""
    if isinstance(v, dict) and "t" in v:
        return np.array(v["t"], float), np.array(v.get("r", (0, 0, 0)), float)
    if isinstance(v, dict):
        v = v[max(v, key=int)] if v else None
    if v:
        return np.array(v, float), np.zeros(3)
    return None


def test_scale(state):
    """How much stronger a glaze really was than its test stroke said, as optical density (-ln t): what this run measured on
    the paints with both a test stroke and a pass, else COLOR_TEST_DENSITY_SCALE (2026-11-20, run H: green's test stroke
    t 0.72 0.93 0.93, its pass 0.22 0.65 0.62 - 6x; transparent black 3.5x; orange's strong channel read right. Not one
    factor for all: used only to decide whether a paint is worth TRYING, never as its value)."""
    v = state.get("test_scale")
    return float(v) if v else float(setting("COLOR_TEST_DENSITY_SCALE", 3.0))


def scaled_test(p, t, r, state):
    """A glaze not yet proven by a pass, as strong as it might be (for the decision to try it): density x test_scale, no
    channel below 0.1. Whites and opaques as they are."""
    t, r = np.asarray(t, float), np.asarray(r, float)
    if is_white(p) or (r > 0).any():
        return t, r
    return np.clip(np.clip(t, 0.03, 1.0) ** test_scale(state), 0.1, 1.0), r


def coat_factor(p, state):
    """What one coat of paint p does: (t, r), after = r + t x before. Measured on this sheet (its last pass), else on the last
    sheet (ko/cp_measured.json), else PAINTS."""
    got = _tr(state.get("learned", {}).get(p))
    if got is not None:
        return got
    try:
        old = _tr(json.load(open(MEASURED)).get(p))
    except (OSError, ValueError):
        old = None
    if old is not None:
        return old
    r = PAINTS[p].get("r", (0, 0, 0))
    return np.array(PAINTS[p]["t"], float), (cup_colour(p) if r == "cup" else np.array(r, float))


def coats(fac, p, k):
    """k coats of p as one (T, R): T = t^k, R = r (1 + t + ... + t^(k-1))."""
    t, r = fac[p]
    T = t ** k
    R = r * np.where(np.abs(1 - t) < 1e-6, k, (1 - T) / np.maximum(1 - t, 1e-6))
    return T, R


def work_shape(plan):
    Hp, Wp = plan["pic"]
    return (Hp // 2, Wp // 2), (Hp, Wp)


def clean(m, Hp, Wp, smooth=None, min_mm=None):
    """A pass mask without specks (on the grid it is given in), smoothed by `smooth` mm (None: the first stage's); pieces
    smaller than min_mm square (default MIN_AREA_MM) left out."""
    m = m.astype(np.uint8)
    if min_mm is None:                                              # (thin lines of the finish keep their 1-pixel parts)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mm_px = (190.0 / max(Hp, Wp)) * Hp / m.shape[0]                 # mm per pixel of this grid (on an A4 painting)
    if smooth is None:
        smooth = per_stage("COLOR_MASK_SMOOTH_MM", 4.0, 1)
    r = int(round(float(smooth) / mm_px))
    if r >= 1:                                                      # no lace: small holes closed, frayed edges rounded, so the
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))   # strokes can be long (2026-11-15: the
        m = cv2.morphologyEx(cv2.morphologyEx(m, cv2.MORPH_CLOSE, k), cv2.MORPH_OPEN, k)   # lace gave 7 mm bits, a maze)
    n, lab_, st, _ = cv2.connectedComponentsWithStats(m, 8)
    min_px = ((MIN_AREA_MM if min_mm is None else float(min_mm)) / mm_px) ** 2
    keep = np.zeros(n, bool)
    keep[1:] = st[1:, cv2.CC_STAT_AREA] >= min_px
    return keep[lab_]


def wet_zone(last, mm_px):
    """The part of the last pass's area a new pass must not touch while it is wet. COLOR_WET_MARGIN_MM > 0: that much
    round it too; < 0: the new pass may reach that far into it, so the two shapes TOUCH (2026-11-18: a 2 mm margin, and the
    thin strip left was smoothed away, left white lines between the shapes). Touching is fine, painting over is not."""
    mg = float(setting("COLOR_WET_MARGIN_MM", -1.0))
    r_ = int(round(abs(mg) / mm_px))
    m = last.astype(np.uint8)
    if r_ < 1:
        return m > 0
    k_ = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r_ + 1, 2 * r_ + 1))
    return (cv2.dilate(m, k_) if mg > 0 else cv2.erode(m, k_)) > 0


def choose(now, target, done, fac, paints, Hp, Wp, stop_gain=None, smooth=None, prefer="glaze", eff=None, small=False,
           lw=1.0, age=None, focus=None, fw=0.0, over_white=None, forbid=None, within=None, fine=False, min_share=None):
    """The next pass from the sheet as it is (now: linear, the paper = 1, on the work grid). Every spot gets the mix of coats
    still to come that brings it nearest to the target; every paint wanted somewhere is a candidate pass, its gain = how much
    nearer one coat of it brings the whole picture (dE, averaged). The best one wins, a lighter one when it is nearly as good.
    eff = {paint: the share of its promises its passes really kept on this sheet} (efficiency()): the promise times that is
    what the choice and the end go by. small: only the spots one coat clearly helps (a pass just for the signature).
    lw: the lightness difference counts lw times where the coats are chosen and passes ranked (COLOR_LIGHTNESS_WEIGHT).
    focus (0..1 per spot, focus_map) x fw: the face and the strong contrasts count (1 + fw x focus) times when ranking.
    CRISP EDGES (COLOR_CRISP_EDGES, Butina: a sharp light-dark break is a form's edge): near the picture's strong lightness
    breaks a pass's area follows the wanted spots exactly instead of the smoothed outline, so figure and ground stay apart.
    age: per spot, how many passes ago it was last painted (0 = the last pass: still wet; None = nothing known). THE WET
    PART LAST: a pass leaves out the area of the last pass when it has enough to do elsewhere (COLOR_AVOID_WET), and spots
    painted in the last passes count less when ranking (COLOR_RECENT_WEIGHT) - the paper still bare and the older, dry parts
    first; the fresh paint dries meanwhile and nothing smears it (2026-11-17: transparent white over wet black = grey mud).
    NO PAINT WHERE IT CANNOT HELP: larger patches inside a pass's smoothed area where its paint makes the spot worse (white
    on bare paper: 19% of the white pass of 2026-11-17) are cut out.
    within: the pass may paint only there (the finish: the picture's lines, its light accents); fine: no smoothing, small
    pieces kept (thin lines); min_share: the smallest pass (default MIN_SHARE).
    NO DARK GLAZE OVER A FRESH WHITE (2026-11-23: transparent black right over the opaque white of the pass before - gouache
    dissolves under a wet glaze, the two cancel out): a dark glaze (dark_glaze) leaves out the spots a white was painted on
    in the last COLOR_DARK_AFTER_WHITE passes (_CTX["recent_white"]), and COLOR_GLAZE_OVER_WHITE counts only for light glazes.
    LATE OPAQUES SMALL (2026-11-23: opaque green on 40% in the detail stage made the picture worse by 3.8): in the detail
    stage a covering paint stays off the figure's inside (COLOR_LATE_OPAQUE_OFF_FIGURE) and paints at most
    COLOR_LATE_OPAQUE_MAX_SHARE of the picture - the spots where it helps most.
    None = no pass helps enough (the end). Returns {paint, mask, gain (the raw promise), expect (ranking: gain x eff, with
    lightness and the wet part weighted), share}."""
    min_share = MIN_SHARE if min_share is None else float(min_share)
    stop_gain = float(setting("COLOR_STOP_GAIN", 0.4)) if stop_gain is None else stop_gain
    cap, look = int(setting("COLOR_MAX_COATS", 6)), int(setting("COLOR_LOOK_AHEAD", 2))
    eff = eff or {}
    eff_of = lambda p_: float(eff.get(p_, eff.get("*", 1.0)))
    # THE STACK OF COATS IS PLANNED COVERING PAINTS FIRST, GLAZES OVER THEM (2026-11-23): white lifts, a glaze over it brings
    # the colour back - before, the whites came last on top, so "white, then a glaze" did not exist for the planner
    usable = sorted([p for p in paints if done.get(p, 0) < cap],
                    key=lambda p: (0 if (is_white(p) and bool(setting("COLOR_COVER_FIRST", True))) else 1, order_key(p, fac)))
    if not usable:
        return None
    mix = int(setting("COLOR_MAX_MIX", 3))
    combos = np.array([c for c in itertools.product(*[range(min(1 if is_white(p) else look, cap - done.get(p, 0)) + 1)
                                                     for p in usable]) if sum(c) <= mix])
    facs = np.ones((len(combos), 3))
    adds = np.zeros((len(combos), 3))
    for j, p in enumerate(usable):                      # each paint's coats applied in this order: T, R of the whole mix
        for k in range(1, int(combos[:, j].max()) + 1):
            sel = combos[:, j] == k
            T, R = coats(fac, p, k)
            facs[sel] = facs[sel] * T[None, :]
            adds[sel] = adds[sel] * T[None, :] + R[None, :]
    cost = COAT_COST * combos.sum(axis=1)
    H_, W_ = now.shape[:2]
    flat, tg = now.reshape(-1, 3), target.reshape(-1, 3)
    d_now = np.linalg.norm(lab_array(flat) - tg, axis=1)
    dw_now = wdist(lab_array(flat), tg, lw)
    best = np.empty(len(flat), int)
    # every spot x every mix of coats: the slow part (2026-11-22: 64% of a forecast). In float32, the spots split over the
    # cores (COLOR_THREADS; numpy lets go of Python's lock while it counts, so the threads really run side by side)
    f32, t32 = flat.astype(np.float32), tg.astype(np.float32)
    fa32, ad32, co32 = facs.astype(np.float32), adds.astype(np.float32), cost.astype(np.float32)

    def _chunk(a, n=4000):
        d = wdist(lab_array(f32[a:a + n, None, :] * fa32[None] + ad32[None]), t32[a:a + n, None, :], lw) + co32[None]
        best[a:a + n] = d.argmin(axis=1)
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(int(setting("COLOR_THREADS", 0) or os.cpu_count() or 1)) as ex_:
        list(ex_.map(_chunk, range(0, len(flat), 4000)))
    wet = None
    wt = np.ones(len(flat))
    if focus is not None and fw > 0:
        wt = wt * (1.0 + float(fw) * focus.ravel())
    mm_px_ = (190.0 / max(Hp, Wp)) * Hp / H_
    edge_zone = None
    sm_ = float(smooth if smooth is not None else per_stage("COLOR_MASK_SMOOTH_MM", 4.0, 1))
    if bool(setting("COLOR_CRISP_EDGES", True)) and sm_ > 1.0:
        Lt = cv2.GaussianBlur(target[..., 0].astype(np.float32), (0, 0), 1.0)
        g = np.hypot(cv2.Sobel(Lt, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(Lt, cv2.CV_32F, 0, 1, ksize=3)) / 8.0 / mm_px_
        edge = (g > float(setting("COLOR_EDGE_DL", 8.0))).astype(np.uint8)           # L per mm
        r_e = max(1, int(round(sm_ / mm_px_)))
        edge_zone = cv2.dilate(edge, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r_e + 1, 2 * r_e + 1))) > 0
    brk = _PLANES.get("brk")
    if brk is not None and brk.shape == (H_, W_) and sm_ > 1.0:          # DEPTH BREAKS: one thing in front of another
        edge_zone = brk if edge_zone is None else (edge_zone | brk)
    if age is not None:
        rw = setting("COLOR_RECENT_WEIGHT", (0.25, 0.6))
        rw = list(rw) if isinstance(rw, (list, tuple)) else [float(rw)]
        a_ = age.ravel()
        for k_, v_ in enumerate(rw):
            wt[a_ == k_] = float(v_)
        if bool(setting("COLOR_AVOID_WET", True)):
            wet = wet_zone(age == 0, (190.0 / max(Hp, Wp)) * Hp / H_)
    # WHITE, THEN A GLAZE OVER IT (the brushographer's way against a milky picture, 2026-11-19): the spots whose last paint
    # is a white, dry now (the sheet was photographed after drying), are not put off as "the last pass's area" for a glaze,
    # and count COLOR_GLAZE_OVER_WHITE times - the white lifted the value, the glaze brings the colour back
    ow_ = over_white.ravel() if over_white is not None else None
    best = cv2.medianBlur(best.reshape(H_, W_).astype(np.uint16), 3).astype(int)
    want = combos[best]                                                   # H x W x paints: coats each spot still wants
    cands = []
    for j, p in enumerate(usable):
        m = want[..., j] >= 1
        if small:                                       # only where one coat clearly helps
            d1 = np.linalg.norm(lab_array(flat * fac[p][0] + fac[p][1]) - tg, axis=1).reshape(H_, W_)
            m = m & (d1 < d_now.reshape(H_, W_) - 3.0)
        if is_white(p) and p.startswith("white") and bool(setting("COLOR_WHITE_ONLY_LIFTS", True)):
            # WHITE ONLY LIFTS THE VALUE (2026-11-19: run G used white 6x to dull colours the camera read stronger than the
            # picture - a milky picture): only where the sheet is darker than wanted; dulling is the glazes' work
            lab_n = lab_array(flat).reshape(H_, W_, 3)
            lifts = lab_n[..., 0] < target[..., 0] - float(setting("COLOR_WHITE_LIFT_DL", 3.0))
            # ... or WHITISH: the picture has little colour there and the sheet clearly more (2026-11-23: the shirt chroma 19
            # on the sheet, 8 in the picture, about as light - white was not allowed to calm it)
            c_t, c_n = np.hypot(target[..., 1], target[..., 2]), np.hypot(lab_n[..., 1], lab_n[..., 2])
            calms = (c_t < float(setting("COLOR_WHITE_NEUTRAL_C", 12.0))) & (c_n > c_t + float(setting("COLOR_WHITE_NEUTRAL_DC", 8.0)))
            m = m & (lifts | calms)
        block = forbid                                  # e.g. still wet (paint elsewhere while it dries)
        if within is not None:
            block = ~within if block is None else (block | ~within)
        dark_p = dark_glaze(p, fac)
        rw_ = _CTX.get("recent_white")
        if dark_p and rw_ is not None and rw_.shape == m.shape:
            block = rw_ if block is None else (block | rw_)
        if block is not None:
            m = m & ~block
        raw_m = m
        if fine:                                        # thin lines, small accents: as they are
            m = clean(m, Hp, Wp, 0.0, min_mm=float(setting("COLOR_FINISH_MIN_MM", 1.0)))
        else:
            m = clean_pass(m, Hp, Wp, smooth)
            if edge_zone is not None:                   # at the picture's edges: exactly the wanted spots
                m = clean((m & ~edge_zone) | (raw_m & edge_zone), Hp, Wp, 1.0)
        if m.mean() < min_share:
            continue
        lab1 = lab_array(flat * fac[p][0] + fac[p][1])
        worse = (wdist(lab1, tg, lw) > dw_now).reshape(H_, W_) & m       # this paint makes these spots worse
        sm = float(smooth if smooth is not None else per_stage("COLOR_MASK_SMOOTH_MM", 4.0, 1))
        r_ = max(1, int(round(max(sm, 1.0) / ((190.0 / max(Hp, Wp)) * Hp / H_))))
        k_ = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r_ + 1, 2 * r_ + 1))
        big = cv2.morphologyEx(worse.astype(np.uint8), cv2.MORPH_OPEN, k_) > 0   # the larger patches, not single strokes' edges
        if big.any():
            m = clean(m & ~big, Hp, Wp, 0.0 if fine else 1.0, min_mm=float(setting("COLOR_FINISH_MIN_MM", 1.0)) if fine else None)
        avoided = False
        wet_p = wet if (wet is None or over_white is None or is_white(p) or dark_p) else (wet & ~over_white)
        if wet_p is not None and (m & wet_p).any():
            m2 = clean(m & ~wet_p, Hp, Wp, 1.0)
            if m2.mean() >= min_share:
                m, avoided = m2, True
        if m.mean() < min_share:
            continue
        if _CTX.get("late") and is_white(p) and within is None:     # LATE OPAQUES SMALL
            fg_ = _CTX.get("figure")
            if fg_ is not None and fg_.shape == m.shape and bool(setting("COLOR_LATE_OPAQUE_OFF_FIGURE", True)):
                m = m & ~fg_
            cap_ = float(setting("COLOR_LATE_OPAQUE_MAX_SHARE", 0.10))
            if m.mean() > cap_:
                gs_ = (d_now - np.linalg.norm(lab1 - tg, axis=1)).reshape(H_, W_)
                thr_ = float(np.quantile(gs_[m], 1.0 - cap_ / max(float(m.mean()), 1e-6)))
                m = clean(m & (gs_ >= thr_), Hp, Wp, 1.0)
            if m.mean() < min_share:
                continue
        if block is not None:                           # once more at the end: smoothing must not reach back in
            m = m & ~block
            if m.mean() < min_share:
                continue
        mi = m.ravel()
        g = float((d_now[mi] - np.linalg.norm(lab1[mi] - tg[mi], axis=1)).sum() / len(flat))
        wt_p = wt
        if ow_ is not None and not is_white(p) and not dark_p:   # a light glaze over a dry white: wanted, not put off
            wt_p = np.where(ow_, np.maximum(wt, 1.0) * float(setting("COLOR_GLAZE_OVER_WHITE", 1.5)), wt)
        gr = float(((dw_now[mi] - wdist(lab1[mi], tg[mi], lw)) * wt_p[mi]).sum() / len(flat))
        cands.append({"paint": p, "mask": m, "gain": g, "expect": gr * eff_of(p), "share": float(m.mean()), "avoided_wet": avoided})
    if not cands:
        return None
    if max(c["gain"] for c in cands) < stop_gain:       # the end by the PROMISE (the camera's end: stalled()); not by the
        return None                                     # trust-scaled one: after two bad passes it fell to 15% and would
    top = max(c["expect"] for c in cands)               # have ended run 20:38 at pass 10 (dE 22). Trust ranks the choice
    if top <= 0:
        return None
    # 2026-11-17: at 0.5 the order (covering paints in ORDER, whites last) gave away half the gain: opaque green 5 times
    # while opaque white would have covered the pink figure (expect 1.35 vs 1.09 for green)
    ok = [c for c in cands if c["expect"] >= float(setting("COLOR_ORDER_SLACK", ORDER_SLACK)) * top]
    return min(ok, key=lambda c: order_key(c["paint"], fac, prefer))


def apply(now, c, fac):
    out = now.copy()
    t, r = fac[c["paint"]]
    out[c["mask"]] = out[c["mask"]] * t + r
    return out


def write_pass(i, plan, c):
    """The pass picture (full size) and its entry in the plan."""
    (Hw, Ww), (Hp, Wp) = work_shape(plan)
    m = cv2.resize(c["mask"].astype(np.uint8), (Wp, Hp), interpolation=cv2.INTER_NEAREST) > 0
    pic = np.full((Hp, Wp, 3), 255, np.uint8)
    # in the paint's colour: an opaque one in its glaze's hue (2026-11-15: opaque magenta was drawn grey, so paint.py called the
    # layer "black"). A white has no hue: dark grey; paint.py gets the real name and colour from JOB_INFO, the cup from --cup
    base = c["paint"][:-2] if c["paint"].endswith("_o") else c["paint"]
    pic[m] = (60, 60, 60) if base not in CC.HUE_OF else CC.name_rgb(base, 235)[::-1]
    name = f"cp_pass{i:02d}.png"
    cv2.imwrite(os.path.join(DIR, name), pic)
    fill = int(c.get("fill", 0))
    n_real = sum(1 for x in plan["passes"] if not x.get("fill")) + (0 if fill else 1)   # this pass's number, fill-ins not counted
    q = {"picture": name, "paint": c["paint"], "show": PAINTS[c["paint"]]["show"],
         "coat": sum(1 for x in plan["passes"] if x["paint"] == c["paint"] and not x.get("fill")) + (0 if fill else 1),
         "share": round(c["share"], 3), "gain": round(c["gain"], 2),
         "stage": STAGE_NAMES[min(stage_of(n_real), len(STAGE_NAMES) - 1)],
         "smooth": float(c.get("smooth", per_stage("COLOR_MASK_SMOOTH_MM", 4.0, n_real))),
         "style": c.get("style") or per_stage("COLOR_INFILL_STYLE", None, n_real)}
    if fill:
        q["fill"] = fill
    if c.get("kind"):                                               # a finishing pass: "edge" or "light"
        q["kind"] = c["kind"]
    if c.get("dE") is not None:
        q["dE"] = round(float(c["dE"]), 2)
    if c.get("expect") is not None:                                 # the promise after what this paint's passes really did
        q["expect"] = round(float(c["expect"]), 2)
    plan["passes"].append(q)
    json.dump(plan, open(PLAN, "w"), indent=1)
    return q, pic


def mask_of(picture, shape_hw):
    """Where a pass picture paints (not white), on the given grid."""
    pic = cv2.imread(os.path.join(DIR, picture))
    m = (pic.min(axis=2) < 250).astype(np.uint8)
    return cv2.resize(m, (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_AREA) > 0


def done_of(passes):
    d = {}
    for q in passes:
        if not q.get("fill"):                                       # a fill-in finishes a coat, it is not another one
            d[q["paint"]] = d.get(q["paint"], 0) + 1
    return d


def missed(q, before, now, fac, target, Hp, Wp):
    """FINISH A COLOUR: the spots of pass q that the dry photo shows (nearly) unchanged although the paint should have changed
    them, and where it is still wanted (it brings them nearer the picture). -> (mask on the work grid or None, gain, faint =
    the share of the pass's own area left unchanged: a dry brush / low cup when it is large)."""
    m = mask_of(q["picture"], now.shape[:2])
    m = cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0      # not the edges (placement, bleeding)
    t, r = fac[q["paint"]]
    exp = np.linalg.norm(before * (t - 1.0) + r, axis=2)                   # what one coat should change there
    act = np.linalg.norm(now - before, axis=2)
    after = now * t + r
    d_now = np.linalg.norm(lab_array(now) - target, axis=2)
    d_after = np.linalg.norm(lab_array(after) - target, axis=2)
    base = m & (exp > 0.03)
    unchanged = base & (act < float(setting("COLOR_FILL_MISS", 0.3)) * exp)
    faint = float(unchanged.sum()) / max(int(base.sum()), 1)
    miss = clean(unchanged & (d_after < d_now), Hp, Wp, smooth=1.0)
    g = float((d_now - d_after)[miss].sum() / miss.size) if miss.any() else 0.0
    # 2026-11-18: white stripes left between the first black bands - a small share, but each spot far off: worth it
    if miss.mean() < float(setting("COLOR_FILL_MIN_SHARE", 0.015)) and g < float(setting("COLOR_FILL_GAIN_ANYWAY", 0.4)):
        return None, 0.0, faint
    return miss, g, faint


def holes(now, target, fac, paints, Hp, Wp, painted=None):
    """THE EMPTY SPOTS (2026-11-19, run G: white paper left in the grey shirt, 1-3 mm holes the blurred eye did not see): the
    sheet still paper-white where the picture is not; the paint that brings most of them nearest gets one pass on the ones it
    suits (a little wider, so it touches what is round them). None = too few holes."""
    L = lab_array(now)
    bare = (L[..., 0] > float(setting("COLOR_HOLE_PAPER_L", 90.0))) & (np.hypot(L[..., 1], L[..., 2]) < 6.0)
    want = (target[..., 0] < 96.0 - float(setting("COLOR_HOLE_DL", 10.0))) | (np.hypot(target[..., 1], target[..., 2]) > 15.0)
    h = bare & want
    if painted is not None:          # only the gaps INSIDE what the passes painted (2026-11-20, run H: at the start all the paper
        h &= painted                 # was "holes": passes 2-5 were hole passes on 37, 27, 4, 20% - and none were left for the end)
    if h.mean() < float(setting("COLOR_HOLE_MIN_SHARE", 0.003)):
        return None
    d0 = np.linalg.norm(L - target, axis=2)
    cand = [p for p in paints if not (is_white(p) and p.startswith("white"))]   # a white on white paper does nothing
    if not cand:
        return None
    D = np.stack([np.linalg.norm(lab_array(now * fac[p][0] + fac[p][1]) - target, axis=2) for p in cand])
    top = D.argmin(axis=0)                               # every hole: the paint that suits it best
    g_ = [float(np.where(h & (top == k) & (D[k] < d0), d0 - D[k], 0.0).sum()) for k in range(len(cand))]
    k = int(np.argmax(g_))
    if g_[k] <= 0:
        return None
    bp, best = cand[k], g_[k]
    ok = h & (top == k) & (D[k] < d0)
    mm_px = (190.0 / max(Hp, Wp)) * Hp / now.shape[0]
    rr = max(1, int(round(float(setting("COLOR_HOLE_GROW_MM", 0.7)) / mm_px)))
    ok = cv2.dilate(ok.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rr + 1, 2 * rr + 1))) > 0
    ok = clean(ok, Hp, Wp, 1.0)
    if ok.mean() < float(setting("COLOR_HOLE_MIN_SHARE", 0.003)):
        return None
    g = float(best / ok.size)
    if g < float(setting("COLOR_HOLE_MIN_GAIN", 0.3)):  # a pass and a drying for 0.05 (2026-11-22): not worth it
        return None
    return {"paint": bp, "mask": ok, "gain": g, "expect": g, "share": float(ok.mean()), "fill": 1, "holes": True, "smooth": 1.0}


def efficiency(passes):
    """HONEST PROMISES: how much of what its passes promised each paint really did on this sheet (the colour error measured
    before and after each pass): {paint: share, "*": all paints}. A paint with few passes leans on the share of all of them.
    2026-11-16 (run of 20:38): the detail passes promised 11.4 and did 4.1 - thin lines cover less than the plan, and the
    end never came because the promises stayed high."""
    lo, hi = setting("COLOR_TRUST_RANGE", (0.15, 1.2))
    w, k = int(setting("COLOR_TRUST_WINDOW", 6)), float(setting("COLOR_TRUST_PRIOR", 1.0))
    got = [q for q in passes if q.get("real") is not None and q.get("gain", 0) > 0]
    rec = got[-w:]
    G = sum(q["gain"] for q in rec)
    g = 1.0 if G < 0.5 else float(np.clip(sum(q["real"] for q in rec) / G, lo, hi))
    eff = {"*": g}
    for p in {q["paint"] for q in got}:
        mine = [q for q in got if q["paint"] == p][-w:]
        eff[p] = float(np.clip((sum(q["real"] for q in mine) + k * g) / (sum(q["gain"] for q in mine) + k), lo, hi))
    return eff


def stalled(passes, de_now):
    """The END BY THE CAMERA: the last COLOR_STOP_PATIENCE passes (fill-ins not counted) did not make the sheet better than
    the best it has been (by COLOR_STOP_MIN_GAIN). -> (stalled?, best dE, after how many passes it was best).
    2026-11-16: the best was after pass 25 (13.7), passes 26-30 only made it worse (14.8)."""
    pat, ming = int(setting("COLOR_STOP_PATIENCE", 3)), float(setting("COLOR_STOP_MIN_GAIN", 0.1))
    des = [q.get("dE") for q in passes] + [de_now]
    best, bi = None, 0
    for j, d in enumerate(des):
        if d is not None and (best is None or d < best - ming):
            best, bi = d, j
    since = sum(1 for q in passes[bi:] if not q.get("fill"))
    last = next((q for q in reversed(passes) if not q.get("fill")), None)
    if last is not None and passes[-1] is last and len(des) >= 2 and des[-2] is not None \
            and des[-2] - de_now >= float(setting("COLOR_STOP_STILL_GAIN", 0.5)):
        return False, best, bi                       # the last pass still made it clearly better: go on (2026-11-17)
    return bool(pat) and since >= pat, best, bi


# --------------------------------------------------------------------------------------------------------- forecast
def archive_last_run():
    """The files of the previous run (plan, pass pictures, photos, G-code) go to ko/cp_runs/<when it was made>/ (kept, not
    deleted). ko/cp_measured.json stays: the paint values for the next run."""
    keep = (os.path.basename(MEASURED),)
    files = [f for f in os.listdir(DIR) if f.startswith("cp_") and f not in keep and os.path.isfile(os.path.join(DIR, f))]
    if not files:
        return
    try:
        stamp = json.load(open(PLAN)).get("made", "").replace(":", "").replace(" ", "_")
    except (OSError, ValueError):
        stamp = ""
    dest = os.path.join(DIR, "cp_runs", stamp or time.strftime("%Y-%m-%d_%H%M"))
    os.makedirs(dest, exist_ok=True)
    for f in files:
        shutil.move(os.path.join(DIR, f), os.path.join(dest, f))
    print(f"the previous run's files -> {os.path.relpath(dest, HERE)}")


def make(image, paints=None, dark=45.0):
    """The forecast: the same choice of passes, on what the paints are expected to do (no camera). ko/cp_preview.png."""
    paints = [p for p in ORDER if p in (paints or ORDER)]
    img = cv2.imread(image)
    if img is None:
        raise SystemExit(f"cannot read {image}")
    h, w = img.shape[:2]
    Hp, Wp = PLAN_H, int(round(w * PLAN_H / h))
    strip = float(setting("COLOR_TEST_STRIP_MM", 0.0) or 0.0)
    plan = {"image": os.path.abspath(image), "paints": paints, "passes": [], "made": time.strftime("%Y-%m-%d %H:%M"),
            "margin_mm": MARGIN_MM, "dark": dark, "pic": [Hp, Wp],
            # room over the picture for the test strokes (kept in the plan: every pass and every photo place it the same)
            "test_room_mm": strip + float(setting("COLOR_TEST_GAP_MM", 6.0)) if strip > 0 else 0.0}
    (Hw, Ww), _ = work_shape(plan)
    target = target_lab(img, dark, (Hw, Ww))
    import color_planes as CPL                                     # DEPTH PLANES (ComfyUI, made once per picture)
    L_full, centres = CPL.compute(image)
    set_planes(None if L_full is None else cv2.resize(L_full, (Ww, Hw), interpolation=cv2.INTER_NEAREST), Hp, Wp,
               focus_map(img, (Hw, Ww)))
    try:                                                           # THE DRAWING for the finish (TEED, made once per picture)
        import color_lines as CL
        lm_ = CL.line_map(image)
        print(f"color_paint: the picture's drawing: {100 * lm_.mean():.1f}% of it on a line (for the edge passes at the end)")
    except Exception as e:
        print(f"(the picture's drawing not made now: {e}; it is made at the end of the run)")
    fac = {p: coat_factor(p, {}) for p in paints}
    now, done, steps, fc = np.ones((Hw, Ww, 3)), {}, [], []
    t0 = time.time()
    while len(fc) < min(max_passes(), 80):
        c = choose(now, target, done, fac, paints, Hp, Wp, smooth=per_stage("COLOR_MASK_SMOOTH_MM", 4.0, len(fc) + 1),
                   prefer=per_stage("COLOR_PREFER", "glaze", len(fc) + 1))
        if c is None:
            break
        now = apply(now, c, fac)
        done[c["paint"]] = done.get(c["paint"], 0) + 1
        fc.append({"paint": c["paint"], "show": PAINTS[c["paint"]]["show"], "coat": done[c["paint"]],
                   "share": round(c["share"], 3), "gain": round(c["gain"], 2)})
        steps.append(to_bgr(now))
    de = float(np.median(np.linalg.norm(lab_array(now) - target, axis=2)))
    plan["forecast"], plan["forecast_dE"] = fc, round(de, 1)
    os.makedirs(DIR, exist_ok=True)
    archive_last_run()
    if L_full is not None:
        CPL.save(image, L_full, centres)
    json.dump(plan, open(PLAN, "w"), indent=1)
    sm = lambda x: cv2.resize(x, (Ww * 2 // 3, Hw * 2 // 3), interpolation=cv2.INTER_AREA)
    tiles = [sm(img)] + [sm(s) for s in steps]
    labels = ["original"] + [f"{i + 1}: {q['show']} ({100 * q['share']:.0f}%)" for i, q in enumerate(fc)]
    cells = []
    for t, l in zip(tiles, labels):
        top = np.full((26, t.shape[1], 3), 255, np.uint8)
        cv2.putText(top, l, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cells.append(np.vstack([top, t]))
    while len(cells) % 5:
        cells.append(np.full_like(cells[0], 255))
    cv2.imwrite(os.path.join(DIR, "cp_preview.png"), np.vstack([np.hstack(cells[i:i + 5]) for i in range(0, len(cells), 5)]))
    print(f"color_paint: forecast for {os.path.basename(image)} with {', '.join(PAINTS[p]['show'] for p in paints)}: "
          f"about {len(fc)} passes, colour error at the end dE {plan['forecast_dE']} (median)  [{time.time() - t0:.0f} s]")
    for i, q in enumerate(fc, 1):
        print(f"   pass {i:2d}: {q['show']:8s} ({100 * q['share']:.0f}% of the picture, nearer by {q['gain']:.2f})")
    print("   ko/cp_preview.png: the original, then the painting after each pass")
    print("   the real passes are chosen from photos of the dry sheet: their number and order can differ")
    return plan


# ------------------------------------------------------------------------------------------------------ the sheet
def camera_points(plan, paper_file=PAPER, upright=None):
    """Where every cell of the planning grid (and a ring of bare paper round the picture) is in the camera picture: the same
    placement paint.py gives the passes (the sheet of pass 1, the signature's room, turned by 90 if it fits larger).
    upright = points in upright artwork mm instead: -> (their camera px, (the picture's upright width, height) in mm)."""
    import paint as PT
    import vision as V
    (Hw, Ww), (Hp, Wp) = work_shape(plan)
    det, pf = V.load_paper(paper_file)
    bottom = None
    if not bool(getattr(S, "SIGN_INSIDE", False)) and PT.signature_file():
        import signature as SG
        _, sh = SG.block_size_mm(S.SIGNATURE, float(S.SIGN_WIDTH_MM), bool(getattr(S, "SIGN_YEAR", True)),
                                 year_h_mm=float(getattr(S, "SIGN_YEAR_MM", 4.0)))
        bottom = float(S.SIGN_GAP_MM) + sh + float(S.SIGN_EDGE_MM)
    pl = V.fit_artwork(pf, Wp, Hp, margin_mm=plan.get("margin_mm", MARGIN_MM), rotate="auto", bottom_mm=bottom,
                       top_mm=plan.get("test_room_mm") or None)
    s = pl["scale_mm_per_px"]
    to_mm = PT.machine_mapper({"placement": pl})
    cal = V.load_calibration()

    def to_px(cols, rows):                                    # picture pixels -> camera pixels
        cols, rows = np.asarray(cols, float), np.asarray(rows, float)
        x, y = (rows * s, cols * s) if pl["rotate"] == 90 else (cols * s, (Hp - rows) * s)
        mx, my = to_mm(x, y)
        return V.mm_to_px_array(cal["H_inv"], np.c_[np.ravel(mx), np.ravel(my)]).reshape(*np.shape(cols), 2)
    if upright is not None:                                   # points in upright artwork mm (x right, y up) -> camera px
        P = np.asarray(upright, float)
        return to_px(P[..., 0] / s, Hp - P[..., 1] / s).astype(np.float32), (Wp * s, Hp * s)
    C, R = np.meshgrid((np.arange(Ww) + 0.5) * Wp / Ww, (np.arange(Hw) + 0.5) * Hp / Hw)
    grid = to_px(C, R)
    # bare paper round the picture (5.5 mm outside it, inside the margin): the exposure of each photo is set by it
    k = 5.5 / s
    ring = np.concatenate([to_px(np.linspace(0, Wp, 60), np.full(60, -k)), to_px(np.linspace(0, Wp, 60), np.full(60, Hp + k)),
                           to_px(np.full(60, -k), np.linspace(0, Hp, 60)), to_px(np.full(60, Wp + k), np.linspace(0, Hp, 60))])
    return grid.astype(np.float32), ring.astype(np.float32)


def eye(img):
    """A camera photo in linear light, blurred only against the camera's noise (COLOR_EYE_BLUR_PX). 2026-11-19: blurred twice
    (2 camera px, then 2 grid px = ~2-3 mm) the planner saw neither the 1-3 mm white holes (white paper among pink dots read
    as the grey of the shirt) nor the paints' real strength: opaque black was believed L 45 (the photo: L 29), opaque green
    L 84 (64) - so black went where a mid grey was wanted (black blobs) and the whole target was made pale."""
    b = float(setting("COLOR_EYE_BLUR_PX", 0.8))
    lin = CC.srgb_to_lin(img[..., ::-1]).astype(np.float32)
    return cv2.GaussianBlur(lin, (0, 0), b) if b > 0 else lin


def sheet_now(photo, plan, blank=None, paper_file=PAPER):
    """What the sheet reflects now on the planning grid, per colour channel, the bare paper = 1: the photo divided by the photo
    of the blank sheet (this takes out uneven light and the paper's own tone), its exposure set on the bare paper round it."""
    blank = cv2.imread(BLANK) if blank is None else blank
    grid, ring = camera_points(plan, paper_file)
    lb, lp = eye(blank), eye(photo)
    samp = lambda im, pts: cv2.remap(im, pts[..., 0], pts[..., 1], cv2.INTER_LINEAR)
    r = samp(lp, grid) / np.maximum(samp(lb, grid), 1e-3)
    rr = samp(lp, ring[None]) / np.maximum(samp(lb, ring[None]), 1e-3)
    r = r / np.median(rr.reshape(-1, 3), axis=0)                  # the bare paper round it = 1 (another exposure, other light)
    gb = float(setting("COLOR_EYE_GRID_BLUR", 0.0))
    return np.clip(cv2.GaussianBlur(r, (0, 0), gb) if gb > 0 else r, 0.02, 1.05)


def learn(q, before, now, state):
    """What pass q really did: the share of the light it let through where it painted (after / before), from now on that
    paint's value (this run, and the next run's start value)."""
    m = mask_of(q["picture"], now.shape[:2])
    m = cv2.erode(m.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0       # away from the edges (placement, bleeding)
    if m.sum() < 200:
        return None
    p = q["paint"]
    if bool(setting("COLOR_LEARN_LANDED", True)):
        # WHERE THE PAINT LANDED: a pass covers its area only partly (gaps between strokes, dabs); the spots it left are
        # not its colour (they are found as holes and filled). Landed = changed at least half as much as the most changed.
        ch_ = np.abs(now - before).max(axis=2)
        thr = max(0.5 * float(np.percentile(ch_[m], 80)), float(setting("COLOR_LEARN_MIN_CHANGE", 0.02)))
        landed = m & (ch_ >= thr)
        if landed.sum() >= 100:
            print(f"   pass {q['picture']} ({q['show']}): the paint landed on {100 * landed.sum() / max(m.sum(), 1):.0f}% of its area "
                  f"- learned from there")
            m = landed
    change = float(np.median(np.abs(now[m] - before[m])))
    if change < float(setting("COLOR_LEARN_MIN_CHANGE", 0.02)):
        print(f"\n   *** pass {q['picture']} ({q['show']}) changed the sheet almost NOT AT ALL (median {change:.3f}): an empty cup, "
              f"or the brush took no paint? Its value is NOT learned from this pass. Look at the cup. ***\n")
        return None
    if not is_white(p):
        t, r = np.clip(np.median((now / np.maximum(before, 1e-3))[m], axis=0), 0.03, 1.0), np.zeros(3)
    else:                                               # a white: after = r + t x before, per colour of light
        t0 = np.array(PAINTS[p]["t"], float)                 # (a covering paint: whites and opaque colours)
        b, a = before[m], now[m]
        t, r = np.empty(3), np.empty(3)
        for ch in range(3):
            if float(b[:, ch].std()) > 0.06:            # different tones under it: both numbers can be fitted
                A = np.c_[b[:, ch], np.ones(len(b))]
                t[ch], r[ch] = np.linalg.lstsq(A, a[:, ch], rcond=None)[0]
            else:                                       # all on one tone (paper): its t as guessed, r from what it gave
                t[ch] = t0[ch]
                r[ch] = float(np.median(a[:, ch] - t0[ch] * b[:, ch]))
        t, r = np.clip(t, 0.0, 1.0), np.clip(r, 0.0, 1.05)
    state.setdefault("learned", {})[p] = {"t": [round(float(v), 4) for v in t], "r": [round(float(v), 4) for v in r]}
    T_ = state.get("tests", {}).get(p)
    if not is_white(p) and T_ and T_.get("sane") and T_.get("t"):         # this paint's test stroke against its real pass
        d_test = -np.log(np.clip(np.array(T_["t"], float), 0.03, 1.0))
        d_pass = -np.log(np.clip(t, 0.03, 1.0))
        k_ = d_test > 0.05
        if k_.any():
            obs = state.setdefault("test_scale_obs", {})
            obs[p] = round(float(np.clip(np.median(d_pass[k_] / d_test[k_]), 1.0, 8.0)), 2)
            state["test_scale"] = round(float(np.median(list(obs.values()))), 2)
            print(f"   {label_of(p)}: its pass is {obs[p]:.1f}x as strong as its test stroke said; a paint known only from "
                  f"its test stroke may be {state['test_scale']:.1f}x as strong - worth trying (COLOR_TEST_DENSITY_SCALE)")
    try:
        kept = json.load(open(MEASURED))
    except (OSError, ValueError):
        kept = {}
    kept[p] = state["learned"][p]
    json.dump(kept, open(MEASURED, "w"), indent=1)
    return t, r


def look_at_sheet(i):
    import paint as PT
    import vision as V
    cap = V.open_camera()
    try:
        photo, _ = PT.capture_fresh(cap)
    finally:
        cap.release()
    cv2.imwrite(os.path.join(DIR, f"cp_before_pass{i:02d}.png"), photo)
    return photo


# --------------------------------------------------------------------------------------------------------- paint
def cup_of(paint):
    """The palette's cup of this paint (None without a palette or when it is empty)."""
    try:
        import cups as CUPS
        cs = CUPS.load()
        return CUPS.cup_for(cs, PAINTS[paint]["row"], PAINTS[paint].get("kind", "transparent")) if cs else None
    except Exception:
        return None


# ------------------------------------------------------------------------------------------------- test strokes
def test_strokes_for(i, plan, q, state, fac):
    """The test strokes this pass paints (paint.py TEST_STROKES): its own paint's, when it is used for the first time, and
    after pass COLOR_TEST_EXTRA_AFTER those of every paint never measured anywhere (an extra dip and rinse each)."""
    if not plan.get("test_room_mm"):
        return []
    tests = state.setdefault("tests", {})
    paints, out = plan["paints"], []
    seq = 1 + max([t.get("order", 0) for t in tests.values()] or [0])

    def add(p, own):
        nonlocal seq
        slot = len(tests)                                       # in the order of use: / and \ by turns
        tests[p] = {"slot": slot, "pass": i, "order": seq}
        seq += 1
        c = None if own else cup_of(p)
        out.append({"label": f"test {label_of(p)}", "bgr": paint_bgr(p, fac), "cup": None if own else (c or {}).get("n"),
                    "slot": slot, "n": len(paints)})
    if q["paint"] not in tests and not q.get("fill"):
        add(q["paint"], True)
    if i == 1 and bool(setting("COLOR_TEST_ALL_AT_START", True)):
        # A SWATCH OF EVERY PAINT FIRST, like a brushographer: every cup with paint gets its band in pass 1, each on bare
        # paper, so every paint is measured as it is TODAY before the choice uses it (2026-11-17: opaque magenta went on with
        # an old value from another mix and turned the whole figure pink; yellow had none and was read off its neighbour)
        for p in sorted(paints, key=lambda p: ORDER.index(p)):
            if p not in tests and cup_of(p) is not None:
                add(p, False)
    n_real = sum(1 for x in plan["passes"][:i] if not x.get("fill"))
    extra_after = int(setting("COLOR_TEST_EXTRA_AFTER", 0) or 0)
    if extra_after and n_real == extra_after and not q.get("fill"):
        try:
            kept = json.load(open(MEASURED))
        except (OSError, ValueError):
            kept = {}
        for p in paints:
            if p not in tests and p not in kept and p not in state.get("learned", {}) and cup_of(p) is not None:
                add(p, False)
    return out


def refl_at(photo, plan, upright, blank=None):
    """What the sheet reflects at points given in upright artwork mm (..., 2) -> (..., 3), the bare paper = 1 (as sheet_now)."""
    blank = cv2.imread(BLANK) if blank is None else blank
    pts, _ = camera_points(plan, upright=upright)
    _, ring = camera_points(plan)
    lb, lp = eye(blank), eye(photo)
    P = pts.reshape(1, -1, 2)
    samp = lambda im, pp: cv2.remap(im, pp[..., 0], pp[..., 1], cv2.INTER_LINEAR)
    r = samp(lp, P) / np.maximum(samp(lb, P), 1e-3)
    rr = samp(lp, ring[None]) / np.maximum(samp(lb, ring[None]), 1e-3)
    r = r / np.median(rr.reshape(-1, 3), axis=0)
    return r.reshape(*np.shape(upright)[:-1], 3)


def _band_hw(u):
    """Half the width of a test band at u (0 = fat end .. 1 = thin end): the stroke's own half width + the band's other lines."""
    nl, gp = int(setting("COLOR_TEST_LINES", 1) or 1), float(setting("COLOR_TEST_LINE_GAP_MM", 1.2))
    return float(setting("COLOR_TEST_FAT_MM", 3.5)) * (1.0 - np.asarray(u, float)) / 2.0 + (nl - 1) / 2.0 * gp


def _bare(p, lines):
    """The points of test band p that lie on bare paper: clear of every other band (their widths taken into account)."""
    from scipy.spatial import cKDTree
    A = lines[p]
    u = np.linspace(0, 1, len(A))
    ok = np.ones(len(A), bool)
    for o, B in lines.items():
        if o != p:
            d, j = cKDTree(B).query(A)
            ok &= d > _band_hw(u) + _band_hw(np.linspace(0, 1, len(B)))[j] + 1.5   # (strokes paint ~1 mm wider than planned)
    return ok


def _centre_offset(photo, plan, A, ok, blank):
    """How far beside its planned line test band A really lies (mm, across it): the camera's placement there can be off by a
    millimetre or two (2026-11-16: the samples missed the stroke, opaque green read 0.97 = paper). None = not found."""
    d = (A[-1] - A[0]) / max(np.linalg.norm(A[-1] - A[0]), 1e-6)
    perp = np.array([-d[1], d[0]])
    idx = np.flatnonzero(ok)
    if len(idx) < 4:
        return None
    s_ = np.linspace(-2.5, 2.5, 26)                             # (the strip's own shift is found first: _strip_shift)
    offs = []
    for k in idx[:: max(1, len(idx) // 8)]:
        prof = A[k][None] + s_[:, None] * perp[None]
        r = refl_at(photo, plan, prof[None], blank)[0]
        dark = np.abs(1.0 - r).max(axis=1)                     # any change from paper (a white over paper is not found)
        if dark.max() < 0.04:
            continue
        hi = dark > 0.5 * dark.max()                            # the dark run nearest the planned line, not a neighbour's
        lab_ = np.cumsum(np.r_[hi[0], hi[1:] & ~hi[:-1]]) * hi
        runs = [np.flatnonzero(lab_ == k) for k in range(1, lab_.max() + 1)]
        run = min(runs, key=lambda r_: float(np.abs(s_[r_]).min()))
        offs.append(float((dark[run] * s_[run]).sum() / dark[run].sum()))
    return (float(np.median(offs)), perp) if offs else None


def sane(p, t, r, pairs):
    """Whether values measured on a test stroke can be real (else they are kept in cp_tests.json but not used). 2026-11-16:
    the stroke was missed and opaque cyan got t (0, .15, 1) r (1.05, .79, 0): brighter than paper, and it was chosen."""
    t, r = np.asarray(t, float), np.asarray(r, float)
    if (r > 1.0).any() or (t + r > 1.03).any():
        return False, "brighter than the paper"
    if is_white(p) and (t > 0.8).any():
        return False, "a covering paint that lets nearly all light through"
    on = [np.asarray(a_) for b_, a_, w_ in pairs if w_ == "paper"]
    if not is_white(p) and ((on and float((1.0 - on[0]).max()) < 0.03) or float(t.min()) > 0.97):
        return False, "nothing seen (stroke missed?)"
    if PAINTS[p].get("row") == "white" and all(w_ == "paper" for _b, _a, w_ in pairs):
        return False, "a white on white paper shows nothing (its first pass measures it)"     # 2026-11-17: taken as
                                                                                               # "lightens a lot" -> 57% white
    if on and PAINTS[p].get("row") != "white" and float((1.0 - on[0]).max()) < 0.03:
        return False, "nothing seen on the paper (no band painted?)"           # 2026-11-17: the yellow band was not there
    if on and PAINTS[p].get("row") != "white":                  # ITS OWN COLOUR? the band's hue vs its cup's (the camera)
        a = lab_array(np.clip(on[0], 0, 1)[None])[0]
        c = lab_array(np.clip(cup_colour(p), 0, 1)[None])[0]
        ca, cc = float(np.hypot(a[1], a[2])), float(np.hypot(c[1], c[2]))
        if ca > 12 and cc > 12:
            dh = abs((np.degrees(np.arctan2(a[2], a[1]) - np.arctan2(c[2], c[1])) + 180.0) % 360.0 - 180.0)
            if dh > float(setting("COLOR_TEST_HUE_MAX", 40.0)):
                return False, f"not the colour of its cup (hue {dh:.0f} deg off: a neighbour's band?)"
    return True, ""


def _strip_shift(photo, plan, lines, bare, fac, blank):
    """Where the test strip really is: the shift (upright mm) that puts every band's planned line on paint of ITS colour
    (one coat of it on paper, as far as it is known). 2026-11-16: out at the sheet's edge the strip lay ~4 mm off, and the
    samples of opaque green read the yellow band beside it. Whites (not seen on paper) do not vote. -> (dx, dy) mm."""
    use = [p for p in lines if p in fac and float(np.abs(1.0 - (fac[p][1] + fac[p][0])).max()) > 0.1]
    if not use:
        return 0.0, 0.0
    pts = {p: lines[p][bare[p] & (np.linspace(0, 1, len(lines[p])) > 0.03) & (np.linspace(0, 1, len(lines[p])) < 0.45)]
           for p in use}
    pts = {p: v for p, v in pts.items() if len(v) >= 4}
    if not pts:
        return 0.0, 0.0
    c0 = np.mean(np.vstack(list(pts.values())), axis=0)
    J0, _ = camera_points(plan, upright=np.array([[c0, c0 + (1, 0), c0 + (0, 1)]]))
    J0 = J0.reshape(3, 2)
    J = np.c_[J0[1] - J0[0], J0[2] - J0[0]]                       # camera px per upright mm, near the strip
    lb = cv2.GaussianBlur(CC.srgb_to_lin(blank[..., ::-1]).astype(np.float32), (0, 0), 2.0)
    lp = cv2.GaussianBlur(CC.srgb_to_lin(photo[..., ::-1]).astype(np.float32), (0, 0), 2.0)
    _, ring = camera_points(plan)
    samp = lambda im, pp: cv2.remap(im, pp[..., 0].astype(np.float32), pp[..., 1].astype(np.float32), cv2.INTER_LINEAR)
    norm = np.median((samp(lp, ring[None]) / np.maximum(samp(lb, ring[None]), 1e-3)).reshape(-1, 3), axis=0)
    px = {p: camera_points(plan, upright=v[None])[0].reshape(-1, 2) for p, v in pts.items()}
    want = {p: lab_array(np.clip(fac[p][1] + fac[p][0], 0, 1)[None])[0] for p in pts}
    best, arg = None, (0.0, 0.0)
    grid = np.arange(-8.0, 8.01, 0.5)
    for dx in grid:
        for dy in grid:
            off = J @ np.array([dx, dy])
            sc = 0.0
            for p, q in px.items():
                qq = (q + off)[None]
                r = samp(lp, qq)[0] / np.maximum(samp(lb, qq)[0], 1e-3) / norm
                sc += float(np.linalg.norm(lab_array(np.median(r, axis=0)[None])[0] - want[p]))
            if best is None or sc < best:
                best, arg = sc, (float(dx), float(dy))
    return arg


def measure_tests(photo, plan, state, fac):
    """The test bands not measured yet, on the first dry photo after they were painted: each paint on bare paper and over
    every band it crosses that was painted before it (before -> after: its t and r on this sheet), and its width along the
    ramp (fat end = deepest). Each band is found across its planned line first (the camera's placement can be off by a mm or
    two). A paint no pass has measured yet takes these values when they can be real (sane). Kept in ko/cp_tests.json."""
    import paint as PT
    tests = state.get("tests", {})
    todo = [p for p, t in tests.items() if "measured" not in t]
    if not todo:
        return
    _, (uw, uh) = camera_points(plan, upright=np.zeros((1, 2)))
    n = len(plan["paints"])
    lines = {p: PT.test_line(t["slot"], n, uw, uh) for p, t in tests.items()}
    blank = cv2.imread(BLANK)
    bare = {p: _bare(p, lines) for p in lines}
    dx, dy = _strip_shift(photo, plan, lines, bare, fac, blank)   # where the strip really is (by the bands' colours)
    lines = {p: A + np.array([dx, dy])[None] for p, A in lines.items()}
    shift = {p: _centre_offset(photo, plan, lines[p], bare[p], blank) for p in lines}   # then each band across itself
    found = [v[0] for v in shift.values() if v]
    common = float(np.median(found)) if found else 0.0           # a band not found (a white on paper): as the others
    for p, A in list(lines.items()):
        d = (A[-1] - A[0]) / max(np.linalg.norm(A[-1] - A[0]), 1e-6)
        o, perp = shift[p] if shift[p] else (common, np.array([-d[1], d[0]]))
        lines[p] = A + o * perp[None]
    print(f"\n   test strokes (the strip found {dx:+.1f}, {dy:+.1f} mm from its planned place, each band {common:+.1f} mm across):")
    for p in todo:
        T = tests[p]
        A = lines[p]
        L = np.linalg.norm(A[-1] - A[0])
        d_dir = (A[-1] - A[0]) / max(L, 1e-6)
        us = np.linspace(0, 1, len(A))
        cross = []                                              # (position along this band 0..1, the other paint, its point)
        for o, B in lines.items():
            if o == p:
                continue
            M = np.array([A[-1] - A[0], -(B[-1] - B[0])]).T
            try:
                u_, v_ = np.linalg.solve(M, B[0] - A[0])
            except np.linalg.LinAlgError:
                continue
            if 0 <= u_ <= 1 and 0 <= v_ <= 1:
                cross.append((u_, o, A[0] + u_ * (A[-1] - A[0]), B, v_))
        pts = A[bare[p] & (us > 0.03) & (us < 0.6)]             # the fat part on bare paper
        pairs = []
        if len(pts) >= 6:
            a = np.median(refl_at(photo, plan, pts[None], blank)[0], axis=0)
            pairs.append((np.ones(3), a, "paper"))
        ahead = (T["pass"], T.get("order", 0))
        for u_, o, X, B, v_ in cross:
            if u_ > 0.8:                                         # the thin end: too narrow to sit on the crossing
                continue
            if (tests[o]["pass"], tests[o].get("order", 0)) >= ahead:
                continue                                         # that one was painted after this one: it lies on top
            ring = np.array([X + rr * np.array([np.cos(a_), np.sin(a_)]) for rr in (0.0, 0.6, 1.2)
                             for a_ in np.linspace(0, 2 * np.pi, 6, endpoint=False)])
            after = np.median(refl_at(photo, plan, ring[None], blank)[0], axis=0)
            LB = max(np.linalg.norm(B[-1] - B[0]), 1e-6)
            eb = (B[-1] - B[0]) / LB
            sin_ = max(abs(float(d_dir[0] * eb[1] - d_dir[1] * eb[0])), 0.1)
            d0 = (float(_band_hw(u_)) + 1.5) / sin_              # along the other band, out of this one's width
            dds = [sgn * dd for sgn in (-1, 1) for dd in (d0, d0 + 1.0, d0 + 2.0)]
            side = np.array([X + dd * eb for dd in dds if 0.0 < v_ + dd / LB < 0.85])
            if len(side) < 2:
                continue
            before = np.median(refl_at(photo, plan, side[None], blank)[0], axis=0)
            pairs.append((before, after, o))
        widths = []
        perp = np.array([-d_dir[1], d_dir[0]])
        for u_ in np.linspace(0.05, 0.95, 10):                   # the band's width along the ramp, where it is on bare paper
            if not bare[p][int(round(u_ * (len(A) - 1)))]:
                continue
            X = A[0] + u_ * (A[-1] - A[0])
            prof = X[None] + np.linspace(-6, 6, 61)[:, None] * perp[None]
            r = refl_at(photo, plan, prof[None], blank)[0]
            dark = 1.0 - (0.2126 * r[:, 0] + 0.7152 * r[:, 1] + 0.0722 * r[:, 2])
            if dark.max() > 0.04:
                widths.append((round(float(u_), 2), round(float((dark > 0.5 * dark.max()).sum() * 0.2), 2)))
        T["measured"] = {"pairs": [{"under": w_, "before": [round(float(v), 4) for v in b],
                                    "after": [round(float(v), 4) for v in a]} for b, a, w_ in pairs],
                         "widths_along": widths, "offset_mm": round(float(shift[p][0]), 2) if shift[p] else None}
        if not pairs:
            print(f"   {PAINTS[p]['show']}: its test stroke was not found (no bare part, no crossing)")
            continue
        B_ = np.array([b for b, _, _ in pairs])
        A_ = np.array([a for _, a, _ in pairs])
        if not is_white(p):
            t, r = np.clip(np.median(A_ / np.maximum(B_, 1e-3), axis=0), 0.03, 1.0), np.zeros(3)
        else:
            t0 = np.array(PAINTS[p]["t"], float)
            t, r = np.empty(3), np.empty(3)
            for ch in range(3):
                if len(B_) >= 3 and float(B_[:, ch].std()) > 0.06:
                    t[ch], r[ch] = np.linalg.lstsq(np.c_[B_[:, ch], np.ones(len(B_))], A_[:, ch], rcond=None)[0]
                else:
                    t[ch] = t0[ch]
                    r[ch] = float(np.median(A_[:, ch] - t0[ch] * B_[:, ch]))
            t, r = np.clip(t, 0.0, 1.0), np.clip(r, 0.0, 1.05)
        tol = float(setting("COLOR_TEST_BRIGHT_TOL", 0.15))       # a bright paint read a little over the paper (gloss,
        if (t + r).max() <= 1.0 + tol and r.max() <= 1.0 + tol:   # the camera): trimmed to paper white, not thrown out
            r = np.minimum(r, np.maximum(1.0 - t, 0.0))                # (2026-11-17: opaque yellow, cyan, magenta rejected)
        T["t"], T["r"] = [round(float(v), 4) for v in t], [round(float(v), 4) for v in r]
        ok, why = sane(p, t, r, pairs)
        T["sane"] = ok
        used = ok and p not in state.get("learned", {})
        if used and bool(setting("COLOR_KEEP_LAST_SHEET", True)):
            # WHAT THE LAST SHEET'S PASSES MEASURED GOES BEFORE A TEST STROKE (2026-11-23, user: "every new picture, the
            # values from before are there but not used"): a test band reads weaker than a real pass (1.2-3.7x), so it
            # replaced good pass values with worse ones at every new run. Kept - unless the band's colour is clearly another
            # (the cup was filled with a different mix): then the band, as before (2026-11-17: the pink figure)
            try:
                old_ = json.load(open(MEASURED)).get(p)
            except (OSError, ValueError):
                old_ = None
            if old_ and old_.get("from") != "test stroke" and _tr(old_) is not None:
                to_, ro_ = _tr(old_)
                c_new = lab_array((t + r)[None])[0]
                c_old = lab_array((to_ + ro_)[None])[0]
                ch_n, ch_o = float(np.hypot(*c_new[1:])), float(np.hypot(*c_old[1:]))
                dh = abs((np.degrees(np.arctan2(c_new[2], c_new[1]) - np.arctan2(c_old[2], c_old[1])) + 180.0) % 360.0 - 180.0)
                if not (ch_n > 12.0 and ch_o > 12.0 and dh > float(setting("COLOR_TEST_HUE_MAX", 40.0))):
                    used = False
                    print(f"   {PAINTS[p]['show']}: the last sheet's passes measured it - that value stays (its band agrees in hue)")
                else:
                    print(f"   {PAINTS[p]['show']}: its band is another colour than the last sheet's ({dh:.0f} deg) - a new mix: the band is used")
        if used:                                                 # no pass of it measured yet: the choice uses these
            state.setdefault("learned", {})[p] = {"t": T["t"], "r": T["r"], "from": "test stroke"}
            fac[p] = (t, r)
            try:
                kept = json.load(open(MEASURED))
            except (OSError, ValueError):
                kept = {}
            if p not in kept:
                kept[p] = {"t": T["t"], "r": T["r"], "from": "test stroke"}
                json.dump(kept, open(MEASURED, "w"), indent=1)
        print(f"   {PAINTS[p]['show']:15s} t {' '.join(f'{v:.2f}' for v in t)}  r {' '.join(f'{v:.2f}' for v in r)}  "
              f"from {len(pairs)} spots ({', '.join(str(w_) for _, _, w_ in pairs)})"
              + ("  -> used from now on" if used else "") + (f"  -> NOT used: {why}" if not ok else "")
              + (f"   band width fat -> thin: {' '.join(f'{w:.1f}' for _, w in widths)} mm" if widths else ""))
    json.dump(tests, open(os.path.join(DIR, "cp_tests.json"), "w"), indent=1)


# ------------------------------------------------------------------------------------------------- the drying view
class DryView:
    """A window while the sheet dries: the camera, a countdown (with the minutes you added), and the last pass's picture
    (seen | target | the pass) in a corner. In the h4xx0r theme when it is on. + / - / Enter work here as in the terminal."""
    def __init__(self, i, plan):
        self.i, self.plan, self.cap, self.win = i, plan, None, "brushography://drying  [+] more  [-] less  [Enter] go on  [Esc] close"
        self.pip = cv2.imread(os.path.join(DIR, f"cp_replan_pass{i - 1:02d}.png"))
        self.last, self.frame, self.off = 0.0, None, False
        try:
            import h4xx0r as HX
            self.hx = HX if (HX.ON or bool(setting("H4XX0R", False))) and HX.has_font() else None
            self.glass = HX.Glass() if self.hx else None
        except Exception:
            self.hx = self.glass = None

    def __call__(self, left_s, added):
        if self.off or not bool(setting("COLOR_DRY_VIEW", True)):
            return None
        if self.cap is None:
            import vision as V
            self.cap = V.open_camera(verbose=False)
        if time.time() - self.last > 0.5 or self.frame is None:
            ok, fr = self.cap.read()
            if ok:
                self.frame = fr
            self.last = time.time()
        if self.frame is None:
            return None
        h, w = self.frame.shape[:2]
        img = cv2.resize(self.frame, (1280, int(round(1280 * h / w))), interpolation=cv2.INTER_AREA)
        if self.pip is not None:                                   # the last pass: seen | target | what it painted
            ph = int(round(600 * self.pip.shape[0] / self.pip.shape[1]))
            pip = cv2.resize(self.pip, (600, ph), interpolation=cv2.INTER_AREA)
            y0 = img.shape[0] - 40 - ph
            img[y0:y0 + ph, 16:16 + 600] = pip
            cv2.rectangle(img, (15, y0 - 1), (16 + 600, y0 + ph), (58, 158, 46), 1)
        m, s_ = divmod(max(0, int(left_s)), 60)
        title = f"DRYING {m:02d}:{s_:02d}" + (f"  ({added:+g} min)" if added else "")
        prev = self.plan["passes"][self.i - 2] if len(self.plan["passes"]) >= self.i - 1 else {}
        lines = [(f"before pass {self.i}: the dry sheet is photographed, then the next pass is chosen", "ink"),
                 (f"last pass: {prev.get('show', '-')}" + (f", fill-in {prev['fill']}" if prev.get("fill") else
                                                           f", coat {prev.get('coat', '-')}") +
                  (f", dE before it {prev['dE']:.1f}" if prev.get("dE") is not None else ""), "dim"),
                 ("[+] / [P] 5 min more   [-] 5 min less   [Enter] go on now   [Esc] close this window", "dim")]
        out = None
        if self.hx:
            out = self.hx.screen(img, title, lines, clock=time.strftime("%H:%M:%S"), t=time.time(), top_h=44 + 24 * len(lines))
            if out is not None:
                out = self.glass.apply(out)
        if out is None:
            out = img
            cv2.putText(out, title, (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
            for k, (t, _) in enumerate(lines):
                cv2.putText(out, t, (16, 64 + 24 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 1, cv2.LINE_AA)
        cv2.imshow(self.win, out)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:                                                # Esc: the window closes, the drying goes on
            self.off = True
            self.close()
            print("   (drying window closed - the countdown goes on here)")
            return None
        return {13: "\r", 10: "\r", ord("+"): "+", ord("="): "+", ord("p"): "p", ord("P"): "p", ord("-"): "-",
                ord("r"): "r", ord("R"): "r"}.get(k)

    def close(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        try:
            cv2.destroyWindow(self.win)
            cv2.waitKey(1)
        except Exception:
            pass


# --------------------------------------------------------------------------------------------------------- paint
def cup_arg(paint):
    """The cup of the 12-cup palette that holds this paint (its row, opaque or transparent): paint.py --cup N. Without a
    palette (CUP_FILE = None): the paint's dish is found by colour as before."""
    if setting("CUP_FILE", "cups.json") is None:
        return []
    import cups as CUPS
    cs = CUPS.load()
    if not cs:
        return ["--cups", CUPS_KIND]
    c = CUPS.cup_for(cs, PAINTS[paint]["row"], PAINTS[paint].get("kind", "transparent"))
    if c is None:
        raise SystemExit(f"color_paint: no filled {PAINTS[paint].get('kind', 'transparent')} cup with {PAINTS[paint]['show']} "
                         f"in the palette (python cups.py detect, cups_detect.png)")
    print(f"   {PAINTS[paint]['show']}: cup {c['n']} ({c['kind']})")
    return ["--cup", str(c["n"])]


_PLAIN = {}                         # the settings as settings.py has them, before a glaze pass changed them


def run_pass(i, plan, extra, first=False, last=False, info=None, tests=()):
    import settings
    # the signature's room is kept in EVERY pass (it decides where the picture sits), painted only in the last one
    settings.SIGN_PAINT = bool(last)
    q = plan["passes"][i - 1]
    style = q.get("style") or per_stage("COLOR_INFILL_STYLE", None, i)   # broad first (bands), detail last (centre lines)
    if style:
        settings.INFILL_STYLE = style
    import paint as PT
    import probe as PB
    argv = [os.path.join("ko", q["picture"]), "--rotate", "auto", "--margin", f"{plan['margin_mm']:g}", "--color", "auto", "--yes"] + cup_arg(q["paint"]) + extra
    probe_file = os.path.join(HERE, "probe_result.json")
    if i > 1 and os.path.isfile(probe_file) and "--probe_file" not in extra:
        argv += ["--probe_file", probe_file,                         # the same sheet: the probe of pass 1, no new lines -
                 "--probe_max_age_h", f"{float(setting('COLOR_PROBE_MAX_AGE_H', 48.0)):g}"]   # valid for the whole run
                                                                     # (2026-11-15: the 6 h limit stopped the run before pass 6)
    argv += ["--paper_keep", PAPER, "--scene_keep", SCENE]           # sheet, dishes, rinse container: as found before pass 1
    settings.RINSE_BEFORE_JOB = first                                # the first pass of a run: the brush comes from the probe's
                                                                     # paint, or from a stopped pass (Ctrl+C skips its rinse)
    settings.REPAIR_AT_END = REPAIR_PASSES
    glaze = not is_white(q["paint"])                                 # a transparent paint: every overlap shows as a darker line
    for k_, v_ in dict(setting("COLOR_GLAZE_OVERRIDES", {}) or {}).items():
        if k_ not in _PLAIN:
            _PLAIN[k_] = getattr(settings, k_, None)
        setattr(settings, k_, v_ if glaze else _PLAIN[k_])
    PB.PROBE_COLOR = PROBE_PAINT                                     # a light paint alone (yellow) cannot be probed: use this
    PT.first_layer_color = lambda *a_, **k_: PROBE_PAINT            # (paint.py would pick the pass's own paint)
    PT.JOB_INFO = info                                               # the real paint's name and colour, the run, for the overlay
    PT.TEST_STROKES = list(tests)
    PT.TEST_ROOM_MM = float(plan.get("test_room_mm") or 0.0)
    _say("narrate_pass", i, q["show"], q["share"], fill=bool(q.get("fill")), last=last, stage=q.get("stage"))
    print(f"\n{'=' * 100}\ncolor_paint: PASS {i}{' (the last one, with the signature)' if last else ''}: {q['show']}, "
          + (f"fill-in {q['fill']} of coat {q['coat']}" if q.get("fill") else f"coat {q['coat']}")
          + f", {100 * q['share']:.0f}% of the picture, {q.get('stage', '')} ({style or 'INFILL_STYLE'}, smooth {q.get('smooth', 0):g} mm)"
          + (f", test strokes: {', '.join(t['label'] for t in tests)}" if tests else "")
          + f"   ({time.strftime('%H:%M')})\n   paint.py " + " ".join(argv))
    PT.LAST_JOB_DONE = False
    try:
        PT.main(argv)
    finally:
        PT.JOB_INFO, PT.TEST_STROKES = None, []
    if not PT.LAST_JOB_DONE:
        raise SystemExit(f"\npass {i} was NOT painted to the end (the reason is above): stopped. "
                         f"Go on with this pass:  python color_paint.py paint {i}")
    final = os.path.join(HERE, "paint_final.png")
    if os.path.isfile(final):
        shutil.copyfile(final, os.path.join(DIR, f"cp_after_pass{i:02d}.png"))


def job_info(i, plan, q, fac):
    """What paint.py's overlay shows of the run: the pass, its real paint and colour, its cup, the passes so far as chips,
    the colour error before each pass."""
    c = cup_of(q["paint"])
    n_real = sum(1 for x in plan["passes"][:i] if not x.get("fill"))
    status = (f"PASS {i} ({n_real}) - {q['show']} - "
              + (f"fill-in {q['fill']}" if q.get("fill") else f"coat {q['coat']}") + f" - {100 * q['share']:.0f}% - {q.get('stage', '')}")
    t_, r_ = fac[q["paint"]]
    return {"label": label_of(q["paint"]), "coat": {"t": [float(v) for v in t_], "r": [float(v) for v in r_]},   # brush check
            "bgr": paint_bgr(q["paint"], fac), "cup": (c or {}).get("n"), "kind": (c or {}).get("kind"),
            "status": status, "chips": [(paint_bgr(x["paint"], fac) if x["paint"] in fac else (128, 128, 128), k == i - 1)
                                        for k, x in enumerate(plan["passes"][:i])],
            "de": [x.get("dE") for x in plan["passes"][:i]], "infill": q.get("style"), "smooth": q.get("smooth")}


def dry_and_look(i, plan, dry_min):
    """The drying before pass i, then the photo of the sheet. THE CAMERA CHECKS IT IS DRY: a photo COLOR_DRY_CHECK_MIN before
    the end of the countdown and one at its end; while the last pass's area still changes between them (more than
    COLOR_DRY_STILL - wet paint dries lighter or darker), it waits that long again (at most COLOR_DRY_CHECK_MAX times). Enter
    in the countdown = go on at once, no check. -> the photo (saved as ko/cp_before_passNN.png)."""
    check, still = float(setting("COLOR_DRY_CHECK_MIN", 0.0) or 0.0), float(setting("COLOR_DRY_STILL", 0.01))
    can = check > 0 and ADAPTIVE and os.path.isfile(BLANK) and os.path.isfile(PAPER)

    def wait(minutes, what):
        _say("narrate_dry", minutes)
        view = DryView(i, plan)
        try:
            return CC.wait_dry(minutes, what, screen=view)
        finally:
            view.close()
    if not can:
        wait(dry_min, f"pass {i}")
        return look_at_sheet(i)
    if wait(max(dry_min - check, 0.0), f"pass {i}"):
        return look_at_sheet(i)                                       # Enter: go on now
    m = mask_of(plan["passes"][i - 2]["picture"], work_shape(plan)[0])
    prev = sheet_now(look_at_sheet(i), plan)
    for k in range(int(setting("COLOR_DRY_CHECK_MAX", 6)) + 1):
        if wait(check, f"pass {i} (the camera checks the sheet is dry)"):
            return look_at_sheet(i)
        photo = look_at_sheet(i)
        cur = sheet_now(photo, plan)
        d = np.abs(cur - prev).mean(axis=2)                           # the change in the last pass's area beyond the
        ch = float(np.median(d[m]) - np.median(d[~m])) if m.any() and (~m).any() else 0.0   # camera's noise (bare spots)
        if ch < still:
            print(f"   dry: the last pass's area changed {ch:.3f} in {check:g} min (under {still:g})")
            return photo
        print(f"   STILL DRYING: the last pass's area changed {ch:.3f} in {check:g} min (over {still:g}) - waiting {check:g} min more")
        prev = cur
    print("   still changing, but the most checks are done: going on")
    return photo


def check_cup(q, faint, learned):
    """THE PAINT CHECK: a pass that left a large share of its own area unchanged (faint >= COLOR_DIP_FAINT), or changed the
    sheet almost not at all, had too little paint on the brush. Its cup is dipped CUP_DEEPER_STEP_MM deeper from now on
    (remembered in cups.json, until  python cups.py refill N); a cup already at its lowest dip has to be refilled."""
    lim = float(setting("COLOR_DIP_FAINT", 0.25) or 0.0)
    if lim <= 0 or q.get("fill") or (faint < lim and learned):
        return
    c = cup_of(q["paint"])
    if not c:
        return
    import cups as CUPS
    msg = CUPS.go_deeper(c["n"])
    print(f"   PAINT CHECK: pass {q['show']} left {100 * faint:.0f}% of its area unchanged -> cup {c['n']}: {msg}")
    _say("narrate_cup", c["n"], q["show"])
    q["faint"] = round(faint, 2)


def keep_blank_if_same_sheet():
    """THE BLANK SHEET WHEN PASS 1 IS RESUMED (2026-11-22: "paint 1" again after a stop took the sheet WITH the aborted paint
    as the white reference). The new blank photo against the one of the run before (kept aside as cp_blank_prev.png): the
    sheet in the same place, most of it unchanged, and a few percent changed like paint -> the same sheet: its first,
    really blank photo stays the reference. A fresh sheet (almost nothing changed) or one placed elsewhere keeps the new photo."""
    if not os.path.isfile(BLANK_OLD):
        return
    try:
        if time.time() - os.path.getmtime(BLANK_OLD) > 3600.0 * float(setting("COLOR_SAME_SHEET_MAX_H", 48.0)):
            return
        new, old = cv2.imread(BLANK), cv2.imread(BLANK_OLD)
        if new is None or old is None or new.shape != old.shape:
            return
        gn, go = (cv2.GaussianBlur(cv2.cvtColor(x, cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 2) for x in (new, old))
        paper = go > np.percentile(go, 60)                          # the light part of the old frame: the sheet
        ratio = gn[paper] / np.maximum(go[paper], 1.0)
        ratio = ratio / np.median(ratio)                            # the light of the day differs: only what changed locally
        changed = float((np.abs(ratio - 1.0) > 0.12).mean())
        lo, hi = float(setting("COLOR_SAME_SHEET_MIN", 0.004)), float(setting("COLOR_SAME_SHEET_MAX", 0.2))
        if lo <= changed <= hi:
            shutil.copyfile(BLANK_OLD, BLANK)
            print(f"color_paint: {100 * changed:.1f}% of the sheet already painted - pass 1 RESUMED on the same sheet: its first "
                  f"blank photo stays the white reference")
        else:
            print(f"color_paint: the blank sheet: {100 * changed:.1f}% differs from the last run's - a new sheet, the new photo")
    except Exception as e:
        print(f"(blank sheet check: {e})")


def finish_pass(now, target, done, fac, paints, Hp, Wp, plan, eff, lw):
    """THE FINISH (2026-11-23, user: "it should keep correcting itself, put the edges right, bring in the lights where they
    are needed"; color_lines.py): when no colour pass helps any more, the passes that ARTICULATE the picture -
      edge:  along the picture's own lines (TEED), only where the sheet is lighter than the picture wants there by more than
             COLOR_EDGE_DL: the paint that brings those spots nearest (no white), as centre lines (the black method);
      light: the picture's light accents (lighter than all round it) where the sheet is darker by more than COLOR_LIGHTS_SHEET_DL:
             a white, small.
    Each kind until COLOR_EDGE_PASSES_MAX / COLOR_LIGHT_PASSES_MAX, or until its last two passes together did not help (the
    camera). The kind with fewer passes goes first, edges on a tie. -> a pass (with kind, style, smooth) or None."""
    if not bool(setting("COLOR_FINISH", True)):
        return None
    H_, W_ = now.shape[:2]
    mm_px = (190.0 / max(Hp, Wp)) * Hp / H_
    fin = [x for x in plan["passes"] if x.get("kind") in ("edge", "light") and not x.get("fill")]
    lmax = {"edge": int(setting("COLOR_EDGE_PASSES_MAX", 6)), "light": int(setting("COLOR_LIGHT_PASSES_MAX", 3))}

    def stopped(kind):
        xs = [x for x in fin if x["kind"] == kind]
        if len(xs) >= lmax[kind]:
            return True
        r = [x.get("real") for x in xs[-2:]]
        # a line makes form, not always a smaller colour error: the kind stops when its last two passes together made the
        # sheet clearly WORSE (COLOR_FINISH_MAX_LOSS, the camera), not when they only did not help
        return len(r) == 2 and None not in r and sum(r) < -float(setting("COLOR_FINISH_MAX_LOSS", 1.0))
    import color_lines as CL
    Lnow = lab_array(now.reshape(-1, 3))[:, 0].reshape(H_, W_)
    dL = Lnow - target[..., 0]                                         # + = the sheet lighter than the picture
    min_share = float(setting("COLOR_FINISH_MIN_SHARE", 0.002))

    def along(m, v, mm):
        """v averaged along the mask's own pixels within about mm (a line judged as a whole, not spot by spot)."""
        s_ = max(1.0, mm / mm_px)
        num = cv2.GaussianBlur(np.where(m, v, 0.0).astype(np.float32), (0, 0), s_)
        den = cv2.GaussianBlur(m.astype(np.float32), (0, 0), s_)
        return num / np.maximum(den, 1e-3)
    # THE CONTRAST, NOT THE LIGHTNESS (2026-11-23, run K: the face on the sheet was DARKER than the picture, L 69 vs 82, yet
    # without a single line - a mush of one mid value; "draw where the sheet is too light" drew nothing there). A line is
    # missing where the picture is darker than its surroundings and the sheet is not (by COLOR_EDGE_DL); a light is missing
    # where the picture is lighter than its surroundings and the sheet is not. That is what the eye reads as form.
    s_ar = max(1.0, float(setting("COLOR_LIGHTS_AROUND_MM", 3.0)) / mm_px)
    Lt = target[..., 0].astype(np.float32)
    ct_t = cv2.GaussianBlur(Lt, (0, 0), s_ar) - Lt                     # + = darker than round it (a line), - = a light
    ct_s = cv2.GaussianBlur(Lnow.astype(np.float32), (0, 0), s_ar) - Lnow.astype(np.float32)
    kinds = sorted(("edge", "light"), key=lambda k: (sum(1 for x in fin if x["kind"] == k), k != "edge"))
    for kind in kinds:
        if stopped(kind):
            continue
        if kind == "edge":
            if _CTX.get("lines") is None or _CTX["lines"].shape != (H_, W_):
                _CTX["lines"] = CL.lines_on_grid(plan["image"], (H_, W_), mm_px)
            ln_ = _CTX["lines"]
            miss_ = (ct_t - ct_s) + np.maximum(dL, 0.0)                    # the line's contrast missing (+ too light overall)
            need = ln_ & (ct_t > 1.0) & (along(ln_, miss_, float(setting("COLOR_EDGE_ALONG_MM", 3.0))) > float(setting("COLOR_EDGE_DL", 4.0)))
            need = clean(need, Hp, Wp, 0.0, min_mm=float(setting("COLOR_EDGE_MIN_MM", 2.0)))
            use = [p for p in paints if not is_white(p) and lum(fac[p][0]) < float(setting("COLOR_EDGE_PAINT_LUM", 0.8))]
        else:
            lt_ = CL.local_lights(target[..., 0], mm_px)
            miss_ = (ct_s - ct_t) - np.minimum(dL, 0.0)                    # the light's contrast missing (+ too dark overall)
            need = lt_ & (along(lt_, miss_, 1.5) > float(setting("COLOR_LIGHTS_SHEET_DL", 8.0)))
            fo_ = _CTX.get("focus")                                      # the lights where the eye goes (figure, face,
            if fo_ is not None and fo_.shape == need.shape:               # strong contrasts) - white spattered over the
                need &= fo_ >= float(setting("COLOR_LIGHTS_FOCUS_MIN", 0.3))   # hills made them milky (test of 11-23)
            need = clean(need, Hp, Wp, 0.0, min_mm=float(setting("COLOR_FINISH_MIN_MM", 1.0)))
            use = [p for p in paints if p.startswith("white")]
        if not use or need.mean() < min_share:
            print(f"   finish: no {kind} pass needed ({100 * need.mean():.2f}% of the picture wants one)")
            continue
        # ONE PAINT FOR THE WHOLE DRAWING (spot by spot broke the lines into specks): of the paints that make the contrast
        # (dark glazes for a line, whites for a light), the one whose colour comes nearest the picture there
        flat, tg = now.reshape(-1, 3), target.reshape(-1, 3)
        mi = need.ravel()
        d0 = wdist(lab_array(flat[mi]), tg[mi], lw)
        best_ = None
        for p in use:
            g_ = float((d0 - wdist(lab_array(flat[mi] * fac[p][0] + fac[p][1]), tg[mi], lw)).sum() / len(flat))
            if best_ is None or g_ > best_[1]:
                best_ = (p, g_)
        c = {"paint": best_[0], "mask": need, "gain": best_[1], "expect": best_[1], "share": float(need.mean())} if best_ else None
        if c is None:
            print(f"   finish: the {kind} spots ({100 * need.mean():.1f}% of the picture) - no paint helps them enough")
            continue
        c.update(kind=kind, style="skeleton", smooth=0.0)
        n_ = sum(1 for x in fin if x["kind"] == kind) + 1
        print(f"   FINISH - {'EDGES: the picture lines where the sheet is too light' if kind == 'edge' else 'LIGHTS: the light accents the sheet lost'}"
              f" ({kind} pass {n_} of at most {lmax[kind]}): {label_of(c['paint'])} on {100 * c['share']:.1f}% of the picture")
        return c
    return None


def paint(start, extra, dry_min):
    if not os.path.isfile(PLAN):
        raise SystemExit("no ko/cp_plan.json: run  python color_paint.py make <picture>  first")
    plan = json.load(open(PLAN))
    if "pic" not in plan or "forecast" not in plan:
        raise SystemExit("ko/cp_plan.json is from the older fixed-pass version: make it again (python color_paint.py make <picture>)")
    paints, (Hw, Ww), (Hp, Wp) = plan["paints"], *work_shape(plan)
    max_passes_ = max_passes()
    fill_tries = int(setting("COLOR_FILL_TRIES", 0) or 0)
    t0 = time.time()
    if start == 1:                                                    # a new run = a new sheet: find everything again
        if os.path.isfile(BLANK):                                     # (kept aside: pass 1 RESUMED on the same sheet)
            shutil.copyfile(BLANK, BLANK_OLD)
        for f in (PAPER, SCENE, BLANK, R_PREV, STATE):
            if os.path.isfile(f):
                os.remove(f)
        state = {}
        if bool(setting("COLOR_REFILL_AT_START", True)):              # a new run = the cups filled up: dips start again on top
            try:
                import cups as CUPS
                if CUPS.refill():
                    print(f"cups: a new run - every cup's dips start again at {setting('CUP_DIP_START_MM', 5.0)} mm over its floor "
                          f"(COLOR_REFILL_AT_START)")
            except Exception as e:
                print(f"(cups not reset: {e})")
    else:
        if not os.path.isfile(PAPER):
            print("NOTE: no ko/cp_paper.json from pass 1: the sheet is detected now (works only while most of it is still white)")
        state = json.load(open(STATE)) if os.path.isfile(STATE) else {}
    plan["passes"] = plan["passes"][:start - 1]
    dark_now = float(plan.get("dark", 45.0))
    target = target_lab(cv2.imread(plan["image"]), dark_now, (Hw, Ww))
    predicted = np.ones((Hw, Ww, 3))                                  # the sheet as expected (without the camera)
    i = start
    chain = 0                                                         # passes painted since the last photo, without waiting
    _DE_NOW[0] = None
    focus, nf = focus_map(cv2.imread(plan["image"]), (Hw, Ww))
    print(f"color_paint: {nf} face(s) found in the picture: the detail passes weigh them most (COLOR_FOCUS_WEIGHT)")
    import color_planes as CPL                                     # DEPTH PLANES (made by make; none = as before)
    fig_p = set_planes(CPL.load((Hw, Ww), plan["image"]), Hp, Wp, (focus, nf))
    if fig_p is not None:                                          # the whole figure counts, not only the face
        focus = np.maximum(focus, float(setting("COLOR_PLANE_FOCUS", 0.5)) * (_PLANES["L"] == fig_p))
        try:                                                       # the narrator says where a pass's paint goes
            inf_ = json.load(open(CPL.INFO)); inf_["figure"] = int(fig_p); json.dump(inf_, open(CPL.INFO, "w"), indent=1)
        except (OSError, ValueError):
            pass
    narrator = None
    if bool(setting("NARRATOR_ON", True)) and bool(setting("VOICE_ON", True)):
        try:                                                       # THE ČOPIČOGRAF TALKS while it paints (narrator_watch.py,
            import subprocess                                      # its own program: it only reads, never disturbs the run)
            narrator = subprocess.Popen([sys.executable, os.path.join(HERE, "narrator_watch.py")], cwd=HERE)
        except Exception as e:
            print(f"(narrator not started: {e})")
    import atexit
    atexit.register(lambda: narrator is not None and narrator.poll() is None and narrator.terminate())
    _CTX.update(recent_white=None, late=False, figure=None, lines=None, focus=None)
    if fig_p is not None:                                          # the figure's inside (its edges stay free for opaques)
        mm_px0 = (190.0 / max(Hp, Wp)) * Hp / Hw
        r0 = max(1, int(round(float(setting("COLOR_LATE_OPAQUE_FIGURE_EDGE_MM", 2.0)) / mm_px0)))
        _CTX["figure"] = cv2.erode((_PLANES["L"] == fig_p).astype(np.uint8),
                                   cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r0 + 1, 2 * r0 + 1))) > 0
    _CTX["focus"] = focus
    while True:
        fac = {p: coat_factor(p, state) for p in paints}
        rw_n = int(setting("COLOR_DARK_AFTER_WHITE", 3) or 0)        # the whites of the last passes: no dark glaze on them
        _CTX["recent_white"] = None
        for x in (plan["passes"][-rw_n:] if rw_n > 0 else []):
            if x["paint"].startswith("white"):
                mk_ = mask_of(x["picture"], (Hw, Ww))
                _CTX["recent_white"] = mk_ if _CTX["recent_white"] is None else (_CTX["recent_white"] | mk_)
        _CTX["late"] = stage_of(sum(1 for x in plan["passes"] if not x.get("fill")) + 1) >= 2
        fill_c = None
        de_now = None
        nowait_c = None
        if (i > start and i >= 2 and bool(setting("COLOR_NO_WAIT", True)) and ADAPTIVE and plan["passes"]
                and not plan["passes"][-1].get("fill") and not state.get("signed") and not state.get("finishing")
                and chain < int(setting("COLOR_NO_WAIT_CHAIN", 2))):
            # THE NEXT PASS WITHOUT WAITING when it touches only dry parts: chosen on the sheet as expected (the model), it
            # must keep COLOR_WET_MARGIN_MM away from the last pass's area; then no drying, no photo - the last pass dries
            # meanwhile and both are measured at the next photo (2026-11-18: the detail passes waited 25 min each for nothing)
            n_r = sum(1 for x in plan["passes"] if not x.get("fill"))
            if n_r < max_passes_:
                ag = np.full((Hw, Ww), 99, np.int16)
                for k_, x in enumerate(reversed(plan["passes"][-3:])):
                    mk = mask_of(x["picture"], (Hw, Ww))
                    ag[mk & (ag == 99)] = k_
                cand = choose(predicted, target, done_of(plan["passes"]), fac, paints, Hp, Wp,
                              smooth=per_stage("COLOR_MASK_SMOOTH_MM", 4.0, n_r + 1), prefer=per_stage("COLOR_PREFER", "glaze", n_r + 1),
                              eff=efficiency(plan["passes"]), lw=float(per_stage("COLOR_LIGHTNESS_WEIGHT", 1.0, n_r + 1)), age=ag,
                              focus=focus, fw=float(per_stage("COLOR_FOCUS_WEIGHT", 0.0, n_r + 1)))
                if cand is not None:
                    # PAINT ELSEWHERE WHILE IT DRIES (2026-11-22, user: "more than half of the sheet is still empty"): the
                    # wet area = every pass since the last photo, grown by COLOR_NO_WAIT_GAP_MM so wet never meets wet
                    mm_px_ = (190.0 / max(Hp, Wp)) * Hp / Hw
                    r_ = max(1, int(round(float(setting("COLOR_NO_WAIT_GAP_MM", 1.5)) / mm_px_)))
                    wet = cv2.dilate(((ag >= 0) & (ag <= chain)).astype(np.uint8),
                                     cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r_ + 1, 2 * r_ + 1))) > 0
                    if not (cand["mask"] & wet).any():
                        nowait_c = cand
                    else:                                    # the best pass that stays on dry or bare paper
                        dry = choose(predicted, target, done_of(plan["passes"]), fac, paints, Hp, Wp,
                                     smooth=per_stage("COLOR_MASK_SMOOTH_MM", 4.0, n_r + 1),
                                     prefer=per_stage("COLOR_PREFER", "glaze", n_r + 1), eff=efficiency(plan["passes"]),
                                     lw=float(per_stage("COLOR_LIGHTNESS_WEIGHT", 1.0, n_r + 1)), age=ag, focus=focus,
                                     fw=float(per_stage("COLOR_FOCUS_WEIGHT", 0.0, n_r + 1)), forbid=wet)
                        if dry is not None and dry["gain"] >= float(setting("COLOR_NO_WAIT_SHARE", 0.3)) * cand["gain"]:
                            nowait_c = dry
                            print(f"   while pass {i - 1} dries: {label_of(dry['paint'])} on the dry and bare parts "
                                  f"({100 * dry['share']:.0f}% of the picture, {dry['gain']:.2f} vs {cand['gain']:.2f} for the best pass)")
        if nowait_c is not None:
            chain += 1
            now = predicted
            de_now = float(np.median(np.linalg.norm(lab_array(now) - target, axis=2)))
            print(f"\ncolor_paint: pass {i} touches only dry parts (none of the last pass's area): NO WAIT, no photo - "
                  f"the sheet as expected (dE ~{de_now:.1f}); measured at the next photo")
        elif i > 1:
            chain = 0
            if ADAPTIVE and os.path.isfile(BLANK) and os.path.isfile(PAPER):
                dmin_ = dry_min                                # GOUACHE DRIES LONGER (2026-11-23): a covering paint since
                since_ = plan["passes"][int(state.get("learned_upto", 0)):]   # the last photo: COLOR_DRY_MIN_OPAQUE
                if any(is_white(x["paint"]) for x in since_):
                    dmin_ = max(dry_min, float(setting("COLOR_DRY_MIN_OPAQUE", 40.0)))
                    if dmin_ > dry_min:
                        print(f"   a covering paint was painted: {dmin_:g} min of drying (COLOR_DRY_MIN_OPAQUE)")
                photo = dry_and_look(i, plan, dmin_) if i > start else look_at_sheet(i)
                print(f"\ncolor_paint: looking at the dry sheet before pass {i}")
                now = sheet_now(photo, plan)
                de_now = float(np.median(np.linalg.norm(lab_array(now) - target, axis=2)))
                q0 = plan["passes"][i - 2]
                first_ = int(state.get("learned_upto", 0)) + 1
                run_ = plan["passes"][first_ - 1:i - 1]
                if len(run_) > 1 and run_[0].get("dE") is not None:   # painted one after another without a photo:
                    tot_ = run_[0]["dE"] - de_now                     # their real gain shared by their promises
                    G_ = sum(max(x.get("gain", 0.0), 1e-6) for x in run_)
                    for x in run_:
                        x["real"] = round(tot_ * max(x.get("gain", 0.0), 1e-6) / G_, 2)
                    print(f"   passes {first_}-{i - 1} (no photo between them) together did {tot_:+.2f}")
                    json.dump(plan, open(PLAN, "w"), indent=1)
                elif q0.get("dE") is not None:                        # what the last pass REALLY did vs its promise
                    q0["real"] = round(q0["dE"] - de_now, 2)
                    print(f"   pass {i - 1} ({q0['show']}{', fill-in' if q0.get('fill') else ''}) promised {q0.get('gain', 0):.2f}"
                          + (f" (expected {q0['expect']:.2f})" if q0.get("expect") is not None else "")
                          + f", really did {q0['real']:+.2f}")
                    json.dump(plan, open(PLAN, "w"), indent=1)
                if state.get("learned_upto", 0) < i - 1:
                    before = np.load(R_PREV) if os.path.isfile(R_PREV) else np.ones_like(now)
                    for x in plan["passes"][first_ - 1:i - 2]:       # the earlier passes of a no-wait chain
                        if not x.get("fill"):
                            g_ = learn(x, before, now, state)
                            if g_ is not None:
                                fac[x["paint"]] = g_
                    got = None if q0.get("fill") else learn(q0, before, now, state)   # a fill-in: its spots only, not learned
                    state["learned_upto"] = i - 1
                    if got is not None:
                        t, r = got
                        exp_t, exp_r = fac[q0["paint"]]
                        print(f"   pass {i - 1} ({q0['show']}) really let through R {t[0]:.2f} G {t[1]:.2f} B {t[2]:.2f} of the light "
                              f"(expected {' '.join(f'{v:.2f}' for v in exp_t)})"
                              + (f", and added R {r[0]:.2f} G {r[1]:.2f} B {r[2]:.2f} (expected {' '.join(f'{v:.2f}' for v in exp_r)})"
                                 if is_white(q0["paint"]) else ""))
                        fac[q0["paint"]] = (t, r)
                    try:
                        measure_tests(photo, plan, state, fac)        # the test strokes painted with the last pass
                    except Exception as e:                           # (they must never stop the run)
                        print(f"   (test strokes not measured: {e})")
                    miss, gain, faint = missed(q0, before, now, fac, target, Hp, Wp)
                    try:
                        check_cup(q0, faint, got is not None)         # too little paint on the brush: that cup deeper
                    except Exception as e:
                        print(f"   (paint check not done: {e})")
                    # FINISH THE COLOUR before the next: spots of the last pass that stayed unchanged get the same paint again,
                    # when it is worth a pass (COLOR_FILL_MIN_SHARE, COLOR_FILL_MIN_GAIN after what this paint really does)
                    kept_ = (q0["real"] / q0["gain"]) if q0.get("real") is not None and q0.get("gain", 0) > 0 else 1.0
                    if miss is not None and kept_ < float(setting("COLOR_FILL_MIN_KEPT", 0.3)):
                        print(f"   pass {i - 1} ({q0['show']}) kept only {100 * kept_:.0f}% of its promise: no fill-in with it")
                        miss = None
                    if fill_tries and int(q0.get("fill", 0)) < fill_tries and miss is not None:
                        e_ = efficiency(plan["passes"])
                        expect = gain * float(e_.get(q0["paint"], e_["*"]))
                        if expect >= float(setting("COLOR_FILL_MIN_GAIN", 0.15)):
                            fill_c = {"paint": q0["paint"], "mask": miss, "gain": gain, "expect": expect,
                                      "share": float(miss.mean()), "fill": int(q0.get("fill", 0)) + 1, "smooth": 1.0}
                            print(f"   pass {i - 1} ({q0['show']}) left {100 * miss.mean():.1f}% of the picture unpainted: "
                                  f"the same paint again there (fill-in {fill_c['fill']} of at most {fill_tries}, expected {expect:.2f})")
                        else:
                            print(f"   pass {i - 1} ({q0['show']}) left {100 * miss.mean():.1f}% unpainted - a fill-in would bring "
                                  f"only {expect:.2f}: not worth a pass")
                    json.dump(plan, open(PLAN, "w"), indent=1)
                np.save(R_PREV, now)
            else:
                if i > start:
                    view = DryView(i, plan)
                    try:
                        CC.wait_dry(dry_min, f"pass {i}", screen=view)
                    finally:
                        view.close()
                if ADAPTIVE:
                    print(f"NOTE: no photo of the blank sheet (ko/cp_blank.png): pass {i} is chosen from the forecast")
                now = predicted
        else:
            now = np.ones((Hw, Ww, 3))
        json.dump(state, open(STATE, "w"), indent=1)
        if de_now is None:
            de_now = float(np.median(np.linalg.norm(lab_array(now) - target, axis=2)))
        print(f"   the sheet vs the picture: dE {de_now:.1f} (median)")
        _DE_NOW[0] = de_now if _DE_NOW[0] is None else min(_DE_NOW[0], de_now)   # the best so far: stages never go back
        done = done_of(plan["passes"])
        n_real = sum(1 for x in plan["passes"] if not x.get("fill"))  # passes so far, fill-ins not counted
        smooth = per_stage("COLOR_MASK_SMOOTH_MM", 4.0, n_real + 1)
        prefer = per_stage("COLOR_PREFER", "glaze", n_real + 1)
        eff = efficiency(plan["passes"])
        lw = float(per_stage("COLOR_LIGHTNESS_WEIGHT", 1.0, n_real + 1))
        age = None
        over_white = None
        if plan["passes"]:                                             # how many passes ago each spot was painted
            age = np.full((Hw, Ww), 99, np.int16)
            for k_, x in enumerate(reversed(plan["passes"][-3:])):
                mk = mask_of(x["picture"], (Hw, Ww))
                age[mk & (age == 99)] = k_
                if is_white(x["paint"]) and x["paint"].startswith("white"):
                    over_white = np.zeros((Hw, Ww), bool) if over_white is None else over_white
                    over_white |= mk & (age == k_)
            if nowait_c is not None or fill_c is not None:
                over_white = None                                  # (no photo after drying: the white may be wet)
        if bool(setting("COLOR_REACHABLE_DARK", True)) and "dark" not in state and any(
                t_.get("sane") for t_ in state.get("tests", {}).values()):
            # the target's darkest: what the paints can reach, from the SWATCHES (on bare paper), once per run; a pass's
            # learned value is no measure for it (2026-11-17 replay: black over wet green read as grey -> L 71)
            sw = {p_: (np.array(t_["t"], float), np.array(t_["r"], float)) for p_, t_ in state["tests"].items() if t_.get("sane")}
            rd = reachable_dark(sw, list(sw)) + 2.0
            state["dark"] = min(float(plan.get("dark", 45.0)) + float(setting("COLOR_REACHABLE_MAX_LIFT", 10.0)),
                                max(float(plan.get("dark", 45.0)), rd))
        if "dark" in state:
            want_dark = float(state["dark"])
            rd = want_dark
            if abs(want_dark - dark_now) > 0.5:
                print(f"   the swatches reach about L {want_dark - 2:.0f} at the darkest: the picture's darkest becomes L {want_dark:.0f} "
                      f"(was {dark_now:.0f}; COLOR_REACHABLE_DARK, at most +{float(setting('COLOR_REACHABLE_MAX_LIFT', 10.0)):g})")
                dark_now = want_dark
                target = target_lab(cv2.imread(plan["image"]), dark_now, (Hw, Ww))
                de_now = float(np.median(np.linalg.norm(lab_array(now) - target, axis=2)))
        if plan["passes"]:
            print(f"   the passes so far kept {100 * eff['*']:.0f}% of their promises"
                  + "".join(f", {PAINTS[p]['show']} {100 * v:.0f}%" for p, v in eff.items() if p != "*"))
        last = False
        fw = float(per_stage("COLOR_FOCUS_WEIGHT", 0.0, n_real + 1))
        hole_c = None
        if (nowait_c is None and fill_c is None and i > 1 and ADAPTIVE and not state.get("signed")
                and stage_of(n_real + 1) >= int(setting("COLOR_HOLE_FROM_STAGE", 1))
                and int(state.get("hole_passes", 0)) < int(setting("COLOR_HOLE_PASSES_MAX", 4))):
            painted = np.zeros((Hw, Ww), bool)
            for x in plan["passes"]:
                painted |= mask_of(x["picture"], (Hw, Ww))
            hole_c = holes(now, target, fac, paints, Hp, Wp, painted=painted)
        if nowait_c is not None:
            c = nowait_c
        elif fill_c is not None:
            c = fill_c
        elif state.get("signed"):                                      # the last real pass is painted (and its fill-ins)
            break
        else:
            stop, best, bi = stalled(plan["passes"], de_now) if plan["passes"] else (False, None, 0)
            use = paints
            if stop:                                                  # SOMETHING ELSE: the paints of the passes that did not
                pat = int(setting("COLOR_STOP_PATIENCE", 3))          # help rest, the others get their turn (2026-11-17: the
                rest = {x["paint"] for x in [x for x in plan["passes"] if not x.get("fill")][-pat:]}   # run ended at dE 24
                use = [p for p in paints if p not in rest]            # with the pink figure uncovered, white never tried)
                print(f"   the sheet was best (dE {best:.1f}) after pass {bi}; the {pat} passes since did not make it better: "
                      f"something else - {', '.join(PAINTS[p]['show'] for p in sorted(rest))} rest")
            st_now = STAGE_NAMES[min(stage_of(n_real + 1), len(STAGE_NAMES) - 1)]
            n_white = sum(1 for x in plan["passes"] if not x.get("fill") and is_white(x["paint"]) and x["paint"].startswith("white")
                          and x.get("stage") == st_now)
            if n_white >= int(setting("COLOR_WHITE_PER_STAGE", 1)) and any(p.startswith("white") for p in use):
                use = [p for p in use if not p.startswith("white")]   # (2026-11-20, run H: 3 white passes on 45% - a milky figure,
                print(f"   {n_white} white pass(es) in the {st_now} stage already: no more white in it (COLOR_WHITE_PER_STAGE)")  # no glaze over it)
            c = None if n_real >= max_passes_ or not use else choose(now, target, done, fac, use, Hp, Wp, smooth=smooth,
                                                                     prefer=prefer, eff=eff, lw=lw, age=age, focus=focus, fw=fw,
                                                                     over_white=over_white)
            if c is not None and not stop and bool(setting("COLOR_TRY_UNUSED", True))                     and n_real + 1 >= int(setting("COLOR_TRY_UNUSED_FROM", 3))                     and len(state.get("tried", [])) < int(setting("COLOR_TRY_UNUSED_MAX", 2)):
                # TRY THE UNUSED PAINT ONCE (2026-11-18: the cyan was never chosen in run G - its test stroke said "weak",
                # and a paint never painted is never measured again): a paint with no pass yet gets one where it promises
                # most, when that is at least COLOR_TRY_UNUSED_SHARE of the best pass's promise; its pass then teaches its
                # real strength. Not the whites (over wet paint they made grey mud, 2026-11-17).
                # KEEP TRYING (2026-11-20): not only the paints never painted - also one not used for COLOR_TRY_AGAIN_AFTER
                # passes (the sheet has changed since); each at most COLOR_TRY_PER_PAINT times, COLOR_TRY_GAP passes apart
                reals_ = [x for x in plan["passes"] if not x.get("fill")]
                recent_ = {x["paint"] for x in reals_[-int(setting("COLOR_TRY_AGAIN_AFTER", 4)):]}
                tried_ = state.get("tried", [])
                gap_ok = n_real - int(state.get("tried_at", -99)) >= int(setting("COLOR_TRY_GAP", 2))
                untried = [p for p in use if p not in recent_ and not is_white(p)
                           and tried_.count(p) < int(setting("COLOR_TRY_PER_PAINT", 2))] if gap_ok else []
                fac_try = dict(fac)                                   # not proven by a pass yet: as strong as it might be
                for p_ in untried:
                    lv_ = state.get("learned", {}).get(p_)
                    if lv_ is None or lv_.get("from") == "test stroke":
                        fac_try[p_] = scaled_test(p_, *fac[p_], state)
                cu = choose(now, target, done, fac_try, untried, Hp, Wp, smooth=smooth, prefer=prefer, eff=eff, lw=lw, age=age,
                            focus=focus, fw=fw) if untried else None
                if cu is not None and cu["gain"] >= float(setting("COLOR_TRY_UNUSED_SHARE", 0.25)) * c["gain"]:
                    print(f"   {label_of(cu['paint'])} has not been painted lately: one pass with it where it promises most "
                          f"({cu['gain']:.2f}, the best paint {c['gain']:.2f}) - then it is known from its own pass (COLOR_TRY_UNUSED)")
                    c = cu
                    state.setdefault("tried", []).append(cu["paint"])
                    state["tried_at"] = n_real
            # THE HOLES ONLY WHEN WORTH IT (2026-11-22, run of 11-21: four hole passes in a row on 0.7-3.6% of the picture,
            # promising 0.05-0.15 each, while a real pass would have promised 2.7 on the 21% never painted): a holes pass
            # only when it brings at least COLOR_HOLE_VS_REAL of what the best real pass would
            if hole_c is not None and (c is None or hole_c["gain"] >= float(setting("COLOR_HOLE_VS_REAL", 0.5)) * c["gain"]):
                c = hole_c
                state["hole_passes"] = int(state.get("hole_passes", 0)) + 1
                print(f"   BARE PAPER where the picture is not white: {100 * c['share']:.1f}% of the picture - "
                      f"{label_of(c['paint'])} there first (holes pass {state['hole_passes']}, COLOR_HOLE_*)")
            elif hole_c is not None:
                print(f"   (bare-paper holes {100 * hole_c['share']:.1f}% would bring {hole_c['gain']:.2f}, the best pass "
                      f"{c['gain']:.2f}: the real pass first, COLOR_HOLE_VS_REAL)")
            if stop and c is None:
                print("   THE END BY THE CAMERA: no other paint helps enough either")
            # THE COLOUR IS DONE when the sheet has not got better than its best for COLOR_END_NO_BEST passes (no pass limit
            # any more, 2026-11-23: run K was best at pass 30 and passes 31-40 only made it worse) - then the finish
            since_best = sum(1 for x in plan["passes"][bi:] if not x.get("fill")) if (plan["passes"] and best is not None) else 0
            if (c is not None and not state.get("finishing") and int(setting("COLOR_END_NO_BEST", 6) or 0) > 0
                    and since_best >= int(setting("COLOR_END_NO_BEST", 6))):
                print(f"   THE COLOUR IS DONE: the sheet was best (dE {best:.1f}) after pass {bi}, the {since_best} passes since "
                      f"did not make it better (COLOR_END_NO_BEST)")
                c = None
            if state.get("finishing") and c is not None and c is not hole_c:
                c = None                                                  # in the finish: no more colour passes
            fin_c = None
            if c is None and not state.get("signed") and i > 1:
                fin_c = finish_pass(now, target, done, fac, paints, Hp, Wp, plan, eff, lw)
                if fin_c is not None and not state.get("finishing"):
                    state["finishing"] = True
                    print("   THE FINISH: the colour is done - now the edges and the lights (COLOR_FINISH)")
            if fin_c is not None:
                c, last = fin_c, False
            elif c is None and not state.get("signed") and i > 1:         # the end, but the signature is not painted yet:
                c = choose(now, target, done, fac, paints, Hp, Wp, stop_gain=-1e9, smooth=1.0, eff=eff, small=True, lw=lw)
                if c is not None:                                         # one small pass with it, only where it clearly helps
                    print("   the painting is done; one more small pass (only where it clearly helps) carries the signature")
                else:
                    print("   NOTE: nothing left to paint: the signature was NOT painted")
                last = True
            elif c is None:
                break
            elif bool(setting("COLOR_FINISH", True)):                     # the finish comes after it: the signature at the end
                last = False
            else:                                                         # the signature goes with the pass expected last
                after = apply(now, c, fac)
                last = n_real + 1 >= max_passes_ or choose(after, target, {**done, c["paint"]: done.get(c["paint"], 0) + 1},
                                                          fac, paints, Hp, Wp, smooth=per_stage("COLOR_MASK_SMOOTH_MM", 4.0, n_real + 2),
                                                          prefer=per_stage("COLOR_PREFER", "glaze", n_real + 2), eff=eff,
                                                          lw=float(per_stage("COLOR_LIGHTNESS_WEIGHT", 1.0, n_real + 2))) is None
        if c is None:
            break
        c["dE"] = de_now
        q, pic = write_pass(i, plan, c)
        cv2.imwrite(os.path.join(DIR, f"cp_replan_pass{i:02d}.png"),
                    np.hstack([to_bgr(now), cv2.cvtColor(np.clip(np.dstack([target[..., 0] * 2.55, target[..., 1] + 128,
                               target[..., 2] + 128]), 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR),
                               cv2.resize(pic, (Ww, Hw), interpolation=cv2.INTER_AREA)]))
        print(f"   pass {i}: {q['show']}" + (f" FILL-IN {q['fill']}" if q.get("fill") else "")
              + f" on {100 * q['share']:.0f}% of the picture, promises {q['gain']:.2f}"
              + (f" (expected {q['expect']:.2f} after what this paint really does)" if q.get("expect") is not None else "")
              + (", the last pass's still-wet area left for later" if c.get("avoided_wet") else "")
              + f", {q['stage']}: {q['style']}, smooth {q['smooth']:g} mm{'  - the LAST pass, with the signature' if last else ''}  "
              f"(ko/cp_replan_pass{i:02d}.png: seen | target | this pass)")
        predicted = apply(now, c, fac)
        tests = test_strokes_for(i, plan, q, state, fac)
        json.dump(state, open(STATE, "w"), indent=1)
        run_pass(i, plan, extra, first=(i == start), last=last, info=job_info(i, plan, q, fac), tests=tests)
        if i == 1 and os.path.isfile(os.path.join(HERE, "paper_frame.png")):
            # pass 1's own photo, taken before the probes and the paint: the blank sheet, the white reference from now on
            shutil.copyfile(os.path.join(HERE, "paper_frame.png"), BLANK)
            keep_blank_if_same_sheet()
        if last:                                                          # signed: only its fill-ins may follow
            state["signed"] = True
            json.dump(state, open(STATE, "w"), indent=1)
            if not fill_tries or not ADAPTIVE:
                break
        i += 1
    print(f"\ncolor_paint: done, {len(plan['passes'])} passes ({(time.time() - t0) / 3600:.1f} h). "
          f"Photos: ko/cp_before_passNN.png, ko/cp_after_passNN.png")
    _say("narrate_done", len(plan["passes"]), (time.time() - t0) / 3600)
    time.sleep(12)                                       # the last sentence is spoken before the program ends


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] not in ("make", "paint", "-h", "--help"):        # python color_paint.py picture.jpg = make + paint
        argv = ["make"] + argv
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make", help="the forecast for a picture AND the painting (--plan_only: only the forecast, nothing moves)")
    m.add_argument("image")
    m.add_argument("--paints", nargs="+", choices=ORDER, help=f"which paints (default all: {', '.join(ORDER)})")
    m.add_argument("--dark", type=float, default=45.0, help="the picture's darkest becomes this L (paper = 100); higher = lighter, fewer coats")
    m.add_argument("--plan_only", action="store_true", help="only the forecast: ko/cp_preview.png, the machine does not move")
    m.add_argument("--dry", type=float, default=DRY_MIN, help=f"minutes of drying between passes (default {DRY_MIN:g})")
    p = sub.add_parser("paint", help="paint the picture of the last make (or go on from pass N, after a stop)")
    p.add_argument("start", nargs="?", type=int, default=1)
    p.add_argument("--dry", type=float, default=DRY_MIN, help=f"minutes of drying between passes (default {DRY_MIN:g})")
    a, extra = ap.parse_known_args(argv)
    os.chdir(HERE)
    if a.cmd == "make":
        make(a.image, a.paints, dark=a.dark)
        if not a.plan_only:                                           # one step: the forecast, then the painting straight away
            print("\npainting starts now (Ctrl+C = stop; go on later with  python color_paint.py paint <pass>)")
            paint(1, extra, a.dry)
    else:
        paint(a.start, extra, a.dry)


if __name__ == "__main__":
    main()
