"""color_planes.py - DEPTH PLANES for colour painting (2026-11-22; docs/depth_planes_comfyui.pdf).

A depth map of the picture (Depth Anything V2, run by ComfyUI on the studio computer through comfy_client.py) is sorted
into a few PLANES (far .. near: e.g. sky + hills, fence, figure, front plants). color_paint then:
  1. smooths each pass's mask INSIDE each plane, never across a plane border (paint of the hills does not fill into the
     head's outline);
  2. smooths each plane by its own amount: the plane with the face (focus) the least, far planes the most;
  3. keeps the mask sharp along DEPTH BREAKS (one thing in front of another), like COLOR_CRISP_EDGES does at lightness breaks.

    python color_planes.py make ko/picture.jpg [--url http://100.x.y.z:8188]   # depth (ComfyUI) + planes -> ko/cp_planes.png
    python color_planes.py show                                                # ko/cp_planes_view.png: the planes over the picture

The depth map is kept per picture in ko/comfy/<name>_depth.png (made once, reused). No ComfyUI, no depth file: no planes,
and color_paint works as before. The planes file ko/cp_planes.png holds the plane number per pixel (0 = farthest).
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
import settings as S

DIR = os.path.join(HERE, "ko")
CACHE = os.path.join(DIR, "comfy")
PLANES = os.path.join(DIR, "cp_planes.png")
INFO = os.path.join(DIR, "cp_planes.json")


def setting(name, default):
    return getattr(S, name, default)


# ------------------------------------------------------------------------------------------------------------ depth
def depth_path(image):
    return os.path.join(CACHE, os.path.splitext(os.path.basename(image))[0] + "_depth.png")


def depth_map(image, url=None, res=1024, wait_s=600):
    """The picture's depth (uint8, light = near, the picture's own size). Made once by ComfyUI, then read from ko/comfy/."""
    img = cv2.imread(image)
    if img is None:
        raise SystemExit(f"cannot read {image}")
    h, w = img.shape[:2]
    p = depth_path(image)
    if os.path.isfile(p) and os.path.getmtime(p) >= os.path.getmtime(image):
        return cv2.resize(cv2.imread(p, 0), (w, h), interpolation=cv2.INTER_AREA)
    import comfy_client as C
    if url:
        C.URL = url.rstrip("/")
    print(f"depth map of {os.path.basename(image)} on ComfyUI ({C.URL}) ...", flush=True)
    name = C.upload(image)
    wf = {"1": {"class_type": "LoadImage", "inputs": {"image": name}},
          "2": {"class_type": "DepthAnythingV2Preprocessor",
                "inputs": {"image": ["1", 0], "resolution": int(res),
                           "ckpt_name": setting("COLOR_DEPTH_MODEL", "depth_anything_v2_vitl.pth")}},
          "3": {"class_type": "SaveImage", "inputs": {"images": ["2", 0], "filename_prefix": "brush_depth"}}}
    pid = json.loads(C._post("/prompt", json.dumps({"prompt": wf}).encode()))["prompt_id"]
    t0 = time.time()
    while time.time() - t0 < wait_s:
        try:
            h_ = json.loads(C._get(f"/history/{pid}", timeout=30))
        except OSError:                                  # the relay drops a request now and then: ask again
            time.sleep(3)
            continue
        st = h_.get(pid, {}).get("status", {})
        if st.get("status_str") == "error":
            raise SystemExit(f"ComfyUI: {json.dumps(st)[:600]}")
        if st.get("completed"):
            img_ = h_[pid]["outputs"]["3"]["images"][0]
            q = urllib.parse.urlencode({"filename": img_["filename"], "subfolder": img_.get("subfolder", ""),
                                        "type": img_.get("type", "output")})
            os.makedirs(CACHE, exist_ok=True)
            open(p, "wb").write(C._get(f"/view?{q}", timeout=120))
            print(f"  -> {os.path.relpath(p, HERE)} ({time.time() - t0:.0f} s)")
            return cv2.resize(cv2.imread(p, 0), (w, h), interpolation=cv2.INTER_AREA)
        time.sleep(2)
    raise SystemExit(f"no depth map after {wait_s} s")


# ----------------------------------------------------------------------------------------------------------- planes
def enabled():
    """COLOR_PLANES: 0 / False = off, a number = that many planes, "auto" = the picture decides."""
    v = setting("COLOR_PLANES", "auto")
    return v == "auto" or bool(v)


def _kmeans(d, k):
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 0.3)
    cv2.setRNGSeed(1)
    _, lab, c = cv2.kmeans(d.reshape(-1, 1), k, None, crit, 5, cv2.KMEANS_PP_CENTERS)
    return lab.ravel(), c.ravel()


def regions_from_depth(depth, mm_px=None, jump=None, min_mm2=None):
    """PLANES FROM THE DEPTH JUMPS (2026-11-22, user: "the planes are not a constant, they are found in each picture"):
    where the depth changes suddenly (more than COLOR_DEPTH_JUMP of the picture's depth range per mm) there is an edge -
    one thing in front of another; the areas between those edges are the planes, as many as the picture has. Areas
    smaller than COLOR_PLANE_MIN_MM2 join their neighbour. Returns (labels 0 = farthest .., mean depth of each)."""
    d = cv2.GaussianBlur(depth.astype(np.float32), (0, 0), 1.5)
    H, W = d.shape
    mm_px = mm_px or 190.0 / max(H, W)
    rng_ = max(1.0, float(np.percentile(d, 98) - np.percentile(d, 2)))
    g = np.hypot(cv2.Sobel(d, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(d, cv2.CV_32F, 0, 1, ksize=3)) / 8.0 / mm_px
    jump = float(jump or setting("COLOR_DEPTH_JUMP", 0.03))
    edge = (g > jump * rng_).astype(np.uint8)
    min_px = float(min_mm2 or setting("COLOR_PLANE_MIN_MM2", 25.0)) / mm_px ** 2
    n, cc, st, _ = cv2.connectedComponentsWithStats(1 - edge, 4)
    markers = np.zeros((H, W), np.int32)
    m = 0
    for j in range(1, n):
        if st[j, cv2.CC_STAT_AREA] >= min_px:
            m += 1
            markers[cc == j] = m
    if m < 2:                                               # no real depth jumps: one plane
        return np.zeros((H, W), np.uint8), [float(d.mean())]
    img = cv2.cvtColor(np.clip(d, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    cv2.watershed(img, markers)                             # the edges and the small bits go to the plane beside them
    lab = markers.copy()
    bad = lab <= 0
    if bad.any():
        lab[bad] = cv2.dilate(np.where(bad, 0, lab).astype(np.float32), np.ones((3, 3), np.uint8)).astype(np.int32)[bad]
    ids = [j for j in range(1, m + 1) if (lab == j).any()][:255]
    means = {j: float(d[lab == j].mean()) for j in ids}
    order = sorted(ids, key=lambda j: means[j])
    L = np.zeros((H, W), np.uint8)
    for r, j in enumerate(order):
        L[lab == j] = r
    return L, [means[j] for j in order]


def planes_from_depth(depth, k=None, min_mm2=None, mm_px=None):
    """Plane number per pixel (0 = farthest). COLOR_PLANES = "auto": from the depth jumps (regions_from_depth), as many as
    the picture has; a number: the depth values sorted into that many groups (k-means). Returns (labels, depths far..near)."""
    d = depth.astype(np.float32)
    k = k or setting("COLOR_PLANES", "auto")
    if k == "auto":
        L, cs = regions_from_depth(depth, mm_px, min_mm2=min_mm2)
        print(f"  depth planes: this picture has {len(cs)} (areas between depth jumps, COLOR_DEPTH_JUMP)")
        return L, cs
    k = int(k)
    lab, c = _kmeans(d, k)
    c = c.reshape(-1, 1)
    order = np.argsort(c.ravel())
    rank = np.zeros(k, np.uint8)
    rank[order] = np.arange(k)
    L = rank[lab.ravel()].reshape(d.shape)
    L = cv2.medianBlur(L, 5)
    # small islands (a few mm) belong to the plane round them
    mm_px = mm_px or 190.0 / max(d.shape)
    min_px = (float(min_mm2 or setting("COLOR_PLANE_MIN_MM2", 25.0))) / mm_px ** 2
    for i in range(k):
        n, cc, st, _ = cv2.connectedComponentsWithStats((L == i).astype(np.uint8), 8)
        for j in range(1, n):
            if st[j, cv2.CC_STAT_AREA] < min_px:
                blob = cc == j
                ring = cv2.dilate(blob.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
                ring &= ~blob
                if ring.any():
                    L[blob] = np.bincount(L[ring], minlength=k).argmax()
    return L, [float(c.ravel()[o]) for o in order]


FIGURE = [None]                 # the figure's plane number when it came from the segmentation (load() / compute() set it)


def person_mask(image, url=None, wait_s=300):
    """WHERE THE PERSON IS (2026-11-22, OneFormerRegions on the studio computer, ADE20K): one clean shape with head, body,
    arms and knees - what depth alone could not give (the knees fell into the front plants' plane). Made once per picture,
    kept in ko/comfy/<name>_person.png. None = no ComfyUI / no person."""
    img = cv2.imread(image)
    h, w = img.shape[:2]
    p = os.path.join(CACHE, os.path.splitext(os.path.basename(image))[0] + "_person.png")
    if not (os.path.isfile(p) and os.path.getmtime(p) >= os.path.getmtime(image)):
        import comfy_client as C
        if url:
            C.URL = url.rstrip("/")
        try:
            name = C.upload(image)
            wf = {"1": {"class_type": "LoadImage", "inputs": {"image": name}},
                  "2": {"class_type": "OneFormerRegions", "inputs": {"image": ["1", 0], "model": "ADE20K (150 classes)",
                                                                       "mask_labels": str(setting("COLOR_FIGURE_LABELS", "person"))}},
                  "3": {"class_type": "MaskToImage", "inputs": {"mask": ["2", 2]}},
                  "4": {"class_type": "SaveImage", "inputs": {"images": ["3", 0], "filename_prefix": "brush_person"}}}
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
                    print("  (no segmentation: ComfyUI reported an error)")
                    return None
                if st.get("completed"):
                    im = h_[pid]["outputs"]["4"]["images"][0]
                    q = urllib.parse.urlencode({"filename": im["filename"], "subfolder": im.get("subfolder", ""),
                                                "type": im.get("type", "output")})
                    os.makedirs(CACHE, exist_ok=True)
                    open(p, "wb").write(C._get(f"/view?{q}", timeout=120))
                    break
                time.sleep(2)
            else:
                return None
        except OSError as e:
            print(f"  (no segmentation: {e})")
            return None
    m = cv2.imread(p, 0)
    if m is None:
        return None
    m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST) >= 128
    return m if m.mean() >= 0.005 else None


def compute(image, url=None):
    """Depth (ComfyUI or the cache) -> planes at the picture's size, (labels, depth centres); (None, None) without depth.
    With a person in the picture (person_mask): the person is ONE plane, the figure, placed by its depth among the others."""
    if not enabled():
        return None, None
    url = url or setting("COMFY_URL", None)
    try:
        depth = depth_map(image, url)
    except (OSError, SystemExit) as e:
        print(f"  no depth planes ({e}); colour painting goes on without them")
        return None, None
    L, cs = planes_from_depth(depth)
    FIGURE[0] = None
    pm = person_mask(image, url) if bool(setting("COLOR_FIGURE_FROM_SEGMENTATION", True)) else None
    if pm is not None and pm.shape == L.shape:
        d = cv2.GaussianBlur(depth.astype(np.float32), (0, 0), 1.5)
        lab = L.astype(np.int32)
        lab[pm] = 1000                                          # the person: one plane
        ids = [j for j in np.unique(lab)]
        means = {j: float(d[lab == j].mean()) for j in ids}
        order = sorted(ids, key=lambda j: means[j])
        L = np.zeros(L.shape, np.uint8)
        for r, j in enumerate(order[:255]):
            L[lab == j] = r
        cs = [means[j] for j in order[:255]]
        FIGURE[0] = order.index(1000)
        print(f"  the figure from the segmentation: {100 * pm.mean():.0f}% of the picture, one plane (COLOR_FIGURE_LABELS)")
    return L, cs


def save(image, L, centres):
    """ko/cp_planes.png (plane number per pixel) + ko/cp_planes.json."""
    os.makedirs(DIR, exist_ok=True)
    cv2.imwrite(PLANES, L)
    info = {"image": os.path.abspath(image), "made": time.strftime("%Y-%m-%d %H:%M"), "k": len(centres),
            "depth_centres": [round(x, 1) for x in centres],
            "share": [round(float((L == i).mean()), 3) for i in range(len(centres))]}
    if FIGURE[0] is not None:
        info["figure"], info["figure_from"] = int(FIGURE[0]), "segmentation"
    json.dump(info, open(INFO, "w"), indent=1)
    big = sorted(range(len(info["share"])), key=lambda i: -info["share"][i])[:4]
    print(f"  depth planes: {len(info['share'])}, the largest " + ", ".join(f"{100 * info['share'][i]:.0f}%" for i in big)
          + f"  ({os.path.relpath(PLANES, HERE)})")
    return L


def make(image, url=None):
    """Depth -> planes -> saved. None when there is no depth."""
    L, centres = compute(image, url)
    return None if L is None else save(image, L, centres)


def load(shape_hw, image=None):
    """The planes on a grid of shape_hw (nearest, no new values), or None (no file, or made for another picture)."""
    if not enabled() or not os.path.isfile(PLANES):
        return None
    if image is not None:
        try:
            same = lambda p_: os.path.normcase(os.path.abspath(p_))
            if same(json.load(open(INFO))["image"]) != same(image):
                return None
        except (OSError, ValueError, KeyError):
            return None
    L = cv2.imread(PLANES, 0)
    try:
        inf = json.load(open(INFO))
        FIGURE[0] = int(inf["figure"]) if inf.get("figure_from") == "segmentation" else None
    except (OSError, ValueError, KeyError):
        FIGURE[0] = None
    H, W = shape_hw
    return cv2.resize(L, (W, H), interpolation=cv2.INTER_NEAREST)


# --------------------------------------------------------------------------------------------- what color_paint uses
def figure_plane(L, focus=None):
    """The figure's plane: the person from the segmentation (FIGURE), else the plane with the face in it (focus_map's
    faces), else None."""
    if FIGURE[0] is not None and FIGURE[0] <= int(L.max()):
        return int(FIGURE[0])
    if isinstance(focus, tuple):                         # color_paint.focus_map gives (map, number of faces found)
        focus = focus[0] if focus[1] else None           # no face: no plane is "the figure"
    if focus is None or focus.shape != L.shape or focus.max() <= 0:
        return None
    k = int(L.max()) + 1
    face = focus >= 0.99                                  # the plane holding MOST of the face (not a speck inside it)
    return int(np.argmax(np.bincount(L[face].ravel(), minlength=k))) if face.any() else None


def factors(L, focus=None):
    """Smoothing factor per plane: COLOR_PLANE_SMOOTH = (far .. near) for the planes WITHOUT the face; the plane with the
    face (figure_plane) gets COLOR_PLANE_SMOOTH_FOCUS instead."""
    k = int(L.max()) + 1
    base = [float(x) for x in setting("COLOR_PLANE_SMOOTH", (1.0, 0.5))]
    if len(base) == k:
        f = base
    else:                                                # any number of planes: from the far value to the near one
        f = [base[0] + (base[-1] - base[0]) * (i / max(1, k - 1)) for i in range(k)]
    fp = figure_plane(L, focus)
    if fp is not None:
        f[fp] = float(setting("COLOR_PLANE_SMOOTH_FOCUS", 0.25))
    return f


def clean_by_plane(m, L, smooth, clean_fn, Hp, Wp, fac=None, floor=1.0):
    """clean_fn(m, Hp, Wp, smooth) done inside each plane with smooth x its factor (not below `floor` mm unless the stage
    itself is finer), then cut to the plane: nothing is filled across a plane border."""
    if L is None or L.shape != m.shape:
        return clean_fn(m, Hp, Wp, smooth)
    fac = fac or [1.0] * (int(L.max()) + 1)
    out = np.zeros(m.shape, bool)
    for i, f in enumerate(fac):
        inside = L == i
        mi = m & inside
        if not mi.any():
            continue
        sm = max(float(smooth) * f, min(float(smooth), floor))
        out |= clean_fn(mi, Hp, Wp, sm) & inside
    return out


def breaks(L, mm_px, width_mm=None):
    """A band along the plane borders (depth breaks), width_mm wide on each side: there a mask follows the wanted spots."""
    if L is None:
        return None
    edge = (np.diff(L.astype(np.int16), axis=0, prepend=L[:1].astype(np.int16)) != 0) | \
           (np.diff(L.astype(np.int16), axis=1, prepend=L[:, :1].astype(np.int16)) != 0)
    r = max(1, int(round(float(width_mm or setting("COLOR_PLANE_EDGE_MM", 1.5)) / mm_px)))
    return cv2.dilate(edge.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))) > 0


def view(image, L, out):
    """The planes coloured over the picture, with their borders."""
    img = cv2.imread(image)
    h, w = img.shape[:2]
    L = cv2.resize(L, (w, h), interpolation=cv2.INTER_NEAREST)
    k = int(L.max()) + 1
    cols = cv2.applyColorMap(np.uint8(np.linspace(30, 230, k)).reshape(-1, 1), cv2.COLORMAP_TURBO)[:, 0]
    ov = cv2.addWeighted(img, 0.45, cols[L], 0.55, 0)
    ov[breaks(L, 190.0 / max(h, w), 0.4)] = (255, 255, 255)
    cv2.imwrite(out, np.hstack([img, ov]))
    return out


if __name__ == "__main__":
    a = sys.argv[1:]
    url = None
    if "--url" in a:
        i = a.index("--url"); url = a[i + 1]; del a[i:i + 2]
    if a and a[0] == "make" and len(a) > 1:
        L = make(a[1], url)
        if L is not None:
            print(view(a[1], L, os.path.join(DIR, "cp_planes_view.png")))
    elif a and a[0] == "show":
        info = json.load(open(INFO))
        print(view(info["image"], cv2.imread(PLANES, 0), os.path.join(DIR, "cp_planes_view.png")))
    else:
        raise SystemExit(__doc__)
