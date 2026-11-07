"""
leaf_style.py - the "leaf" infill (INFILL_STYLE = "leaf"): every shape is split into leaf-shaped Voronoi cells that follow its form,
and every leaf is painted with 'vein' strokes from tip to tip, all in one direction, a fixed spacing apart (no paint piled up).

  * leaf width follows the shape: as many leaves across as fit at about LEAF_TARGET_MM each (a thin arm = one leaf as wide as the
    arm, a broad back = several), between LEAF_WIDTH_MM (min, max), with some random variety (LEAF_VARIETY)
  * a leaf grows along the form as long as the shape lets it, up to LEAF_ASPECT x its width
  * the outermost veins run half a real brush mark inside the leaf's edge: where the leaf meets the outline, the paint stops ON it
  * each half of a leaf has veins at fixed distances from its own edge; a vein ends where the leaf gets too narrow for it, so the
    brush does not go over the same place again (simulated on men.png: x1.18 coats instead of x1.97, 43 % less brush travel)
  * veins in one leaf vary in width (LEAF_VEIN_VARIETY) and neighbours run opposite ways (LEAF_DIRECTION = "alternate"),
    so both tips of a leaf get heavy starts and light tails
  * smaller leaves (LEAF_TIERS) fill what the big ones leave open

Used by paint.py (_plan_piece); also runs on its own for a picture:  python leaf_style.py men.png red out.png
"""
import math
import random

import cv2
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage import measure


def _smooth(p, k=5):
    if len(p) < 2 * k + 1:
        return p
    ker = np.ones(k) / k
    return np.c_[np.convolve(np.pad(p[:, 0], k // 2, mode="edge"), ker, "valid"),
                 np.convolve(np.pad(p[:, 1], k // 2, mode="edge"), ker, "valid")]


def _resample(p, step):
    s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(p, axis=0).T))]
    if s[-1] < step:
        return p
    t = np.arange(0.0, s[-1] + 1e-9, step)
    return np.c_[np.interp(t, s, p[:, 0]), np.interp(t, s, p[:, 1])]


def leaf_cells(area, fx, fy, width_px, aspect, rng, relax=3):
    """Like paint.voronoi_cells, but every seed has its OWN width (width_px: a map, px) and length (aspect x width, random):
    wide leaves where the shape is broad, narrow ones where it is thin. Returns (labels 1..M, M)."""
    H, W = area.shape
    ys, xs = np.nonzero(area)
    if xs.size == 0:
        return np.zeros(area.shape, np.int32), 0

    def frame(pts):
        xi = np.clip(np.round(pts[:, 0]).astype(int), 0, W - 1)
        yi = np.clip(np.round(pts[:, 1]).astype(int), 0, H - 1)
        return np.c_[fx[yi, xi], fy[yi, xi]], width_px[yi, xi]

    g = max(1, int(np.percentile(width_px[area], 10) / 3))
    sel = (xs % g == 0) & (ys % g == 0)
    cand = np.c_[xs[sel], ys[sel]].astype(float)
    if len(cand) == 0:
        cand = np.c_[xs[:1], ys[:1]].astype(float)
    cand = cand[np.array(rng.sample(range(len(cand)), len(cand)))]
    t_c, w_c = frame(cand)
    bucket = max(2.0, float(width_px[area].max()) * aspect[1])
    grid, seeds, lens, wids = {}, [], [], []
    for p, t, wd in zip(cand, t_c, w_c):
        L = rng.uniform(aspect[0], aspect[1]) * wd
        bx, by = int(p[0] // bucket), int(p[1] // bucket)
        ok = True
        for j in (j for gx in (bx - 1, bx, bx + 1) for gy in (by - 1, by, by + 1) for j in grid.get((gx, gy), ())):
            dx, dy = p[0] - seeds[j][0], p[1] - seeds[j][1]
            a = (dx * t[0] + dy * t[1]) / (0.5 * (L + lens[j]))
            b = (-dx * t[1] + dy * t[0]) / (0.5 * (wd + wids[j]))
            if a * a + b * b < 1.0:
                ok = False
                break
        if ok:
            grid.setdefault((bx, by), []).append(len(seeds))
            seeds.append(p); lens.append(L); wids.append(wd)
    seeds, lens, wids = np.array(seeds), np.array(lens), np.array(wids)
    P = np.c_[xs, ys].astype(np.float32)

    def label(seeds, lens, wids):
        ts, _ = frame(seeds)
        k = min(len(seeds), 24)
        tree = cKDTree(seeds)
        out = np.empty(len(P), np.int32)
        for i0 in range(0, len(P), 100000):
            p = P[i0:i0 + 100000]
            _, idx = tree.query(p, k=k)
            idx = idx.reshape(len(p), k)
            d = p[:, None, :] - seeds[idx]
            t = ts[idx]
            a = (d[..., 0] * t[..., 0] + d[..., 1] * t[..., 1]) / (0.5 * lens[idx])
            b = (-d[..., 0] * t[..., 1] + d[..., 1] * t[..., 0]) / (0.5 * wids[idx])
            out[i0:i0 + len(p)] = idx[np.arange(len(p)), np.argmin(a * a + b * b, axis=1)]
        return out

    for _ in range(max(0, int(relax))):
        own = label(seeds, lens, wids)
        cnt = np.bincount(own, minlength=len(seeds))
        cx = np.bincount(own, P[:, 0], len(seeds)) / np.maximum(cnt, 1)
        cy = np.bincount(own, P[:, 1], len(seeds)) / np.maximum(cnt, 1)
        live = cnt > 0
        new = np.c_[cx, cy]
        xi, yi = np.round(new[:, 0]).astype(int).clip(0, W - 1), np.round(new[:, 1]).astype(int).clip(0, H - 1)
        seeds = np.where(area[yi, xi][:, None], new, seeds)[live]
        lens, wids = lens[live], wids[live]
    lab = np.zeros(area.shape, np.int32)
    lab[ys, xs] = label(seeds, lens, wids) + 1
    orphan = np.zeros(area.shape, bool)                                 # a cell must be one piece
    for i, sl in enumerate(ndimage.find_objects(lab), 1):
        if sl is None:
            continue
        cell = (lab[sl] == i).astype(np.uint8)
        n, cc = cv2.connectedComponents(cell, connectivity=4)
        if n > 2:
            keep = 1 + int(np.argmax(np.bincount(cc.ravel())[1:]))
            orphan[sl] |= (cell > 0) & (cc != keep)
    if orphan.any():
        lab[orphan] = 0
        iy, ix = ndimage.distance_transform_edt(lab == 0, return_distances=False, return_indices=True)
        lab[orphan] = lab[iy, ix][orphan]
    return lab, len(seeds)


def _long_axis_tips(mask):
    """The two ends of the shape's longest skeleton path (x, y), or None: where a whole-shape leaf has its tips."""
    from skimage.morphology import skeletonize
    ys, xs = np.nonzero(skeletonize(mask))
    if len(xs) < 3:
        return None
    idx = {p: i for i, p in enumerate(zip(ys.tolist(), xs.tolist()))}
    nbrs = [[idx[(y + dy, x + dx)] for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy or dx) and (y + dy, x + dx) in idx]
            for y, x in zip(ys.tolist(), xs.tolist())]

    def farthest(s):                                                      # breadth-first along the skeleton
        dist, queue = {s: 0}, [s]
        for u in queue:
            for v in nbrs[u]:
                if v not in dist:
                    dist[v] = dist[u] + 1
                    queue.append(v)
        return max(dist, key=dist.get)

    a = farthest(0)
    b = farthest(a)
    return np.array([xs[a], ys[a]], float), np.array([xs[b], ys[b]], float)


def veins(cell, tipA, tipB, m, maxn=25, spacing=0.78, rng=None, variety=0.0, straight=0.45):
    """The vein strokes of one leaf. cell: bool crop, tipA/tipB: its two tips (x, y) in the crop, m: what one stroke really paints
    at full width (px). The leaf's outline is split at the tips into a left and a right edge; the veins of each half run at FIXED
    distances from their own edge, so they never squeeze together or pile up in the tips: a vein just ends where its half of the leaf
    gets too narrow. A middle vein closes the seam. Every vein has its own width (1 - variety .. 1 of the full width, random) and
    the distances are worked out from those widths, so thin and wide veins still just touch.
    Returns (polyline (x, y) from the A end to the B end, width share, place across the leaf: 0 = left edge ...) tuples."""
    cs, _ = cv2.findContours(cell.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        return []
    c = max(cs, key=len).reshape(-1, 2)
    if len(c) < 8:
        return []
    ia = int(np.argmin(np.hypot(*(c - tipA).T)))
    ib = int(np.argmin(np.hypot(*(c - tipB).T)))
    if ia == ib:
        return []
    i0, i1 = sorted((ia, ib))
    h, w = cell.shape
    d = []
    for arc in (c[i0:i1 + 1], np.r_[c[i1:], c[:i0 + 1]]):
        img = np.ones((h, w), np.uint8)
        img[arc[:, 1], arc[:, 0]] = 0
        d.append(cv2.distanceTransform(img, cv2.DIST_L2, 5))
    dL, dR = d
    width = dL + dR
    wmax = float(width[cell].max())
    s = spacing * m                                                       # vein to vein at full width (a little less than one real mark)

    def share():
        return 1.0 - variety * (rng.random() if rng is not None else 0.0)

    # (level field, region, width share, place across): veins at FIXED distances from each edge, so they never squeeze together; each
    # half of the leaf follows its own edge and the two halves meet in the middle, where a middle vein closes the seam
    plan = []
    if wmax < 2.0 * s:
        plan.append((dL - dR, cell, share(), 0))                         # a narrow leaf: one vein down its middle
    else:
        plan.append((dL - dR, cell & (width >= 1.0 * m), share(), 0))
        for side, (dd, other) in enumerate(((dL, dR), (dR, dL))):
            f = 1.0                                                       # the outermost vein is always full width: a crisp edge
            t = 0.5 * f * m                                               # the outermost vein: its edge on the leaf's edge
            k = 0
            while t + 0.5 * s <= 0.5 * wmax:
                # this vein only where it stays clear of the middle vein: its own half must be at least t + s/2 wide
                plan.append((dd - t, cell & (dd + other >= 2.0 * t + s), f, (-1 if side == 0 else 1) * (100 - k)))
                f_next = share()
                t += spacing * 0.5 * (f + f_next) * m                     # the next one just touches it, whatever their widths
                f, k = f_next, k + 1
    out = []
    for g, valid, f, place in plan[:maxn]:
        if valid.sum() < 6:
            continue
        cts = measure.find_contours(np.where(valid, g, np.nan), 0.0)
        if not cts:
            continue
        ct = max(cts, key=len)[:, ::-1]                                   # (row, col) -> (x, y)
        if len(ct) < 4:
            continue
        L = float(np.hypot(*np.diff(ct, axis=0).T).sum())
        if L < 1.5 * m or np.hypot(*(ct[-1] - ct[0])) < straight * L:     # a stub, or a vein bending round on itself: left out
            continue
        if np.hypot(*(ct[0] - tipA)) > np.hypot(*(ct[-1] - tipA)):
            ct = ct[::-1]
        out.append((ct, f, place))
    out.sort(key=lambda v: v[2])                                          # left edge -> middle -> right edge
    return [(ct, f, i) for i, (ct, f, _) in enumerate(out)]


def leaf_strokes(area, dmask, res, cfg, rng, centre_line, extend_ends, field):
    """Leaf fill of one piece. centre_line / extend_ends / field = paint._cell_centre_line, paint._extend_ends,
    paint.voronoi_field (passed in so this file does not import paint.py). Returns (points px, widths mm, tier, leaf) tuples."""
    H, W = area.shape
    out = []
    if not area.any():
        return out
    brush = max(2.0, cfg["brush_w"] * res)
    cap = max(brush, float(cfg.get("thin_max") or cfg["brush_w"] * 2.0) * res)
    m = cap * min(1.0, max(0.3, float(cfg.get("voro_mark_share", 0.7))))            # what one stroke really paints, px
    wmin, wmax = (float(v) * res for v in cfg.get("leaf_width_mm", (4.0, 14.0)))
    wmin = max(wmin, m)
    target_px = float(cfg.get("leaf_target_mm", 9.0)) * res                        # the usual leaf width
    aspect = tuple(cfg.get("leaf_aspect", (3.0, 7.0)))
    variety = float(cfg.get("leaf_variety", 0.25))
    tiers = tuple(cfg.get("leaf_tiers", (1.0, 0.5, 0.3)))
    vein_variety = min(0.7, max(0.0, float(cfg.get("leaf_vein_variety", 0.4))))   # veins between (1 - this) and 1 x the widest mark
    alternate = str(cfg.get("leaf_direction", "alternate")) == "alternate"         # neighbouring veins run opposite ways ("hand" = all one way)
    hd = math.radians(float(cfg.get("voro_hand_deg", 45.0)))
    hand = (math.cos(hd), math.sin(hd))
    step = max(1.0, 0.5 * res)
    fx, fy = field(area, res, float(cfg.get("voro_field_sigma_mm", 5.0)), float(cfg.get("voro_axis_weight", 0.3)))
    # local thickness of the shape: twice the largest distance to the outline nearby (the medial width)
    thick = 2.0 * ndimage.maximum_filter(np.where(area, dmask, 0), size=max(3, int(wmax)))
    # how busy the outline is around each point (hands, hat brims, faces): outline length in a window of one leaf width, against
    # what a straight edge would have there (1 = calm or straight, 2+ = busy). Busy places get smaller leaves (LEAF_DETAIL)
    rim = cv2.morphologyEx(area.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)).astype(np.float32)
    win = max(3, int(target_px) | 1)
    busy = cv2.GaussianBlur(cv2.boxFilter(rim, -1, (win, win), normalize=False) / (2.0 * win), (0, 0), 0.5 * win)
    shrink = 1.0 / (1.0 + float(cfg.get("leaf_detail", 0.6)) * np.clip(busy - 1.0, 0.0, 3.0))
    # a little variety that changes slowly over the shape (neighbouring leaves differ, but not like noise)
    noise = cv2.GaussianBlur(np.array([[rng.uniform(-1, 1) for _ in range(W)] for _ in range(H)], np.float32), (0, 0), max(1.0, wmax))
    noise = noise / max(1e-6, float(np.abs(noise).max()))
    covered = np.zeros((H, W), np.uint8)
    leaf_id = 0

    def at(img, x, y):
        return img[min(max(int(round(y)), 0), H - 1), min(max(int(round(x)), 0), W - 1)]

    def emit(vs, xs0, ys0, tier, taper_mm=None):
        """The veins of one leaf into out (middle first, neighbours opposite ways), and what they paint into covered."""
        nonlocal leaf_id
        leaf_id += 1
        flip = rng.random() < 0.5                                         # which neighbour starts at which tip: differs leaf to leaf
        mid = (len(vs) - 1) / 2
        for k in sorted(range(len(vs)), key=lambda k: abs(k - mid)):      # middle vein first, then outwards
            v, f, place = vs[k]
            if alternate and (place % 2 == 1) != flip:
                v = v[::-1]                                               # neighbours run opposite ways: both tips get starts and tails
            p = _resample(_smooth(v + (xs0, ys0)), step)
            if len(p) < 3:
                continue
            if tier > 0 and float(np.hypot(*np.diff(p, axis=0).T).sum()) < 1.5 * res:
                continue                                                  # a dot, not a vein
            s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(p, axis=0).T))]
            tt = s / max(s[-1], 1e-9)
            # tapered ends: 15 % of a leaf vein, or taper_mm of a long whole-shape vein
            taper = 0.15 if taper_mm is None else min(0.15, taper_mm * res / max(s[-1], 1e-9))
            ww = cap * f * np.clip(0.55 + 0.45 * np.minimum(tt, 1.0 - tt) / max(taper, 1e-3), 0.55, 1.0)   # its own width
            ww = np.maximum(ww, brush)
            out.append((p, ww / res, tier, leaf_id))
            q = np.round(p).astype(np.int32)
            for j in range(len(q) - 1):
                cv2.line(covered, tuple(map(int, q[j])), tuple(map(int, q[j + 1])), 1,
                         max(1, int(round(0.5 * m / cap * (ww[j] + ww[j + 1])))))

    outline = bool(cfg.get("leaf_outline", True))
    # with the contour on, the leaves stop one mark inside the real outline: the contour paints that band alone (no double coat)
    inner = area & (dmask >= 0.85 * m) if outline else area
    # LEAF_WHOLE: a long form (an arm, a petal, a thick brush line) is ONE leaf: a few long veins from its one end to the other,
    # following its curve - instead of a row of short leaves. Used when those few veins fill it about as well as the leaves would
    whole, whole_cov = None, 0.0
    nmax = int(cfg.get("leaf_whole_max", 4))
    if nmax > 0 and inner.any():
        tips = _long_axis_tips(inner)
        if tips is not None:
            vs = veins(inner, tips[0], tips[1], m, maxn=nmax + 1, rng=rng, variety=vein_variety, straight=0.1)
            if 0 < len(vs) <= nmax:
                test = np.zeros((H, W), np.uint8)
                for v, f, _ in vs:
                    cv2.polylines(test, [np.round(v).astype(np.int32).reshape(-1, 1, 2)], False, 1, max(1, int(round(m * f))))
                whole, whole_cov = vs, float((test.astype(bool) & inner).sum()) / float(inner.sum())
    for tier, share in enumerate(tiers):
        todo = inner & (covered == 0)
        if tier > 0:
            todo = cv2.morphologyEx(todo.astype(np.uint8), cv2.MORPH_OPEN, _disk(0.3 * m)) > 0
        # how many leaves fit across the shape here (at about LEAF_TARGET_MM each, varied): a thin arm = one leaf as wide as the arm,
        # a broad back = several. The thickness is then shared evenly between them
        target = target_px * share * (1.0 + variety * noise) * shrink
        across = np.maximum(1.0, np.round(thick / np.maximum(target, 1.0)))
        width = np.clip(thick / across, max(m, wmin * share), wmax * share).astype(np.float32)
        if todo.sum() < 0.3 * float(width[todo].mean() if todo.any() else 1) ** 2:
            continue
        lab, M = leaf_cells(todo, fx, fy, width, aspect, rng, relax=int(cfg.get("voro_relax", 3)))
        if M == 0:
            continue
        edge = ~inner
        for dy, dx in ((0, 1), (1, 0)):
            a, b = lab[: H - dy, : W - dx], lab[dy:, dx:]
            diff = (a != b) & (a > 0) & (b > 0)
            edge[: H - dy, : W - dx] |= diff
            edge[dy:, dx:] |= diff
        dcell = cv2.distanceTransform((~edge).astype(np.uint8), cv2.DIST_L2, 5)
        starts = ndimage.maximum_position(np.where(lab > 0, dcell, 0), lab, index=np.arange(1, M + 1))
        boxes = ndimage.find_objects(lab)
        cy_, cx_ = np.nonzero(lab)
        li = lab[cy_, cx_]
        cnt = np.maximum(np.bincount(li, minlength=M + 1), 1).astype(float)
        mx, my = np.bincount(li, cx_, M + 1) / cnt, np.bincount(li, cy_, M + 1) / cnt
        sxx = np.bincount(li, cx_ * cx_.astype(float), M + 1) / cnt - mx * mx
        syy = np.bincount(li, cy_ * cy_.astype(float), M + 1) / cnt - my * my
        sxy = np.bincount(li, cx_ * cy_.astype(float), M + 1) / cnt - mx * my
        ang = 0.5 * np.arctan2(2 * sxy, sxx - syy)
        for i, (y0, x0) in enumerate(starts, 1):
            sl = boxes[i - 1]
            if sl is None:
                continue
            hi = aspect[1] * float(width[y0, x0])
            c = centre_line(i, x0, y0, (math.cos(ang[i]), math.sin(ang[i])), lab, dcell, fx, fy, at, step, 0.25 * brush, brush, hi)
            if (c[-1, 0] - c[0, 0]) * hand[0] + (c[-1, 1] - c[0, 1]) * hand[1] < 0:
                c = c[::-1]                                               # one hand: every leaf is painted the same way round
            ys0, xs0 = max(0, sl[0].start - 2), max(0, sl[1].start - 2)
            ys1, xs1 = min(H, sl[0].stop + 2), min(W, sl[1].stop + 2)
            cell = lab[ys0:ys1, xs0:xs1] == i
            ce = extend_ends(c, 2.0 * hi, 1.0, lambda x, y: 0 <= x <= W - 1 and 0 <= y <= H - 1 and at(lab, x, y) == i)
            vs = veins(cell, ce[0] - (xs0, ys0), ce[-1] - (xs0, ys0), m, rng=rng, variety=vein_variety)
            if not vs:
                vs = [(c - (xs0, ys0), 1.0, 0)]
            emit(vs, xs0, ys0, tier)
    if whole is not None:
        leaf_cov = float((covered.astype(bool) & inner).sum()) / float(max(1, inner.sum()))
        if whole_cov >= leaf_cov - float(cfg.get("leaf_whole_cover", 0.05)):
            out.clear()                                                   # the long strokes do (nearly) as well: they replace the leaves
            covered[:] = 0
            emit(whole, 0, 0, 0, taper_mm=3.0)
    # the contour (LEAF_OUTLINE): one stroke along the shape's real outline, half a mark inside it, painted after the leaves.
    # It only follows the true outline (where the shape meets paper), never the seam to a thin part painted as a line
    if outline:
        rim_d = np.where(area, dmask, np.nan)
        for ct in measure.find_contours(rim_d, 0.5 * m):
            p = _resample(_smooth(ct[:, ::-1]), step)
            if len(p) < 4 or float(np.hypot(*np.diff(p, axis=0).T).sum()) < 3.0 * res:
                continue
            tt = np.linspace(0.0, 1.0, len(p))
            ww = cap * np.clip(0.6 + 0.4 * np.minimum(tt, 1.0 - tt) / 0.05, 0.6, 1.0)        # full width, very short tapered ends
            leaf_id += 1
            out.append((p, ww / res, len(tiers), leaf_id))
    return out


def _disk(r_px):
    r = max(1, int(round(r_px)))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def picture(mask, strokes, res, share, out_png, title=""):
    """Left: every leaf (or blob) in its own colour with its strokes and direction arrows. Right: the paint at the REAL width
    (share x planned). strokes: (points px, widths mm, group) - rows run bottom -> top, as in the planner."""
    H, W = mask.shape
    leaves = np.zeros((H, W, 3), np.uint8)
    leaves[mask] = (255, 255, 255)
    paint = leaves.copy()
    rs = np.random.RandomState(1)
    cols = {}
    for p, w, grp in strokes:
        col = cols.setdefault(grp, tuple(int(v) for v in rs.randint(60, 230, 3)))
        q = np.round(p).astype(np.int32)
        for j in range(len(q) - 1):
            th = max(1, int(round(0.5 * share * (w[j] + w[j + 1]) * res)))
            cv2.line(leaves, tuple(map(int, q[j])), tuple(map(int, q[j + 1])), col, th, cv2.LINE_AA)
            cv2.line(paint, tuple(map(int, q[j])), tuple(map(int, q[j + 1])), (64, 0, 184), th, cv2.LINE_AA)
    for p, w, grp in strokes:
        q = np.round(p).astype(np.int32)
        cv2.polylines(leaves, [q.reshape(-1, 1, 2)], False, (0, 0, 0), 1, cv2.LINE_AA)
        if len(q) >= 2:
            cv2.arrowedLine(leaves, tuple(map(int, q[-2])), tuple(map(int, q[-1])), (0, 0, 0), 1, cv2.LINE_AA, tipLength=3.0)
    missed = mask & (paint[..., 0] == 255) & (paint[..., 1] == 255)
    cov = 1.0 - missed.sum() / max(1, mask.sum())
    img = np.hstack([leaves[::-1], np.full((H, 12, 3), 40, np.uint8), paint[::-1]])
    ys, xs = np.nonzero(mask[::-1])
    if ys.size:
        img = img[max(0, ys.min() - 40):min(H, ys.max() + 20)]
    cv2.putText(img, f"{title}  {len(cols)} leaves/blobs, {len(strokes)} strokes, real coverage {100 * cov:.0f}%", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
    cv2.imwrite(out_png, img)
    return cov


if __name__ == "__main__":
    import sys
    import paint as P
    img, colour, out_png = sys.argv[1], sys.argv[2], sys.argv[3]
    width_mm = float(sys.argv[4]) if len(sys.argv) > 4 else 117.0
    cfg = P.default_config()
    cfg.update({"brush_w": 1.0, "thin_max": float(sys.argv[5]) if len(sys.argv) > 5 else 2.79, "line_depth": 2.8})
    res = P.RES
    mask = [L for L in P.find_layers(P.load_rgb(img, width_mm, rotate=90)) if L["name"] == colour][0]["mask"]
    dmask = cv2.distanceTransform(np.pad(mask.astype(np.uint8), 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
    n, lab = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    rng = random.Random(3)
    strokes = []
    for k in range(1, n):
        comp = lab == k
        if comp.sum() < 4 * res * res:
            continue
        for p, w, tier, leaf in leaf_strokes(comp, dmask, res, cfg, rng, P._cell_centre_line, P._extend_ends, P.voronoi_field):
            strokes.append((p, w, (k, leaf)))
    cov = picture(mask, strokes, res, float(cfg["voro_mark_share"]), out_png, f"leaf style, {colour}")
    print(f"{len(strokes)} strokes, {sum(np.hypot(*np.diff(p, axis=0).T).sum() for p, *_ in strokes) / res / 1000:.2f} m, real coverage {100 * cov:.1f}%")
