"""One picture per pass that shows how color_paint chose it: the sheet as the camera saw it, what it reflects, the
target, the difference, what every paint would do (where it is wanted, its gain), the winner and why, and afterwards
what the pass really did (expected vs measured). Reads ko/ only (safe during a run: no camera, no machine).
   python color_explain.py            -> ko/cp_explain/cp_explain_passNN.png
   python color_explain.py <folder>   -> into that folder"""
import os, sys, json, itertools
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import color_paint as CP          # read only: nothing in it is changed or run except its pure functions
import color_chart as CC

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(CP.DIR, "cp_explain")
os.makedirs(OUT, exist_ok=True)
SHOW = {"orange": (0.95, 0.80, 0.10), "red": (0.85, 0.15, 0.55), "blue": (0.15, 0.65, 0.90), "black": (0.15, 0.15, 0.15)}
NAME = {"orange": "yellow", "red": "magenta", "blue": "cyan", "black": "black"}

plan = json.load(open(CP.PLAN))
(Hw, Ww), (Hp, Wp) = CP.work_shape(plan)
paints = plan["paints"]
target = CP.target_lab(cv2.imread(plan["image"]), plan.get("dark", 45.0), (Hw, Ww))
target_bgr = cv2.cvtColor(np.clip(np.dstack([target[..., 0] * 2.55, target[..., 1] + 128, target[..., 2] + 128]), 0, 255)
                          .astype(np.uint8), cv2.COLOR_LAB2BGR)
blank = cv2.imread(CP.BLANK)
grid, _ = CP.camera_points(plan)


def straight(photo):
    return cv2.remap(photo, grid[..., 0], grid[..., 1], cv2.INTER_LINEAR)


def rgb(bgr):
    return bgr[..., ::-1]


def analyse(now, done, fac):
    """choose() step by step, keeping what it computes on the way."""
    cap, look = int(CP.setting("COLOR_MAX_COATS", 6)), int(CP.setting("COLOR_LOOK_AHEAD", 2))
    usable = [p for p in paints if done.get(p, 0) < cap]
    combos = np.array(list(itertools.product(*[range(min(look, cap - done.get(p, 0)) + 1) for p in usable])))
    facs = np.ones((len(combos), 3))
    for j, p in enumerate(usable):
        facs *= fac[p][None, :] ** combos[:, j:j + 1]
    cost = CP.COAT_COST * combos.sum(axis=1)
    flat, tg = now.reshape(-1, 3), target.reshape(-1, 3)
    d_now = np.linalg.norm(CP.lab_array(flat) - tg, axis=1)
    best, d_best = np.empty(len(flat), int), np.empty(len(flat))
    for a in range(0, len(flat), 8000):
        d = np.linalg.norm(CP.lab_array(flat[a:a + 8000, None, :] * facs[None]) - tg[a:a + 8000, None, :], axis=2) + cost[None]
        best[a:a + 8000] = d.argmin(axis=1)
        d_best[a:a + 8000] = d.min(axis=1)
    best = cv2.medianBlur(best.reshape(Hw, Ww).astype(np.uint16), 3).astype(int)
    want = combos[best]
    cands = {}
    for j, p in enumerate(usable):
        m = CP.clean(want[..., j] >= 1, Hp, Wp)
        mi = m.ravel()
        gain = 0.0
        if mi.any():
            d_after = np.linalg.norm(CP.lab_array(flat[mi] * fac[p]) - tg[mi], axis=1)
            gain = float((d_now[mi] - d_after).sum() / len(flat))
        cands[p] = {"paint": p, "mask": m, "want": want[..., j], "gain": gain, "share": float(m.mean()),
                    "ok": m.mean() >= CP.MIN_SHARE}
    return d_now.reshape(Hw, Ww), d_best.reshape(Hw, Ww), cands


def decide(cands):
    ok = [c for c in cands.values() if c["ok"]]
    if not ok:
        return None, "no paint is wanted anywhere: the end"
    top = max(c["gain"] for c in ok)
    if top < CP.setting("COLOR_STOP_GAIN", 0.4):
        return None, f"the best gain {top:.2f} < stop {CP.setting('COLOR_STOP_GAIN', 0.4)}: the end"
    good = [c for c in ok if c["gain"] >= CP.ORDER_SLACK * top]
    win = min(good, key=lambda c: CP.ORDER.index(c["paint"]))
    best = max(ok, key=lambda c: c["gain"])
    if win is best:
        why = f"{NAME[win['paint']]} has the biggest gain ({win['gain']:.2f})"
    else:
        why = (f"{NAME[best['paint']]} has the biggest gain ({best['gain']:.2f}), but {NAME[win['paint']]} "
               f"({win['gain']:.2f}) is lighter and gives at least {CP.ORDER_SLACK:.0%} of it: lighter paint first")
    return win, why


def tint(c):
    pic = np.ones((Hw, Ww, 3))
    col = np.array(SHOW[c["paint"]])
    w = np.clip(c["want"], 0, 2)
    for k, a in ((1, 0.45), (2, 0.85)):
        sel = c["mask"] & (w >= k)
        pic[sel] = 1 - a * (1 - col)
    return pic


# ---------------------------------------------------------------------------------- pass by pass, from the photos
fac = {p: CP.PAINTS[p]["t"] for p in paints}
fac = {p: np.array(v, float) for p, v in fac.items()}
book = {p: fac[p].copy() for p in paints}
done, prev_now, made = {}, None, []
for i, q in enumerate(plan["passes"], 1):
    f_before = os.path.join(CP.DIR, f"cp_before_pass{i:02d}.png")
    if i == 1:
        now, photo = np.ones((Hw, Ww, 3)), straight(blank)
    elif os.path.isfile(f_before):
        photo_full = cv2.imread(f_before)
        now, photo = CP.sheet_now(photo_full, plan), straight(photo_full)
    else:
        print(f"pass {i}: no photo before it, skipped")
        break
    learned_txt = ""
    if i > 1 and prev_now is not None:                   # what the previous pass really did (the same as learn())
        q0 = plan["passes"][i - 2]
        m = CP.mask_of(q0["picture"], now.shape[:2])
        m = cv2.erode(m.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        if m.sum() >= 200:
            t = np.clip(np.median((now / np.maximum(prev_now, 1e-3))[m], axis=0), 0.03, 1.0)
            made[-1]["measured"] = t
            fac[q0["paint"]] = t
    d_now, d_best, cands = analyse(now, done, fac)
    win, why = decide(cands)
    after = CP.apply(now, win, fac) if win else now
    d_after = np.linalg.norm(CP.lab_array(after) - target, axis=2)
    made.append({"i": i, "q": q, "now": now, "photo": photo, "d_now": d_now, "d_best": d_best, "cands": cands,
                 "win": win, "why": why, "after": after, "d_after": d_after, "fac_used": {p: fac[p].copy() for p in paints}})
    if win and win["paint"] != q["paint"]:
        print(f"NOTE pass {i}: recomputed choice {win['paint']} differs from the real one {q['paint']} "
              f"(start values of the paints are not known exactly)")
    done[q["paint"]] = done.get(q["paint"], 0) + 1
    prev_now = now

# ---------------------------------------------------------------------------------------------------- the pictures
for k, r in enumerate(made):
    i, q = r["i"], r["q"]
    nxt = made[k + 1] if k + 1 < len(made) else None
    fig = plt.figure(figsize=(16, 13.6), dpi=110)
    gs = fig.add_gridspec(3, 4, hspace=0.32, wspace=0.08, left=0.02, right=0.98, top=0.90, bottom=0.03)
    win = r["win"]
    fig.suptitle(f"Pass {i}: how the next layer was chosen  ->  {q['show'].upper()} on {100 * q['share']:.0f}% of the picture",
                 fontsize=17, weight="bold", y=0.985)
    fig.text(0.5, 0.945, "Why: " + r["why"], ha="center", fontsize=11.5, wrap=True)

    def show(pos, img, title, sub=None):
        ax = fig.add_subplot(pos)
        ax.imshow(img)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, fontsize=11.5, weight="bold")
        if sub:
            ax.set_xlabel(sub, fontsize=9.5)
        return ax

    # row 1: the sheet and the target
    show(gs[0, 0], rgb(r["photo"]), "1. Camera photo (straightened)",
         "the dry sheet before this pass" if i > 1 else "the blank sheet")
    show(gs[0, 1], rgb(CP.to_bgr(r["now"])), "2. What the sheet reflects",
         "photo / blank-sheet photo: uneven light removed")
    show(gs[0, 2], rgb(target_bgr), "3. Target", "the picture, tones mapped to paper..dark")
    ax = show(gs[0, 3], r["d_now"], f"4. Difference now: median dE {np.median(r['d_now']):.1f}",
              "bright = far from the target (Lab colour distance)")
    ax.images[0].set_cmap("magma"); ax.images[0].set_clim(0, 45)

    # row 2: every paint as a candidate
    for j, p in enumerate(paints[:4]):
        c = r["cands"].get(p)
        if c is None:
            show(gs[1, j], np.ones((Hw, Ww, 3)), f"{NAME[p]}: used up (max coats)")
            continue
        mark = "  <- CHOSEN" if win and win["paint"] == p else ""
        ax = show(gs[1, j], tint(c), f"{NAME[p]}: gain {c['gain']:.2f}{mark}",
                  f"wanted on {100 * c['share']:.0f}% (pale = 1 coat ahead, strong = 2)")
        if mark:
            for s in ax.spines.values():
                s.set_edgecolor("red"); s.set_linewidth(4)
    fig.text(0.5, 0.605, "5. Every spot tries all mixes of up to 2 more coats of each paint (81 mixes) and keeps the one "
             "nearest the target. Each paint's map = where that best mix contains it. Gain = how much one coat of it there "
             "brings the WHOLE picture nearer (mean dE).", ha="center", fontsize=9.5, style="italic", wrap=True)

    # row 3: the result
    show(gs[2, 0], rgb(CP.to_bgr(r["after"])), "6. Expected after this pass",
         f"median dE {np.median(r['d_after']):.1f}  (was {np.median(r['d_now']):.1f})")
    if nxt is not None:
        show(gs[2, 1], rgb(CP.to_bgr(nxt["now"])), "7. Really, after drying",
             f"median dE {np.median(nxt['d_now']):.1f}  (photo before pass {i + 1})")
        diff = np.linalg.norm(CP.lab_array(nxt["now"]) - CP.lab_array(r["after"]), axis=2)
        ax = show(gs[2, 2], diff, "8. Expected vs real: dE", "bright = the paint did something else than expected")
        ax.images[0].set_cmap("viridis"); ax.images[0].set_clim(0, 20)
    else:
        show(gs[2, 1], np.ones((Hw, Ww, 3)), "7. Really, after drying", "not photographed yet")
        show(gs[2, 2], np.ones((Hw, Ww, 3)), "8. Expected vs real", "not photographed yet")
    ax = fig.add_subplot(gs[2, 3])
    p = q["paint"]
    used, meas = r["fac_used"][p], r.get("measured")
    x = np.arange(3)
    ax.bar(x - 0.2, used, 0.38, color="#9aa", label="used for the choice")
    if meas is not None:
        ax.bar(x + 0.2, meas, 0.38, color=SHOW[p], label="measured on the sheet")
    ax.set_xticks(x); ax.set_xticklabels(["red light", "green light", "blue light"])
    ax.set_ylim(0, 1.05); ax.legend(fontsize=8.5, loc="lower right")
    ax.set_title(f"9. One coat of {NAME[p]} lets through", fontsize=11.5, weight="bold")
    ax.set_xlabel("1 = no effect, 0 = blocks all light", fontsize=9.5)
    name = os.path.join(OUT, f"cp_explain_pass{i:02d}.png")
    fig.savefig(name)
    plt.close(fig)
    print("wrote", name, "|", r["why"])
