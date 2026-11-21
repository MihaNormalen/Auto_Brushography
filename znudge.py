"""znudge.py - your own Z correction for the brush on the paper, any time, also while a job runs.

    python znudge.py              # show it
    python znudge.py down 0.2     # the brush 0.2 mm deeper (it flies too high)
    python znudge.py up 0.1       # 0.1 mm higher (it presses too hard)
    python znudge.py 0            # back to plain (what the probe measured)

It adds up (down 0.2 twice = 0.4 deeper) and stays until you change it (also for the next run). paint.py reads it again every
few seconds and applies it to the painting moves only (the strokes on the paper, never the dips, rinse or travel), so a change
counts from the next stroke. Kept in z_nudge.json; limits ZNUDGE_MAX_DOWN_MM / ZNUDGE_MAX_UP_MM in settings.py.
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
FILE = os.path.join(HERE, "z_nudge.json")


def limits():
    try:
        sys.path.insert(0, HERE)
        import settings as S
        return float(getattr(S, "ZNUDGE_MAX_DOWN_MM", 1.0)), float(getattr(S, "ZNUDGE_MAX_UP_MM", 1.5))
    except Exception:
        return 1.0, 1.5


def read():
    """mm, + = higher, - = deeper; 0 when there is no file or it is broken."""
    try:
        with open(FILE, encoding="utf-8") as f:
            v = float(json.load(f).get("dz", 0.0))
    except Exception:
        return 0.0
    dn, up = limits()
    return max(-dn, min(up, v))


def write(v):
    dn, up = limits()
    v = round(max(-dn, min(up, float(v))), 2)
    tmp = FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"dz": v, "date": time.strftime("%Y-%m-%d %H:%M")}, f)
    os.replace(tmp, FILE)                         # never half a file for a running job
    return v


def say(v):
    if abs(v) < 1e-9:
        return "0: the brush paints at the height the probe measured"
    return f"{abs(v):.2f} mm {'DEEPER' if v < 0 else 'HIGHER'} than the probe measured"


def main(argv):
    if not argv:
        print(say(read()))
        return
    a = argv[0].lower()
    try:
        if a in ("down", "deeper", "d") and len(argv) > 1:
            v = write(read() - abs(float(argv[1])))
        elif a in ("up", "higher", "u") and len(argv) > 1:
            v = write(read() + abs(float(argv[1])))
        else:
            v = write(float(a))                       # a plain number: set it (- = deeper)
    except ValueError:
        raise SystemExit(__doc__)
    print("now " + say(v) + " (from the next stroke if a job runs)")


if __name__ == "__main__":
    main(sys.argv[1:])
