"""znudge_keys.py - your Z correction with the keyboard, in its own window, also while a job runs.

    python znudge_keys.py

    arrow UP     the brush 0.05 mm HIGHER        Page Up     0.2 mm higher
    arrow DOWN   the brush 0.05 mm DEEPER        Page Down   0.2 mm deeper
    0            back to plain (what the probe measured)
    Q / Esc      close this window (the correction stays)

The same as  python znudge.py up/down ...  (z_nudge.json, limits ZNUDGE_MAX_DOWN_MM / ZNUDGE_MAX_UP_MM in settings.py):
paint.py reads it every 2 s and applies it to the strokes on the paper only, from the next stroke on.
"""
import msvcrt
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import znudge as ZN

STEP, BIG = 0.05, 0.2


def show(v, what=""):
    print(f"\r{what:<12} now {ZN.say(v):<70}", end="", flush=True)


def main():
    print(__doc__.split("\n\n", 1)[1].split("The same")[0].rstrip())
    print()
    v = ZN.read()
    show(v, "")
    while True:
        ch = msvcrt.getwch()
        if ch in ("\x00", "\xe0"):                        # arrows and Page keys come as two characters
            k = msvcrt.getwch()
            d = {"H": STEP, "P": -STEP, "I": BIG, "Q": -BIG}.get(k)
            if d is None:
                continue
            v = ZN.write(ZN.read() + d)
            show(v, "higher" if d > 0 else "deeper")
        elif ch == "0":
            v = ZN.write(0.0)
            show(v, "plain")
        elif ch.lower() == "q" or ch == "\x1b":
            print("\nclosed; the correction stays: " + ZN.say(ZN.read()))
            return


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nclosed; the correction stays: " + ZN.say(ZN.read()))
