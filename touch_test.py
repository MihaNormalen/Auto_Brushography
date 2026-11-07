"""Where is the paper really? Lowers the brush by hand at a few spots of the sheet and compares with the probe result.

    python touch_test.py

Use a CLEAN, DRY brush (a drop of paint touches before the bristles do). At every spot the brush stops 1 mm above the
height the probe says the paper is at. Then:
    Enter  = 0.1 mm lower        u = 0.1 mm higher        t = it touches the paper NOW        s = skip this spot
The heights are saved to touch_test.json and compared with probe_result.json."""
import json

import numpy as np

import machine as M
import settings as S

STEP = 0.1
ABOVE = 1.0


def main():
    res = json.load(open("probe_result.json"))
    touch, cx, cy = float(res["touch_center"]), float(res["center_mm"][0]), float(res["center_mm"][1])
    tx, ty = float(res["tilt_x"]), float(res["tilt_y"])

    def probe_z(x, y):                                       # where the probe result puts the paper (touch) at (x, y)
        return touch + tx * (x - cx) + ty * (y - cy)

    C = np.array(res["paper_corners_mm"], float)
    mid = C.mean(axis=0)
    spots = [("middle", mid)] + [(f"towards corner {i + 1}", mid + 0.6 * (c - mid)) for i, c in enumerate(C)]
    print(f"probe result {res['created']}: touch Z {touch:.2f} in the middle, tilt {res['plane_spread_mm']:.2f} mm across the probes")
    m = M.open_machine(home=True)
    out = []
    try:
        for name, (x, y) in spots:
            pz = probe_z(x, y)
            z = pz + ABOVE
            m.send_line(f"G0 Z{float(S.HIGH_MM) + touch:.3f} F3000")
            m.send_line(f"G0 X{x:.3f} Y{y:.3f} F6000", timeout=60)
            m.send_line(f"G1 Z{z:.3f} F600")
            m.send_line("M400", timeout=60)
            print(f"\n{name} (X {x:.0f} Y {y:.0f}): the probe says the paper is at Z {pz:.2f}. Brush at Z {z:.2f}.")
            while True:
                k = input("  Enter = lower, u = higher, t = touches now, s = skip: ").strip().lower()
                if k == "t":
                    out.append({"spot": name, "x": x, "y": y, "probe_z": pz, "real_z": z})
                    print(f"  touches at Z {z:.2f}: the paper is {z - pz:+.2f} mm from where the probe says (+ = higher)")
                    break
                if k == "s":
                    break
                z += STEP if k == "u" else -STEP
                if z < pz - 3.0:
                    print("  3 mm below the probe height: stopping here (check the brush)")
                    break
                m.send_line(f"G1 Z{z:.3f} F300")
                m.send_line("M400", timeout=60)
                print(f"  Z {z:.2f}")
    finally:
        m.send_line(f"G0 Z{float(S.HIGH_MM) + touch:.3f} F3000")
        m.disconnect()
    json.dump(out, open("touch_test.json", "w"), indent=1)
    if out:
        d = np.array([o["real_z"] - o["probe_z"] for o in out])
        print(f"\nreal paper minus probe: " + ", ".join(f"{o['spot']} {o['real_z'] - o['probe_z']:+.2f}" for o in out))
        print(f"mean {d.mean():+.2f} mm (+ = the paper is higher than the probe says, - = lower: the brush then paints in the air)")
        print("saved touch_test.json")


if __name__ == "__main__":
    main()
