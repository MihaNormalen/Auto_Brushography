"""
rinse_container.py - writes rinse_container.stl (and rinse_container.scad) for the brush rinse container.

    python hardware/rinse_container.py              # a plain trough (put your own scrub material in)
    python hardware/rinse_container.py --ribs       # with printed scrub ribs across the rinse compartment

Outside LENGTH x WIDTH x HEIGHT. A long rinse compartment (with scrub ribs across it, their tops RIB_BELOW_RIM under the
rim, where the brush reaches) and, behind a full-height divider, a small dry compartment for a sponge / folded towel to
blot the brush. The brush reaches only about 6 mm below the paper: mount the container so its rim is about level with the
paper (sunk like the paint dishes), and fill the water close to the rim.

The mesh is built as a height field on a grid (every feature is a box standing on the bed), so it comes out closed and
watertight without any CAD library. Print it as it lies (open side up), no supports.
"""
import argparse
import os
import struct

LENGTH = 180.0          # outside, mm
WIDTH = 20.0
HEIGHT = 30.0
WALL = 2.0              # side walls, end walls and the divider
FLOOR = 2.0
DRY_INSIDE = 30.0       # the blot compartment, inside length (the rinse compartment gets the rest: 180 - 3 x 2 - 30 = 144)
RIB_THICK = 1.2         # scrub ribs across the rinse compartment
RIB_PITCH = 8.0         # rib to rib
RIB_BELOW_RIM = 4.0     # rib tops this far under the rim (the brush scrubs over them)
RIB_FREE_ENDS = 10.0    # no ribs this close to the ends of the rinse compartment (the brush goes down there)


def boxes(ribs=True):
    """The part as boxes standing on the bed: (x0, x1, y0, y1, height)."""
    b = [(0, LENGTH, 0, WIDTH, FLOOR),                                              # floor
         (0, LENGTH, 0, WALL, HEIGHT), (0, LENGTH, WIDTH - WALL, WIDTH, HEIGHT),     # long walls
         (0, WALL, 0, WIDTH, HEIGHT), (LENGTH - WALL, LENGTH, 0, WIDTH, HEIGHT)]     # end walls
    xd = LENGTH - WALL - DRY_INSIDE - WALL                                          # the divider
    b.append((xd, xd + WALL, 0, WIDTH, HEIGHT))
    if ribs:
        x = WALL + RIB_FREE_ENDS
        while x + RIB_THICK <= xd - RIB_FREE_ENDS + 1e-9:
            b.append((x, x + RIB_THICK, 0, WIDTH, HEIGHT - RIB_BELOW_RIM))
            x += RIB_PITCH
    return b, xd


def heightfield_mesh(bx):
    """A closed triangle mesh of the union of boxes standing on z = 0 (each grid cell gets the tallest box over it)."""
    xs = sorted({v for b in bx for v in b[:2]})
    ys = sorted({v for b in bx for v in b[2:4]})
    nx, ny = len(xs) - 1, len(ys) - 1
    h = [[0.0] * ny for _ in range(nx)]
    for i in range(nx):
        cx = 0.5 * (xs[i] + xs[i + 1])
        for j in range(ny):
            cy = 0.5 * (ys[j] + ys[j + 1])
            for x0, x1, y0, y1, z in bx:
                if x0 <= cx <= x1 and y0 <= cy <= y1:
                    h[i][j] = max(h[i][j], z)
    levels = sorted({0.0} | {z for b in bx for z in (b[4],)})
    tris = []

    def quad(a, b, c, d):                       # counter-clockwise seen from outside
        tris.append((a, b, c))
        tris.append((a, c, d))

    def H(i, j):
        return h[i][j] if 0 <= i < nx and 0 <= j < ny else 0.0
    for i in range(nx):
        for j in range(ny):
            z = h[i][j]
            if z <= 0:
                continue
            x0, x1, y0, y1 = xs[i], xs[i + 1], ys[j], ys[j + 1]
            quad((x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z))               # top
            quad((x0, y0, 0), (x0, y1, 0), (x1, y1, 0), (x1, y0, 0))               # bottom
    # vertical faces, split at every height level so the edges along each vertical line meet exactly (watertight)
    for i in range(nx + 1):
        for j in range(ny):
            a, b = H(i - 1, j), H(i, j)
            if a == b:
                continue
            x, y0, y1 = xs[i], ys[j], ys[j + 1]
            lo, hi = min(a, b), max(a, b)
            zz = [z for z in levels if lo <= z <= hi]
            for z0, z1 in zip(zz, zz[1:]):
                if a > b:   # the solid is on the -x side: the face looks +x
                    quad((x, y0, z0), (x, y1, z0), (x, y1, z1), (x, y0, z1))
                else:
                    quad((x, y0, z0), (x, y0, z1), (x, y1, z1), (x, y1, z0))
    for j in range(ny + 1):
        for i in range(nx):
            a, b = H(i, j - 1), H(i, j)
            if a == b:
                continue
            y, x0, x1 = ys[j], xs[i], xs[i + 1]
            lo, hi = min(a, b), max(a, b)
            zz = [z for z in levels if lo <= z <= hi]
            for z0, z1 in zip(zz, zz[1:]):
                if a > b:   # solid on the -y side: the face looks +y
                    quad((x0, y, z0), (x0, y, z1), (x1, y, z1), (x1, y, z0))
                else:
                    quad((x0, y, z0), (x1, y, z0), (x1, y, z1), (x0, y, z1))
    return tris


def check_closed(tris):
    """Every edge used exactly once in each direction = closed and consistently oriented."""
    from collections import Counter
    e = Counter()
    for t in tris:
        for k in range(3):
            e[(t[k], t[(k + 1) % 3])] += 1
    bad = [k for k, n in e.items() if n != 1 or e.get((k[1], k[0]), 0) != 1]
    return len(bad)


def write_stl(tris, path):
    with open(path, "wb") as f:
        f.write(b"rinse_container".ljust(80, b" "))
        f.write(struct.pack("<I", len(tris)))
        for a, b, c in tris:
            ux, uy, uz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
            vx, vy, vz = c[0] - a[0], c[1] - a[1], c[2] - a[2]
            n = (uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx)
            ln = max((n[0] ** 2 + n[1] ** 2 + n[2] ** 2) ** 0.5, 1e-12)
            f.write(struct.pack("<12fH", n[0] / ln, n[1] / ln, n[2] / ln, *a, *b, *c, 0))


def write_scad(bx, path, xd, ribs):
    rib = (f"    // scrub ribs across the rinse compartment, tops {RIB_BELOW_RIM:g} mm under the rim\n"
           f"    for (x = [wall + rib_free : rib_pitch : {xd:g} - rib_free - rib_thick]) translate([x, 0, 0]) "
           f"cube([rib_thick, W, H - rib_below_rim]);\n") if ribs else ""
    open(path, "w").write(f"""// rinse container for the brush (written by rinse_container.py; change the numbers and render in OpenSCAD)
L = {LENGTH:g}; W = {WIDTH:g}; H = {HEIGHT:g};      // outside
wall = {WALL:g}; floor_t = {FLOOR:g};
dry_inside = {DRY_INSIDE:g};                    // blot compartment, inside length
rib_thick = {RIB_THICK:g}; rib_pitch = {RIB_PITCH:g}; rib_below_rim = {RIB_BELOW_RIM:g}; rib_free = {RIB_FREE_ENDS:g};
xd = L - wall - dry_inside - wall;              // the divider
union() {{
    difference() {{
        cube([L, W, H]);
        translate([wall, wall, floor_t]) cube([L - 2 * wall, W - 2 * wall, H]);
    }}
    translate([xd, 0, 0]) cube([wall, W, H]);   // divider (full height: the blot side stays dry)
{rib}}}
""")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ribs", action="store_true", help="scrub ribs across the rinse compartment (default: a plain trough, "
                                                        "with something put in it, easier to print and to clean)")
    a = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    bx, xd = boxes(a.ribs)
    tris = heightfield_mesh(bx)
    bad = check_closed(tris)
    stl = os.path.join(here, "rinse_container.stl")
    write_stl(tris, stl)
    write_scad(bx, os.path.join(here, "rinse_container.scad"), xd, a.ribs)
    n_ribs = sum(1 for b in bx if b[4] == HEIGHT - RIB_BELOW_RIM)
    print(f"{stl}: {len(tris)} triangles, " + ("closed and watertight" if bad == 0 else f"{bad} OPEN EDGES (do not print)"))
    print(f"   outside {LENGTH:g} x {WIDTH:g} x {HEIGHT:g} mm; rinse compartment {xd - WALL:g} x {WIDTH - 2 * WALL:g} mm inside"
          + (f" with {n_ribs} scrub ribs (tops {RIB_BELOW_RIM:g} mm under the rim)" if n_ribs else "")
          + f"; blot compartment {DRY_INSIDE:g} x {WIDTH - 2 * WALL:g} mm inside; walls and floor {WALL:g} mm")


if __name__ == "__main__":
    main()
