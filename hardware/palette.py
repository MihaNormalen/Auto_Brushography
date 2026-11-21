"""
palette.py - writes palette_holder.stl and palette_cup.stl: a 12-colour palette of removable cups.

    python hardware/palette.py

The HOLDER is printed twice (two pieces side by side make the full palette, 2 x HOLDER_L x HOLDER_W): each piece holds
3 x 2 cups in pockets POCKET_H deep, with finger notches in the long walls to lift a cup out. Glue the pieces down
(hot glue) end to end. The CUPS (print 12) are plain open boxes that stand in the pockets with CLEARANCE all round.

Both are built as height fields from boxes standing on the bed (closed, watertight, no CAD library), printed as they lie,
no supports. Uses the mesh code of rinse_container.py.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rinse_container import check_closed, heightfield_mesh, write_stl      # noqa: E402

HOLDER_L = 140.0        # one holder piece (two make the palette: 280 x 60)
HOLDER_W = 60.0
HOLDER_FLOOR = 2.0
POCKET_H = 10.0         # pocket depth above the floor (the cups stand on the floor, held by the walls)
OUTER_WALL = 2.5
DIVIDER = 2.0
COLS, ROWS = 3, 2       # cups per piece
NOTCH_W = 14.0          # finger notches in the long outer walls, one per cup
NOTCH_LEFT = 3.0        # wall height left at a notch (above the floor)

CLEARANCE = 0.3         # play between cup and pocket, each side
CUP_H = 25.0            # cup height (the paint depth is CUP_H - CUP_FLOOR)
CUP_WALL = 1.6
CUP_FLOOR = 1.6


def pocket_size():
    px = (HOLDER_L - 2 * OUTER_WALL - (COLS - 1) * DIVIDER) / COLS
    py = (HOLDER_W - 2 * OUTER_WALL - (ROWS - 1) * DIVIDER) / ROWS
    return px, py


def holder_boxes():
    px, py = pocket_size()
    top = HOLDER_FLOOR + POCKET_H
    b = [(0, HOLDER_L, 0, HOLDER_W, HOLDER_FLOOR)]                                   # floor
    # end walls and the dividers across
    b += [(0, OUTER_WALL, 0, HOLDER_W, top), (HOLDER_L - OUTER_WALL, HOLDER_L, 0, HOLDER_W, top)]
    for c in range(1, COLS):
        x = OUTER_WALL + c * px + (c - 1) * DIVIDER
        b.append((x, x + DIVIDER, 0, HOLDER_W, top))
    # the long divider between the two rows
    y = OUTER_WALL + py
    b.append((0, HOLDER_L, y, y + DIVIDER, top))
    # long outer walls, with a finger notch in the middle of every cup
    for y0, y1 in ((0, OUTER_WALL), (HOLDER_W - OUTER_WALL, HOLDER_W)):
        xs = [0.0]
        for c in range(COLS):
            cx = OUTER_WALL + c * (px + DIVIDER) + px / 2
            xs += [cx - NOTCH_W / 2, cx + NOTCH_W / 2]
        xs.append(HOLDER_L)
        for k in range(0, len(xs), 2):                                               # wall pieces between the notches
            b.append((xs[k], xs[k + 1], y0, y1, top))
        for k in range(1, len(xs) - 1, 2):                                           # the notches: low wall
            b.append((xs[k], xs[k + 1], y0, y1, HOLDER_FLOOR + NOTCH_LEFT))
    return b


def cup_boxes():
    px, py = pocket_size()
    L, W = px - 2 * CLEARANCE, py - 2 * CLEARANCE
    return [(0, L, 0, W, CUP_FLOOR), (0, L, 0, CUP_WALL, CUP_H), (0, L, W - CUP_WALL, W, CUP_H),
            (0, CUP_WALL, 0, W, CUP_H), (L - CUP_WALL, L, 0, W, CUP_H)], L, W


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    for name, bx in (("palette_holder.stl", holder_boxes()), ("palette_cup.stl", cup_boxes()[0])):
        tris = heightfield_mesh(bx)
        bad = check_closed(tris)
        write_stl(tris, os.path.join(here, name))
        print(f"{name}: {len(tris)} triangles, " + ("closed and watertight" if bad == 0 else f"{bad} OPEN EDGES (do not print)"))
    px, py = pocket_size()
    _, L, W = cup_boxes()
    print(f"   holder piece {HOLDER_L:g} x {HOLDER_W:g} x {HOLDER_FLOOR + POCKET_H:g} mm (print 2: palette {2 * HOLDER_L:g} x {HOLDER_W:g}), "
          f"{COLS} x {ROWS} pockets {px:.1f} x {py:.1f} mm, finger notches {NOTCH_W:g} mm")
    print(f"   cup (print {2 * COLS * ROWS}) outside {L:.1f} x {W:.1f} x {CUP_H:g} mm, inside {L - 2 * CUP_WALL:.1f} x {W - 2 * CUP_WALL:.1f} mm, "
          f"{CUP_H - CUP_FLOOR:.1f} mm deep, about {(L - 2 * CUP_WALL) * (W - 2 * CUP_WALL) * (CUP_H - CUP_FLOOR) / 1000:.0f} ml")


if __name__ == "__main__":
    main()
