// rinse container for the brush (written by rinse_container.py; change the numbers and render in OpenSCAD)
L = 180; W = 20; H = 30;      // outside
wall = 2; floor_t = 2;
dry_inside = 30;                    // blot compartment, inside length
rib_thick = 1.2; rib_pitch = 8; rib_below_rim = 4; rib_free = 10;
xd = L - wall - dry_inside - wall;              // the divider
union() {
    difference() {
        cube([L, W, H]);
        translate([wall, wall, floor_t]) cube([L - 2 * wall, W - 2 * wall, H]);
    }
    translate([xd, 0, 0]) cube([wall, W, H]);   // divider (full height: the blot side stays dry)
}
