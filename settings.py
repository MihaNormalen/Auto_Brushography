# =====================================================================================================
#  settings.py  -  EVERY number lives here (this file replaces config.py + the old settings.py).
#
#  Edit a number, save, run again. The command line can still override a few of them for one run.
#  Priority:  command line  >  this file  >  built-in default.
#  A value of None means "not set here": the brush profile / the built-in default decides.
#
#  HEIGHTS (one idea):  machine Z = TOUCH_Z + height.   All heights below are mm ABOVE the touch height,
#  except the ones marked "machine Z". TOUCH_Z is where the brush tip touches the paper; the ladder measures
#  it exactly (BRUSH_PROFILE below). Moved or re-mounted the brush? Run the ladder again.
# =====================================================================================================
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent            # files are always found next to the scripts


# =============================== 1. HEIGHTS ==========================================================
TOUCH_Z = 4.2                # 2026-11-01: brush re-mounted, touches at Z 4 (by hand; the probe said 4.09). Was ~6.1 (2026-10-28)
#   machine Z where the brush tip touches the paper (rough). TRAINING: the ladder starts 0.5 mm above it and goes
#   down to machine Z 0.2. PAINTING: used only when there is NO brush profile; with a profile the measured contact is used.

HOME_LIFT_Z = 8.0
#   machine Z the brush is lifted to right after homing Z and BEFORE homing X and Y. Keep it well ABOVE the touch height:
#   homing X/Y drags the brush across the paper if it is still touching. (This used to be Z_SAFE = 2.8: too low.)

CAL_HOVER_MM = 3.0
#   calibration: hover this far above the touch height over a corner (low = less parallax error, but never touching).

# --- painting (mama / paint.py) ---
TRAVEL_MM = 3.0            # lift between strokes
HIGH_MM = 15.0              # high travel (to and from a dish)
WIPE_EXIT_MM = 9.4         # height of the wipe after the dip
PAINT_DIP_Z = 2.0          # how far the brush goes into a paint dish (above the touch height)
DIP_DEEPEN_MM = 0.0        # every dip this much lower than the one before (the paint level sinks as it is used). 0 = always PAINT_DIP_Z
DIP_FLOOR_Z = None         # ... but never below this machine Z (measure the bottom of the dish, keep a margin). Needed for DIP_DEEPEN_MM

# --- the water dish used by TRAINING (names kept from the old config.py) ---
Z_SAFE_DIP = 20.2          # clearance height around the water dish
DIP_Z = 2.0                # depth into the water
Z_WIPE_TRAVEL = 8.2        # height while wiping on the rim


# =============================== 2. SPEEDS ===========================================================
#   shared: the ladder paints at the same speed as paint.py, so the brush profile matches what you paint.
PAINT_FEED = 3000           # painting speed, mm/min
PAINT_ACCEL = 200          # acceleration while painting, mm/s2
PLUNGE_FEED = 2000          # speed of the brush going down onto the paper, mm/min
WIPE_FEED = 2000           # speed of the wipe move after a dip, mm/min
FEED_TRAVEL = 12000        # travel speed, mm/min
ACCEL_TRAVEL = 12000       # travel acceleration, mm/s2


# =============================== 3. THE BRUSH PROFILE ================================================
BRUSH_PROFILE = None
#   Name of a ladder file in ladder_dataset (with or without .json), or None = no profile.
#   Create one with:   python train.py --name S4Z32
#   With a profile, every mark is painted at the depth where the REAL brush makes that width.


# =============================== 4. TRAINING (train.py = the ladder) =================================
LADDER_LEVELS = 12         # how many depths
LADDER_REPEATS = 2         # strokes per depth (1 is fast but noisy: one bad stroke moves the contact height)
LADDER_LENGTH_MM = 40      # length of each stroke
LADDER_ABOVE_MM = 0.5      # the first depth is this far ABOVE the touch height, so the contact is found, not guessed
TRAIN_DEPTH_MM = None      # how far below TOUCH_Z the ladder goes. None = down to machine Z 0.2
TRAIN_AREA_MM = None       # (x0, x1, y0, y1) in machine mm where strokes may go. None = the paper the camera finds
PARK_X = 0.0               # where the toolhead waits while the camera looks (machine mm); must be out of the picture
PARK_Y = 0.0


# =============================== 5. PAINTING (paint.py) ==============================================
ARTWORK = None             # e.g. "PoletnaModra.png": then you can run  python paint.py  with nothing after it
OUTPUT = None              # e.g. "out.gcode"  (None = the artwork's name + .gcode)
ARTWORK_WIDTH_MM = None    # None = as large as fits on the paper
ROTATE = "auto"            # "auto" = the artwork is turned by 90 degrees when it then fits larger (landscape art on portrait paper), 0 = never, 90 = always
H4XX0R = True             # True = every program in the h4xx0r theme (the same as --h4xx0r: paint, preview, calibrate*)
SIGNATURE = "podpis2.png"  # painted FIRST (right after the probes), in the bottom margin under the artwork's right edge. None = no signature
SIGN_INSIDE = True         # True: the signature goes INSIDE the artwork's bottom right corner (no extra bottom margin: the artwork is
                           # bigger, and paints over the signature where they meet). False: in its own margin under the artwork
SIGN_INSET_MM = 5.0        # SIGN_INSIDE: distance of the signature from the artwork's bottom and right edges
SIGN_WIDTH_MM =35.0       # width of the signature (the height follows the picture)
SIGN_YEAR = True           # the year (of the run) to the right of the signature
SIGN_GAP_MM = 4.0          # between the artwork's bottom edge and the signature
SIGN_EDGE_MM = 6.0         # paper left free below the signature (the bottom margin grows to fit it)
SIGN_LINE_MM = 1.6         # mark width of the signature lines (sets their depth from the probe curve). 1.3 = the shallowest depth: broken lines (2026-10-30)
SIGN_DEPTH_MM = 0.1        # the signature is pressed this far below the touch height (the paper tilt is still followed), not through the probe
                           # curve: SIGN_LINE_MM 1.6 landed 0.75 mm deep there and painted 3-4 mm blobs (2026-11-05). None = depth from SIGN_LINE_MM
SIGN_SPUR_MM = 1.0        # signature pieces shorter than this that touch another piece (spurs at cusps and crossings) are dropped; the rest is joined into long strokes
SIGN_YEAR_MM = 6.0        # height of the year. 4 mm came out as dashes with this brush

Z_OFFSET = None            # machine Z of the touch height. None = the profile's contact (or TOUCH_Z). A number overrides it.
LINE_DEPTH = 3.0           # (2026-10-29 eve: 4.0 pressed too deep on vic; the 29.10 probe gave 2.85 mm at 3.0 vs 3.36 at 4.0, deeper only squashes the brush. Before: 4.0, brush of 28.10. probes clean to 4.0 deep / 4.5 mm; 2.8 flattened every mark over 3.3 mm) None = as deep as the profile goes (or all the room below). A number = at most this deep, mm.
WIDTH_MIN = None           # only WITHOUT a profile: marks this narrow or narrower are painted at the touch height
WIDTH_REF = None           # only without a profile: marks this wide or wider are painted at the full LINE_DEPTH
BRUSH_WIDTH = None         # None = the thinnest line of the profile (or 1.0). Spacing of the lines in filled shapes.
MAX_STROKE_WIDTH = None    # marks narrower than this are ONE centre-line stroke, wider shapes are filled. None = the widest the brush makes.
POINT_STEP = None          # distance between points along a centre-line stroke, mm (None = 0.4)
INFILL_TYPE = "lines"      # styles "lines"/"bands" only: wide shapes filled with "lines" (parallel) or "rings" (following the outline). --infill overrides
OUTLINE = False            # styles "lines"/"bands" only: one stroke along every shape's outline too. --no_outline switches it off
PLAN_WORKERS = 0           # CPU cores used for planning (0 = all, at most 8)

DIP_MIN_MM = 70.0          # painted path before the next dip: shortest ...
DIP_MAX_MM = 90.0        # ... and longest. Each dip picks a random distance in between. Smaller = dips more often.
DIP_BY_WIDTH = True        # wide strokes use more paint: each painted mm counts by its width / DIP_REF_WIDTH_MM (x0.5 .. x3)
DIP_REF_WIDTH_MM = 1.5     # at this width DIP_MIN/MAX_MM hold as they are (3 mm wide = dip after half the distance, 1 mm = 1.5x as far)
LINE_ONE_STROKE = True     # a line is always painted in ONE stroke, never cut by a dip: a line that does not fit into the paint left gets its
                           # dip BEFORE it and is then painted to its end, even if longer than DIP_MAX_MM. False = cut at the dip, as before
                           # (2026-11-03: ko/24 had 20 lines cut by 36 dips, the long architecture lines)

# paint dishes: how the brush dips and wipes (the water dish of training has its own numbers further below)
PAINT_DIP_JITTER = 10.0    # the dip point wanders this far from the centre, so one spot is not used up
PAINT_DIP_SPIRAL_R = 15.0  # radius of the little spiral in the dish (shrinks for a small pool)
PAINT_DIP_SPIRAL_LOOPS = 1.0
PAINT_WIPE_R = 31.0        # the wipe point is this far from the dish centre, towards the next stroke

# finding the dishes in the camera picture
PALETTE_RADIUS_MM = 200.0  # a paint dish must be within this distance of DIP_X / DIP_Y
POOL_MIN_AREA_MM2 = 300.0  # smaller pools are specks, not dishes
POOL_MIN_ROUNDNESS = 0.3
POOL_MIN_SAT_PASTEL = 35   # how colourful a light (pastel) paint in a dish must be to be found. Lower = paler paints found, more false dishes
# colours of the ARTWORK: which pixels are a colour (own layer, own dish) and which are black
DARK_CHROMA = 70.0         # a dark pixel with less colour than this (saturation x value / 255) is black, not a colour
LAYER_MIN_CHROMA = 90.0    # a whole colour layer that is dark (value < 150) and weaker than this is merged into black (navy cast of ink)
LAYER_FRINGE_SHARE = 0.8   # a colour layer with this share of its pixels within 1 mm of black is the tinted halo of black strokes: black
MACHINE_X_MIN = -1e9       # the start of X travel (-1e9 = no limit); a bigger sheet is used only from here
MACHINE_Y_MIN = -1e9       # the start of Y travel (-1e9 = no limit)
MACHINE_X_MAX = 289.0      # the end of X travel (2026-10-28: a probe line sent to X 296 stopped at ~290); a bigger sheet is used only up to here
MACHINE_Y_MAX = 453.0      # the end of Y travel (the snapshot position)
SNAPSHOT_X = 0.0          # where the toolhead waits while the camera photographs the paper
SNAPSHOT_Y = 453.0           # the far end of Y travel: out of the picture of a big sheet
WATER_X = None             # optional rinse dish between colours (machine mm, radius)
WATER_Y = None
WATER_R = 15.0


# =============================== 6. THE WATER DISH (training) AND THE FALLBACK DISH ==================
DIP_X = 148.0              # position of the water dish (training); paint.py uses it for colours with no dish (black, white)
DIP_Y = 0.0
DIP_WIPE_R = 32.0          # wipe point distance from the dish centre
DIP_JITTER = 8.0
DIP_SPIRAL_LOOPS = 1.0
DIP_SPIRAL_R = 8.0


# =============================== 7. MACHINE, CAMERA, CANVAS (rarely changed) =========================
SERIAL_PORT = "COM9"
SERIAL_BAUD = 115200
SERIAL_TIMEOUT = 5

CAMERA_INDEX = 0
CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1080
CAMERA_EXPOSURE = -5
CAMERA_BACKEND = "auto"    # auto | dshow | msmf | v4l2 | avfoundation
CAMERA_FOURCC = "MJPG"
CAMERA_SET_EXPOSURE = False   # True = apply CAMERA_EXPOSURE manually

# canvas corners (mm) = the four calibration dots: bottom-left, bottom-right, top-right, top-left
CANVAS_CORNERS_MM = [
    (80.0, 200.0),
    (200.0, 200.0),
    (200.0, 430.0),
    (80.0, 430.0),
]
CALIB_MARKER = True        # an extra origin dot MARKER_OFFSET_MM to the +X side of the bottom-left dot (works for any camera rotation)
MARKER_OFFSET_MM = 12.0
CAMERA_X_AXIS = "down"     # only used when CALIB_MARKER = False: where machine +X / +Y point in the camera image
CAMERA_Y_AXIS = "left"
DOT_DWELL_MS = 300         # calibration with dots: how long the brush rests on a dot
DOT_FEED = 500
X_OFFSET = 0.0
Y_OFFSET = 0.0
VERSION = "clean 1"        # shown in the calibration_verify window

CALIBRATION_FILE = str(BASE_DIR / "calibration.npz")
PAPER_FILE = str(BASE_DIR / "paper.json")
LENS_FILE = str(BASE_DIR / "lens.json")           # the camera lens, measured ONCE with calibrate_grid.py (a photo of a ruled sheet)

# --- calibration with dots (calibrate_dots.py): the machine paints a grid of dots on any white paper, the camera finds them ---
CAL_DOTS_X = 6                  # dots along the paper's long side (machine Y) ...
CAL_DOTS_Y = 5                  # ... and across it (machine X). 30 dots take about 4 minutes.
CAL_DOT_MARGIN_MM = 15.0        # keep this far from the paper edge
CAL_DOT_PRESS_MM = 1.0          # press this far below the contact height: a clear dot about 2 mm wide
CAL_DOT_COLOR = "auto"          # which paint dish to use: "auto" = the black dish if there is one (a dark colour shows best)
CAL_DOTS_PER_DIP = 4            # dots per dip (the brush runs dry after a few)

# --- ramp probes (probe.py, or paint.py --probe): measure where THIS paint touches THIS paper, on the white margin ---
PROBES = True                  # every paint.py job starts with the probes (the contact depends on paint, paper, speed, brush). False = never; --no_probe = skip once
PROBE_ASK = False               # True = ask before painting: "Paint the probes...? [y/N]" (or, when this sheet's probe is reused,
                                #   "Start painting now? [y/N]"). False = never ask: probes straight away, artwork after a 10 s countdown
PROBE_COUNT = 2                 # 4 = one line along each edge: touch height, width curve, brush offset AND the tilt of the paper (checked).
                                # 2 = left + right edge only (the paper lies flattest there): the height follows a straight slope from one
                                #     line to the other, the brush offset along the lines comes from the last probe; no cross-check
PROBE_PAIR = ("left", "right")  # PROBE_COUNT = 2: which two opposite edges. ("left", "right") = as before; ("top", "bottom") where the
                                #     sides are bent (2026-11-08, the big 411 x 277 sheet). Top / bottom as the camera picture shows them
PROBE_LINE_MM = 140.0            # length of each line (centred on its edge; one dip has to last for it)
PROBE_STEP_STROKES = False      # 2026-11-01: ko/15 with True looked worse to the user. (with PROBE_COUNT = 2) a row of short strokes along the TOP edge, each landed and lifted exactly like an
                                # artwork line at its own depth: the touch height and the width curve are taken from THEM (the sloping
                                # lines only give the tilt between left and right). False = height and curve from the sloping lines
PROBE_STEP_COUNT = 10           # how many short strokes
PROBE_STEP_LEN_MM = 5.0         # length of each (its middle is measured, where the depth is constant)
PROBE_STEP_GAP_MM = 3.5         # paper between two strokes (the glide out of one and into the next must not touch the other)
PROBE_STEP_ABOVE_MM = 0.75      # the first stroke is this far ABOVE the expected contact (should stay in the air or barely mark) ...
PROBE_STEP_BELOW_MM = 2.75      # ... the last this far below it; the others evenly in between
PROBE_END_MM = 8.0              # paper left free at each end of an edge
PROBE_MARGIN_MM = 0.25          # the paint reference = the measured touch minus this (a little deeper)
PROBE_Z_EXTRA_MM = 0.0          # your own correction on top of the measured touch: minus = paint deeper, plus = higher
PROBE_SAMPLE_MM = 1.0           # one point of the line (one Z step, one width measured) every this many mm
PROBE_RETRIES = 0               # when a line already marks at its start: paint all lines again this many times, starting higher (0 = no retry)
PROBE_RETRY_MM = 1.0            # ... each retry starts this much higher
PROBE_APPLY_OFFSET = True       # True = shift the job by where the probes saw the paint land (brush tip offset). False = paint where told
PROBE_FILE = "probe_result.json"   # where the probe result is saved and read from (next to the scripts)
PROBE_ABOVE_MM = 4.0           # (2026-11-08: 4 for the big 411 x 277 sheet, it touched above 4.55; was 2.0) a ramp starts this far ABOVE the contact you expect (in the air); the real contact varies over the paper ...
PROBE_BELOW_MM = 4.0            # ... and ends this far below it (never below machine Z 0.3). The deepest point measured = the widest
                                # mark a job uses: 3.0 stopped at 2.75 mm deep / 2.88 mm wide while the line was still widening (2026-10-25)
PAINT_COLOR = "auto"           # "black" = the whole artwork is painted in black, whatever colours the picture has (by design);
                                # "auto" = every colour of the picture becomes a layer with its own dish. --color overrides it
PROBE_COLOR = "auto"           # fallback dish for the probes (normally the darkest colour of the artwork is used) (a dark colour shows best on white paper)
PROBE_CONTACT_WIDTH_MM = 0.9    # "contact" = where the line first reaches this width (about your thinnest line)
PROBE_USE_ANISO = False         # True = press deeper / shallower by stroke direction (the probe's 'aniso'). Off: from only two lines per
                                # direction it is noise (0.68/1.24, 0.95/0.95, 0.79/1.03 for the same brush) and made curves go up and
                                # down by about 1 mm (2026-10-30)
PROBE_INSET_MM = 5.0            # distance of the ramps from the edge of the usable paper
PROBE_OFFSET_MAX_MM = 10.0      # the probes measure where the paint lands vs. where the machine is told; a larger offset is not used
                                # (was 6: a lowered brush on 2026-10-26 measured 6.35 mm, cleanly, and the job landed 6 mm off)
PROBE_OFFSET_SCATTER_MM = 0.5   # ... nor one where the four lines disagree with one common shift by more than this (RMS)

# --- the touch check while painting (check.py): at every dip the camera looks at the new strokes, a depth map is learned, missing strokes are painted again ---
CHECK = False                   # EXPERIMENTAL. True = every job runs the touch check (the same as paint.py --check); --no_check skips it once
CHECK_EVERY_DIPS = 1            # a check every this many dips (a dip comes every 90-110 mm of paint)
CHECK_SETTLE_S = 1.0            # wait after parking so the picture is sharp
CHECK_MAX_CORR_MM = 0.3         # the learned correction never goes beyond this (deeper or shallower)
CHECK_REPAINT = True            # paint the strokes that are still missing at the end of a layer
# painterly infill (off = plain lines, as before)
INFILL_STYLE = "skeleton"  # every shape = centre-line strokes, depth from the probe width curve; wider shapes get side strokes the same way (2026-10-28)
# INFILL_STYLE = "voronoi" # blobs: calligraphic Voronoi cells, one stroke each, big ones first
# INFILL_STYLE = "voronoi" # calligraphic dashes: one thin-thick-thin stroke per small cell, biggest first (run of 2026-10-25 18:44)
# INFILL_STYLE = "cells"   # blobs: strokes laid cell by cell, following the shape (the previous default)
INFILL_STYLE_BY_COLOR = {} # mix styles on one artwork, e.g. {"red": "leaf", "blue": "voronoi"}; colours not listed use INFILL_STYLE
# INFILL_STYLE = "wide"    # fixed-width strokes, one width apart (fast, what you had in the last run)
# INFILL_STYLE = "bands"   # parallel strokes along each piece's direction, width follows the shape
# INFILL_STYLE = "field"   # strokes traced along the edge direction (experimental, short strokes)
# INFILL_STYLE = "painterly"  # older broken-stroke fill (experimental)
# INFILL_STYLE = "lines"   # plain parallel lines (anything not listed above falls back to this)
PAINTERLINESS = 0.6           # 0 = close to lines, 1 = very loose
PAINTERLY_LEN_MM = (8.0, 45.0) # length range of the broken strokes, in mm
INFILL_COVERAGE = 1.0  

# GIF made automatically at the end of each job
AUTO_GIF = False               # False = no GIF
OVERLAY_SHOW = "art"           # <name>_overlay.png shows on the paper: the original artwork ("art") or the planned strokes ("strokes")
STROKE_PHOTOS = "off"        # a photo with the head out of view after every leaf ("leaf"), every stroke ("stroke") or never ("off")
                               # -> folder strokes_<date>_<time>\ and strokes_<date>_<time>.mp4 at the end
STROKE_PHOTO_SECONDS = 5.0     # only for the time estimate printed before the job: how long one stroke photo takes
STROKE_PHOTO_LINES = 1         # thin lines (no leaf): one photo every this many strokes
STROKE_PHOTO_SETTLE_S = 0.4    # wait after parking before the photo (raise if the photos are blurred)
STROKE_PHOTO_FPS = 12          # video speed of the stroke time-lapse
TIMELAPSE = False              # record timelapse_<date>_<time>.mp4 of the whole job from the camera (works with the photos on or off)
TIMELAPSE_EVERY_S = 2.0        # a frame every this many seconds (smaller = smoother, longer video)
TIMELAPSE_FPS = 15             # video speed: 15 frames per second -> every 2 s of painting = 1/15 s of video (30x faster)
TIMELAPSE_MIN_CHANGE = 0.4     # frames that hardly differ from the last one (drying, waiting) are left out; 0 = keep every frame
PHOTO_EVERY_DIPS = 0           # 0 = no photo every few dips (start and final photos still taken)
REPAIR_AT_END = True      # repair the white spots after drying (False = the job ends with the final photo)
DRY_WAIT_MIN = 1         # wait before every photo that gets judged or repaired

# blob style (INFILL_STYLE = "cells")
BLOB_MIN_MM = 6.0          # shortest blob kept, mm. Raise for longer strokes, lower for more dots
BLOB_CLEAR = 0.4           # how close a blob may come to the paint beside it (0.4 = slight overlap, 0.5 = just touching)
CELL_LEN_MM = (10, 60)     # blob length range, mm (min, max)
BLOB_PASSES = 1        # 1 = big blobs only (default). 2 or 4 = add smaller passes in the first run
BLOB_LEFTOVER = False  # True = also paint small leftover strokes in the first run
CELL_LEN_SMALL = (6, 15)      # blob length range for small shapes, mm
CELL_LEN_LARGE = (15, 60)     # blob length range for large shapes, mm
SHAPE_SMALL_MM2 = 20          # shapes this small (or smaller) get the small range
SHAPE_LARGE_MM2 = 70         # shapes this large (or larger) get the large range
BLOB_RAMP_MM = 2.5           # length of the lead-in and lead-out ramp on each blob (0 = no ramp, the old behaviour)
FAINT_LINE_THR = 190         # thin light-grey lines (grey up to this, 0 = black, 255 = white) are painted too, as thin lines. Without it only
                             # what is darker than 140 is painted: the faint 1 px lines of a small picture were left out (ko/48). None = off
LINE_SMOOTH_PX =0.2         # how much skeleton lines are smoothed (gaussian sigma in planning pixels, 8 px = 1 mm). 2.0 = as before; smaller
                             # keeps bends and small wiggles sharper (1.0 = about half), 0 = raw pixel steps (jagged)
LINE_TIP_MM = 0.5           # lines run on to the pointed ends of their shape (down to where it is this wide) and lift out of them: thin-thick-thin. 0 = off
LINE_WIDTH_SLACK = 1.5     # a line up to this x the widest mark stays ONE line stroke (not cut into line / leaf pieces where it gets a bit wider; skeleton style: no side strokes, its edges come out a bit thin)
PROBE_TOUCH_W_MAX = 1.5      # a probe whose brush is already wider than this 'at the touch' measured the paper too low: it is not reused (a new probe is painted)
# Along a line ONLY its width sets the depth (probe curve). Landing and lifting happen outside the line, while moving: the brush comes
# straight down to LINE_LAND_HIGH_MM above the paper, LINE_LEAD_IN_MM before the line's start (in line with it), and glides down to the
# start; at the end it glides on the same way while rising. Also at a dip in the middle of a line, and in the repair pass.
LINE_LEAD_IN_MM = 3.0        # 0 = straight down onto the line and straight up (leaves a dot where it lands)
LINE_LAND_HIGH_MM = 2.0      # the straight-down / straight-up part stays this far above the touch height
BLOB_LEAD_AIR_MM = 0.6      # how far above the paint the brush starts each blob
BLOB_DEPTH_FROM_WIDTH = True # False = blobs keep the fixed contact depth, as before

# Leaves (INFILL_STYLE = "leaf"). Every run also saves <output>_<colour>_strokes.png: the leaves in colour with their strokes
LEAF_TARGET_MM = 10.0        # (6 = finer, more photos) usual leaf width: as many leaves across a shape as fit at about this width (thin arm = one leaf)
LEAF_WIDTH_MM = (4.0, 14.0)  # narrowest and widest leaf
LEAF_WHOLE_MAX = 4           # a long shape that at most this many full-length strokes fill (end to end, along its curve) is painted so: no small leaves. 0 = off
LEAF_WHOLE_COVER = 0.05      # ... when they cover at most this much less of it than the leaves would (0.05 = 5 % less; else the usual leaves)
LEAF_ASPECT = (4.0, 10.0)    # leaf length = this many times its width: it grows along the form as far as the shape lets it
LEAF_VARIETY = 0.25          # how much leaf sizes vary over the shape (0 = all alike)
LEAF_TIERS = (1.0, 0.5, 0.3) # leaf sizes: big leaves first, then smaller ones for the gaps and corners
LEAF_VEIN_VARIETY = 0.4      # veins in one leaf vary between (1 - this) and full width (0 = all as wide as the brush goes)
LEAF_DIRECTION = "alternate" # "alternate" = neighbouring veins run opposite ways (both tips even); "hand" = all one way (VORO_HAND_DEG)
LEAF_OUTLINE = False         # one contour stroke along every shape's outline, after its leaves (crisp silhouettes; leaves stop inside it)
LEAF_DETAIL = 0.6            # smaller leaves where the outline is busy (hands, hat, face): 0 = off, 1 = strong
REGION_STYLES = {}           # leaves everywhere (your choice). {"elongated": "voronoi"} = long thin pieces (ripples, grass) in another style; {} = leaves everywhere
ELONGATED_RATIO = 5.0        # a piece this many times longer than wide counts as "elongated"

# Voronoi blobs (INFILL_STYLE = "voronoi"): one calligraphic stroke per cell, one direction, biggest blobs first (picture: blob_method.png)
VORO_TIERS = (1.0, 0.6, 0.35)  # blob sizes, as shares of the widest mark: big ones where they fit, then smaller ones for corners and gaps
VORO_BLOB_ASPECT = (2.5, 5.0)  # blob length = this many times its width (random in between). Higher = longer, more line-like
VORO_HAND_DEG = 45.0         # every blob is painted this way round: 0 = left to right, 90 = bottom to top, 45 = diagonal up-right
VORO_CALLIGRAPHY = True      # thin entry, swells to full width at 40 % of the length, fine tail. False = width only from the cell
VORO_MARK_SHARE = 0.7        # share of the planned width the brush REALLY paints (run of 2026-10-25: ~0.7)
VORO_CELL_WIDTH = 0.7        # cell width as a share of what one blob really paints at its widest (below 1 = neighbouring swells overlap)
VORO_OVERLAP_MM = 0.3        # how far a blob reaches past its own cell into its neighbours (never past the outline)
VORO_RELAX = 3               # smoothing steps for the cells (0 = raw, more = more even cells)
VORO_FIELD_SIGMA_MM = 5.0    # how smoothly the stroke direction follows the shape (smaller = follows every bend)
VORO_AXIS_WEIGHT = 0.3       # pull towards each piece's long axis (0 = follow the outline only)

# repair at the end of the job (gap method)
REPAIR_BY_GAPS = True      # True = paint only the empty spots of the artwork (recommended)
REPAIR_GAP_DONE_PX = 30   # stop repairing when fewer empty pixels than this are left
REPAIR_TRAVEL_MM = 2.0     # lift between repair strokes, mm above the paint contact
REPAIR_MIN_GAP_MM2 = 2.0     # (was 8: left all the 5-8 mm2 white spots in solid shapes, 2026-10-27) ignore empty specks smaller than this (mm²). Raise = fewer small fills, big white spots still filled
REPAIR_RIM_MM = 3.0          # edge strips thinner than this are not counted as empty. Raise = less overpainting of edges
REPAIR_EDGE_BAND_MM = 0.5    # ... but only within this distance of the outline: further inside, thin white strips (between veins) ARE repaired
REPAIR_SPECK_MERGE_MM = 1.5  # pinholes closer than this merge into one patch that is repainted as a whole (0 = off)
REPAIR_MIN_PROGRESS = 1.25   # stop when a pass keeps more than this fraction of the previous empty pixels
REPAIR_MIN_STROKE_MM = 1.2   # repair strokes shorter than this are not painted (they would only be dots)
REPAIR_WIDTH_SHARE = 0.9     # a repair stroke is pressed for at most this share of the widest mark (less deep = paler)
REPAIR_TAPER = True          # a repair stroke blends in (no 'stairs'): smooth width along it, and each end runs on over the paint around
                             #   it (inside the artwork's shape), thinning to the thinnest mark - the brush glides in and out in an arc
REPAIR_TAPER_RATIO = 2.5     # ... each run is this many times the stroke's widest point long ...
REPAIR_TAPER_MIN_MM = 1.5    # ... but at least this ...
REPAIR_TAPER_MAX_MM = 6.0    # ... and at most this (mm). False / REPAIR_TAPER = False = the old square-ended repair strokes
REPAIR_FADED_SHARE = 0.5     # black that is painted but less than this share as dark as the job's well-painted strokes counts as
                             #   missing too: the repair gives it a second coat. 0 = off (only empty spots are repaired)
REPAIR_TAPER_MAX_TURN_DEG = 30.0   # a run stops where it would cut across the painted lines around it by more than this (degrees)
REPAIR_PASSES = 1000         # at most this many repair passes (1000 = until done or no progress; REPAIR_MIN_PROGRESS stops it)
REPAIR_STEP_MM = 0.2         # each repair pass presses this much deeper than the one before (2026-11-08: was 0.15) ...
REPAIR_MAX_DEEPER_MM = 2.0   # ... but never more than this below the first pass (was 1.0); never below machine Z 0.3 either
REPAIR_MAX_SHIFT_MM = 1.5    # no repair when the paint sits further than this from where the plan put it (the photo would mislead it)
REPAIR_LINE_MAX_MM = 2.5     # thin lines of the artwork (narrower than this) are checked along their centre line ...
REPAIR_LINE_NEAR_MM = 0.7    # ... and repainted where no paint lies within this distance of it ...
REPAIR_MIN_LINE_MM = 2.0     # ... over at least this length
REPAIR_MAX_STROKES = 400     # stroke method only (REPAIR_BY_GAPS = False): at most this many strokes repainted per pass
JUDGE_BY_DIFFERENCE = True   # the final photo is judged against the start photo (what got darker); False = the final photo alone

# terminal
VERBOSE = False            # False = key lines only; True = everything


# =============================== PERSONAL OVERRIDES ==================================================
# settings_local.py (next to this file, NOT in git: .gitignore) may set any of the values above for this computer only,
# e.g. another machine, camera or H4XX0R = True. It is read last, so what it sets wins. Without the file nothing changes.
try:
    from settings_local import *          # noqa: F401,F403
except ImportError:
    pass
