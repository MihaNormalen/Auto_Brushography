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
TOUCH_Z = 0.0                # Z 0 = the Z switch = the brush touches the paper (the probe measures the real touch every run)
Z_MIN_MM = -6.0              # the lowest Z the program ever sends (below the touch); the axis has ~15 mm below the switch
Z_CONTROLLER_SHIFT_MM = 15.0  # (2026-11-21) the controller allows only Z 0..50 (M211): every Z is sent + this (and M206 sets its switch to it). Our Z range: -15 .. +35. 0 = off
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
Z_ACCEL = None             # Z acceleration sent to the controller at the start of a job (M201 Z..), mm/s2. None = the controller's own (Lerdge: 100, which makes every cup dip and every travel with Z slow). Try 300; if Z ever skips steps (brush too high/low, hits a rim), go back to None


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
H4XX0R = False            # (2026-11-21: off for now, kept) True = every program in the h4xx0r theme (the same as --h4xx0r: paint, preview, calibrate*)
SIGNATURE = "podpis2.png"  # painted FIRST (right after the probes), in the bottom margin under the artwork's right edge. None = no signature
SIGN_PAINT = True          # False = the signature's room is kept (the artwork is placed the same) but it is not painted: color_paint.py
                           # paints it only in its LAST pass, so all passes of one sheet sit in the same place
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

# color_paint.py (glazes): NO fixed number of passes. Before every pass the dry sheet is photographed and the pass that brings
# the painting nearest to the picture is chosen (which paint, where); it goes on until another pass would hardly help.
COLOR_MAX_PASSES = 0       # (2026-11-23, user: "it ends when it is done, not after a number of passes": 40 -> 0 = NO LIMIT;
                           # the end is COLOR_END_NO_BEST + the finish) at most this many passes, fill-ins not counted (one pass with its drying is about 45-70 min). 2026-11-16: 20 -> 40,
                           # only a safety net now: the run ends when the camera sees no more progress (COLOR_STOP_PATIENCE). (2026-11-15: 12 -> 20, the run of 13:18 stopped at 12 while passes still helped 0.5-0.6)
COLOR_MAX_COATS = 6        # at most this many coats of one paint on any spot (2026-11-13: one coat of the black took only ~10% of the light)
COLOR_STOP_GAIN = 0.25     # (2026-11-15: 0.4 -> 0.25) the end: when the best next pass would bring the painting less than this much nearer to the picture
                           # (colour difference dE, averaged over the whole picture; a pass that improves 10% of it by 4 = 0.4).
                           # (the raw promise; which pass is chosen goes by the promise x what that paint really kept, COLOR_TRUST_*)
# HONEST PROMISES (2026-11-16, run of 20:38: the detail passes promised 11.4 and did 4.1, so the end never came): after every
# pass the camera measures what it really did; each promise is then scaled by the share its paint's recent passes kept, and
# the choice goes by that (a paint that keeps failing loses to the others). The end: COLOR_STOP_GAIN and COLOR_STOP_PATIENCE.
COLOR_TRUST_WINDOW = 6     # the last this many passes (of each paint, and of all) count
COLOR_TRUST_PRIOR = 1.0    # a paint with few passes leans on all passes' share as much as this many dE of its own promises
COLOR_TRUST_RANGE = (0.15, 1.2)  # the share stays within these
# THE END BY THE CAMERA: when the last COLOR_STOP_PATIENCE passes (fill-ins not counted) did not make the sheet better than the
# best it has been by COLOR_STOP_MIN_GAIN, the run ends (run of 20:38: best after pass 25, passes 26-30 only made it worse). 0 = off
COLOR_STOP_PATIENCE = 3
COLOR_STOP_MIN_GAIN = 0.1
COLOR_STOP_STILL_GAIN = 0.5  # (2026-11-17) not stalled while the last pass still made the sheet this much better (dE)
# when it IS stalled, the run does not end: the paints of those COLOR_STOP_PATIENCE passes rest and the others get their
# turn (2026-11-17: it ended at dE 24 with the pink figure uncovered, opaque white never tried). The end = nothing else helps
COLOR_ORDER_SLACK = 0.85     # (2026-11-17: was 0.5 in color_paint) of the passes expected to help at least this share of the
COLOR_LIGHTNESS_WEIGHT = (2.0, 1.5, 1.0)  # (2026-11-17, Butina) per stage: the lightness difference counts this many times where
                             # the coats are chosen and passes ranked - light and dark first, the colour after. 1 = plain dE
COLOR_REACHABLE_DARK = True  # the picture's darkest becomes what the swatches can really reach, never darker than --dark ...
COLOR_REACHABLE_MAX_LIFT = 10.0  # ... and at most this much lighter (L) than --dark
COLOR_AVOID_WET = True       # (2026-11-17) a pass leaves out the LAST pass's area (still wet) when it has enough to do elsewhere:
COLOR_WET_MARGIN_MM = -1.0   # the bare paper and older, dry parts first; < 0: a pass may reach this far into the last (wet) pass's area,
                           # so the shapes touch (2026-11-18: +2 mm left white lines between them); > 0: keep this far away
COLOR_RECENT_WEIGHT = (0.25, 0.6)  # spots painted in the last pass / the one before count this much when ranking passes
COLOR_NO_WAIT = True       # (2026-11-18) a pass that touches none of the last pass's area (plus COLOR_WET_MARGIN_MM) goes on at once:
COLOR_NO_WAIT_GAP_MM = 1.5  # (2026-11-22) ... the wet area grown by this: wet never meets wet (no run, no mud)
COLOR_NO_WAIT_SHARE = 0.3  # ... when the best pass would touch wet paint: the best pass on dry or bare paper goes on at once,
                           # if it brings at least this share of the best (user: "paint elsewhere while it dries")
COLOR_NO_WAIT_CHAIN = 2    # no drying, no photo, at most this many in a row; they are measured together at the next photo
                             # best one, the preferred order picks (lighter / covering first, whites last); higher = the best one
COLOR_LOOK_AHEAD = 2       # choosing a pass: every spot looks this many coats of each paint ahead (green = yellow AND cyan)
# "I have about a hundred outlines, but I'll shape it up." - Seen, in Style Wars (1983)
# BROAD FIRST, DETAIL LAST (like a brushographer): the passes go through stages. COLOR_STAGE_FROM = the first pass of each stage;
# per stage its smoothing and its infill style. Passes 1-4: broad areas (4 mm smoothing, long bands), 5-8: 2 mm, from 9: the
# detail (1 mm, centre-line strokes for edges and features). A single number / name instead of a tuple = the same for all.
COLOR_STAGE_FROM = (1, 5, 9)
COLOR_STAGE_DE = (None, 30.0, 25.0)  # (2026-11-18) ... and a stage only once the sheet is that near the picture (the best dE so far):
                           # middle at dE 26, detail at 17 - the structure first, details only when it is there (details came at dE 21 as blobs)
                           # 2026-11-22: 26/17 -> 30/25: the portrait stalled at dE ~27 in the BROAD stage after 8 passes,
                           # never reaching the detail (no articulation); user agreed to loosen it
COLOR_FOCUS_WEIGHT = (0.0, 1.0, 2.0)  # (Butina) per stage: the faces (found in the picture) and strong contrasts count 1 + this when ranking
COLOR_CRISP_EDGES = True   # (Butina) near the picture's sharp light-dark breaks a pass follows the wanted spots exactly (not the smoothed outline)
COLOR_EDGE_DL = 8.0        # ... a break = the lightness changes by more than this per mm
COLOR_MODULATION = 5.0     # (Butina, Cezanne's modulation) the lights this much warmer, the shadows this much cooler (Lab units at the
                           # ends of the picture's lightness range; the paper stays white); 0 = the photo's own colours
COLOR_TRY_UNUSED = True    # a paint not used lately (never, or not in the last COLOR_TRY_AGAIN_AFTER passes; not a white) gets a pass where it promises most, when that is at least
COLOR_TRY_UNUSED_SHARE = 0.25  # ... this share of the best pass's promise - its own pass then shows what it really does
COLOR_TRY_UNUSED_FROM = 3  # ... from this pass on (the first ones lay in the picture)
COLOR_TRY_UNUSED_MAX = 6   # ... at most this many such tries in one run (2026-11-20: 2 -> 6, it keeps trying)
COLOR_TRY_PER_PAINT = 2    # ... each paint at most this many times in one run
COLOR_TRY_AGAIN_AFTER = 4  # ... a paint not used in this many passes counts as unused again (the sheet has changed since)
COLOR_TRY_GAP = 2          # ... at least this many passes between two tries
COLOR_TEST_DENSITY_SCALE = 3.0  # (2026-11-20) deciding whether to TRY a glaze known only from its test stroke: it may be this
                           # many times as strong (optical density) - test strokes read glazes 3-6x too weak (run H: cyan never
                           # painted). The run measures its own from the paints with a test stroke AND a pass. Its pass then counts
COLOR_WHITE_PER_STAGE = 99  # (2026-11-23: 1 -> 99 = no limit, user: white wherever it belongs) (2026-11-20) at most this many white passes per stage (run H: 3 whites on 45% - milky, no glaze over it)
COLOR_EYE_BLUR_PX = 0.8    # (2026-11-19) the camera photo is blurred only this much (camera px, against noise); it was 2 px + 2 grid px:
COLOR_EYE_GRID_BLUR = 0.0  # ... ~2-3 mm, white holes unseen, black believed mid grey. This: an extra blur on the planning grid (0 = none)
COLOR_LEARN_LANDED = True  # a pass's paint is learned where it landed (changed at least half the most), not from the gaps between strokes
COLOR_HOLE_PASSES_MAX = 4  # bare paper where the picture is not white: at most this many hole passes in one run (the best paint for them)
COLOR_HOLE_MIN_SHARE = 0.003  # ... when the holes are at least this share of the picture
COLOR_HOLE_PAPER_L = 90.0  # ... a spot counts as bare paper above this L (and nearly no colour)
COLOR_HOLE_DL = 10.0       # ... where the picture is at least this much darker than the paper (or clearly coloured)
COLOR_HOLE_FROM_STAGE = 1  # (2026-11-20) hole passes from this stage on (0 broad, 1 middle, 2 detail), and only in the gaps INSIDE
                           # what earlier passes painted (run H: all the bare paper at the start counted - 4 hole passes on 88%)
COLOR_GLAZE_OVERRIDES = {}  # e.g. {"BAND_SPACING": 1.0, "PATCH_MAX_COVERED": 0.5, "VORO_OVERLAP_MM": 0.0, "BLOB_CLEAR": 0.5}: OFF -
                           # (2026-11-20) tested on run H passes 3 and 7: painted twice 36 -> 37%, 44 -> 45% - no help; the overlap
                           # is in the plan itself (side strokes at least one mark wide). A TRANSPARENT paint's pass would use these
COLOR_HOLE_GROW_MM = 0.7   # ... the hole pass a little wider than the holes, so it touches what is round them
COLOR_HOLE_MIN_GAIN = 0.3  # (2026-11-22) ... and only when it brings at least this (dE); the run of 11-21 got stuck in 4 hole
COLOR_HOLE_VS_REAL = 0.5   # ... passes of 0.05-0.15 each; and only when it brings at least this share of the best real pass
                           # (that would have brought 2.7 on the 21% of the picture never painted)
COLOR_GLAZE_OVER_WHITE = 1.5  # (2026-11-23: LIGHT glazes only - a dark glaze never goes on a fresh white, COLOR_DARK_AFTER_WHITE) a dry white's area counts this many times for a glaze (white lifts the value, the glaze over
                           # it brings the colour back: no milky picture); 1 = no preference
COLOR_WHITE_ONLY_LIFTS = True  # the whites only where the sheet is darker than wanted (to lift the value), never to dull a
COLOR_WHITE_LIFT_DL = 3.0  # ... colour (run G: milky); ... at least this much darker (L)
# DEPTH PLANES (2026-11-22, color_planes.py, docs/depth_planes_comfyui.pdf): a depth map of the picture (Depth Anything V2 on
# ComfyUI, the studio computer; kept in ko/comfy/<picture>_depth.png) sorted into planes far .. near. A pass's area is
# smoothed INSIDE each plane (nothing filled across a plane border), by its own amount, and kept sharp along depth breaks.
VOICE_ON = True            # (2026-11-22) voice.py: the Čopičograf speaks on the speakers (edge-tts, online); False = only printed
VOICE = "en-US-AndrewNeural"  # ... the voice (warm, conversational, male); Slovene: "sl-SI-RokNeural" with VOICE_LANG "sl"
VOICE_MIC = "USB Audio"    # voice_lab.py: the microphone (a part of its name: "USB Audio", "T89", "JBL"; None = Windows' default).
                           # A Bluetooth headset's mic is 8 kHz phone sound: fine to talk, poor for training the voice
VOICE_WHISPER = "small"    # ... speech to text on this computer: "small" (fast) or "medium" (better Slovene, slower)
VOICE_HEAR_LANGS = ("sl", "en")  # ... the languages it listens for
TALK_BRAIN = "ollama"      # talk.py: "ollama" = local, on the studio computer (no key, no cost); "claude" = Claude API (key)
TALK_OLLAMA_URL = "http://100.115.80.121:11434"  # ... Ollama on domimik (OLLAMA_HOST=0.0.0.0 there)
TALK_OLLAMA_MODEL = "gemma2:2b"  # ... the local model: gemma2:2b (1.6 GB, quick, simple); better: gemma2:9b (5.4 GB, fits 8 GB)
TALK_OLLAMA_CTX = 8192     # ... its memory in tokens (the character sheet alone is ~4500)
TALK_MODEL = "claude-opus-5-5"  # talk.py: the brain (Claude API, ANTHROPIC_API_KEY); "claude-haiku-5-5" = cheaper and faster
TALK_EFFORT = "low"        # ... how much it thinks before answering (low = quick spoken answers)
TALK_RECORD = True         # ... every sentence you say is also kept in ko/voice_dataset (for training your voice)
VOICE_SL = "sl-SI-RokNeural"  # ... the voice for answers in Slovene
PYLIBS = r"C:\py\pylibs" # the anthropic SDK is installed here (the Store Python's own folder is too long a path for it)
PAINT_VOICE = False        # (2026-11-22, user: "throw it out") the painting says nothing (pass start, drying, cups, end)
NARRATOR_ON = False       # (OFF, user) color_paint starts narrator_watch.py: the Čopičograf talks about what it does while it paints
NARRATOR_EVERY_S = 150     # ... at most this often (seconds) about brushography; real events (halfway, the cup sinking) sooner
VOICE_LANG = "en"          # ... the narrator's language: "en" or "sl" (the Slovene voices sounded stiff, 2026-11-22)
VOICE_LOCAL = "Microsoft David Desktop"  # ... when the online voice does not answer: this computer's own (offline) voice
VOICE_RATE = "-12%"        # ... speed: "-10%" slower, "+10%" faster (2026-11-22, user: talks too fast)
COMFY_URL = "http://100.115.80.121:8188"  # ComfyUI on the studio computer (Tailscale address); comfy_client.py, color_planes.py
COLOR_PLANES = "auto"      # depth planes: "auto" = the areas between DEPTH JUMPS, as many as the picture has (2026-11-22, user:
                           # "the planes are not a constant, they are found in each picture"); a number = that many depth groups;
                           # 0 = off (the whole picture as one surface). 4 fixed groups tested: dE 10.73 -> 10.71 (no harm, no gain
                           # in the forecast; the gain is on the sheet: no paint across a plane's edge, the edge kept sharp)
COLOR_FIGURE_FROM_SEGMENTATION = True  # (2026-11-22) the figure = ONE plane from OneFormerRegions on ComfyUI (head, body, arms AND knees;
COLOR_FIGURE_LABELS = "person"         # ... depth alone put the knees with the plants); these region names (comma separated) are the figure
COLOR_DEPTH_JUMP = 0.03   # ... a depth jump = the depth changes by more than this share of the picture's depth range per mm
                           # (mati_mahnic: 0.03 -> 17 planes, the figure apart from the fence and the leaves; 0.05 joined them)
COLOR_PLANE_SMOOTH = (1.0, 0.5)  # smoothing factor of the farthest .. nearest plane (times COLOR_MASK_SMOOTH_MM; not below 1 mm)
COLOR_PLANE_SMOOTH_FOCUS = 0.25  # ... the plane with the face (the figure) instead: broad stage 4 mm -> 1 mm on the figure
COLOR_PLANE_FOCUS = 0.5    # the whole figure plane counts this much in the focus map (the face itself 1.0, COLOR_FOCUS_WEIGHT)
COLOR_PLANE_EDGE_MM = 1.5  # along a depth break, this far on each side a pass follows the wanted spots exactly
COLOR_PLANE_MIN_MM2 = 25.0  # plane islands smaller than this (mm2) join the plane round them
COLOR_DEPTH_MODEL = "depth_anything_v2_vitl.pth"  # the depth model on ComfyUI
COLOR_MARGIN_MM = 16.0     # color_paint: the picture this far from the sheet's edge. (2026-11-22: was 10, fixed in the code: the probe
                           # ramps lie PROBE_INSET_MM 5 from the edge with a fat end up to 6.4 mm wide, and the picture's edge painted
                           # over the top and bottom probes on the 356 x 247 sheet.) 16 = 5 + 6.4 + ~4.5 mm room
COLOR_MASK_SMOOTH_MM = (4.0, 2.0, 1.5)  # color_paint: each pass's area is smoothed by this much (holes closed, frayed edges rounded) so strokes can be long; 0 = off (2026-11-15: pass 5 was a lace of 128 holes -> 7 mm strokes, a maze)
COLOR_PREFER = ("opaque", "glaze", "glaze")  # per stage: of the passes that help nearly as much, which go first - "opaque" = the covering paints (an opaque
                           # UNDERPAINTING, glazes on top later), "glaze" = the light glazes (2026-11-16: glazes first left the figure thin and light
                           # while the heavy opaque green went all round it)
COLOR_INFILL_STYLE = ("bands", "bands", "bands")  # (2026-11-23: detail "skeleton" -> "bands": centre lines on AREA corrections
                           # broke into short white dabs - the legs; bands paint the large parts in bands and the narrow ones as
                           # centre lines anyway. The real drawing is the finish: COLOR_FINISH) (2026-11-23: detail smoothing 1.0 -> 1.5)
                           # color_paint: the style per stage (None = INFILL_STYLE); "bands" = long parallel strokes along each area (smoothed pass 5: median 33 mm vs skeleton 17, painterly 11)
COLOR_FILL_TRIES = 2       # FINISH A COLOUR before the next: spots of a pass that the dry photo shows unchanged get the same paint again, at most this many times (0 = off)
COLOR_FILL_MISS = 0.3      # a spot counts as missed when it changed less than this share of what the paint should have done there
COLOR_FILL_MIN_SHARE = 0.015  # a fill-in pass only when the missed spots are at least this share of the picture (2026-11-16: 0.004 -> 0.015, 9 fill-ins on
                              # 0.5-3% each brought -0.14..+0.07: a whole pass and its drying for nothing)
COLOR_FILL_MIN_GAIN = 0.15    # ... and when it is expected to bring at least this (dE, after what that paint's passes really did)
COLOR_FILL_GAIN_ANYWAY = 0.4  # (2026-11-18) ... or, however small the missed share (white stripes between the bands), when it brings this much
COLOR_FILL_MIN_KEPT = 0.3  # no fill-in with a paint whose pass kept less than this share of its promise (white: 18% -> 2 useless fill-ins)
# THE PAINT CHECK: a pass that left this share of its own area unchanged (or changed the sheet almost not at all) had too little
# paint on the brush: its cup is dipped CUP_DEEPER_STEP_MM deeper from then on (cups.json, until python cups.py refill N). 0 = off
COLOR_DIP_FAINT = 0.25
CUP_DEEPER_STEP_MM = 1.0
# THE BRUSH CHECK DURING A PASS (paint.py BrushCheck): after the first dip, every COLOR_BRUSH_CHECK_EVERY-th dip and the dip after a
# faint one, the head parks out of view and the camera looks at the first stroke after that dip. Did it change the sheet less than
# COLOR_BRUSH_CHECK_MIN of what one coat of that paint should? Then the cup goes CUP_DEEPER_STEP_MM deeper (now, and remembered),
# the brush dips again and paints that stroke once more. Each check costs ~10 s. 0 = off
COLOR_BRUSH_CHECK_EVERY = 8
COLOR_BRUSH_CHECK_MIN = 0.35
COLOR_BRUSH_CHECK_MIN_EXP = 0.06  # (2026-11-18) only a stroke where this paint must clearly show is judged (over its own colour
                           # nothing shows and the check said OK); the swatches of other cups are never judged
COLOR_BRUSH_LOWER_MM = 0.2      # (2026-11-17) a faint stroke that does NOT fade along (paint on the brush, it touched too lightly):
COLOR_BRUSH_LOWER_MAX_MM = 0.6  # the brush goes this much lower for the rest of the pass first (at most this in all); still faint
                                # after that = too little paint -> the deeper dip (CUP_DEEPER_STEP_MM). 0 = never lower
# TEST STROKES in a strip above the picture (the side opposite the signature): when a paint is used for the first time, its
# first dip paints one stroke there, fat at one end and thin at the other; every stroke crosses all the others, so the camera
# sees each paint on bare paper and over every paint before it (what it really does: t, r) and its width at each depth.
COLOR_TEST_STRIP_MM = 36.0  # (2026-11-16: 30 -> 36 for the bands) the strip's width (0 = no test strokes, no strip). The picture gets smaller by this + the gap: use a bigger sheet
COLOR_TEST_FAT_MM = 5.0     # (2026-11-17: 3.5 -> 5, too thin to read) a test stroke's fat end is this wide (the probe curve gives the depth), its thin end the touch height
COLOR_TEST_GAP_MM = 6.0     # between the picture's top and the strip
COLOR_TEST_LEN_MM = 200.0   # a stroke's length at most (it is shorter when the picture is narrower); longer = the bands lie further apart
COLOR_TEST_RUN = 0.1       # (2026-11-17: 0.2 -> 0.1) a band runs along the strip this share of its length while it crosses it (steep: the bands lie far apart)
COLOR_TEST_CROSS = False   # (2026-11-17) False = all bands the same way, side by side, each on its own bare paper (crossing a wet
                           # band carried green into the yellow cup and blue into the white); True = by turns / and \, crossing
COLOR_TEST_ALL_AT_START = True  # (2026-11-17) pass 1 paints a band of EVERY paint with a cup (a swatch, like a brushographer):
                           # each paint measured as it is today before it is chosen (opaque magenta went on with an old value
                           # from another mix and turned the figure pink). An extra dip and rinse each, in pass 1 only
COLOR_TEST_HUE_MAX = 40.0  # (2026-11-17) a band whose hue is more than this many degrees off its cup's (as the camera sees it) is
COLOR_TEST_BRIGHT_TOL = 0.15  # (2026-11-17) a band read up to this much brighter than the paper (gloss) is trimmed to paper white,
                           # not thrown out (opaque yellow, cyan and magenta were rejected). A white's band on paper alone is never used
                           # not used: it was read off a neighbour (the missing yellow band read as dark green)
COLOR_TEST_LINES = 3        # each test is a BAND of this many parallel strokes from one dip (2026-11-16: one stroke was too thin to see)
COLOR_TEST_LINE_GAP_MM = 1.2  # between the band's strokes (wider bands leave no bare paper between them in the strip)
COLOR_TEST_EXTRA_AFTER = 2  # after this pass, paints never measured anywhere yet (not in ko/cp_measured.json) get their test stroke too (an extra dip and rinse each, no extra drying), so the choice knows them. 0 = off
COLOR_PROBE_MAX_AGE_H = 96.0  # (2026-11-16: 48 -> 96 for runs of up to 40 passes) color_paint: the probe of pass 1 stays valid this long for the later passes (paint.py alone: 6 h)
COLOR_LEARN_MIN_CHANGE = 0.02 # color_paint: a pass that changed the sheet less than this (median, linear) is not learned from (empty cup)

Z_OFFSET = None            # machine Z of the touch height. None = the profile's contact (or TOUCH_Z). A number overrides it.
LINE_DEPTH = 6.0           # (2026-10-29 eve: 4.0 pressed too deep on vic; the 29.10 probe gave 2.85 mm at 3.0 vs 3.36 at 4.0, deeper only squashes the brush. Before: 4.0, brush of 28.10. probes clean to 4.0 deep / 4.5 mm; 2.8 flattened every mark over 3.3 mm) None = as deep as the profile goes (or all the room below). A number = at most this deep, mm.
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
DIP_REF_WIDTH_MM = 4.5     # at this width DIP_MIN/MAX_MM hold as they are (6 mm wide = dip after half the distance, 2 mm = 1.5x as far)
                           # (2026-11-15: 1.5 -> 3.0, the wide glaze strokes dipped every 25-30 mm; 2026-11-22: 3.0 -> 4.5, user: "dips
                           # too often" - pass 1 black 197 dips; the probe now reaches 6.4 mm, so most band strokes counted x2. Watch
                           # for dry strokes at the end of a dip)
THIN_OVER_BANDS = False    # (2026-11-22) "bands" infill: the centre-line strokes of the narrow parts are CUT where they run over the
THIN_BAND_CLEAR = 0.35     # bands (a band nearer than this x the mark's width); pass 1 black of 11-22 had 60% of its area painted
                           # 2+ times (89% of it these strokes): now 11%. True = as before
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
# --- the long rinse container (rinse.py; hardware/rinse_container.stl). The camera finds it by its black rim every job
#     (python rinse.py detect  to check now). Or by hand:  python rinse.py calibrate  (then the hand points are used)
RINSE_FILE = "rinse.json"  # its last known place (None = never rinse). No container seen and no file = no rinsing
RINSE_BOX_MM = (180.0, 20.0, 29.0)   # the container outside: length, width, height (it stands on the board, like the paper)
RINSE_WALL_MM = 2.0        # its walls and divider
RINSE_DRY_INSIDE_MM = 30.0 # the blot compartment, inside length
RINSE_PAPER_MM = 0.3       # the paper's thickness (the rim is RINSE_BOX_MM[2] - this above the paper)
RINSE_SCRUB_BELOW_RIM_MM = 17.0  # the brush tip scrubs this far under the rim (+ RINSE_PRESS_MM): in the water, in your insert
BLOT_BELOW_RIM_MM = 4.0    # the top of the blot pad, under the rim
RINSE_END_MARGIN_MM = 15.0 # the scrub stays this far from the slot's ends (2026-11-12: 6 was too close, more room for error)
RINSE_SEEN_BIGGER = 1.037  # the camera sees the rim this much bigger than it is (its height): measured with clean water
                           # 2026-11-12. Fixed, because coloured water made the measured value wrong. None = measure every time
RINSE_LOCK_MM = 5.0        # the container is fixed: a camera reading within this of its saved place keeps the saved place (dark
                           # rinse water let the reading drift 2 mm overnight, 2026-11-13); further away = it was moved, taken over
RINSE_CLEAR_MM = 6.0      # the brush moves to, from and over the container this far above its rim
RINSE_KEEP_OUT_MM = 8.0    # the safety check: no other move may pass lower than the rim within this distance of the container
RINSE_PRESS_MM = 3.0       # scrubs this much deeper than that (or than the depth recorded by hand). 2026-11-21: 0 -> 3 (user: 3 mm lower)
RINSE_PASSES = 2           # along the slot and back, this many times (2026-11-12: 3 was more than needed)
RINSE_BOBS = 4             # then up and down at this many spots along the slot ...
RINSE_BOB_MM = 4.0         # ... by this much
RINSE_FEED = 2000          # speed in the water, mm/min
BLOT_TIMES = 3             # presses on the blot pad
BLOT_PRESS_MM = 1.5        # how far into the pad (below where the tip touched it)
BLOT_STEP_MM = 3.0         # each press this much further along
RINSE_AT_END = True        # rinse once more when the job (repairs included) is over: the brush does not dry with paint in it

# --- the 12-cup palette (cups.py; hardware/palette.py): 2 holders end to end, 6 cups along, 2 across. Found by the camera
#     every job (python cups.py detect  to check now); while it is found, its cups are the paint dishes (the old round dishes
#     are not searched). Cups 1-6 opaque, 7-12 transparent.
CUP_FILE = "cups.json"     # the cups' places, colours and dip counts (None = no palette: the old round dishes)
CUP_HOLDER_MM = (140.0, 60.0)   # one holder piece, outside (hardware/palette.py HOLDER_L, HOLDER_W)
CUP_INSIDE_MM = (39.9, 22.7)    # a cup's opening inside: along the palette, across
CUP_RIM_MM = 27.0          # the cups' rims over the board (holder floor 2 + cup 25)
CUP_FLOOR_MM = 3.6         # a cup's inside floor over the board (holder floor 2 + cup floor 1.6)
CUP_CLEAR_MM = 6.0         # the brush travels this far over the rims whenever it is over the palette
CUP_KEEP_OUT_MM = 8.0      # the safety check: no move outside a dip may pass lower than that within this distance of the palette
CUP_OPAQUE_SIDE = "paper"  # the column nearer the paper holds OPAQUE paint ("far" = the other one)
CUP_KIND = "any"           # which cups a job may dip into: "any", "opaque", "transparent" (paint.py --cups)
CUP_ROW_SAME_COLOUR = True # a row holds ONE colour: opaque in front, the same colour transparent behind. A transparent cup (it looks
                           # nearly black, deep) then takes its colour from its opaque partner, which the camera reads well
CUP_WHITE_CUPS = (3, 9)    # cups holding WHITE paint (3 opaque, 9 transparent): white in a white cup cannot be seen, so declared here
COLOR_MAX_MIX = 3          # color_paint: when choosing a pass, every spot tries mixes of at most this many coats still to come
                           # (with 7 paints the mixes grow fast: 3 keeps the choice to a few seconds)
CUP_SHIFT_MM = (0.0, 0.0)  # a correction by hand of where the camera puts every cup, mm: (along the palette, TOWARDS THE PAPER).
                           # The brush stands too close to the frame = a positive second number (e.g. (0.0, 4.0) = 4 mm towards the paper)
CUP_LOCK_MM = 3.0          # the palette is glued down: a camera reading within this of the saved places keeps the saved places
CUP_EMPTY_DE = 12.0        # a cup looking at least this different (Lab) from the picture of the empty cups holds paint
CUP_DIP_START_MM = 3.0    # the first dip in a cup goes this far over its floor ...
CUP_DIP_STEP_MM = 0.05     # ... every later dip this much lower (the paint level sinks as it is used; counted per cup, also over
                           # jobs: python cups.py refill  after filling up) ...
CUP_DIP_END_MM = 0.5       # ... but never lower than this over the floor (keep at least this much paint in the cups)
COLOR_REFILL_AT_START = True  # color_paint: a new run (pass 1) counts every cup as filled up: dips start again at CUP_DIP_START_MM, the paint
                              # check's deeper dips are forgotten (python cups.py refill N still does it by hand, e.g. in a drying pause)
CUP_DIP_FEED = 3000        # speed going down into the paint, mm/min (no splashing, the bristles fill; 2026-11-14: 600 was slow)
CUP_STIR_MM = 4.0          # the small stir in the paint (radius)
CUP_STIR_LOOPS = 1
CUP_STIR_FEED = 3000       # speed of the stir in the paint, mm/min (2026-11-15: 1500 -> 3000, faster dips)
CUP_JITTER_SHARE = 0.6     # the dip point wanders this share of the room left in the cup, so one spot is not used up
CUP_WIPE_BELOW_RIM_MM = 4.5   # (2026-11-15 evening: 0.5 -> 2.5, the user: wipe a few mm lower, then lift and go. Watch for dry strokes) (2026-11-15: 1.5 -> 0.5, it scraped too much paint off: 13 mm per dip) the wipe: the tip this far under the rim, drawn across it on the side facing the paper
# removing the drops when the brush leaves a cup (as remove_drops in the earlier painters, D2Woptimized.py):
CUP_SOAK_MS = 0            # a pause in the paint after the stir (ms), so the bristles take up paint
CUP_DRIP_RISE_MM = 6.0     # out of the paint slowly, first this far up over the dip depth ...
CUP_LIFT_FEED = 1500       # ... at this speed (mm/min): the drops run back into the cup. Also the lift after the wipe (was 400)
CUP_DRIP_MS = 150          # a pause inside the cup, over the paint, before the wipe: drops fall back in (was 800)
CUP_WIPE_FEED = 2000       # the wipe across the rim, not fast (fast flings the drops off; was 600)
CUP_WIPE_SPREAD_MM = 6.0   # the wipe point wanders this far along the wall, so the rim is not always wiped at one spot
CUP_WIPE_ACCEL = 1500      # (was 300) acceleration (mm/s2) from leaving the paint until over the rim: gentle (copicograf.py remove_drops acc)


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
CAMERA_ZOOM = 0              # (2026-11-21) the camera's own zoom / pan / tilt, SET EVERY TIME the camera opens (vision.py): another app
CAMERA_PAN = 0               # ... (the Windows Camera) can change them and the picture then moves like a bumped camera. 0/0/0 =
CAMERA_TILT = 0              # ... the full sensor picture (before: zoom 25, pan -129, tilt -21). None = leave the camera as it is
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
CAL_DOT_PRESS_MM = 1.5          # press this far below the contact height (TOUCH_Z): a clear dot. 2026-11-21: 1.0 -> 1.5 (new Z zero)
CAL_DOT_COLOR = "auto"          # which paint dish to use: "auto" = the black dish if there is one (a dark colour shows best)
CAL_DOTS_PER_DIP = 4            # dots per dip (the brush runs dry after a few)

# --- ramp probes (probe.py, or paint.py --probe): measure where THIS paint touches THIS paper, on the white margin ---
PROBES = True                  # every paint.py job starts with the probes (the contact depends on paint, paper, speed, brush). False = never; --no_probe = skip once
PROBE_ASK = False               # True = ask before painting: "Paint the probes...? [y/N]" (or, when this sheet's probe is reused,
                                #   "Start painting now? [y/N]"). False = never ask: probes straight away, artwork after a 10 s countdown
PROBE_COUNT = 4                 # (2026-11-10: 4 again, ko/84 landed 1.4 mm off in X: 2 lines never measure the offset along them) 4 = one line along each edge: touch height, width curve, brush offset AND the tilt of the paper (checked).
                                # 2 = left + right edge only (the paper lies flattest there): the height follows a straight slope from one
                                #     line to the other, the brush offset along the lines comes from the last probe; no cross-check
PROBE_BEND = True               # PROBE_COUNT = 4: a BENT sheet (top / bottom higher or lower than left / right, ko/85: 1 mm) is followed,
                                #   all four touches matched; False = a flat tilted plane only (was 0.5 mm off at every line on ko/85)
PROBE_PAIR = ("left", "right")  # PROBE_COUNT = 2: which two opposite edges. ("left", "right") = as before; ("top", "bottom") where the
                                #     sides are bent (2026-11-08, the big 411 x 277 sheet). Top / bottom as the camera picture shows them
PROBE_LINE_MM = 100.0            # the LONGEST a probe line may be. Its real length is set so the ramp gets PROBE_BELOW_MM below the
                                 #   expected touch within what one dip paints (measured by the last probe, probe_reach.json)
PROBE_LINE_MIN_MM = 50.0        # ... and the shortest: an edge with less free room (no paint) is not used
PROBE_CLEAR_MM = 8.0            # new probe lines keep this far from paint already along the edge (old probe lines, any day): the
                                #   camera finds the free stretches; no room on the PROBE_PAIR edges = the other pair is used
PROBE_REACH_MM = 50.0           # how far one dip paints after the touch, until a probe has measured it
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
ZNUDGE_MAX_DOWN_MM = 3.0        # (2026-11-18; 2026-11-20 1.0 -> 3.0, the user needed more) python znudge.py down 0.2 / up 0.1 / 0: your correction WHILE a job runs (z_nudge.json,
ZNUDGE_MAX_UP_MM = 1.5          # read every 2 s, the strokes on the paper only); never more than this deeper / higher
PROBE_SAMPLE_MM = 1.0           # one point of the line (one Z step, one width measured) every this many mm
PROBE_RETRIES = 2               # when a line already marks at its start: paint all lines again this many times, starting higher (0 = no retry)
PROBE_RETRY_MM = 1.0            # ... each retry starts this much higher
PROBE_APPLY_OFFSET = True       # True = shift the job by where the probes saw the paint land (brush tip offset). False = paint where told
PROBE_FILE = "probe_result.json"   # where the probe result is saved and read from (next to the scripts)
PROBE_ABOVE_MM = 4.0           # (2026-11-08: 4 for the big 411 x 277 sheet, it touched above 4.55; was 2.0) a ramp starts this far ABOVE the contact you expect (in the air); the real contact varies over the paper ...
PROBE_BELOW_MM = 6.0            # ... and ends this far below it (never below machine Z 0.3). The deepest point measured = the widest
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
DRY_EXTEND_MIN = 5.0     # the drying countdown between colour passes: + (or P) adds this many minutes, - takes them off, Enter goes on now
COLOR_DRY_MIN_OPAQUE = 40.0  # (2026-11-23) after a covering paint (gouache: opaque colours, whites) at least this many minutes
COLOR_DRY_MIN = 25.0     # color_paint: minutes of drying between passes (python color_paint.py ... --dry N for one run). 2026-11-16: humid days
COLOR_DRY_CHECK_MIN = 3.0  # ... then THE CAMERA CHECKS IT IS DRY: a photo this long before the end and one at the end; while the last pass's area
                         # still changes (wet paint dries lighter or darker), it waits this long again. 0 = no check
COLOR_DRY_STILL = 0.002  # ... dry = its area changed less than this more than the rest of the sheet (linear light, medians). Run of 20:38:
                         # the whole wet -> dry change was 0.007-0.027, the camera's noise ~0.0045 (a guess for 3 min, watch it)
COLOR_DRY_CHECK_MAX = 6  # ... at most this many extra waits
COLOR_DRY_VIEW = True    # color_paint: a camera window while the sheet dries (countdown, the last pass seen | target | pass; + - Enter work there too)

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
BAND_SPACING = 0.9           # "bands" infill: the bands are this share of the widest mark apart (below 1 = they overlap a little)
PATCH_MAX_COVERED = None     # a planned repair stroke (the plan's own gaps) is left out when the other strokes paint this share of
                             # it already (0.5); None = all kept. COLOR_GLAZE_OVERRIDES sets it for glazes
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
REPAIR_MIN_PROGRESS = 0.9    # stop when a pass keeps more than this fraction of the previous empty pixels (2026-11-08: was 1.25,
                             #   which only stopped when a pass made it WORSE: ko/70 ran passes 9-10 for nothing, ko/74 pass 3)
REPAIR_MIN_STROKE_MM = 1.2   # repair strokes shorter than this are not painted (they would only be dots)
REPAIR_WIDTH_SHARE = 0.9     # a repair stroke is pressed for at most this share of the widest mark (less deep = paler)
REPAIR_TAPER = True          # a repair stroke blends in (no 'stairs'): smooth width along it, and each end runs on over the paint around
                             #   it (inside the artwork's shape), thinning to the thinnest mark - the brush glides in and out in an arc
REPAIR_TAPER_RATIO = 2.5     # ... each run is this many times the stroke's widest point long ...
REPAIR_TAPER_MIN_MM = 1.5    # ... but at least this ...
REPAIR_TAPER_MAX_MM = 6.0    # ... and at most this (mm). False / REPAIR_TAPER = False = the old square-ended repair strokes
REPAIR_FADED_SHARE = 0.5     # black that is painted but less than this share as dark as the job's well-painted strokes counts as
                             #   missing too: the repair gives it a second coat. 0 = off (only empty spots are repaired)
SPUR_MARKS = 2.0             # a short stroke (under this many x the brush's thinnest mark, from the probe) whose spot the strokes
SPUR_COVERED = 0.6           #   around it already paint for this share is left out: a side branch that only lands a knob on a line
DOT_DAB_MARKS = 1.2          # a mark shorter than this many x the thinnest mark is a DOT: straight down, DOT_DWELL_MS, straight up
DOT_DWELL_MS = 120           #   (round, no comma tail from gliding in). 0 = dots are painted like lines
REPAIR_EDGE_INSIDE = True    # a repair stroke's EDGE (not its centre) goes on the artwork's outline: a brush wider than the missing
                             #   strip lies over the paint already there instead of sticking out as a bulge ('stairs'). False = centred
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


# ------------------------------------------------------------------------------------------- 2026-11-23: the end and the finish
COLOR_END_NO_BEST = 6        # THE COLOUR IS DONE when the sheet has not got better than its best for this many passes (no pass
                             # limit any more; run K was best at pass 30, passes 31-40 only worse). 0 = off
COLOR_FINISH = True          # then THE FINISH (color_lines.py): edge passes along the picture's lines (TEED on ComfyUI, else a
                             # local line drawing) where the sheet is too light there, and light accents with white; then the signature
COLOR_LINES_SOURCE = "teed"  # "teed" (ComfyUI, cached in ko/comfy/<name>_teed.png; local when ComfyUI is off) or "local"
COLOR_LINES_TEED_MIN = 0.5   # a TEED pixel counts as a line above this (0..1)
COLOR_EDGE_LINE_MM = 1.2     # an edge pass's lines this wide on the sheet (the mask; the brush's width curve paints them)
COLOR_EDGE_DL = 4.0          # a line needs a pass where its CONTRAST is missing on the sheet: the picture is darker there than round it
                             # (COLOR_LIGHTS_AROUND_MM) by this much more (L) than the sheet is (judged along COLOR_EDGE_ALONG_MM)
COLOR_EDGE_ALONG_MM = 3.0    # ... judged along the line over about this length, so whole lines are drawn, not specks
COLOR_EDGE_MIN_MM = 2.0      # ... line pieces smaller than this square are left out
COLOR_DARK_GLAZES = ("black",)  # these glazes count as dark always (the black measured weak, 0.66, on run K's sheet)
COLOR_EDGE_PASSES_MAX = 4    # at most this many edge passes (a kind stops when its last two passes together made dE worse by
COLOR_FINISH_MAX_LOSS = 1.0  # more than this)
COLOR_EDGE_PAINT_LUM = 0.8   # a line is drawn with a glaze that lets through less than this (a dark one), whichever comes nearest
COLOR_LIGHTS_DL = 6.0        # a light accent = the picture this much lighter (L) than its surroundings ...
COLOR_LIGHTS_AROUND_MM = 3.0 # ... within about this distance
COLOR_LIGHTS_SHEET_DL = 8.0  # ... and its contrast missing on the sheet by more than this (L)
COLOR_LIGHT_PASSES_MAX = 3   # at most this many light passes
COLOR_LIGHTS_FOCUS_MIN = 0.3  # lights only where the focus map is at least this (the figure 0.5, the face 1; not the hills)
COLOR_FINISH_MIN_SHARE = 0.002  # a finishing pass only for at least this share of the picture ...
COLOR_FINISH_MIN_GAIN = 0.04 # ... and when it brings at least this (dE, the whole picture)
COLOR_FINISH_MIN_MM = 1.0    # pieces of a finishing pass smaller than this square are left out
COLOR_DARK_AFTER_WHITE = 3   # NO DARK GLAZE OVER A FRESH WHITE (run K: black over the white of the pass before, 32% of it): a dark
COLOR_DARK_GLAZE_LUM = 0.6   # glaze (one coat lets through less than this of the light) leaves out what a white got in the last 3 passes
COLOR_LATE_OPAQUE_MAX_SHARE = 0.10  # LATE OPAQUES SMALL (run K pass 31: opaque green on 40% -> 3.8 worse): in the detail stage a
COLOR_LATE_OPAQUE_OFF_FIGURE = False  # covering paint on at most this share, where it helps most, and off the figure's inside ...
COLOR_LATE_OPAQUE_FIGURE_EDGE_MM = 2.0  # ... (the figure's edge this wide stays free for it)
COLOR_KEEP_LAST_SHEET = True # a paint measured by the last sheet's PASSES keeps that value; the new test band replaces it only when
                             # it is clearly another colour (a new mix). Before: every new run took the weaker test-band values
COLOR_SAME_SHEET_MIN = 0.004 # pass 1 resumed: the new blank photo differs from the run before's in this .. COLOR_SAME_SHEET_MAX of
COLOR_SAME_SHEET_MAX = 0.2   # the sheet (paint) -> the same sheet, its first blank photo stays the white reference
COLOR_SAME_SHEET_MAX_H = 48.0
CUP_KEEP_LOWER = True        # the brush check's "brush too high, lower" stays for the next picture (refill no longer forgets it)
COLOR_COVER_FIRST = True     # (2026-11-23) a spot's stack of coats is planned covering paints (whites, opaques) first, glazes over
COLOR_WHITE_NEUTRAL_C = 12.0 # a white also where the picture is whitish (chroma under this) ...
COLOR_WHITE_NEUTRAL_DC = 8.0 # ... and the sheet more coloured than the picture by more than this
