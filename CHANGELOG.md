# Changes since v1.0

## Colour painting, 2026-11-21 .. 23 (runs J, K and the run of 23 Nov)

- **No double strokes** (`trim_over_bands`, `THIN_BAND_CLEAR`): thin centre-line strokes are cut where they would run over
  the bands - painted twice or more 60% -> 11% of a pass. Fewer dips (`DIP_REF_WIDTH_MM` 4.5); forecast 4x faster
  (float32, threads).
- **Paint while it dries** (`COLOR_NO_WAIT_*`, `choose(forbid=)`): when the best pass would touch wet paint, the best pass
  on dry or bare paper goes on at once.
- **Depth planes from ComfyUI** (`color_planes.py`, `comfy_client.py`): Depth Anything + OneFormer segmentation; a pass is
  smoothed inside each plane, sharp along depth breaks; the figure as one plane.
- **The finish** (`color_lines.py`, `finish_pass`): when the colour is done, edge passes along the picture's lines (TEED)
  where the line's contrast is missing on the sheet, and light accents in white on the figure; then the signature.
- **No pass limit** (`COLOR_MAX_PASSES` 0, `COLOR_END_NO_BEST` 6): the run ends when it is done.
- **White the way a brushographer uses it** (`COLOR_COVER_FIRST`, `COLOR_WHITE_NEUTRAL_*`): each spot's stack of coats is
  planned covering paints first, glazes over them ("lift with white, then glaze"); white also where the picture is
  whitish and the sheet too coloured; no white limit; a dark glaze never on a white of the last 3 passes.
- **Late opaques small** (`COLOR_LATE_OPAQUE_MAX_SHARE`), **gouache dries longer** (`COLOR_DRY_MIN_OPAQUE` 40),
  **detail stage in bands** (no more short white dabs).
- **What carries over**: the last sheet's measured paints are kept (test bands no longer replace them), the brush check's
  "lower" is kept (`CUP_KEEP_LOWER`), a resumed pass 1 keeps the first blank photo as the white reference.
- **The 12-paint test sheet** (`density_test.py`): a 12 x 12 matrix of overlaps in both orders and, per paint, dabs,
  short strokes and a zig-zag from one dip - for a colour model measured on strokes (see the notes "What the brush
  really does" and "Deli, oblike in meje" on the website).
- **The talking brushograph** (speech in and out, a cloned voice) was built too; it stays local for now.

## Colour painting (color_paint.py)

After run H of 2026-11-19/20 (mati_mahnic, 15 passes, dE 18.7 - worse than G at the same pass; flat opaque green,
no cyan, milky figure, dark stripes):

- **Holes only in the gaps** (`COLOR_HOLE_FROM_STAGE` 1, `holes(..., painted=)`): bare paper counts as a hole only inside
  what earlier passes painted, and only from the middle stage on. In run H all the paper at the start was a "hole":
  passes 2-5 were hole passes (opaque green 37%, black 27%, magenta 4%, black 20%), no stage logic, and none were left
  for the real gaps at the end (7.5% bare paper).
- **It keeps trying** (`COLOR_TRY_UNUSED_MAX` 2 -> 6, `COLOR_TRY_PER_PAINT` 2, `COLOR_TRY_AGAIN_AFTER` 4, `COLOR_TRY_GAP` 2):
  a paint not used in the last 4 passes (not only one never used) gets a pass where it promises most, each paint up to
  twice, 2 passes apart.
- **An unproven glaze may be strong** (`COLOR_TEST_DENSITY_SCALE` 3, `test_scale()`, `scaled_test()`): test strokes read
  glazes 3-6x too weak (green's test stroke t 0.72, its pass 0.22; transparent black 0.86 / 0.59), so cyan, known only
  from its test stroke, never looked worth a pass. For the decision to TRY a paint, one not proven by a pass counts as
  that much stronger (optical density, no channel below 0.1); its value for everything else stays as measured, and its
  pass then counts. The run measures its own factor from the paints with a test stroke and a pass.
- **One white per stage** (`COLOR_WHITE_PER_STAGE` 1): run H painted 3 white passes on 45% - the milky figure, with no
  glaze over it.
- **Glaze overrides, built but OFF** (`COLOR_GLAZE_OVERRIDES` = {}; paint.py `BAND_SPACING`, `PATCH_MAX_COVERED`,
  `drop_covered()`): a transparent paint's pass can get its own settings (bands one mark apart, no planned repair stroke
  that the others cover by half, blobs not into each other). Tested by planning run H's passes 3 and 7 both ways and
  counting the coats with the probe's width curve: painted twice or more 36 -> 37%, 44 -> 45% - no help. The overlap is
  in the plan itself (the side strokes and planned repairs are at least one mark wide); the real fix is in the planning.
- **Keyboard Z correction** (`znudge_keys.py`): arrows 0.05 mm, Page Up/Down 0.2 mm, 0 = plain, in its own window while
  a job runs (the same `z_nudge.json`).

After run G of 2026-11-18/19 (mati_mahnic, 22 passes, dE 48 -> 10.3: the best so far; no-wait chains worked):

- **Shapes touch** (`COLOR_WET_MARGIN_MM` -1.0, `wet_zone()`): a pass may reach 1 mm into the last (wet) pass's area; the
  +2 mm margin, its thin strip smoothed away, left white lines between the shapes.
- **Your Z correction while a job runs** (`znudge.py`, `z_nudge.json`, `ZNUDGE_MAX_DOWN_MM` / `ZNUDGE_MAX_UP_MM`):
  `python znudge.py down 0.2` / `up 0.1` / `0`; paint.py reads it every 2 s and moves only the strokes on the paper (also
  the strokes the brush check paints again), never the dips, rinse or travel.
- **An unused paint gets one try** (`COLOR_TRY_UNUSED`, `_SHARE` 0.25, `_FROM` 3, `_MAX` 2): the transparent cyan was never
  chosen in run G - its test stroke said weak, and a paint never painted is never measured again.
- **Modulation** (`COLOR_MODULATION` 5, Butina after Cezanne): the target's lights a little warmer, its shadows a little
  cooler (Lab units at the ends of its lightness range; the paper stays white). dE is measured against this target.
- **Sharp eyes** (`COLOR_EYE_BLUR_PX` 0.8, `COLOR_EYE_GRID_BLUR` 0): the sheet photo was blurred twice (~2-3 mm): run G's
  1-3 mm white holes were not seen, and the paints were believed far lighter (opaque black L 45, the photo L 29) - black
  went where a mid grey was wanted, and the target was made pale. The same sheet read sharp: dE 12.9 instead of 10.5.
- **Learned where the paint landed** (`COLOR_LEARN_LANDED`): a pass's value from the spots it really changed, not from the
  gaps between its strokes.
- **Holes passes** (`COLOR_HOLE_*`): bare paper where the picture is not white gets a pass first, each hole with the paint
  that suits it best (at most 4 per run; on run G's end: magenta on 2%, then transparent black on 1%).
- **White lifts, the glaze colours** (`COLOR_WHITE_ONLY_LIFTS`, `COLOR_WHITE_LIFT_DL` 3, `COLOR_GLAZE_OVER_WHITE` 1.5): run G
  used white 6 times, mostly to dull colours the camera read stronger than the picture (chroma 30 vs 22) - milky. Now a
  white goes only where the sheet is darker than wanted, and a dry white's area is wanted (not put off) for a glaze. Run G
  after pass 10 again: transparent magenta over the greens (7% on the dry white) instead of white on 53%.

After run F of 2026-11-17/18 (mati_mahnic, 22 passes, dE 48 -> 16: the best so far; stopped, the details came out as blobs):

- **No wait when the next pass is on dry parts** (`COLOR_NO_WAIT`, `COLOR_NO_WAIT_CHAIN` 2): a pass chosen on the sheet as expected
  that keeps `COLOR_WET_MARGIN_MM` away from the last pass's area goes on at once - no drying, no photo. The passes painted
  without a photo are measured together at the next one (their areas do not overlap: each is learned from the same photos,
  their real gain shared by their promises). In run F not one pass touched the one before it.
- **Stages by progress** (`COLOR_STAGE_DE` 26 / 17): the middle and detail stages begin only when the sheet is that near the
  picture (the best dE so far; stages never go back). Run F began its details at pass 11, dE 21.
- **Butina: focus** (`COLOR_FOCUS_WEIGHT` 0 / 1 / 2 per stage): the faces in the picture (OpenCV's face finder) and its strong
  local contrasts count up to 3x when passes are ranked. **Butina: crisp edges** (`COLOR_CRISP_EDGES`, `COLOR_EDGE_DL` 8 L/mm):
  near the picture's sharp light-dark breaks a pass follows the wanted spots, not the smoothed outline.
- **Fill-ins:** also for a small missed share when the spots are far off (`COLOR_FILL_GAIN_ANYWAY` 0.4: the white stripes between
  the first black bands); never with a paint whose pass kept under `COLOR_FILL_MIN_KEPT` (30%) of its promise (2 useless white
  fill-ins in run F).
- **Brush check:** never judges another cup's swatch (it judged the transparent magenta swatch as black, found it faint and
  painted it over with black); judges only a stroke where its paint must clearly show (`COLOR_BRUSH_CHECK_MIN_EXP`; over its
  own colour nothing shows and it said OK); a brush found too high stays lower for the rest of the run (cups.json `lower`,
  a new run forgets it).

After run E of 2026-11-17 (mati_mahnic again, stopped after pass 3: transparent white over the wet black made grey mud):

- **The wet part last** (`COLOR_AVOID_WET`, `COLOR_WET_MARGIN_MM`, `COLOR_RECENT_WEIGHT`): a pass leaves out the area of the last
  pass (still wet) when it has enough to do elsewhere, and spots painted in the last two passes count less when passes are
  ranked. The bare paper and the older, dry parts come first; the fresh paint dries meanwhile. Replayed before pass 3: white on
  57% (54% of it on the wet black) became a pass on bare paper only.
- **No paint where it cannot help:** larger patches inside a pass's smoothed area where its paint makes the spot worse are cut
  out (19% of the white pass lay on bare paper, where white does nothing).
- **Light and dark first** (`COLOR_LIGHTNESS_WEIGHT` per stage 2 / 1.5 / 1; after Butina, *Mala likovna teorija*): where the coats
  are chosen and passes ranked, the lightness difference counts double in the broad stage. The promise (`gain`) and the end
  stay in plain dE.
- **The darkest the paints can reach** (`COLOR_REACHABLE_DARK`, `COLOR_REACHABLE_MAX_LIFT` 10): once the swatches are measured,
  the picture's darkest becomes what they can really make (run E: L 53 -> target darkest L 55 instead of 45), at most 10 lighter.
- **Swatches:** a white's band on white paper alone is never used (it was read as "lightens a lot" and gave white 57% of the
  picture); a bright paint read up to `COLOR_TEST_BRIGHT_TOL` over the paper is trimmed to paper white instead of being
  rejected (opaque yellow, cyan, magenta now pass). The bogus white value was removed from ko/cp_measured.json.

After the run of 2026-11-16 21:02 (mati_mahnic, 9 passes, dE 48 -> 24, then the camera ended it):

- **What went wrong.** Pass 3 (opaque yellow) used a value read off the green band next to it (the yellow band was not
  painted): it put yellow where dark green belonged (-4.5). Pass 6 (opaque magenta) used an old value from another mix and
  turned the whole figure pink (-4.5). Then opaque green 3 more times, and the run ended at dE 24, with the pink uncovered.
- **A swatch of every paint first** (`COLOR_TEST_ALL_AT_START`): pass 1 paints a band of every cup with paint, so each paint is
  measured as it is today before it is chosen.
- **The bands no longer cross** (`COLOR_TEST_CROSS` False): all the same way, side by side, 13.6 mm apart, each on its own bare
  paper (crossing wet bands carried green into the yellow cup and blue into the white). Fatter (`COLOR_TEST_FAT_MM` 3.5 -> 5),
  steeper (`COLOR_TEST_RUN` 0.2 -> 0.1).
- **A band must be there and be its own colour:** no change on the paper = not painted; a hue more than `COLOR_TEST_HUE_MAX`
  (40 deg) off its cup's colour (as the camera sees the cup) = a neighbour's band. Neither is used. Run D's yellow reading is
  now rejected (57 deg off); every real band was within 10 deg.
- **Something else instead of the end:** when the last `COLOR_STOP_PATIENCE` passes did not beat the best, the paints of those
  passes rest and the others get their turn; the end comes only when nothing else helps. Not stalled while the last pass still
  made it `COLOR_STOP_STILL_GAIN` (0.5) better.
- **The best pass wins more often:** `COLOR_ORDER_SLACK` 0.5 -> 0.85. At 0.5 the order (covering paints in their order, whites
  last) gave away nearly half the gain: before pass 10 opaque white (expected 1.35) lost to green (1.09). Replayed: now white.
- **Brush check: too high or too dry?** A faint stroke that fades along it = too dry (deeper dip, as before). Otherwise the
  brush first goes `COLOR_BRUSH_LOWER_MM` (0.2) lower for the rest of the pass (at most `COLOR_BRUSH_LOWER_MAX_MM` 0.6) and
  paints the stroke again; still faint after that = the deeper dip.

After the run of 2026-11-15 20:38 (30 passes, dE 48 -> 14.8, best 13.7 after pass 25):

- **Honest promises.** After every pass the camera measures what it really did (`real` in cp_plan.json) against what it
  promised (`gain`). Each paint's promises are scaled by the share its recent passes kept (`COLOR_TRUST_*`) and the choice
  goes by that. The detail passes of that run promised 11.4 and did 4.1.
- **The end by the camera.** The run ends when the last `COLOR_STOP_PATIENCE` (3) passes did not make the sheet better than
  its best by `COLOR_STOP_MIN_GAIN`. That run would have ended after pass 28 instead of 30. The signature now goes with
  the last real pass (its fill-ins after it); a run ended by the camera signs with a small pass only where it clearly helps.
  Fixes 21 real passes against a maximum of 20.
- **Opaque underneath, glazes on top.** `COLOR_PREFER` per stage: the broad stage puts covering paints first.
- **The transparent white does not lighten** (measured t .92, r 0): its start value is now that, not a guess that lightened.
- **Fill-ins only when worth it:** `COLOR_FILL_MIN_SHARE` 0.004 -> 0.015 and `COLOR_FILL_MIN_GAIN` 0.15 (9 fill-ins brought
  -0.14..+0.07).
- **Brush check during a pass** (paint.py `BrushCheck`, `COLOR_BRUSH_CHECK_EVERY` 8, `COLOR_BRUSH_CHECK_MIN` 0.35): after the first
  dip, every 8th dip and the dip after a faint one, the head parks and the camera compares the first stroke after that dip with
  the photo from before the pass (the strongest change within ~1 mm of its line). Under 35% of what the paint should do: the cup
  goes `CUP_DEEPER_STEP_MM` deeper at once (the dips still to come in the job too) and in cups.json, the brush dips again and
  paints that stroke once more. Tested offline on run 20:38's passes: real strokes 46-235%, an empty brush 0%.
- **Transparent white is used again:** start value t .8 r .15 (a veil that lightens dark spots a little), the 20:38 value
  (measured on wet paint) is dropped; its first pass or test band measures the real one. `COLOR_REFILL_AT_START`: a new run
  resets the cups' dip counts. `COLOR_MAX_PASSES` 20 -> 40 (the camera ends the run), probe valid 96 h.
- **Paint check:** a pass that left `COLOR_DIP_FAINT` of its area unchanged makes its cup dip `CUP_DEEPER_STEP_MM` deeper
  from then on (cups.json `deeper`, reset by `python cups.py refill N`); a cup already at its lowest dip asks for a refill.
- **Test strokes:** bands of `COLOR_TEST_LINES` 3 strokes, steeper (`COLOR_TEST_RUN` 0.2) so they no longer touch (at 0.45
  same-direction strokes lay 3.6 mm apart and the camera read green and yellow as one), strip 36 mm, length 200. Measuring
  first finds where the strip really is by the bands' colours (it lay ~5 mm off at the sheet edge), then each band across
  itself; values that cannot be real (brighter than paper, nothing seen) are not used (`sane`).
- **Drying:** `COLOR_DRY_MIN` 25 min, then the camera checks the last pass's area stopped changing (`COLOR_DRY_CHECK_MIN`,
  `COLOR_DRY_STILL`, `COLOR_DRY_CHECK_MAX`). Esc closes the drying window (the countdown goes on).

- **Broad first, detail last.** The passes go through stages (`COLOR_STAGE_FROM`): passes 1-4 paint broad areas (4 mm
  smoothing, long bands), 5-8 use 2 mm, from pass 9 on it is 1 mm and centre-line strokes for edges and features
  (`COLOR_MASK_SMOOTH_MM`, `COLOR_INFILL_STYLE` now take one value per stage).
- **Each colour is finished before the next.** After a pass dries, the spots of it that the photo shows unchanged get the
  same paint from the same cup again, up to `COLOR_FILL_TRIES` times (`COLOR_FILL_MISS`, `COLOR_FILL_MIN_SHARE`). A fill-in
  is not another coat and does not count towards `COLOR_MAX_PASSES`.
- **Test strokes** in a strip over the picture (`COLOR_TEST_STRIP_MM`, `COLOR_TEST_GAP_MM`, `COLOR_TEST_LEN_MM`,
  `COLOR_TEST_FAT_MM`). When a paint is used for the first time, its first dip paints one stroke there, fat at one end and
  thin at the other. The strokes cross the strip as / and \ by turns, so each crosses the strokes of the other direction.
  On the next dry photo the camera measures each paint on bare paper and over the strokes under it (its real t and r on
  this sheet), and its width along the ramp (`ko/cp_tests.json`). After pass `COLOR_TEST_EXTRA_AFTER`, paints that were
  never measured get their stroke too (an extra dip and rinse each), so the choice can use them with real values.
- **Longer runs:** `COLOR_MAX_PASSES` 12 -> 20, `COLOR_STOP_GAIN` 0.4 -> 0.25.
- **Drying can be extended:** in the countdown, `+` (or `P`) adds `DRY_EXTEND_MIN` minutes, `-` takes them off, and Enter
  goes on now. A drying window (`COLOR_DRY_VIEW`) shows the camera, the countdown and the last pass (seen | target | pass).
- **The real paint in the overlay.** A covering paint or a white is shown under its own name and colour ("cup 2 - opaque
  magenta", "cup 9 - transparent white"), not as "black dip". The cup is drawn as a rectangle at its rim, where the camera
  sees it, not as a round dish at paper height.
- **The live preview shows more:** the pass ("PASS 5 (5/20) - magenta - coat 1 - 32% - broad", infill and smoothing), the
  brush's state in the title (DIP / WIPE / PAINT / RINSE / TRAVEL), the cup, its dip and how much paint is left, the depth
  and width while painting, the last terminal lines, the passes as colour chips, a curve of the colour error, and a
  minimap of the sheet with the brush. All of this is drawn in the h4xx0r theme.
- **The wipe on the cup rim is 2 mm lower** (`CUP_WIPE_BELOW_RIM_MM` 0.5 -> 2.5).

## Repair (the passes after the first painting)

- **Repair strokes blend in.** No more "stairs" where a repair is added to a line: the width along a repair stroke is
  smoothed, and each end runs on over the paint already there, thinning out, so the brush glides in and out in an arc
  (`REPAIR_TAPER`, `REPAIR_TAPER_RATIO`, `REPAIR_TAPER_MIN_MM`, `REPAIR_TAPER_MAX_MM`). A run only continues *along* the
  painted lines around it, never across them (`REPAIR_TAPER_MAX_TURN_DEG`).
- **Faded black gets a second coat.** Black that is painted but much paler than the job's good strokes is repaired too,
  at the normal depth (`REPAIR_FADED_SHARE`, 0 = off).
- **Deeper every pass.** Empty spots are repainted one `REPAIR_STEP_MM` deeper on every pass, up to `REPAIR_MAX_DEEPER_MM`
  (it used to stop going deeper after two passes).
- **A repair stroke's edge stays on the artwork's outline** (`REPAIR_EDGE_INSIDE`): a brush wider than the missing piece no
  longer paints past the shape.
- **Cleaner travel.** After a dip the brush goes to the next stroke in one diagonal move (was up, across, down), and there is
  no extra down-and-up move before a dip.

## Planning

- **Rings and closed shapes are painted all the way round.** An "O", a ring or any closed line was cut to half.
- **No knobs on lines.** A short side branch whose spot the strokes around it already paint is left out
  (`SPUR_MARKS`, `SPUR_COVERED`).
- **Round dots.** A mark shorter than about one brush width is painted as a dab, straight down and up
  (`DOT_DAB_MARKS`, `DOT_DWELL_MS`), instead of a little comma.
- **A style per area:** an artwork can have a `<name>.styles.json` next to it that gives boxes on the picture their own
  `INFILL_STYLE`, e.g. a test sheet with every style side by side.
- [docs/infill_styles.pdf](docs/infill_styles.pdf): every infill style on the same test shapes and on an example artwork.

## Probes

- **Four probe lines again** (`PROBE_COUNT = 4`). With only left + right lines the brush-tip offset *along* them was never
  measured, so it was carried over from an older brush mounting and whole paintings landed 1.4 mm off (ko/84). Four lines
  measure X and Y fresh every time and check each other.
- **Tip offset read at the tip.** Where the paint lands sideways is measured over the first few mm after each line's touch,
  where only the tip paints; deeper, a brush can lean or splay to one side (`PROBE_OFFSET_TIP_MM`).
- **Bent paper is followed** (`PROBE_BEND`). A sheet whose top and bottom lie higher (or lower) than its sides cannot be
  matched by a flat tilted plane (ko/85: 0.5 mm off at every line, so thin marks were painted in the air near two edges).
  With four lines a bend is fitted too, every stroke follows it, and all four measured touches are met exactly.
- Probe lines go where the sheet edge is still free of paint (the camera checks), so a sheet can be probed again; the
  ramp is made steep enough for what one dip of this brush paints (`probe_reach.json`).
- No false "ran out of paint" on a brush that paints a little unevenly, and no fake tilt when the lines touched at
  different points of their landing.
- `PROBE_PAIR`: with `PROBE_COUNT = 2`, choose which two opposite edges carry the probe lines (left + right, or top + bottom
  when the sides of the sheet are bent).
- `PROBE_ASK` now decides every question: `True` asks before painting, `False` never asks (the artwork starts after a
  10 s countdown, Ctrl+C stops it).

## Screen

- **h4xx0r theme** (from pull request #3): a green phosphor terminal look for the live window, `preview.py` and the
  calibration tools. `--h4xx0r`, or `H4XX0R = True` in `settings.py`.
- The live window is drawn so it also works on macOS.
- `preview.py`: S saves the camera picture without the overlay.
