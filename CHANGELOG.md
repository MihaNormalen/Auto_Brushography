# Changes since v1.0

## Repair (the passes after the first painting)

- **Repair strokes blend in.** No more "stairs" where a repair is added to a line: the width along a repair stroke is
  smoothed, and each end runs on over the paint already there, thinning out, so the brush glides in and out in an arc
  (`REPAIR_TAPER`, `REPAIR_TAPER_RATIO`, `REPAIR_TAPER_MIN_MM`, `REPAIR_TAPER_MAX_MM`). A run only continues *along* the
  painted lines around it, never across them (`REPAIR_TAPER_MAX_TURN_DEG`).
- **Faded black gets a second coat.** Black that is painted but much paler than the job's good strokes is repaired too,
  at the normal depth (`REPAIR_FADED_SHARE`, 0 = off).
- **Deeper every pass.** Empty spots are repainted one `REPAIR_STEP_MM` deeper on every pass, up to `REPAIR_MAX_DEEPER_MM`
  (it used to stop going deeper after two passes).
- **Cleaner travel.** After a dip the brush goes to the next stroke in one diagonal move (was up, across, down), and there is
  no extra down-and-up move before a dip.

## Planning

- **Rings and closed shapes are painted all the way round.** An "O", a ring or any closed line was cut to half.
- **A style per area:** an artwork can have a `<name>.styles.json` next to it that gives boxes on the picture their own
  `INFILL_STYLE`, e.g. a test sheet with every style side by side.
- [docs/infill_styles.pdf](docs/infill_styles.pdf): every infill style on the same test shapes and on an example artwork.

## Probes

- `PROBE_PAIR`: with `PROBE_COUNT = 2`, choose which two opposite edges carry the probe lines (left + right, or top + bottom
  when the sides of the sheet are bent).
- `PROBE_ASK` now decides every question: `True` asks before painting, `False` never asks (the artwork starts after a
  10 s countdown, Ctrl+C stops it).

## Screen

- **h4xx0r theme** (from pull request #3): a green phosphor terminal look for the live window, `preview.py` and the
  calibration tools. `--h4xx0r`, or `H4XX0R = True` in `settings.py`.
- The live window is drawn so it also works on macOS.
- `preview.py`: S saves the camera picture without the overlay.
