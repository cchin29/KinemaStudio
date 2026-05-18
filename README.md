# KinemaStudio

**Markerless biomechanics from a single video.** KinemaStudio turns an
ordinary RGB clip into a driven OpenSim musculoskeletal simulation — pose
and hand tracking with Google MediaPipe, fused into an OpenSim-ready
marker trajectory, then solved for joint kinematics, joint moments, and
rendered as a muscle-activity video. No motion-capture suit, no markers,
no force plates.

The walkthrough below is the bundled `demo/Wieniawski2` clip
(solo violin performance, ~17 s, 500 frames @ 30 fps) carried through
every stage.

## The pipeline

### 1 · Input video

A plain handheld recording — the only input the pipeline needs.

![input clip](docs/assets/01_input.gif)

### 2 · MediaPipe tracking  (`mp2trc.py`)

Each frame is run through MediaPipe: a full-body pose plus a detailed
21-landmark HandLandmarker on an ROI cropped around each wrist. The
landmarks (overlaid below) are unified into one world frame, low-pass
filtered, and written as an OpenSim `.trc` marker file with a generated
IK setup. One command does Stage 1 (video → landmark CSVs + annotated
video) and Stage 2 (CSVs → `.trc`).

![mediapipe landmarks](docs/assets/02_mediapipe.gif)

### 3 · Inverse kinematics & dynamics  (`run_ik.py` → `run_id.py`)

`run_ik.py` solves robust per-frame Inverse Kinematics of the
`combined_body_model` against the marker trajectory (a non-converging
frame holds the last good pose instead of aborting the whole solve).
`run_id.py` then runs Inverse Dynamics to recover the net joint moments
that produced the motion.

### 4 · Musculoskeletal render  (`viz_osim.py`)

Headless VTK renders the posed skeleton and every muscle path, tiled
front / left / right / back, with muscles colored by the inverse-dynamics
joint moment (blue → red).

![opensim render](docs/assets/03_opensim.gif)

## Quick start

Environment setup (one Python 3.12 venv runs everything — OpenSim,
MediaPipe, VTK) is in **[docs/SETUP.md](docs/SETUP.md)**:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -U pip wheel
.venv/bin/pip install -r requirements.txt
```

Run the **whole pipeline** on the bundled clip with the top-level
wrapper (defaults: heavy pose model, ROI hands):

```bash
.venv/bin/python run_pipeline.py demo/Wieniawski2.mp4
```

Outputs land in `runs/Wieniawski2/`. Options after `--` pass straight
to Stage 1; e.g. the faster Holistic-only path:

```bash
.venv/bin/python run_pipeline.py my_clip.mp4 --no-viz -- \
    --pose-model holistic --no-roi-hands --hand-source registered
```

Or drive each stage yourself:

```bash
.venv/bin/python src/mp2trc.py   my_clip.mp4 --outdir runs
.venv/bin/python src/run_ik.py   runs/my_clip/my_clip.roihands.trc
.venv/bin/python src/run_id.py   runs/my_clip/my_clip.roihands.ik.mot
.venv/bin/python src/viz_osim.py models/combined_body_model/combined_body_model.osim \
                                 runs/my_clip/my_clip.roihands.ik.mot
```

Every script (and the wrapper) takes `--help`. `scale_model.py`
optionally scales the model to the subject before IK for more
anatomically faithful joint angles.

## Repository layout

```
run_pipeline.py               one-command wrapper for all 4 stages
src/                          pipeline scripts
  mpipe_pipeline.py           Stage 1/2 engine (imported by mp2trc)
  mp2trc.py                   video -> .trc + IK setup
  scale_model.py              optional subject scaling
  run_ik.py  run_id.py        OpenSim inverse kinematics / dynamics
  viz_osim.py                 musculoskeletal playback render
models/
  combined_body_model/        the .osim model + Geometry/ meshes
  mediapipe/                  MediaPipe .task bundles (auto-downloaded)
templates/TEMPLATE_IK.xml     IK setup skeleton
docs/SETUP.md                 environment reproduction guide
demo/Wieniawski2.mp4          bundled example clip (run output is regenerable)
requirements*.txt             pinned dependencies
```

## The model

`combined_body_model.osim` is a full-body OpenSim model with detailed
articulated hands, posed from MediaPipe markers (75.59 kg default
subject). Geometry meshes resolve from `models/combined_body_model/
Geometry/`. Pass `--model` to any script to use a different `.osim`.

## Notes

- Defaults are tuned for the accurate path: separate **heavy**
  PoseLandmarker, ROI-cropped hands feeding IK (`--hand-source roi`),
  with debug ROI boxes drawn on the annotated video. Use
  `--pose-model holistic --no-roi-hands --hand-source registered` for
  the faster Holistic-only path.
- Markerless capture has no measured muscle activity or force plates, so
  Inverse Dynamics returns *net* joint moments, not muscle-resolved
  forces. MediaPipe's shoulder is its noisiest landmark (a surface
  point, not the glenohumeral center); `scale_model.py` excludes the
  humerus from scaling for this reason.
