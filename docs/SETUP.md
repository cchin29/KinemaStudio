# Environment setup

A single Python 3.12 virtualenv (`.venv/` at the repo root) runs **every
`src/*.py` script**: `mp2trc.py`, `mpipe_pipeline.py`, `run_ik.py`,
`run_id.py`, `scale_model.py`, `viz_osim.py`.

OpenSim ships a PyPI wheel (`opensim==4.6`, cp312 universal2), so no
conda environment is needed — `opensim`, `mediapipe`, `vtk`, and OpenCV
all coexist in one venv (verified: `cv2.dnn.DictValue` works in this
combination).

## Repository layout

```
KinemaStudio/
├── src/                         # the 6 pipeline scripts
├── models/
│   ├── combined_body_model/     # combined_body_model.osim + Geometry/*.vtp
│   └── mediapipe/               # MediaPipe .task bundles (auto-downloaded)
├── templates/TEMPLATE_IK.xml    # IK setup skeleton
├── docs/SETUP.md                # this file
├── requirements.txt             # pinned direct deps
└── requirements.lock.txt        # full transitive lock
```

## Prerequisites

- **Python 3.12** (verified on 3.12.13). Not 3.13/3.14 — `mediapipe`,
  `opensim`, and `vtk` wheels target 3.12.
- macOS arm64 (Apple Silicon) is the verified platform. The pins are
  all wheels; Linux x86_64 should resolve equivalently but is unverified.
- No system `ffmpeg` needed: `imageio-ffmpeg` (pinned) bundles its own
  ffmpeg binary, which `viz_osim.py` uses via imageio for H.264 output.

## Install

From the repo root:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -U pip wheel
.venv/bin/pip install -r requirements.txt
```

`requirements.txt` is the readable, pinned direct-dependency list. For a
byte-for-byte transitive reproduction use the full lock instead:

```bash
.venv/bin/pip install -r requirements.lock.txt
```

## Verify

```bash
.venv/bin/python - <<'EOF'
import importlib
for m in ["opensim","mediapipe","cv2","numpy","pandas","scipy",
          "sklearn","vtk","imageio","yaml","tqdm"]:
    mod = importlib.import_module(m)
    print(f"{m:12s}", getattr(mod,"__version__",getattr(mod,"VTK_VERSION","?")))
import cv2, opensim as osim
print("cv2.dnn.DictValue ok:", hasattr(cv2.dnn, "DictValue"))
osim.Model(); print("opensim Model() ok")
EOF
```

Expected: all 11 versions print, `cv2.dnn.DictValue ok: True`,
`opensim Model() ok`. Then every CLI responds to `--help`:

```bash
for s in mp2trc run_ik run_id scale_model viz_osim; do
  .venv/bin/python src/$s.py --help >/dev/null && echo "$s OK"
done
```

## Running the scripts

Always use the venv interpreter explicitly (no activation needed). The
default model and IK template resolve automatically from the repo
layout — pass `--model` / `--template` only to override.

```bash
.venv/bin/python src/mp2trc.py      <clip.mp4> [...]   # video -> .trc + IK setup
.venv/bin/python src/scale_model.py <name>.trc [...]   # optional model scaling
.venv/bin/python src/run_ik.py      <name>.trc [...]   # inverse kinematics -> .mot
.venv/bin/python src/run_id.py      <name>.ik.mot [...] # inverse dynamics
.venv/bin/python src/viz_osim.py    <model> <mot> [...] # render model+motion video
```

## Notes / troubleshooting

- **First import is slow** (~30 s): `opensim` + `mediapipe` + `vtk`
  load large native libraries. Subsequent runs are fast.
- **MediaPipe `.task` bundles** (~14 MB each) auto-download to
  `models/mediapipe/` on first `mp2trc.py` run (git-ignored).
- **`tqdm`** progress bars are a soft dependency (`--no-progress` to
  hide); they're pinned so behavior is deterministic.
- **`landmark_projection_calculator … NORM_RECT` warning** (one per
  frame) comes from inside the prebuilt MediaPipe `.task` graph, not
  this code — immaterial to the output. Filter with
  `grep -v landmark_projection` if noisy.
- **Regenerating the lock:** `.venv/bin/pip freeze > requirements.lock.txt`
  after any deliberate dependency change (keep `requirements.txt` in sync).
