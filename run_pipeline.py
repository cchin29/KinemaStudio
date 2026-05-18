#!/usr/bin/env python3
"""One-command markerless pipeline: video -> TRC -> IK -> ID -> video.

Chains the four stages on a user-supplied clip, in-process, using the
project defaults (heavy PoseLandmarker, ROI hands):

    .venv/bin/python run_pipeline.py my_clip.mp4
    .venv/bin/python run_pipeline.py my_clip.mp4 --outdir runs --no-viz

Anything after a literal `--` is forwarded verbatim to src/mp2trc.py,
so every Stage-1/2 option is still available, e.g. the faster
Holistic-only path:

    .venv/bin/python run_pipeline.py my_clip.mp4 -- \\
        --pose-model holistic --no-roi-hands --hand-source registered

Stages: src/mp2trc.py (video -> .trc + IK setup) -> src/run_ik.py
(inverse kinematics) -> src/run_id.py (inverse dynamics) ->
src/viz_osim.py (musculoskeletal render). Run it with the project
.venv -- it needs opensim + mediapipe + vtk (see docs/SETUP.md).
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
DEFAULT_MODEL = ROOT / "models" / "combined_body_model" / "combined_body_model.osim"


def _split_passthrough(argv):
    """Split argv at the first literal '--': (wrapper_args, mp2trc_args)."""
    if "--" in argv:
        i = argv.index("--")
        return argv[:i], argv[i + 1:]
    return argv, []


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    mine, passthrough = _split_passthrough(argv)

    ap = argparse.ArgumentParser(
        prog="run_pipeline.py",
        description="Run the full markerless pipeline on a video. "
                    "Options after '--' are passed to src/mp2trc.py.")
    ap.add_argument("video", help="input clip (any format ffmpeg reads)")
    ap.add_argument("--outdir", default="runs",
                    help="parent dir for the per-clip output folder "
                         "(default: runs/; outputs land in "
                         "<outdir>/<video-stem>/)")
    ap.add_argument("--model", default=str(DEFAULT_MODEL),
                    help="OSIM model for IK/ID/viz (default: the bundled "
                         "combined_body_model)")
    ap.add_argument("--no-id", action="store_true",
                    help="skip the Inverse Dynamics stage")
    ap.add_argument("--no-viz", action="store_true",
                    help="skip the musculoskeletal render stage")
    a = ap.parse_args(mine)

    if any(o == "--outdir" or o.startswith("--outdir=") for o in passthrough):
        ap.error("set --outdir on run_pipeline.py itself, not after '--' "
                 "(the wrapper must know where mp2trc writes to find the "
                 "TRC for the next stage)")

    video = Path(a.video).expanduser()
    if not video.is_file():
        ap.error(f"video not found: {video}")
    model = str(Path(a.model).expanduser().resolve())
    outdir = Path(a.outdir).expanduser()
    stem = video.stem
    dest = outdir / stem

    sys.path.insert(0, str(SRC))
    import mp2trc, run_ik, run_id, viz_osim

    print(f"\n=== [1/4] mp2trc  {video.name}  (-> {dest}/) ===", flush=True)
    mp2trc.main([str(video), "--outdir", str(outdir), *passthrough])

    # Pick the marker file IK should consume: prefer the ROI-hand TRC
    # (the accurate default path); fall back to the registered one.
    trcs = (sorted(dest.glob(f"{stem}*roihands.trc"))
            or sorted(p for p in dest.glob(f"{stem}*.trc")
                      if "roihands" not in p.name))
    if not trcs:
        sys.exit(f"run_pipeline: no .trc produced under {dest}/")
    trc = trcs[0]
    mot = trc.with_suffix(".ik.mot")  # matches run_ik's default naming

    print(f"\n=== [2/4] run_ik  {trc.name} ===", flush=True)
    run_ik.main([str(trc), "--model", model, "--out", str(mot)])
    if not mot.is_file():
        sys.exit(f"run_pipeline: IK did not produce {mot}")

    if a.no_id:
        print("\n=== [3/4] run_id  skipped (--no-id) ===", flush=True)
    else:
        print(f"\n=== [3/4] run_id  {mot.name} ===", flush=True)
        run_id.main([str(mot), "--model", model])

    if a.no_viz:
        print("\n=== [4/4] viz_osim  skipped (--no-viz) ===", flush=True)
    else:
        print(f"\n=== [4/4] viz_osim  {mot.name} ===", flush=True)
        viz_osim.main([model, str(mot)])

    print(f"\nDone. Outputs in {dest}/", flush=True)


if __name__ == "__main__":
    main()
