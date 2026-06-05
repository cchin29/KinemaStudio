#!/usr/bin/env python3
"""One-command markerless pipeline: video -> TRC -> IK -> ID -> video.

Chains the four stages on a user-supplied clip, in-process. Default
hand source is "combined" (heavy PoseLandmarker ROI hands + Holistic-
recovered rows for ROI-missing frames -- see src/combine_hands.py):

    .venv/bin/python run_pipeline.py my_clip.mp4
    .venv/bin/python run_pipeline.py my_clip.mp4 --outdir runs --no-viz
    .venv/bin/python run_pipeline.py my_clip.mp4 -- --hand-source roi  # baseline

Anything after a literal `--` is forwarded verbatim to src/mp2trc.py,
so every Stage-1/2 option is still available, e.g. the faster
Holistic-only path:

    .venv/bin/python run_pipeline.py my_clip.mp4 -- \\
        --pose-model holistic --no-roi-hands --hand-source registered

Typical settings can live in an optional YAML (src/pipeline_config.py)
that the wrapper AND every stage script read, so re-running one stage
standalone honours the same config. Precedence, low -> high: built-in
defaults < the YAML < explicit command-line flags / `--` passthrough.
Auto-loaded from ./kinemastudio.yaml (or the repo root); point
elsewhere with --config PATH, ignore with --no-config, write a
commented template with --dump-config [PATH]:

    .venv/bin/python run_pipeline.py --dump-config kinemastudio.yaml
    .venv/bin/python run_pipeline.py my_clip.mp4 --config tuned.yaml

--scale (or scale: true in the YAML) inserts a measurement-based
model-scaling stage after mp2trc: src/scale_model.py fits the model's
segment lengths to the subject from the Stage-1 TRC, and the scaled
.osim is then used by IK/ID/viz (scale_model's own knobs -- averaging
window, mass, humerus -- come from the YAML's scale_model: block).

--so (or so: true in the YAML) inserts a Static Optimization stage
after Inverse Dynamics: src/run_so.py resolves the joint moments into
per-muscle forces (-> <root>.so_force.sto / .so_activation.sto). It is
slow and only approximate on markerless data (no measured external
loads on a floating base), so it is opt-in; its knobs live in the
YAML's run_so: block.

--grf (or grf: true) inserts a ground-reaction-force estimation stage
after IK: src/run_grf.py derives the GRF from the kinematics alone (no
force plates) and writes an ExternalLoads .xml that ID and SO then
apply, so the base residual collapses to ~0 and both become
dynamically consistent for any motion. Use the same lowpass across
run_grf/run_id/run_so (config) for the tightest residuals.

--debug echoes each stage's exact, copy-pasteable invocation (the
command-line args the wrapper passes it) just before running it.

Stages: src/mp2trc.py (video -> .trc + IK setup) -> [src/scale_model.py
(optional, --scale)] -> src/run_ik.py (inverse kinematics) ->
[src/run_grf.py (optional, --grf: ground reaction forces)] ->
src/run_id.py (inverse dynamics) -> [src/run_so.py (optional, --so:
per-muscle forces)] -> src/viz_osim.py (musculoskeletal render) ->
[src/combine_viz.py (default on, --no-combine: stack the annotated
source over the render + audio)]. Run it with the project .venv -- it
needs opensim + mediapipe + vtk (see docs/SETUP.md).
"""
import argparse
import shlex
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
DEFAULT_MODEL = ROOT / "models" / "combined_body_model" / "combined_body_model.neck.osim"

sys.path.insert(0, str(SRC))
import pipeline_config as PC


def _split_passthrough(argv):
    """Split argv at the first literal '--': (wrapper_args, mp2trc_args)."""
    if "--" in argv:
        i = argv.index("--")
        return argv[:i], argv[i + 1:]
    return argv, []


def _has_opt(argv, name):
    """True if --name (or --name=...) appears in an argv list."""
    return any(o == name or o.startswith(name + "=") for o in argv)


def _invoke(stage, fn, args, debug):
    """Call a stage's main(args); with --debug, echo the exact
    invocation (a copy-pasteable command) first."""
    if debug:
        cmd = " ".join(shlex.quote(str(x))
                       for x in [f"src/{stage}.py", *args])
        print(f"[debug] {stage}: .venv/bin/python {cmd}", flush=True)
    return fn(args)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    mine, passthrough = _split_passthrough(argv)

    ap = argparse.ArgumentParser(
        prog="run_pipeline.py",
        description="Run the full markerless pipeline on a video. "
                    "Settings can come from a YAML (--config / "
                    "./kinemastudio.yaml), also read by each stage when "
                    "run standalone; explicit flags and anything after "
                    "'--' override it. Options after '--' go to mp2trc.")
    ap.add_argument("video", nargs="?", default=None,
                    help="input clip (any format ffmpeg reads); omit "
                         "only with --dump-config")
    ap.add_argument("--outdir", default="runs",
                    help="parent dir for the per-clip output folder "
                         "(default: runs/; outputs land in "
                         "<outdir>/<video-stem>/)")
    ap.add_argument("--model", default=str(DEFAULT_MODEL),
                    help="OSIM model for IK/ID/viz (default: the bundled "
                         "combined_body_model)")
    ap.add_argument("--scale", dest="scale", action="store_true",
                    default=False,
                    help="scale the model to the subject (measurement-"
                         "based, from the Stage-1 TRC) before IK; the "
                         "scaled model is then used by IK/ID/viz")
    ap.add_argument("--no-scale", dest="scale", action="store_false",
                    help="skip scaling even if the config sets scale: true")
    ap.add_argument("--no-id", dest="no_id", action="store_true",
                    default=False, help="skip the Inverse Dynamics stage")
    ap.add_argument("--id", dest="no_id", action="store_false",
                    help="force-run Inverse Dynamics even if the config "
                         "sets no_id: true")
    ap.add_argument("--so", dest="so", action="store_true", default=False,
                    help="run Static Optimization after ID to estimate "
                         "per-muscle forces (-> <root>.so_force.sto / "
                         ".so_activation.sto); slow, and approximate on "
                         "markerless data (see src/run_so.py)")
    ap.add_argument("--no-so", dest="so", action="store_false",
                    help="skip Static Optimization even if the config "
                         "sets so: true")
    ap.add_argument("--grf", dest="grf", action="store_true", default=False,
                    help="estimate ground reaction forces from the "
                         "kinematics (no force plates) after IK and feed "
                         "them to ID/SO as ExternalLoads, so the base "
                         "residual collapses to ~0 (see src/run_grf.py)")
    ap.add_argument("--no-grf", dest="grf", action="store_false",
                    help="skip GRF estimation even if the config sets "
                         "grf: true")
    ap.add_argument("--lowpass", type=float, default=6.0,
                    help="shared low-pass cutoff (Hz) applied to the GRF/"
                         "ID/SO dynamics stages when --grf, so their "
                         "coordinate differentiation matches (required "
                         "for the base residual to cancel; <0 = off, "
                         "default: 6). Ignored without --grf.")
    ap.add_argument("--no-viz", dest="no_viz", action="store_true",
                    default=False, help="skip the musculoskeletal render "
                                        "stage")
    ap.add_argument("--viz", dest="no_viz", action="store_false",
                    help="force-run the render even if the config sets "
                         "no_viz: true")
    ap.add_argument("--combine", dest="combine", action="store_true",
                    default=True,
                    help="after the render, stack the annotated source "
                         "video above it and mux audio into "
                         "<stem>.combo.mp4 (src/combine_viz.py; default on, "
                         "needs the render + the mp2trc _merged.mp4)")
    ap.add_argument("--no-combine", dest="combine", action="store_false",
                    help="skip the combine_viz post-step")
    ap.add_argument("--debug", action="store_true",
                    help="echo each stage invocation (the exact "
                         "command-line args passed to it) before running")
    PC.add_args(ap)
    ap.add_argument("--dump-config", nargs="?", const="-", metavar="PATH",
                    help="write a template pipeline-config YAML to PATH "
                         "(or stdout if omitted) and exit")
    a = ap.parse_args(mine)

    if a.dump_config is not None:
        tpl = PC.template()
        if a.dump_config == "-":
            print(tpl, end="")
        else:
            Path(a.dump_config).expanduser().write_text(tpl)
            print(f"wrote {a.dump_config}")
        return

    # Wrapper's own knobs from the YAML top level; re-parse so an
    # explicit CLI flag still beats the file. Stages load their own
    # sections themselves (we forward --config below).
    cfg_path = PC.apply(ap, None, mine)
    a = ap.parse_args(mine)
    if cfg_path:
        print(f"run_pipeline: config {cfg_path}", flush=True)

    if a.video is None:
        ap.error("video is required (omit it only with --dump-config)")
    if _has_opt(passthrough, "--outdir"):
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

    import mp2trc, scale_model, run_ik, run_grf, run_id, run_so
    import viz_osim, combine_viz

    # Base stages: mp2trc, run_ik, run_id, viz. --scale inserts a stage
    # before IK; --grf one after IK (before ID); --so one between ID/viz;
    # --combine appends a render-stacking post-step (skipped if --no-viz).
    combine = a.combine and not a.no_viz
    nst = (4 + (1 if a.scale else 0) + (1 if a.grf else 0)
           + (1 if a.so else 0) + (1 if combine else 0))
    step = 0

    # Forward the resolved config so every stage reads the same file
    # (single source of truth) regardless of its working directory.
    if a.no_config:
        cfg_args = ["--no-config"]
    elif cfg_path is not None:
        cfg_args = ["--config", str(cfg_path)]
    else:
        cfg_args = []

    # mp2trc gets the `--` passthrough verbatim (explicit -> wins over
    # the YAML). The wrapper's historical default is the combined hand
    # source; only inject it if neither the passthrough nor the config's
    # mp2trc block already chose one.
    mp2trc_extra = list(passthrough)
    if not _has_opt(mp2trc_extra, "--hand-source"):
        mp_defaults, _ = PC.load("mp2trc", a.config, a.no_config)
        if "hand_source" not in mp_defaults:
            mp2trc_extra += ["--hand-source", "combined"]

    step += 1
    print(f"\n=== [{step}/{nst}] mp2trc  {video.name}  (-> {dest}/) ===",
          flush=True)
    _invoke("mp2trc", mp2trc.main,
            [str(video), "--outdir", str(outdir), *cfg_args, *mp2trc_extra],
            a.debug)

    # Pick the marker file IK should consume: prefer the combined-hand
    # TRC (ROI + Holistic-recovered frames), then the ROI-hand TRC, then
    # fall back to the registered one.
    trcs = (sorted(dest.glob(f"{stem}*combhands.trc"))
            or sorted(dest.glob(f"{stem}*roihands.trc"))
            or sorted(p for p in dest.glob(f"{stem}*.trc")
                      if "roihands" not in p.name
                      and "combhands" not in p.name))
    if not trcs:
        sys.exit(f"run_pipeline: no .trc produced under {dest}/")
    trc = trcs[0]
    mot = trc.with_suffix(".ik.mot")  # matches run_ik's default naming

    # Optional measurement-based scaling: fit the model to the subject
    # from the just-produced TRC, then hand the scaled .osim to IK/ID/viz.
    # scale_model writes it beside the original model so its relative
    # Geometry/ still resolves (see scale_model.py); per-clip name keeps
    # multiple subjects from colliding.
    if a.scale:
        step += 1
        scaled = Path(model).with_name(f"{trc.stem}.scaled.osim")
        print(f"\n=== [{step}/{nst}] scale_model  {trc.name}  "
              f"(-> {scaled.name}) ===", flush=True)
        _invoke("scale_model", scale_model.main,
                [str(trc), "--model", model, "--out", str(scaled),
                 *cfg_args], a.debug)
        if not scaled.is_file():
            sys.exit(f"run_pipeline: scaling did not produce {scaled}")
        model = str(scaled)

    step += 1
    print(f"\n=== [{step}/{nst}] run_ik  {trc.name} ===", flush=True)
    _invoke("run_ik", run_ik.main,
            [str(trc), "--model", model, "--out", str(mot), *cfg_args],
            a.debug)
    if not mot.is_file():
        sys.exit(f"run_pipeline: IK did not produce {mot}")

    # Optional GRF estimation: derive ground reactions from the IK
    # trajectory and feed the resulting ExternalLoads to ID and SO, so
    # both are dynamically consistent (base residual ~0) instead of
    # dumping the net wrench on the free base joint. Use the same lowpass
    # across run_grf/run_id/run_so (config) for the tightest residuals.
    # With --grf, GRF/ID/SO must share one lowpass so their coordinate
    # differentiation matches; otherwise the GRF (filtered) cannot cancel
    # ID's (unfiltered, noisy) base residual. This explicit flag wins
    # over any per-stage config lowpass -- inconsistent cutoffs here are
    # simply wrong.
    extloads_args = []
    dyn_args = ["--lowpass", str(a.lowpass)] if a.grf else []
    if a.grf:
        step += 1
        print(f"\n=== [{step}/{nst}] run_grf  {mot.name} ===", flush=True)
        _invoke("run_grf", run_grf.main,
                [str(mot), "--model", model, *cfg_args, *dyn_args], a.debug)
        xml = Path(run_grf.xml_out_path(run_grf.grf_out_path(mot)))
        if not xml.is_file():
            sys.exit(f"run_pipeline: GRF did not produce {xml}")
        extloads_args = ["--external-loads", str(xml)]

    step += 1
    if a.no_id:
        print(f"\n=== [{step}/{nst}] run_id  skipped (--no-id) ===",
              flush=True)
    else:
        print(f"\n=== [{step}/{nst}] run_id  {mot.name} ===", flush=True)
        _invoke("run_id", run_id.main,
                [str(mot), "--model", model, *cfg_args, *extloads_args,
                 *dyn_args], a.debug)

    # Static Optimization: per-muscle forces from the IK trajectory on the
    # same (possibly scaled) model. Independent of ID -- both consume the
    # IK .mot -- so it runs whether or not ID was skipped.
    if a.so:
        step += 1
        print(f"\n=== [{step}/{nst}] run_so  {mot.name} ===", flush=True)
        _invoke("run_so", run_so.main,
                [str(mot), "--model", model, *cfg_args, *extloads_args,
                 *dyn_args], a.debug)

    step += 1
    if a.no_viz:
        print(f"\n=== [{step}/{nst}] viz_osim  skipped (--no-viz) ===",
              flush=True)
    else:
        print(f"\n=== [{step}/{nst}] viz_osim  {mot.name} ===", flush=True)
        _invoke("viz_osim", viz_osim.main,
                [model, str(mot), *cfg_args], a.debug)

    # Combine post-step: stack the mp2trc _merged annotated video above the
    # render and mux audio. Cosmetic and dependent on external bits (the
    # _merged.mp4, ffmpeg), so a failure here is non-fatal -- the core
    # outputs are already written.
    if combine:
        step += 1
        viz_mp4 = viz_osim.viz_out_path(mot)
        print(f"\n=== [{step}/{nst}] combine_viz  {viz_mp4.name} ===",
              flush=True)
        try:
            _invoke("combine_viz", combine_viz.main,
                    [str(viz_mp4), *cfg_args], a.debug)
        except Exception as exc:
            print(f"combine_viz skipped (non-fatal): {exc}", flush=True)

    print(f"\nDone. Outputs in {dest}/", flush=True)


if __name__ == "__main__":
    main()
