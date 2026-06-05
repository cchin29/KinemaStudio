#!/usr/bin/env python3
"""Run OpenSim Inverse Dynamics on a model + an IK .mot, robustly.

    .venv/bin/python src/run_id.py path/to/<name>.ik.mot

The step *after* run_ik.py: given the IK joint-angle trajectory it
solves the net generalized forces (joint moments / torques) that, with
gravity and segment inertia, produce that motion. Muscles are excluded
(`forces_to_exclude = Muscles`, same as the legacy *.RH.ID setup
files): markerless capture has no measured muscle activity or force
plates, so ID returns the *net* joint loads, not muscle-resolved ones.

Needs `opensim` (in the project .venv; see docs/SETUP.md), exactly
like run_ik.py. Inverse Dynamics double-differentiates the coordinates,
so it amplifies noise: markerless IK is noisy and a low-pass on the
coordinates (`--lowpass HZ`, e.g. 6) is usually advisable. Default is
-1 (no filtering) to match OpenSim's default and the legacy setups --
opt in explicitly so the choice is visible.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mpipe_pipeline as M
import pipeline_config as PC


def id_out_path(mot):
    """Output .sto next to the coordinates file: a trailing `.ik.mot`
    becomes `.id.sto` (so `<root>.roihands.ik.mot` ->
    `<root>.roihands.id.sto`); any other `.mot`/name just gains
    `.id.sto`."""
    mot = Path(mot)
    name = mot.name
    if name.endswith(".ik.mot"):
        name = name[: -len(".ik.mot")] + ".id.sto"
    elif name.endswith(".mot"):
        name = name[: -len(".mot")] + ".id.sto"
    else:
        name = name + ".id.sto"
    return mot.with_name(name)


def run(mot, model=None, out=None, start=None, end=None,
        lowpass=-1.0, exclude=("Muscles",), external_loads=None):
    """Solve Inverse Dynamics; returns the output .sto path.

    `mot` is an IK coordinates file (.mot/.sto, as written by
    run_ik.py). Time range defaults to the file's full span; `--start`
    /`--end` clip it. `lowpass` < 0 disables coordinate filtering.
    `external_loads` is an OpenSim ExternalLoads .xml (e.g. from
    run_grf.py) -- with it the base residual collapses to ~0 and the
    joint moments become physically meaningful; without it ID is
    top-down and the unbalanced wrench lands on the free base joint."""
    import opensim as osim

    mot = Path(mot).resolve()
    if not mot.exists():
        raise FileNotFoundError(mot)
    model_path = Path(model).resolve() if model else M.DEFAULT_MODEL
    if not Path(model_path).exists():
        raise FileNotFoundError(model_path)
    out = Path(out).resolve() if out else id_out_path(mot).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    extloads = Path(external_loads).resolve() if external_loads else None
    if extloads and not extloads.exists():
        raise FileNotFoundError(extloads)

    # Full span from the coordinates file unless clipped. The tool would
    # clamp an unset range itself, but read it so the printed summary
    # and --start/--end are exact.
    sto = osim.Storage(str(mot))
    t0 = sto.getFirstTime() if start is None else start
    t1 = sto.getLastTime() if end is None else end

    idt = osim.InverseDynamicsTool()
    idt.setModelFileName(str(model_path))
    idt.setCoordinatesFileName(str(mot))
    idt.setStartTime(t0)
    idt.setEndTime(t1)
    idt.setLowpassCutoffFrequency(float(lowpass))
    ex = osim.ArrayStr()
    for f in exclude:
        ex.append(f)
    idt.setExcludedForces(ex)
    if extloads:
        idt.setExternalLoadsFileName(str(extloads))
    # The tool writes results_dir/<output_gen_force_file> (filename
    # only), so split the chosen path into the two properties.
    idt.setResultsDir(str(out.parent))
    idt.setOutputGenForceFileName(out.name)

    ok = idt.run()

    if not out.exists():
        raise RuntimeError(
            f"InverseDynamicsTool.run() returned {ok} and produced no "
            f"output at {out}. Common causes: the .mot coordinates do "
            f"not match the model's coordinate set, or the model failed "
            f"to load. Check that <mot> came from run_ik.py on this "
            f"same --model.")

    nrows = sto.getSize()
    print(f"model : {model_path}")
    print(f"mot   : {mot}")
    print(f"sto   : {out}")
    print(f"time range : {t0:.4f} .. {t1:.4f} s  ({nrows} input rows)")
    print(f"forces excluded : {', '.join(exclude) or '(none)'}  |  "
          f"lowpass : {'off' if lowpass < 0 else f'{lowpass:g} Hz'}")
    print(f"external loads : {extloads if extloads else '(none -- '
          'top-down; base residual carries the net wrench)'}")
    return str(out)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="OpenSim Inverse Dynamics on a model + an IK .mot "
                    "(needs opensim; use the project .venv -- see "
                    "docs/SETUP.md). Run after run_ik.py.")
    ap.add_argument("mot", help="IK coordinates file (.mot/.sto from "
                                "run_ik.py)")
    ap.add_argument("--model", default=None,
                    help=f"OSIM model (default: {M.DEFAULT_MODEL.name}; "
                         f"must match the one IK was solved against)")
    ap.add_argument("--out", default=None,
                    help="output .sto (default: <root>.id.sto next to "
                         "the .mot)")
    ap.add_argument("--start", type=float, default=None,
                    help="start time (s); default = mot start")
    ap.add_argument("--end", type=float, default=None,
                    help="end time (s); default = mot end")
    ap.add_argument("--lowpass", type=float, default=-1.0,
                    help="low-pass cutoff (Hz) for the coordinates "
                         "before differentiation; <0 = no filter "
                         "(default). Markerless IK is noisy -- 6 is a "
                         "sensible starting point.")
    ap.add_argument("--include-muscles", action="store_true",
                    help="do NOT exclude muscle forces (default excludes "
                         "'Muscles': markerless has no measured muscle "
                         "activity, so ID gives net joint moments)")
    ap.add_argument("--external-loads", default=None, metavar="XML",
                    help="OpenSim ExternalLoads .xml (e.g. "
                         "<root>.externalloads.xml from run_grf.py) so the "
                         "ground reaction enters the dynamics; without it "
                         "the net wrench lands on the free base joint")
    PC.add_args(ap)
    argv = list(sys.argv[1:] if argv is None else argv)
    _cfg = PC.apply(ap, "run_id", argv)
    a = ap.parse_args(argv)
    if _cfg:
        print(f"run_id: config {_cfg}", flush=True)
    run(a.mot, model=a.model, out=a.out, start=a.start, end=a.end,
        lowpass=a.lowpass,
        exclude=() if a.include_muscles else ("Muscles",),
        external_loads=a.external_loads)


if __name__ == "__main__":
    main()
