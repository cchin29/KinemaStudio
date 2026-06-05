#!/usr/bin/env python3
"""Run OpenSim Static Optimization on a model + an IK .mot to estimate
per-muscle forces.

    .venv/bin/python src/run_so.py path/to/<name>.ik.mot

The step *beyond* run_id.py. Inverse Dynamics gives the net generalized
force at each joint; Static Optimization (SO) goes further and resolves
those net joint moments into the individual muscle forces that produce
them, by minimizing the sum of muscle activations^p at every frame
(subject to each muscle's force-generating capacity). This is the true
muscle-force decomposition viz_osim's muscle colouring approximates.

Writes two .sto next to the .mot:
    <root>.so_force.sto        per-actuator force (N)   <- the muscle forces
    <root>.so_activation.sto   per-actuator activation (0..1)

Markerless capture has no ground-reaction / external-load measurement
and the model's base is a free joint, so SO cannot be reproduced by
muscles alone (nothing balances gravity at the floating base, and some
coordinates -- fingers, base translations -- have no spanning muscle).
By default we therefore append a reserve/residual CoordinateActuator to
every coordinate (OpenSim's standard fix). The reserves absorb whatever
the muscles cannot, so SO stays solvable; muscle-force estimates are
trustworthy where muscle actuation dominates and the matching reserve
sits near zero. A large reserve force on a muscled coordinate is the
flag that SO leaned on the reserve there. --reserve-force tunes the
optimal force (low = expensive = muscles preferred), --no-reserves
drops them (expect non-convergence on a floating-base full-body model).
Supplying --external-loads (e.g. run_grf.py's ExternalLoads) removes
the base residual the reserves would carry, so SO converges far better.

SO is a per-frame optimization, so it is slow over a long clip. Two
independent speed-ups:
  * --jobs N  solves the time range in N parallel processes (frames are
    independent; near-linear on a multi-core box). The per-frame SO
    maths is unchanged; only the cross-frame warm start is lost at chunk
    seams.
  * --stride N  solves every Nth frame (muscle activation is low-
    bandwidth, so this previews fast); viz_osim's nearest-time colour
    lookup handles the coarser output. With stride the cutoff is clamped
    to the decimated Nyquist.

Needs `opensim` (in the project .venv; see docs/SETUP.md), like run_ik
/ run_id. SO double-uses the IK coordinates' accelerations, so the same
--lowpass advice as run_id applies (markerless IK is noisy).
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pipeline_config as PC


def so_out_path(mot):
    """Force-output .sto next to the coordinates file: a trailing
    `.ik.mot` becomes `.so_force.sto` (so `<root>.combhands.ik.mot` ->
    `<root>.combhands.so_force.sto`); any other `.mot`/name just gains
    `.so_force.sto`. The activation file swaps `so_force` -> `so_activation`."""
    mot = Path(mot)
    name = mot.name
    if name.endswith(".ik.mot"):
        name = name[: -len(".ik.mot")] + ".so_force.sto"
    elif name.endswith(".mot"):
        name = name[: -len(".mot")] + ".so_force.sto"
    else:
        name = name + ".so_force.sto"
    return mot.with_name(name)


def _activation_path(force_out):
    """The activation .sto beside a `.so_force.sto` force file."""
    force_out = Path(force_out)
    if force_out.name.endswith(".so_force.sto"):
        return force_out.with_name(
            force_out.name[: -len(".so_force.sto")] + ".so_activation.sto")
    return force_out.with_name(force_out.stem + ".activation.sto")


def _split_sto(path):
    """Split an OpenSim .sto into (header_lines_incl_endheader,
    column_header_line, [data_lines])."""
    lines = Path(path).read_text().splitlines()
    ie = next(i for i, l in enumerate(lines)
              if l.strip().lower() == "endheader")
    data = [d for d in lines[ie + 2:] if d.strip() != ""]
    return lines[: ie + 1], lines[ie + 1], data


def _decimate_mot(mot, stride, out_path):
    """Write every `stride`-th row of an OpenSim coordinates .mot/.sto to
    out_path (header + column line preserved, nRows fixed). Returns the
    kept-row count."""
    hdr, col, data = _split_sto(mot)
    kept = data[::stride]
    newhdr = [f"nRows={len(kept)}" if l.lower().startswith("nrows=") else l
              for l in hdr]
    Path(out_path).write_text("\n".join(newhdr + [col] + kept) + "\n")
    return len(kept)


def _concat_sto(paths, out):
    """Concatenate same-schema .sto files (chunk outputs, already in time
    order) into one, fixing nRows. Column header is taken from the first."""
    hdr, col, rows = _split_sto(paths[0])
    for p in paths[1:]:
        rows = rows + _split_sto(p)[2]
    newhdr = [f"nRows={len(rows)}" if l.lower().startswith("nrows=") else l
              for l in hdr]
    Path(out).write_text("\n".join(newhdr + [col] + rows) + "\n")


def _solve_chunk(args):
    """Worker: Static Optimization over one [t0, t1] on a prebuilt
    (reserve-augmented) model file, into its own temp results dir, then
    move the force/activation .sto to the given paths. Module-level and
    picklable so it runs under ProcessPoolExecutor. Returns
    (force_path, activation_path|None)."""
    (aug_model, mot, force_out, act_out, t0, t1, lowpass, act_exp,
     use_phys, extloads) = args
    import shutil
    import tempfile

    import opensim as osim

    rdir = tempfile.mkdtemp(prefix="run_so_")
    try:
        so = osim.StaticOptimization()
        so.setStartTime(float(t0))
        so.setEndTime(float(t1))
        so.setUseModelForceSet(True)        # muscles + reserves in the model
        so.setActivationExponent(float(act_exp))
        so.setUseMusclePhysiology(bool(use_phys))

        # XML round-trip: setModel on a live object segfaults / doesn't
        # build states-from-coordinates; reconstructing AnalyzeTool from a
        # setup file does the full input setup before solving. Absolute
        # paths so it resolves regardless of cwd; a private results dir so
        # parallel workers never collide.
        tool = osim.AnalyzeTool()
        tool.setName("so")
        tool.setModelFilename(str(aug_model))
        tool.setCoordinatesFileName(str(mot))
        tool.setLowpassCutoffFrequency(float(lowpass))
        tool.setInitialTime(float(t0))
        tool.setFinalTime(float(t1))
        tool.setResultsDir(rdir)
        tool.setSolveForEquilibrium(False)
        if extloads:
            tool.setExternalLoadsFileName(str(extloads))
        tool.getAnalysisSet().cloneAndAppend(so)
        setup = str(Path(rdir) / "so_setup.xml")
        tool.printToXML(setup)

        osim.AnalyzeTool(setup).run()

        pf = Path(rdir) / "so_StaticOptimization_force.sto"
        pa = Path(rdir) / "so_StaticOptimization_activation.sto"
        if not pf.exists():
            raise RuntimeError(
                f"SO produced no force file for [{t0:.3f}, {t1:.3f}] s. "
                f"Common causes: the .mot does not match the model, the "
                f"model failed to load, or SO could not converge (try "
                f"--reserve-force higher, --lowpass 6, or --external-loads).")
        shutil.move(str(pf), str(force_out))
        has_act = pa.exists()
        if has_act:
            shutil.move(str(pa), str(act_out))
        return (str(force_out), str(act_out) if has_act else None)
    finally:
        shutil.rmtree(rdir, ignore_errors=True)


def run(mot, model=None, out=None, start=None, end=None, lowpass=-1.0,
        activation_exponent=2.0, reserve_force=1.0, append_reserves=True,
        use_muscle_physiology=True, external_loads=None, stride=1, jobs=1):
    """Solve Static Optimization; returns the force .sto path.

    `mot` is an IK coordinates file (.mot/.sto from run_ik.py). Time
    range defaults to the file's full span; `--start`/`--end` clip it.
    `lowpass` < 0 disables coordinate filtering. `external_loads` is an
    OpenSim ExternalLoads .xml (e.g. from run_grf.py). `stride` solves
    every Nth frame; `jobs` solves the range in that many parallel
    processes."""
    from concurrent.futures import ProcessPoolExecutor

    import numpy as np
    import opensim as osim
    import mpipe_pipeline as M

    mot = Path(mot).resolve()
    if not mot.exists():
        raise FileNotFoundError(mot)
    model_path = Path(model).resolve() if model else M.DEFAULT_MODEL
    if not Path(model_path).exists():
        raise FileNotFoundError(model_path)
    extloads = Path(external_loads).resolve() if external_loads else None
    if extloads and not extloads.exists():
        raise FileNotFoundError(extloads)
    out = Path(out).resolve() if out else so_out_path(mot).resolve()
    act_out = _activation_path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    stride = max(1, int(stride))
    jobs = max(1, int(jobs))

    # --stride: decimate the coordinates so SO solves fewer frames.
    work_mot, tmp_mot = mot, None
    if stride > 1:
        tmp_mot = out.parent / (out.stem + f".stride{stride}.mot")
        _decimate_mot(mot, stride, tmp_mot)
        work_mot = tmp_mot.resolve()

    coords = osim.Storage(str(work_mot))
    times = [coords.getStateVector(k).getTime()
             for k in range(coords.getSize())]
    sel = [k for k, t in enumerate(times)
           if (start is None or t >= start) and (end is None or t <= end)]
    if not sel:
        raise RuntimeError("no frames in the requested time range")
    fs = 1.0 / float(np.median(np.diff(times))) if len(times) > 1 else 0.0

    # Guard the cutoff against the (decimated) Nyquist.
    eff_lowpass = lowpass
    if stride > 1 and lowpass and lowpass > 0 and fs > 0 \
            and lowpass >= 0.5 * fs:
        eff_lowpass = 0.45 * fs
        print(f"note: stride {stride} -> {fs:.1f} Hz; clamping lowpass "
              f"{lowpass:g} -> {eff_lowpass:.2f} Hz (decimated Nyquist)",
              flush=True)

    # Build the reserve-augmented model ONCE; all workers point at it
    # (read-only). PID-tagged so concurrent run_so processes don't clash.
    osim_model = osim.Model(str(model_path))
    nreserve = 0
    if append_reserves:
        osim_model.initSystem()
        cs = osim_model.getCoordinateSet()
        for i in range(cs.getSize()):
            c = cs.get(i)
            ca = osim.CoordinateActuator()
            ca.setCoordinate(c)
            ca.setName(c.getName() + "_reserve")
            ca.setOptimalForce(float(reserve_force))
            ca.setMinControl(-1e9)
            ca.setMaxControl(1e9)
            osim_model.addForce(ca)
            nreserve += 1
        osim_model.initSystem()
    aug_model = model_path.with_name(
        f"{model_path.stem}.so_tmp.{os.getpid()}.osim")
    osim_model.printToXML(str(aug_model))

    # Split the selected frames into `jobs` contiguous, non-overlapping
    # index chunks (disjoint time ranges => no duplicate rows at seams).
    nf = len(sel)
    jobs = min(jobs, nf)
    chunks = []
    for j in range(jobs):
        a, b = j * nf // jobs, (j + 1) * nf // jobs
        if a < b:
            chunks.append((times[sel[a]], times[sel[b - 1]]))
    single = len(chunks) == 1

    cf = [out if single else out.parent / f"{out.stem}.chunk{j}.force.sto"
          for j in range(len(chunks))]
    ca_ = [act_out if single else out.parent / f"{out.stem}.chunk{j}.act.sto"
           for j in range(len(chunks))]
    tasks = [(str(aug_model), str(work_mot), str(cf[j]), str(ca_[j]),
              float(c0), float(c1), float(eff_lowpass),
              float(activation_exponent), bool(use_muscle_physiology),
              str(extloads) if extloads else None)
             for j, (c0, c1) in enumerate(chunks)]

    try:
        if len(tasks) == 1:
            res = [_solve_chunk(tasks[0])]
        else:
            print(f"running SO in {len(tasks)} parallel jobs "
                  f"({nf} frames)...", flush=True)
            with ProcessPoolExecutor(max_workers=jobs) as ex:
                res = list(ex.map(_solve_chunk, tasks))
        have_act = all(r[1] for r in res)
        if not single:
            _concat_sto([str(p) for p in cf], out)
            if have_act:
                _concat_sto([str(p) for p in ca_], act_out)
    finally:
        Path(aug_model).unlink(missing_ok=True)
        if tmp_mot:
            Path(tmp_mot).unlink(missing_ok=True)
        if not single:
            for p in cf + ca_:
                Path(p).unlink(missing_ok=True)

    print(f"model : {model_path}")
    print(f"mot   : {mot}")
    print(f"force : {out}")
    print(f"act   : {act_out if have_act else '(none)'}")
    print(f"time range : {chunks[0][0]:.4f} .. {chunks[-1][1]:.4f} s  |  "
          f"{nf} frames solved" + (f" (stride {stride})" if stride > 1
                                    else "") + f"  |  jobs {jobs}")
    print(f"reserves : {nreserve} CoordinateActuators "
          f"(optimal force {reserve_force:g})" if append_reserves
          else "reserves : none (--no-reserves)")
    print(f"activation exponent : {activation_exponent:g}  |  muscle "
          f"physiology : {'on' if use_muscle_physiology else 'off'}  |  "
          f"lowpass : {'off' if eff_lowpass < 0 else f'{eff_lowpass:g} Hz'}")
    print(f"external loads : {extloads if extloads else '(none -- '
          'reserves carry the base residual)'}")
    return str(out)


def main(argv=None):
    import mpipe_pipeline as M
    ap = argparse.ArgumentParser(
        description="OpenSim Static Optimization on a model + an IK .mot "
                    "-> per-muscle forces (needs opensim; use the project "
                    ".venv -- see docs/SETUP.md). Run after run_ik.py.")
    ap.add_argument("mot", help="IK coordinates file (.mot/.sto from "
                                "run_ik.py)")
    ap.add_argument("--model", default=None,
                    help=f"OSIM model (default: {M.DEFAULT_MODEL.name}; "
                         f"must match the one IK was solved against)")
    ap.add_argument("--out", default=None,
                    help="output force .sto (default: <root>.so_force.sto "
                         "next to the .mot; activations go to the matching "
                         "<root>.so_activation.sto)")
    ap.add_argument("--start", type=float, default=None,
                    help="start time (s); default = mot start")
    ap.add_argument("--end", type=float, default=None,
                    help="end time (s); default = mot end")
    ap.add_argument("--lowpass", type=float, default=-1.0,
                    help="low-pass cutoff (Hz) for the coordinates before "
                         "differentiation; <0 = no filter (default). "
                         "Markerless IK is noisy -- 6 is a sensible start.")
    ap.add_argument("--activation-exponent", type=float, default=2.0,
                    help="exponent p in the minimised sum(activation^p) "
                         "(default: 2)")
    ap.add_argument("--reserve-force", type=float, default=1.0,
                    help="optimal force of the per-coordinate reserve "
                         "actuators (default: 1.0; lower = reserves more "
                         "expensive = muscles preferred)")
    ap.add_argument("--no-reserves", action="store_true",
                    help="do NOT append reserve actuators (expect SO to "
                         "fail on a floating-base full-body model)")
    ap.add_argument("--no-muscle-physiology", action="store_true",
                    help="treat muscles as ideal force generators (ignore "
                         "force-length-velocity); faster + more robust, "
                         "less physiological")
    ap.add_argument("--external-loads", default=None, metavar="XML",
                    help="OpenSim ExternalLoads .xml (e.g. "
                         "<root>.externalloads.xml from run_grf.py); "
                         "supplying the ground reaction collapses the base "
                         "residual and greatly improves SO convergence")
    ap.add_argument("--jobs", type=int, default=1,
                    help="solve the time range in this many parallel "
                         "processes (frames are independent; near-linear "
                         "speed-up, default: 1)")
    ap.add_argument("--stride", type=int, default=1,
                    help="solve every Nth frame only (fast preview; the "
                         "cutoff is clamped to the decimated Nyquist, "
                         "default: 1 = every frame)")
    PC.add_args(ap)
    argv = list(sys.argv[1:] if argv is None else argv)
    _cfg = PC.apply(ap, "run_so", argv)
    a = ap.parse_args(argv)
    if _cfg:
        print(f"run_so: config {_cfg}", flush=True)
    run(a.mot, model=a.model, out=a.out, start=a.start, end=a.end,
        lowpass=a.lowpass, activation_exponent=a.activation_exponent,
        reserve_force=a.reserve_force, append_reserves=not a.no_reserves,
        use_muscle_physiology=not a.no_muscle_physiology,
        external_loads=a.external_loads, stride=a.stride, jobs=a.jobs)


if __name__ == "__main__":
    main()
