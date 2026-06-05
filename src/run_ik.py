#!/usr/bin/env python3
"""Run OpenSim Inverse Kinematics on a model + TRC, robustly.

    .venv/bin/python src/run_ik.py path/to/<name>.trc

Solves IK frame-by-frame with InverseKinematicsSolver instead of
InverseKinematicsTool.run(): the tool aborts the *entire* solve if any
single frame fails to converge (common with markerless data on an
unscaled model). Here a non-converging frame keeps the last good pose,
is logged, and the solve continues -- one bad frame no longer kills a
20 s clip. Writes a standard IK .mot and a marker-error summary.

Needs `opensim` (in the project .venv; see docs/SETUP.md). The
mediapipe->TRC stage (mp2trc.py) deliberately does not run IK; this is
that separate step. Per-frame failures are a model/data mismatch, not a
solver bug -- scale the model (see scale_model) for accurate results.
"""
import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mpipe_pipeline as M
import pipeline_config as PC


def run(trc, model=None, out=None, start=None, end=None,
        accuracy=None, weight=None, report_errors=True, revert_rms=0.10,
        progress=True, auto_weights=False, marker_weights=None,
        use_crop_meta=True):
    """Robust per-frame IK; returns the output .mot path.

    On a non-converged frame OpenSim leaves the state at Ipopt's best
    iterate; keep it (logged "soft") when its marker RMS <= revert_rms,
    else treat as diverged and hold the previous pose ("reverted")."""
    import opensim as osim

    trc = Path(trc).resolve()
    if not trc.exists():
        raise FileNotFoundError(trc)
    model_path = Path(model).resolve() if model else M.DEFAULT_MODEL
    out = Path(out).resolve() if out else trc.with_suffix(".ik.mot")
    acc = accuracy if accuracy is not None else M.DEFAULT_IK_ACCURACY
    wt = weight if weight is not None else M.DEFAULT_IK_WEIGHT

    osim_model = osim.Model(str(model_path))
    state = osim_model.initSystem()

    # Per-marker IK weights: data-driven from rigid-segment consistency
    # (--auto-weights) and/or explicit overrides (e.g. de-weight the
    # unreliable shoulders). A non-empty MarkerWeightSet makes
    # MarkersReference track exactly the listed markers, so every TRC
    # marker is listed.
    trc_names, _ = M.read_trc(trc)
    wmap = (M.auto_marker_weights(str(trc), base=wt) if auto_weights
            else {n: wt for n in trc_names})

    # Stage 2 sidecar: body markers cropped out of frame. MediaPipe
    # still emits a (often stably wrong) world position for them, so
    # rigid-CV auto-weights miss it -- force their weight to 0 here.
    # Explicit --marker-weight overrides still win (applied after).
    crop_zeroed = []
    if use_crop_meta:
        meta = M.read_ik_meta(trc)
        if meta:
            for n in meta.get("zeroed_in_ik", []):
                if n in wmap:
                    wmap[n] = 0.0
                    crop_zeroed.append(n)
            if crop_zeroed:
                print(f"cropped (weight 0 from {M.ik_meta_path(trc).name}, "
                      f"{len(crop_zeroed)}): {', '.join(sorted(crop_zeroed))}")

    for n, w in (marker_weights or {}).items():
        if n in wmap:
            wmap[n] = w
    wset = osim.SetMarkerWeights()
    for n in trc_names:
        wset.cloneAndAppend(osim.MarkerWeight(n, float(wmap[n])))
    nd = sorted((w, n) for n, w in wmap.items() if abs(w - wt) > 1e-6)
    if nd:
        head = ", ".join(f"{n}={w:.2f}" for w, n in nd[:8])
        print(f"non-default weights ({len(nd)}): {head}"
              + (" ..." if len(nd) > 8 else ""))

    markers = osim.MarkersReference()
    markers.set_default_weight(wt)
    markers.initializeFromMarkersFile(str(trc), wset)
    coord_refs = osim.SimTKArrayCoordinateReference()
    ik = osim.InverseKinematicsSolver(osim_model, markers, coord_refs)
    ik.setAccuracy(acc)

    def marker_errs():
        # OpenSim 4.6's computeCurrentMarkerError takes the marker NAME
        # (the int-index overload present in 4.5 was removed), so map
        # each in-use index to its name.
        return [ik.computeCurrentMarkerError(ik.getMarkerNameForIndex(i))
                for i in range(ik.getNumMarkersInUse())]

    table = markers.getMarkerTable()
    times = [table.getIndependentColumn()[i]
             for i in range(table.getNumRows())]

    coords = osim_model.getCoordinateSet()
    ncoord = coords.getSize()
    is_rot = [coords.get(i).getMotionType() == osim.Coordinate.Rotational
              for i in range(ncoord)]

    labels = osim.StdVectorString()
    for i in range(ncoord):
        labels.append(coords.get(i).getName())
    tbl = osim.TimeSeriesTable()
    tbl.setColumnLabels(labels)

    # Assemble once at the first frame (COM-centred -> near the model's
    # default pose, easy) then track sequentially with warm starts --
    # cold-assembling to an arbitrary mid-motion frame is the hard case.
    # start/end bound only the written output, not where tracking begins.
    # assemble() can hit Ipopt max-iter at the unreachable ~1e-5
    # tolerance while the seed is actually fine (sub-cm); keep that
    # best iterate (same policy as the per-frame loop) and only abort if
    # it genuinely diverged -- which usually means a bad model/units or
    # an un-centred TRC (regenerate with --center-on-model-com).
    state.setTime(times[0])
    try:
        ik.assemble(state)
    except RuntimeError as exc:
        n = ik.getNumMarkersInUse()
        e = marker_errs()
        rms = math.sqrt(sum(v * v for v in e) / n) if n else float("inf")
        if not (math.isfinite(rms) and rms <= revert_rms):
            raise RuntimeError(
                f"initial assemble() diverged (marker RMS {rms:.3f} m > "
                f"{revert_rms} m). The TRC is likely far from the model's "
                f"default pose -- regenerate it with "
                f"--center-on-model-com (and/or scale the model).") from exc
        print(f"note: assemble() non-converged but seed is good "
              f"(RMS {rms:.4f} m); continuing.")
    soft, reverted, rms_all = [], [], []
    last_q = None
    bar = None
    if progress:
        try:
            from tqdm import tqdm
            ntrack = sum(1 for t in times if end is None or t <= end)
            bar = tqdm(total=ntrack, unit="frame", desc="IK")
        except ImportError:
            bar = None
    for k, t in enumerate(times):
        if end is not None and t > end:
            break
        if bar is not None:
            bar.update(1)
        state.setTime(t)
        try:
            ik.track(state)
            ok = True
        except RuntimeError:
            # Ipopt hit max-iter, but OpenSim leaves the state at its
            # BEST iterate -- for markerless data that is usually still
            # sub-cm and far better than freezing to a stale pose. Keep
            # it; only revert if it actually diverged.
            ok = False

        n = ik.getNumMarkersInUse()
        e = marker_errs()
        rms = math.sqrt(sum(v * v for v in e) / n) if n else 0.0

        if not ok:
            if math.isfinite(rms) and rms <= revert_rms:
                soft.append(t)                    # non-converged but good
            else:
                reverted.append(t)                # diverged -> hold pose
                if last_q is not None:
                    for i in range(ncoord):
                        coords.get(i).setValue(state, last_q[i], False)
                    osim_model.assemble(state)
                    e = marker_errs()
                    rms = math.sqrt(sum(v * v for v in e) / n) if n else 0.0
        q = [coords.get(i).getValue(state) for i in range(ncoord)]
        last_q = q

        if start is not None and t < start:
            continue                              # warm-up, not recorded

        if report_errors:
            rms_all.append(rms)

        row = osim.RowVector(ncoord, 0.0)
        for i in range(ncoord):
            row[i] = math.degrees(q[i]) if is_rot[i] else q[i]
        tbl.appendRow(t, row)

    if bar is not None:
        bar.close()
    tbl.addTableMetaDataString("inDegrees", "yes")
    out.parent.mkdir(parents=True, exist_ok=True)
    osim.STOFileAdapter().write(tbl, str(out))

    nrec = tbl.getNumRows()
    print(f"model : {model_path}")
    print(f"trc   : {trc}")
    print(f"mot   : {out}")
    print(f"written {nrec} frames  |  soft (non-converged, kept best "
          f"iterate): {len(soft)}  |  reverted (diverged, held pose): "
          f"{len(reverted)}")
    for label, lst in (("soft", soft), ("reverted", reverted)):
        if lst:
            ft = ", ".join(f"{t:.3f}s" for t in lst[:8])
            more = "" if len(lst) <= 8 else f" (+{len(lst) - 8} more)"
            print(f"  {label} @ {ft}{more}")
    if report_errors and rms_all:
        print(f"marker RMS: mean={sum(rms_all) / len(rms_all):.4f} m  "
              f"max={max(rms_all):.4f} m  over {len(rms_all)} frames")
    return str(out)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Robust per-frame OpenSim IK on a model + TRC (needs "
                    "opensim; use the project .venv -- see docs/SETUP.md)")
    ap.add_argument("trc", help="TRC marker file")
    ap.add_argument("--model", default=None,
                    help=f"OSIM model (default: {M.DEFAULT_MODEL.name})")
    ap.add_argument("--out", default=None,
                    help="output .mot (default: <trc>.ik.mot)")
    ap.add_argument("--start", type=float, default=None,
                    help="start time (s)")
    ap.add_argument("--end", type=float, default=None,
                    help="end time (s)")
    ap.add_argument("--accuracy", type=float, default=None,
                    help=f"IK accuracy (default: {M.DEFAULT_IK_ACCURACY})")
    ap.add_argument("--weight", type=float, default=None,
                    help=f"uniform marker weight (default: "
                         f"{M.DEFAULT_IK_WEIGHT})")
    ap.add_argument("--no-report-errors", action="store_true",
                    help="skip the marker-error report")
    ap.add_argument("--revert-rms", type=float, default=0.10,
                    help="marker RMS (m) above which a non-converged "
                         "frame is treated as diverged and the previous "
                         "pose held (default: 0.10)")
    ap.add_argument("--no-progress", action="store_true",
                    help="hide the per-frame progress bar")
    ap.add_argument("--auto-weights", action="store_true",
                    help="data-driven per-marker weights: down-weight "
                         "markers on kinematically inconsistent bones "
                         "(adapts per clip; general unreliable-marker fix)")
    ap.add_argument("--marker-weight", action="append", default=[],
                    metavar="NAME=W",
                    help="explicit per-marker weight override, repeatable "
                         "(e.g. --marker-weight LEFT_SHOULDER=0.1)")
    ap.add_argument("--no-crop-meta", action="store_true",
                    help="ignore the <trc>.ik_meta.json sidecar (do NOT "
                         "auto-zero markers cropped out of frame)")
    PC.add_args(ap)
    argv = list(sys.argv[1:] if argv is None else argv)
    _cfg = PC.apply(ap, "run_ik", argv)
    a = ap.parse_args(argv)
    if _cfg:
        print(f"run_ik: config {_cfg}", flush=True)
    mw = {}
    for kv in a.marker_weight:
        k, v = kv.split("=")
        mw[k] = float(v)
    run(a.trc, model=a.model, out=a.out, start=a.start, end=a.end,
        accuracy=a.accuracy, weight=a.weight,
        report_errors=not a.no_report_errors, revert_rms=a.revert_rms,
        progress=not a.no_progress, auto_weights=a.auto_weights,
        marker_weights=mw, use_crop_meta=not a.no_crop_meta)


if __name__ == "__main__":
    main()
