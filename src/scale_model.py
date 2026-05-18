#!/usr/bin/env python3
"""Scale the combined_body_model to a subject from a markerless TRC.

    .venv/bin/python src/scale_model.py <name>.trc
    .venv/bin/python src/run_ik.py <name>.trc --model <...>.scaled.osim

OpenSim ModelScaler, measurement-based, segment lengths averaged over a
time range (default: the whole clip -- robust, no static T-pose needed),
mass distribution preserved, marker placement left as the model defines
it (the MediaPipe marker locations are intentional -- see model README).

The humerus is EXCLUDED by default: MediaPipe Pose's shoulder landmark
is a surface point, not the glenohumeral joint center, and a raised /
across-body arm is depth-foreshortened, so shoulder->elbow comes out
~40 % short (subject upper-arm < forearm, anatomically impossible).
Scaling it would distort the skeleton. Verified the model's shoulder/
elbow markers are correctly placed, so this is a data artifact, not a
model-marker problem. --include-humerus overrides (uses the raw, likely
corrupted, measurement).

Needs `opensim` (in the project .venv; see docs/SETUP.md).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mpipe_pipeline as M

# measurement name -> (marker A, marker B, [bodies scaled by that span])
MEASUREMENTS = [
    ("torso",     "LEFT_SHOULDER", "RIGHT_SHOULDER", ["torso"]),
    ("pelvis",    "LEFT_HIP",      "RIGHT_HIP",      ["pelvis"]),
    ("forearm_l", "LEFT_ELBOW",    "LEFT_WRIST",     ["ulna_l", "radius_l"]),
    ("forearm_r", "RIGHT_ELBOW",   "RIGHT_WRIST",    ["ulna_r", "radius_r"]),
    ("femur_l",   "LEFT_HIP",      "LEFT_KNEE",      ["femur_l"]),
    ("femur_r",   "RIGHT_HIP",     "RIGHT_KNEE",     ["femur_r"]),
    ("tibia_l",   "LEFT_KNEE",     "LEFT_ANKLE",     ["tibia_l"]),
    ("tibia_r",   "RIGHT_KNEE",    "RIGHT_ANKLE",    ["tibia_r"]),
]
HUMERUS = [
    ("humerus_l", "LEFT_SHOULDER",  "LEFT_ELBOW",  ["humerus_l"]),
    ("humerus_r", "RIGHT_SHOULDER", "RIGHT_ELBOW", ["humerus_r"]),
]


def _measurement(osim, name, a, b, bodies):
    m = osim.Measurement()
    m.setName(name)
    m.getMarkerPairSet().cloneAndAppend(osim.MarkerPair(a, b))
    for body in bodies:
        bs = osim.BodyScale()
        bs.setName(body)
        ax = osim.ArrayStr()
        for x in "XYZ":
            ax.append(x)
        bs.setAxisNames(ax)
        m.getBodyScaleSet().cloneAndAppend(bs)
    return m


def run(trc, model=None, out=None, start=None, end=None,
        include_humerus=False, mass=-1.0):
    """Scale the model to the TRC subject; returns the scaled .osim path."""
    import math

    import numpy as np
    import opensim as osim

    trc = Path(trc).resolve()
    if not trc.exists():
        raise FileNotFoundError(trc)
    model_path = Path(model).resolve() if model else M.DEFAULT_MODEL
    # Beside the original model so its relative Geometry/ still resolves.
    out = (Path(out).resolve() if out else
           model_path.with_name(trc.stem + ".scaled.osim"))

    spec = list(MEASUREMENTS) + (HUMERUS if include_humerus else [])

    # Report target factors (subject mean span / model default span).
    om = osim.Model(str(model_path))
    s = om.initSystem()
    mk = om.getMarkerSet()
    lines = [x for x in trc.read_text().splitlines() if x != ""]
    names = [t for t in lines[3].split("\t")[2:] if t]
    arr = np.array([[float(v) for v in r.split("\t")[2:]]
                    for r in lines[5:]]).reshape(len(lines) - 5,
                                                 len(names), 3)
    times = [float(r.split("\t")[1]) for r in lines[5:]]
    t0 = start if start is not None else times[0]
    t1 = end if end is not None else times[-1]
    sl = [i for i, t in enumerate(times) if t0 <= t <= t1]

    def mspan(a, b):
        pa, pb = mk.get(a).getLocationInGround(s), mk.get(b).getLocationInGround(s)
        return math.dist([pa.get(i) for i in range(3)],
                         [pb.get(i) for i in range(3)])

    print(f"model : {model_path}")
    print(f"trc   : {trc}  (avg t=[{t0:.2f},{t1:.2f}] s)")
    print(f"{'segment':10s} {'model':>8s} {'subj':>8s} {'factor':>7s}")
    for name, a, b, _ in spec:
        mo = mspan(a, b)
        ia, ib = names.index(a), names.index(b)
        su = float(np.linalg.norm(arr[sl, ia, :] - arr[sl, ib, :],
                                  axis=1).mean())
        flag = "  <-- excluded humerus, likely corrupt" if name.startswith(
            "humerus") else ""
        print(f"{name:10s} {mo:8.4f} {su:8.4f} {su / mo:7.3f}{flag}")

    mset = osim.MeasurementSet()
    for name, a, b, bodies in spec:
        mset.cloneAndAppend(_measurement(osim, name, a, b, bodies))

    scaler = osim.ModelScaler()
    scaler.setApply(True)
    scaler.setPreserveMassDist(True)
    scaler.setMeasurementSet(mset)
    scaler.setMarkerFileName(str(trc))
    tr = osim.ArrayDouble()
    tr.append(t0)
    tr.append(t1)
    scaler.setTimeRange(tr)
    order = osim.ArrayStr()
    order.append("measurements")
    scaler.setScalingOrder(order)

    model_obj = osim.Model(str(model_path))
    if not scaler.processModel(model_obj, "", mass):
        raise RuntimeError("ModelScaler.processModel failed")
    model_obj.printToXML(str(out))
    print(f"scaled: {out}")
    return str(out)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Scale combined_body_model to a subject from a TRC "
                    "(needs opensim; use the project .venv -- see docs/SETUP.md)")
    ap.add_argument("trc", help="TRC marker file")
    ap.add_argument("--model", default=None,
                    help=f"OSIM to scale (default: {M.DEFAULT_MODEL.name})")
    ap.add_argument("--out", default=None,
                    help="scaled .osim (default: <trc>.scaled.osim beside "
                         "the model)")
    ap.add_argument("--start", type=float, default=None,
                    help="averaging start time (s; default: clip start)")
    ap.add_argument("--end", type=float, default=None,
                    help="averaging end time (s; default: clip end)")
    ap.add_argument("--include-humerus", action="store_true",
                    help="also scale the humerus (uses the depth-corrupted "
                         "shoulder->elbow span; not recommended)")
    ap.add_argument("--mass", type=float, default=-1.0,
                    help="final total mass kg (default: -1 = keep model's)")
    a = ap.parse_args(argv)
    run(a.trc, model=a.model, out=a.out, start=a.start, end=a.end,
        include_humerus=a.include_humerus, mass=a.mass)


if __name__ == "__main__":
    main()
