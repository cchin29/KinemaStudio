#!/usr/bin/env python3
"""Estimate ground reaction forces from kinematics alone (no force plates).

    .venv/bin/python src/run_grf.py path/to/<name>.ik.mot

Markerless capture has no measured external loads, so Inverse Dynamics
is "top-down" and dumps the whole unbalanced wrench into the floating
base (the ground_pelvis residual), and Static Optimization then needs
huge reserves. This stage closes that gap WITHOUT force plates, the
Route-A workaround: the whole-body Newton-Euler net external wrench is
*fully determined* by the motion + inertia, so we compute it from the
posed IK frames and attribute it to the feet.

    F_ext      = sum_i m_i (a_i - g)                       (exact)
    M_ext(O)   = sum_i [ r_i x m_i(a_i - g) + I_i a_i^rot          ]
                 + sum_i [ w_i x (I_i w_i) ]               (exact)

Only the *distribution* across simultaneous contacts is under-
determined. Per frame we detect which feet are down (contact points
near an estimated floor and slow), then:
  * single contact -> the entire wrench goes to that foot, at its true
    centre of pressure on the floor plane, plus a free vertical torque;
  * double contact -> the vertical/shear force is split between the feet
    by where the net COP projects onto the line joining them, and the
    residual moment is applied as per-foot free torque.
Either way the *net* wrench is reproduced exactly, so ID/SO become
dynamically consistent (base residual ~ 0) for ANY motion, not just a
planted-base pose. The approximation is only in how realistic each
individual foot load is during double support -- see Ren et al. 2008,
"Whole-body inverse dynamics over a complete gait cycle based only on
measured kinematics".

Writes, next to the .mot:
    <root>.grf.sto             per-foot force / point (COP) / torque
    <root>.externalloads.xml   OpenSim ExternalLoads -> feed run_id/run_so
                               with --external-loads (the wrapper does
                               this automatically with --grf).

Needs `opensim` + `scipy` (project .venv; see docs/SETUP.md). Use the
SAME --lowpass here and in run_id/run_so so their internal coordinate
differentiation matches this stage's (otherwise small residuals leak
back in).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mpipe_pipeline as M
import pipeline_config as PC

# Default contact feet: (label, body the force is applied to, contact
# marker points used for detection + centre-of-pressure). MediaPipe
# gives heel + foot-index (toe) on each foot -- ideal for heel-strike /
# toe-off. Falls back to the calcn/toes body origins if absent.
DEFAULT_FEET = [
    ("R", "calcn_r", ["RIGHT_HEEL", "RIGHT_FOOT_INDEX"], ["calcn_r", "toes_r"]),
    ("L", "calcn_l", ["LEFT_HEEL", "LEFT_FOOT_INDEX"], ["calcn_l", "toes_l"]),
]


def grf_out_path(mot):
    """The force .sto next to the coordinates file (`.ik.mot` ->
    `.grf.sto`)."""
    name = Path(mot).name
    for suf in (".ik.mot", ".mot", ".sto"):
        if name.endswith(suf):
            return Path(mot).with_name(name[: -len(suf)] + ".grf.sto")
    return Path(mot).with_name(name + ".grf.sto")


def xml_out_path(grf_sto):
    """The ExternalLoads .xml beside a `.grf.sto`."""
    p = Path(grf_sto)
    if p.name.endswith(".grf.sto"):
        return p.with_name(p.name[: -len(".grf.sto")] + ".externalloads.xml")
    return p.with_name(p.stem + ".externalloads.xml")


def _butter(arr, cutoff, fs, order=4):
    """Zero-lag low-pass each column of arr (T, ...) at `cutoff` Hz; a
    cutoff <= 0 or a too-short signal returns arr unchanged."""
    import numpy as np
    from scipy.signal import butter, filtfilt
    if cutoff is None or cutoff <= 0:
        return arr
    nyq = 0.5 * fs
    wn = min(0.99, cutoff / nyq)
    if wn <= 0:
        return arr
    b, a = butter(order, wn)
    pad = 3 * max(len(a), len(b))
    if arr.shape[0] <= pad:
        return arr
    flat = arr.reshape(arr.shape[0], -1)
    out = filtfilt(b, a, flat, axis=0)
    return out.reshape(arr.shape)


def run(mot, model=None, out=None, xml_out=None, lowpass=6.0,
        height_thresh=0.05, vel_thresh=0.8, floor_pct=5.0, feet=None):
    """Estimate GRF from the IK trajectory; returns (grf_sto, xml) paths."""
    import numpy as np
    import opensim as osim
    from scipy.spatial.transform import Rotation

    mot = Path(mot).resolve()
    if not mot.exists():
        raise FileNotFoundError(mot)
    model_path = Path(model).resolve() if model else M.DEFAULT_MODEL
    out = Path(out).resolve() if out else grf_out_path(mot).resolve()
    xml = Path(xml_out).resolve() if xml_out else xml_out_path(out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    feet = feet or DEFAULT_FEET

    osim_model = osim.Model(str(model_path))
    state = osim_model.initSystem()
    g = np.array([osim_model.getGravity().get(i) for i in range(3)])

    # ---- read + filter the coordinate trajectory -------------------
    sto = osim.Storage(str(mot))
    in_deg = sto.isInDegrees()
    labels = sto.getColumnLabels()
    col = {labels.get(i): i - 1 for i in range(1, labels.getSize())}
    coords = osim_model.getCoordinateSet()
    cmap = []  # (Coordinate, data_index, is_rotational)
    for i in range(coords.getSize()):
        c = coords.get(i)
        if c.getName() in col:
            cmap.append((c, col[c.getName()],
                         c.getMotionType() == osim.Coordinate.Rotational))
    if not cmap:
        raise RuntimeError(
            f"no .mot column matches a model coordinate -- is {mot.name} "
            f"an IK output for {model_path.name}?")

    times, data = [], []
    for k in range(sto.getSize()):
        sv = sto.getStateVector(k)
        times.append(sv.getTime())
        d = sv.getData()
        data.append([d.get(j) for j in range(d.getSize())])
    times = np.asarray(times)
    n = len(times)
    if n < 5:
        raise RuntimeError(f"need >=5 frames, got {n}")
    dt = float(np.mean(np.diff(times)))
    fs = 1.0 / dt

    # Filter each used coordinate (rotational -> radians) before posing,
    # so the body kinematics we differentiate are smooth.
    q = np.zeros((n, len(cmap)))
    for j, (c, di, is_rot) in enumerate(cmap):
        series = np.array([row[di] for row in data])
        if is_rot and in_deg:
            series = np.radians(series)
        q[:, j] = series
    q = _butter(q, lowpass, fs)

    # ---- per-body constants ----------------------------------------
    bodies = osim_model.getBodySet()
    nb = bodies.getSize()
    mass = np.zeros(nb)
    Ibody = np.zeros((nb, 3, 3))   # inertia about COM, body frame
    com_local = []
    for i in range(nb):
        b = bodies.get(i)
        mass[i] = b.getMass()
        I = b.getInertia()
        mo, pr = I.getMoments(), I.getProducts()
        Ibody[i] = [[mo.get(0), pr.get(0), pr.get(1)],
                    [pr.get(0), mo.get(1), pr.get(2)],
                    [pr.get(1), pr.get(2), mo.get(2)]]
        com_local.append(b.getMassCenter())
    Mtot = float(mass.sum())

    # ---- pose each frame: COM positions, body rotations, contacts --
    p = np.zeros((n, nb, 3))       # body COM in ground
    R = np.zeros((n, nb, 3, 3))    # body rotation in ground
    mk_set = osim_model.getMarkerSet()
    have_mk = {mk_set.get(i).getName() for i in range(mk_set.getSize())}
    # Resolve each foot's contact stations: markers if present, else the
    # named body origins.
    foot_pts = []
    for lab, body, mks, fallback in feet:
        pts = [m for m in mks if m in have_mk]
        use_markers = bool(pts)
        foot_pts.append((lab, body, pts, fallback, use_markers))
    cpt = {lab: np.zeros((n, max(1, len(pts or fb)), 3))
           for lab, body, pts, fb, um in foot_pts}

    def setpose(k):
        for j, (c, di, is_rot) in enumerate(cmap):
            try:
                c.setValue(state, q[k, j], False)
            except Exception:
                pass
        osim_model.realizePosition(state)

    try:
        from tqdm import tqdm
        rng = tqdm(range(n), unit="frame", desc="GRF pose")
    except ImportError:
        rng = range(n)
    for k in rng:
        setpose(k)
        for i in range(nb):
            b = bodies.get(i)
            loc = b.findStationLocationInGround(state, com_local[i])
            p[k, i] = [loc.get(0), loc.get(1), loc.get(2)]
            Rot = b.getTransformInGround(state).R()
            R[k, i] = [[Rot.get(r, c2) for c2 in range(3)] for r in range(3)]
        for lab, body, pts, fb, um in foot_pts:
            names = pts if um else fb
            for c2, nm in enumerate(names):
                if um:
                    loc = mk_set.get(nm).getLocationInGround(state)
                else:
                    bb = bodies.get(nm)
                    loc = bb.findStationLocationInGround(
                        state, osim.Vec3(0, 0, 0))
                cpt[lab][k, c2] = [loc.get(0), loc.get(1), loc.get(2)]

    # ---- velocities / accelerations (finite difference) ------------
    a = np.gradient(np.gradient(p, dt, axis=0), dt, axis=0)   # COM accel
    # Angular velocity from relative rotation (central diff), then accel.
    omega = np.zeros((n, nb, 3))
    for k in range(n):
        km, kp = max(0, k - 1), min(n - 1, k + 1)
        span = (kp - km) * dt
        if span <= 0:
            continue
        Rrel = np.einsum("bij,bkj->bik", R[kp], R[km])   # R[kp] @ R[km]^T
        omega[k] = Rotation.from_matrix(Rrel).as_rotvec() / span
    alpha = np.gradient(omega, dt, axis=0)

    # ---- net external wrench about ground origin O -----------------
    F = np.zeros((n, 3))
    Mo = np.zeros((n, 3))
    for k in range(n):
        fk = mass[:, None] * (a[k] - g[None, :])             # (nb,3)
        F[k] = fk.sum(axis=0)
        # translational moment r x f
        m_t = np.cross(p[k], fk).sum(axis=0)
        # rotational: I_g alpha + omega x (I_g omega), I_g = R I R^T
        Ig = np.einsum("bij,bjk,blk->bil", R[k], Ibody, R[k])
        Ia = np.einsum("bij,bj->bi", Ig, alpha[k])
        Iw = np.einsum("bij,bj->bi", Ig, omega[k])
        m_r = (Ia + np.cross(omega[k], Iw)).sum(axis=0)
        Mo[k] = m_t + m_r

    # ---- floor + per-foot contact detection ------------------------
    all_y = np.concatenate([cpt[lab][:, :, 1].ravel() for lab, *_ in foot_pts])
    y_floor = float(np.percentile(all_y, floor_pct))
    contact = {}        # lab -> (n,) bool
    centroid = {}       # lab -> (n,3) on floor plane (NaN if no contact)
    for lab, body, pts, fb, um in foot_pts:
        P = cpt[lab]                                          # (n, npts, 3)
        vel = np.gradient(P, dt, axis=0)
        hsp = np.linalg.norm(vel[:, :, [0, 2]], axis=2)      # horiz speed
        down = ((P[:, :, 1] - y_floor) < height_thresh) & (hsp < vel_thresh)
        contact[lab] = down.any(axis=1)
        cen = np.full((n, 3), np.nan)
        for k in range(n):
            if down[k].any():
                xz = P[k, down[k]][:, [0, 2]].mean(axis=0)
                cen[k] = [xz[0], y_floor, xz[1]]
        centroid[lab] = cen

    # ---- decompose net wrench -> per-foot force/point/torque -------
    labs = [lab for lab, *_ in foot_pts]
    grf = {lab: np.zeros((n, 9)) for lab in labs}   # vx vy vz px py pz tx ty tz

    def cop_on_floor(Fk, Mk):
        """Centre of pressure on the plane y=y_floor for wrench (Fk,Mk)
        about O, and the free vertical torque there. Returns (cop, ty)
        or None if the vertical force is ~0."""
        Fy = Fk[1]
        if abs(Fy) < 1e-6:
            return None
        cx = (Mk[2] + y_floor * Fk[0]) / Fy
        cz = (y_floor * Fk[2] - Mk[0]) / Fy
        ty = Mk[1] - (cz * Fk[0] - cx * Fk[2])
        return np.array([cx, y_floor, cz]), ty

    n_single = n_double = n_flight = 0
    for k in range(n):
        active = [lab for lab in labs if contact[lab][k]]
        if not active:
            n_flight += 1
            continue
        if len(active) == 1:
            n_single += 1
            lab = active[0]
            cop = cop_on_floor(F[k], Mo[k])
            pt = cop[0] if cop else centroid[lab][k]
            ty = cop[1] if cop else 0.0
            grf[lab][k, 0:3] = F[k]
            grf[lab][k, 3:6] = pt
            grf[lab][k, 6:9] = [0.0, ty, 0.0]
            continue
        # double (or more) support: split by COP projection onto the
        # line between the two foot centroids; residual moment -> free
        # per-foot torque so the NET wrench is still reproduced exactly.
        n_double += 1
        a_lab, b_lab = active[0], active[1]
        ca, cb = centroid[a_lab][k], centroid[b_lab][k]
        cop = cop_on_floor(F[k], Mo[k])
        copxz = (cop[0] if cop else 0.5 * (ca + cb))
        d = (cb - ca)[[0, 2]]
        denom = float(d @ d)
        wb = 0.5 if denom < 1e-9 else float(
            ((copxz - ca)[[0, 2]] @ d) / denom)
        wb = min(1.0, max(0.0, wb))
        wa = 1.0 - wb
        Fa, Fb = wa * F[k], wb * F[k]
        dM = Mo[k] - (np.cross(ca, Fa) + np.cross(cb, Fb))
        for lab, Ff, cc, ww in ((a_lab, Fa, ca, wa), (b_lab, Fb, cb, wb)):
            grf[lab][k, 0:3] = Ff
            grf[lab][k, 3:6] = cc
            grf[lab][k, 6:9] = ww * dM

    # Any foot with no contact this frame keeps a valid (last) point so
    # the COP column never holds NaN/0 jumps the tool dislikes; use the
    # foot centroid when available, else the floor under the body.
    for lab in labs:
        last = np.array([0.0, y_floor, 0.0])
        for k in range(n):
            if np.allclose(grf[lab][k, 0:3], 0.0):
                cen = centroid[lab][k]
                grf[lab][k, 3:6] = last if np.isnan(cen).any() else cen
            if not np.isnan(centroid[lab][k]).any():
                last = centroid[lab][k]

    # ---- write the GRF .sto ----------------------------------------
    comps = ["force_vx", "force_vy", "force_vz",
             "force_px", "force_py", "force_pz",
             "torque_x", "torque_y", "torque_z"]
    col_labels, ident = [], {}
    for lab in labs:
        ident[lab] = (f"{lab}_ground_force_v", f"{lab}_ground_force_p",
                      f"{lab}_ground_torque_")
        for c2 in comps:
            col_labels.append(f"{lab}_ground_{c2}")
    tbl = osim.TimeSeriesTable()
    sv = osim.StdVectorString()
    for cl in col_labels:
        sv.append(cl)
    tbl.setColumnLabels(sv)
    ncol = len(col_labels)
    for k in range(n):
        row = osim.RowVector(ncol, 0.0)
        for fi, lab in enumerate(labs):
            for c2 in range(9):
                row[fi * 9 + c2] = float(grf[lab][k, c2])
        tbl.appendRow(float(times[k]), row)
    tbl.addTableMetaDataString("inDegrees", "no")
    osim.STOFileAdapter().write(tbl, str(out))

    # ---- build the ExternalLoads .xml ------------------------------
    el = osim.ExternalLoads()
    el.setDataFileName(str(out))
    for lab, body, pts, fb, um in foot_pts:
        fid, pid, tid = ident[lab]
        ef = osim.ExternalForce()
        ef.setName(f"{lab}_grf")
        ef.setAppliedToBodyName(body)
        ef.setForceExpressedInBodyName("ground")
        ef.setPointExpressedInBodyName("ground")
        ef.setForceIdentifier(fid)
        ef.setPointIdentifier(pid)
        ef.setTorqueIdentifier(tid)
        el.cloneAndAppend(ef)
    el.printToXML(str(xml))

    # ---- summary ---------------------------------------------------
    fy_tot = np.sum([grf[lab][:, 1] for lab in labs], axis=0)
    print(f"model : {model_path}")
    print(f"mot   : {mot}")
    print(f"grf   : {out}")
    print(f"xml   : {xml}")
    print(f"frames {n} @ {fs:.1f} Hz  |  lowpass "
          f"{'off' if lowpass <= 0 else f'{lowpass:g} Hz'}  |  floor y="
          f"{y_floor:.3f} m (p{floor_pct:g})")
    print(f"contact phases: single {n_single}  double {n_double}  "
          f"flight {n_flight}  (of {n})")
    print(f"bodyweight {Mtot * abs(g[1]):.1f} N  |  mean total vertical "
          f"GRF {float(fy_tot.mean()):.1f} N  (quasi-static check)")
    for lab in labs:
        print(f"  foot {lab}: peak |Fy| {np.abs(grf[lab][:, 1]).max():.1f} "
              f"N  |  contact {int(contact[lab].sum())}/{n} frames")
    return str(out), str(xml)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Estimate ground reaction forces from an IK .mot "
                    "(kinematics-only, no force plates) and write an "
                    "OpenSim ExternalLoads for run_id/run_so. Needs "
                    "opensim + scipy (project .venv -- see docs/SETUP.md).")
    ap.add_argument("mot", help="IK coordinates file (.mot/.sto from "
                                "run_ik.py)")
    ap.add_argument("--model", default=None,
                    help=f"OSIM model (default: {M.DEFAULT_MODEL.name}; "
                         f"must match the one IK was solved against)")
    ap.add_argument("--out", default=None,
                    help="output force .sto (default: <root>.grf.sto next "
                         "to the .mot; the .xml goes beside it)")
    ap.add_argument("--xml-out", default=None,
                    help="output ExternalLoads .xml (default: "
                         "<root>.externalloads.xml)")
    ap.add_argument("--lowpass", type=float, default=6.0,
                    help="low-pass cutoff (Hz) for the coordinates before "
                         "differentiation (default: 6; <=0 disables). Use "
                         "the SAME value in run_id/run_so.")
    ap.add_argument("--height-thresh", type=float, default=0.05,
                    help="contact-point height above the floor (m) below "
                         "which it counts as in contact (default: 0.05)")
    ap.add_argument("--vel-thresh", type=float, default=0.8,
                    help="contact-point horizontal speed (m/s) below which "
                         "it counts as planted (default: 0.8)")
    ap.add_argument("--floor-pct", type=float, default=5.0,
                    help="percentile of contact-point heights used as the "
                         "floor level (default: 5)")
    PC.add_args(ap)
    argv = list(sys.argv[1:] if argv is None else argv)
    _cfg = PC.apply(ap, "run_grf", argv)
    a = ap.parse_args(argv)
    if _cfg:
        print(f"run_grf: config {_cfg}", flush=True)
    run(a.mot, model=a.model, out=a.out, xml_out=a.xml_out,
        lowpass=a.lowpass, height_thresh=a.height_thresh,
        vel_thresh=a.vel_thresh, floor_pct=a.floor_pct)


if __name__ == "__main__":
    main()
