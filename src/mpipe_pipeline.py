#!/usr/bin/env python3
"""MediaPipe -> TRC markerless pipeline for the combined_body_model OpenSim model.

Two stages, both reusable from thin notebooks or the CLI:

  Stage 1  video_to_csv(video)   ->  <root>_lh.csv / _rh.csv / _pose.csv
  Stage 2  csv_to_trc(root,mode) ->  <root>.trc (+ optional <root>.IK.xml)

Stage 2 fuses the three CSVs into one OpenSim-ready TRC. The detailed
21-landmark MediaPipe-Hand streams live in their own normalized image
space (near-zero depth, zero confidence) detached from the pose world
frame, so each hand is placed into the body by a per-frame *similarity*
registration onto that side's coarse pose hand points. Body landmarks get
one documented MediaPipe->OpenSim rotation + one global metric scale.

cv2/mediapipe are imported lazily inside video_to_csv so Stage 2 (which
needs only numpy/pandas/scipy/sklearn) runs in environments where the
OpenCV/mediapipe import is unavailable or broken.
"""
import json
import re
from pathlib import Path
from string import Template

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.signal import butter, filtfilt
from sklearn.linear_model import LinearRegression

# Default zero-phase low-pass cutoff (Hz) for landmark trajectories.
# Metric world landmarks are accurate in scale but noisier per-frame
# than the old (implicitly z-smoothed) normalized path; 6 Hz is the
# standard biomechanics cutoff and recovers full joint-angle smoothness.
DEFAULT_LOWPASS_HZ = 6.0

# This module lives in <repo>/src; the model lives under <repo>/models/
# and the IK template under <repo>/templates/.
SRC = Path(__file__).resolve().parent
REPO = SRC.parent
DEFAULT_MODEL = REPO / "models" / "combined_body_model" / "combined_body_model.osim"
DEFAULT_IK_TEMPLATE = REPO / "templates" / "TEMPLATE_IK.xml"

# Default-pose mass-center of combined_body_model.osim in the ground
# frame (m), from opensim Model.calcMassCenterPosition() at the default
# state (total mass 75.59 kg). Cached so the opensim-free CLI can use
# it; model_com() recomputes live whenever opensim is importable.
MODEL_COM_DEFAULT = np.array([-0.082546, -0.029509, 0.000029])

# Stage 1 uses the modern MediaPipe Tasks HolisticLandmarker (the legacy
# mp.solutions.holistic was removed in mediapipe >=0.10.15). The .task
# bundle is auto-downloaded to models/mediapipe/ on first use.
MODELS_DIR = REPO / "models" / "mediapipe"
HOLISTIC_TASK = MODELS_DIR / "holistic_landmarker.task"
HOLISTIC_TASK_URL = (
    "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/"
    "holistic_landmarker/float16/latest/holistic_landmarker.task")

# HolisticLandmarker ships as ONE fixed bundle (no lite/full/heavy) and
# has no `model_complexity`. The lite/full/heavy variants exist only for
# the standalone PoseLandmarker task (pose only, no hands). pose_model=
# {lite,full,heavy} runs a separate PoseLandmarker for the _pose.csv
# (more accurate 3D body) while HolisticLandmarker still provides the
# detailed 21-pt hands -- so the pose-model variable is isolated and the
# validated hand-registration path is unchanged. heavy ~31 MB.
POSE_MODELS = ("lite", "full", "heavy")
POSE_TASK_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_{v}/float16/latest/pose_landmarker_{v}.task")

# Standalone HandLandmarker .task (the "alternate hands" path). Run on a
# small square crop taken around each wrist from the heavy pose, NOT on
# the full frame, so it sees the hand much larger -> finer landmarks
# than HolisticLandmarker's full-frame hands. Auto-downloaded on use.
HAND_TASK = MODELS_DIR / "hand_landmarker.task"
HAND_TASK_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task")

# The two CLI shortcut knobs. Each broadcasts to two genuine
# HolisticLandmarkerOptions fields (see MP_SHORTCUTS); a --mp-config
# YAML can override any individual field at higher precedence. Defaults
# preserve the historical behaviour (tracking 0.90, detection 0.50).
DEFAULT_MIN_DET_CONF = 0.50
DEFAULT_MIN_TRK_CONF = 0.90

# Genuine tunable HolisticLandmarkerOptions fields and their library
# defaults (mediapipe Tasks API; mirrors the dataclass so the opensim-
# and mediapipe-free import path stays clean -- cross-checked against
# the real dataclass at call time). base_options/running_mode/
# result_callback are pipeline-controlled and intentionally excluded.
# NOTE: the legacy mp.solutions.holistic `model_complexity` knob does
# NOT exist in the Tasks API and is intentionally absent here.
MP_OPTION_DEFAULTS = {
    "min_face_detection_confidence": 0.5,
    "min_face_suppression_threshold": 0.5,
    "min_face_landmarks_confidence": 0.5,
    "min_pose_detection_confidence": 0.5,
    "min_pose_suppression_threshold": 0.5,
    "min_pose_landmarks_confidence": 0.5,
    "min_hand_landmarks_confidence": 0.5,
    "output_face_blendshapes": False,
    "output_segmentation_mask": False,
}
_MP_BOOL_OPTS = {"output_face_blendshapes", "output_segmentation_mask"}

# How each shortcut broadcasts onto genuine fields.
MP_SHORTCUTS = {
    "min_detection_confidence": ("min_pose_detection_confidence",
                                 "min_face_detection_confidence"),
    "min_tracking_confidence": ("min_pose_landmarks_confidence",
                                "min_hand_landmarks_confidence"),
}


def _coerce_mp_value(key, raw):
    """Type/range-check one config value against MP_OPTION_DEFAULTS."""
    if key in _MP_BOOL_OPTS:
        if isinstance(raw, bool):
            return raw
        s = str(raw).strip().lower()
        if s in ("true", "1", "yes"):
            return True
        if s in ("false", "0", "no"):
            return False
        raise ValueError(f"{key}: expected a boolean, got {raw!r}")
    try:
        v = float(raw)
    except (TypeError, ValueError):
        raise ValueError(f"{key}: expected a number in [0,1], got {raw!r}")
    if not 0.0 <= v <= 1.0:
        raise ValueError(f"{key}: {v} out of range [0,1]")
    return v


def parse_mp_config(path):
    """Load an mp-config YAML -> validated {field: value} dict for the
    genuine HolisticLandmarkerOptions knobs. Unknown keys are rejected
    (with the valid set listed) so a typo or a legacy `model_complexity`
    fails loudly instead of being silently ignored."""
    import yaml
    raw = yaml.safe_load(Path(path).read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: top level must be a mapping")
    bad = set(raw) - set(MP_OPTION_DEFAULTS)
    if bad:
        raise ValueError(
            f"{path}: unknown option(s) {sorted(bad)}. Valid keys: "
            f"{sorted(MP_OPTION_DEFAULTS)} (note: the legacy "
            f"'model_complexity' is not a Tasks-API option)")
    return {k: _coerce_mp_value(k, raw[k]) for k in raw}


def mp_config_template():
    """A ready-to-edit mp-config YAML (all genuine knobs at their
    library defaults) with a header documenting the shortcuts +
    precedence."""
    lines = [
        "# MediaPipe Tasks HolisticLandmarker tuning for mp2trc.py",
        "#   mp2trc.py video.mp4 --mp-config this_file.yaml",
        "#",
        "# Precedence (low -> high):",
        "#   library defaults  <  --min-detection-confidence /",
        "#   --min-tracking-confidence (each broadcasts to two fields",
        "#   below)  <  the explicit per-field values in THIS file.",
        "#",
        "#   --min-detection-confidence -> min_pose_detection_confidence,",
        "#                                 min_face_detection_confidence",
        "#   --min-tracking-confidence  -> min_pose_landmarks_confidence,",
        "#                                 min_hand_landmarks_confidence",
        "#",
        "# Confidences/thresholds are floats in [0,1]; output_* are bools.",
        "# Delete any line to leave that knob at the broadcast/default.",
        "# (The legacy mp.solutions.holistic `model_complexity` is NOT a",
        "#  Tasks-API option and is intentionally unavailable.)",
        "",
    ]
    for k, v in MP_OPTION_DEFAULTS.items():
        lines.append(f"{k}: {str(v).lower() if isinstance(v, bool) else v}")
    return "\n".join(lines) + "\n"

# Canonical MediaPipe Pose 33-landmark order (matches the existing
# *_pose.csv headers; kept explicit so the CSV schema is stable and
# independent of mediapipe internals).
POSE_LANDMARKS = [
    "NOSE", "LEFT_EYE_INNER", "LEFT_EYE", "LEFT_EYE_OUTER",
    "RIGHT_EYE_INNER", "RIGHT_EYE", "RIGHT_EYE_OUTER",
    "LEFT_EAR", "RIGHT_EAR", "MOUTH_LEFT", "MOUTH_RIGHT",
    "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW",
    "LEFT_WRIST", "RIGHT_WRIST", "LEFT_PINKY", "RIGHT_PINKY",
    "LEFT_INDEX", "RIGHT_INDEX", "LEFT_THUMB", "RIGHT_THUMB",
    "LEFT_HIP", "RIGHT_HIP", "LEFT_KNEE", "RIGHT_KNEE",
    "LEFT_ANKLE", "RIGHT_ANKLE", "LEFT_HEEL", "RIGHT_HEEL",
    "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX",
]

# 21 MediaPipe-Hand landmark leaf names, in HandLandmark enum order. The
# OpenSim model carries these verbatim under LH_/RH_ prefixes.
HAND_LEAF = [
    "WRIST",
    "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP",
    "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP",
    "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP",
]

# Pose landmarks the model actually has markers for (identity-named).
# The model intentionally omits the 6 coarse pose hand points
# (LEFT/RIGHT_INDEX/PINKY/THUMB) -- superseded by the detailed hands.
BODY_MARKERS = [
    "NOSE",
    "LEFT_EYE_INNER", "LEFT_EYE", "LEFT_EYE_OUTER",
    "RIGHT_EYE_INNER", "RIGHT_EYE", "RIGHT_EYE_OUTER",
    "LEFT_EAR", "RIGHT_EAR", "MOUTH_LEFT", "MOUTH_RIGHT",
    "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW",
    "LEFT_WRIST", "RIGHT_WRIST", "LEFT_HIP", "RIGHT_HIP",
    "LEFT_KNEE", "RIGHT_KNEE", "LEFT_ANKLE", "RIGHT_ANKLE",
    "LEFT_HEEL", "RIGHT_HEEL", "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX",
]

# When a detailed hand is ABSENT, the coarse pose hand points stand in for
# the corresponding fingertip markers (user rule).
COARSE_HAND_FALLBACK = {
    "LEFT_INDEX": "LH_INDEX_FINGER_TIP",
    "LEFT_PINKY": "LH_PINKY_TIP",
    "LEFT_THUMB": "LH_THUMB_TIP",
    "RIGHT_INDEX": "RH_INDEX_FINGER_TIP",
    "RIGHT_PINKY": "RH_PINKY_TIP",
    "RIGHT_THUMB": "RH_THUMB_TIP",
}

# Hand-landmark -> pose-landmark correspondences used to register a hand
# into the body frame, per side. The MediaPipe-Pose LEFT/RIGHT
# INDEX/PINKY/THUMB points sit at the knuckle (MCP) level, ~0.04-0.07 m
# from the wrist -- NOT the fingertip -- so they pair with the hand's MCP
# landmarks (pairing them to TIPs shrinks the hand ~30%). Wrist is
# weighted highest: most stable shared point and the hand's kinematic
# anchor.
_REG_HAND_PTS = ["WRIST", "INDEX_FINGER_MCP", "PINKY_MCP", "THUMB_MCP"]
_REG_POSE_SFX = ["WRIST", "INDEX", "PINKY", "THUMB"]
_REG_WEIGHTS = np.array([3.0, 1.0, 1.0, 1.0])

# The coarse pose hand-points are noisy/foreshortened, so registration
# supplies the hand's position + orientation only; its absolute size is
# fixed from a flexion-invariant intrinsic bony span (wrist -> index MCP;
# both are palm landmarks that don't move when the fingers curl). Same
# 0.090 m reference the original pipeline used.
_HAND_SCALE_REF = ("WRIST", "INDEX_FINGER_MCP")
HAND_REF_M = 0.090

# Single validated MediaPipe -> OpenSim rotation.
#   MediaPipe: X right, Y down, Z toward camera-back
#   OpenSim:   X forward, Y up,  Z left
# = rotate 90 deg about Y then 180 about X (orthogonal, det +1).
MP_TO_OS = np.array([
    [0,  0, -1],
    [0, -1,  0],
    [-1, 0,  0],
], dtype=float)


# --------------------------------------------------------------------------
# Stage 2 helpers reused verbatim from mpipe2trc4hand_osv.ipynb
# --------------------------------------------------------------------------
def read_csv_prefix(csv_file, prefix):
    """Read a MediaPipe landmark CSV, prefixing the :X/:Y/:Z/:C columns."""
    df = pd.read_csv(csv_file)
    pat = r":[CXYZ]$"
    df.rename(columns={c: prefix + c for c in df.columns if re.search(pat, c)},
              inplace=True)
    return df


def interp_na(df):
    """Forward/back-fill the first/last rows, then linearly interpolate."""
    if df.iloc[0].isna().any():
        r0 = df.dropna(how="any").iloc[0]
        r0["Frame Number"] = 1
        df.iloc[0] = r0
    if df.iloc[-1].isna().any():
        rz = df.dropna(how="any").iloc[-1]
        rz["Frame Number"] = df.shape[0]
        df.iloc[-1] = rz
    df.interpolate(inplace=True)
    return df


def calc_z_scale(node1_coords, node2_coords):
    """Z-axis scale that makes the node1<->node2 distance time-invariant."""
    def dist(zs):
        a = node1_coords.copy()
        b = node2_coords.copy()
        a[:, 2] *= zs
        b[:, 2] *= zs
        return np.sqrt(np.sum((a - b) ** 2, axis=1))

    return minimize(lambda zs: np.var(dist(zs)), [1.0],
                    bounds=[(0, None)]).x[0]


# --------------------------------------------------------------------------
# Stage 2: load + condition + register + assemble
# --------------------------------------------------------------------------
def _lowpass(xyz, fps, hz):
    """Zero-phase 4th-order Butterworth low-pass along the frame axis.
    No-op if disabled, sub-Nyquist-impossible, or the clip is too short
    for filtfilt's edge padding."""
    if hz is None or fps <= 0:
        return xyz
    nyq = fps / 2.0
    if hz >= nyq or xyz.shape[0] < 20:
        return xyz
    b, a = butter(4, hz / nyq)
    return filtfilt(b, a, xyz, axis=0)


def _df_to_xyz(df, prefix):
    """(F,J,3) array + leaf-name list from an interpolated landmark df."""
    cols = [c for c in df.columns if re.search(r":[XYZ]$", c)]
    leaves, seen = [], set()
    for c in cols:
        leaf = c[:-2]
        if prefix and leaf.startswith(prefix):
            leaf = leaf[len(prefix):]
        if leaf not in seen:
            seen.add(leaf)
            leaves.append(leaf)
    xyz = df[cols].to_numpy().reshape(df.shape[0], len(leaves), 3)
    return xyz, leaves


def load_landmark_frames(root, lowpass_hz=DEFAULT_LOWPASS_HZ):
    """Read <root>_{lh,rh,pose}.csv (+ optional _{lh,rh}_roi.csv and
    _{lh,rh}_roi.combined.csv), build a common frame superset, gap-fill,
    low-pass each landmark trajectory, and return (times, sources) where
    sources maps name -> (xyz, leaves). Hands absent on disk are
    returned as None. lowpass_hz=None disables the filter. The _roi
    hands are the alternate ROI-cropped HandLandmarker stream, already
    in the pose world frame; the _comb (combined) hands are _roi plus
    Holistic-recovered rows for ROI-missing frames (see combine_hands)."""
    root = Path(root)
    pose_p = root.with_name(root.name + "_pose.csv")
    lh_p = root.with_name(root.name + "_lh.csv")
    rh_p = root.with_name(root.name + "_rh.csv")
    lh_roi_p = root.with_name(root.name + "_lh_roi.csv")
    rh_roi_p = root.with_name(root.name + "_rh_roi.csv")
    lh_comb_p = root.with_name(root.name + "_lh_roi.combined.csv")
    rh_comb_p = root.with_name(root.name + "_rh_roi.combined.csv")
    if not pose_p.exists():
        raise FileNotFoundError(pose_p)

    bddf = read_csv_prefix(pose_p, "")
    lhdf = read_csv_prefix(lh_p, "LEFTHAND_") if lh_p.exists() else None
    rhdf = read_csv_prefix(rh_p, "RIGHTHAND_") if rh_p.exists() else None
    lh_roi_df = (read_csv_prefix(lh_roi_p, "LEFTHAND_")
                 if lh_roi_p.exists() else None)
    rh_roi_df = (read_csv_prefix(rh_roi_p, "RIGHTHAND_")
                 if rh_roi_p.exists() else None)
    lh_comb_df = (read_csv_prefix(lh_comb_p, "LEFTHAND_")
                  if lh_comb_p.exists() else None)
    rh_comb_df = (read_csv_prefix(rh_comb_p, "RIGHTHAND_")
                  if rh_comb_p.exists() else None)

    maxfn = bddf["Frame Number"].max()
    for d in (lhdf, rhdf, lh_roi_df, rh_roi_df, lh_comb_df, rh_comb_df):
        if d is not None:
            maxfn = max(maxfn, d["Frame Number"].max())
    alldf = pd.DataFrame({"Frame Number": range(1, int(maxfn) + 1)})

    tmodel = LinearRegression().fit(bddf[["Frame Number"]], bddf["Time (s)"])
    times = tmodel.predict(alldf[["Frame Number"]]).astype(float)

    def cond(df):
        return interp_na(pd.merge(alldf, df, on="Frame Number", how="outer"))

    dt = np.diff(times)
    fps = 1.0 / np.median(dt) if len(dt) else 0.0

    def src(df, prefix):
        if df is None:
            return None
        xyz, leaves = _df_to_xyz(cond(df), prefix)
        return _lowpass(xyz, fps, lowpass_hz), leaves

    sources = {"pose": src(bddf, ""), "lh": src(lhdf, "LEFTHAND_"),
               "rh": src(rhdf, "RIGHTHAND_"),
               "lh_roi": src(lh_roi_df, "LEFTHAND_"),
               "rh_roi": src(rh_roi_df, "RIGHTHAND_"),
               "lh_comb": src(lh_comb_df, "LEFTHAND_"),
               "rh_comb": src(rh_comb_df, "RIGHTHAND_")}
    return times, sources


def condition_pose(pose_xyz, joints, ref_pair, ref_dist_m,
                   z_rescale=False):
    """Pose landmarks -> OpenSim-frame metric pose.

    Default (z_rescale=False): metric `pose_world_landmarks` input --
    just the single MP->OS rotation; depth is already physical and the
    true per-subject scale is preserved (model scaling, not a fixed
    reference, reconciles subject vs model).

    Legacy (z_rescale=True): old normalized-image-landmark input --
    (1) variance-min Z-rescale so ref_pair is time-invariant,
    (2) MP->OS rotation, (3) uniform scale so ref_pair == ref_dist_m.
    Both `ref_*` args are used only on this legacy path."""
    if not z_rescale:
        return np.tensordot(pose_xyz, MP_TO_OS, axes=(2, 0))

    i1, i2 = joints.index(ref_pair[0]), joints.index(ref_pair[1])
    zs = calc_z_scale(pose_xyz[:, i1, :], pose_xyz[:, i2, :])
    xyz = np.tensordot(pose_xyz * np.array([1.0, 1.0, zs]),
                       MP_TO_OS, axes=(2, 0))
    seg = np.linalg.norm(xyz[:, i1, :] - xyz[:, i2, :], axis=1).mean()
    return xyz * (ref_dist_m / seg)


def _umeyama(src, dst, w):
    """Weighted similarity transform (scale*R@src + t ~= dst) for one
    frame. src,dst: (N,3); w: (N,). Returns transformed src."""
    w = w / w.sum()
    mu_s = (w[:, None] * src).sum(0)
    mu_d = (w[:, None] * dst).sum(0)
    s0, d0 = src - mu_s, dst - mu_d
    cov = (w[:, None] * d0).T @ s0
    U, S, Vt = np.linalg.svd(cov)
    D = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    var = (w * (s0 ** 2).sum(1)).sum()
    scale = (S * np.diag(D)).sum() / var
    return mu_d, R, scale, mu_s


def register_hand_to_pose(hand_xyz, hand_leaves, pose_xyz, pose_joints, side,
                           hand_ref_m=HAND_REF_M):
    """Place a raw 21-pt hand into the conditioned body frame: position
    and orientation from a per-frame weighted fit onto that side's pose
    hand points, absolute scale from the hand's own intrinsic bony span
    (pose-derived scale is discarded as unreliable). Returns (F,21,3) in
    the body/OpenSim metric frame."""
    pre = "LEFT_" if side == "left" else "RIGHT_"
    hi = [hand_leaves.index(p) for p in _REG_HAND_PTS]
    pi = [pose_joints.index(pre + s) for s in _REG_POSE_SFX]
    ra, rb = (hand_leaves.index(j) for j in _HAND_SCALE_REF)
    raw_ref = np.linalg.norm(
        hand_xyz[:, ra, :] - hand_xyz[:, rb, :], axis=1).mean()
    scale = hand_ref_m / raw_ref
    out = np.empty_like(hand_xyz)
    for f in range(hand_xyz.shape[0]):
        src = hand_xyz[f, hi, :]
        dst = pose_xyz[f, pi, :]
        mu_d, R, _, mu_s = _umeyama(src, dst, _REG_WEIGHTS)
        out[f] = (scale * (hand_xyz[f] - mu_s) @ R.T) + mu_d
    return out


def build_marker_table(times, sources, mode, ref_pair, ref_dist_m,
                       z_rescale=False, hand_source="roi"):
    """Return ordered dict {marker_name: (F,3)} per mode + the locked
    body-vs-coarse-hand rule. mode in
    {combined,left_hand,right_hand,body}.

    hand_source: "registered" places the raw Holistic hand_world
    landmarks (_lh/_rh.csv) by per-frame similarity registration onto
    the coarse pose hand points. "roi" (the default) instead uses
    the alternate ROI-cropped HandLandmarker stream (_lh/_rh_roi.csv),
    which is already unified into the pose world frame, so it needs only
    the same MediaPipe->OpenSim rotation as the pose -- no registration.
    The marker names are identical either way, so the TRC/IK marker set
    is unchanged; only the hand coordinates differ."""
    if mode not in ("combined", "left_hand", "right_hand", "body"):
        raise ValueError(f"bad mode {mode!r}")
    if hand_source not in ("registered", "roi", "combined"):
        raise ValueError(f"bad hand_source {hand_source!r}")

    pose_xyz, pose_joints = sources["pose"]
    pose_m = condition_pose(pose_xyz, pose_joints, ref_pair, ref_dist_m,
                            z_rescale=z_rescale)
    suf = {"roi": "_roi", "combined": "_comb", "registered": ""}[hand_source]

    def hand_block(base, side, prefix):
        key = base + suf
        if sources.get(key) is None:
            return None
        h_xyz, h_leaves = sources[key]
        if hand_source in ("roi", "combined"):
            # Already in the raw pose world frame (wrist anchored to the
            # pose wrist); the single MP->OS rotation lands it in the
            # same OpenSim frame as the conditioned pose. "combined" is
            # _roi plus Holistic-recovered rows in the identical frame.
            placed = np.tensordot(h_xyz, MP_TO_OS, axes=(2, 0))
        else:
            placed = register_hand_to_pose(h_xyz, h_leaves, pose_m,
                                           pose_joints, side)
        idx = [h_leaves.index(l) for l in HAND_LEAF]
        return {f"{prefix}{l}": placed[:, idx[k], :]
                for k, l in enumerate(HAND_LEAF)}

    table = {}
    want_lh = mode in ("combined", "left_hand")
    want_rh = mode in ("combined", "right_hand")
    lh = hand_block("lh", "left", "LH_") if want_lh else None
    rh = hand_block("rh", "right", "RH_") if want_rh else None

    if mode == "left_hand":
        if lh is None:
            raise ValueError(f"left_hand mode but no _lh{suf}.csv data")
        return lh
    if mode == "right_hand":
        if rh is None:
            raise ValueError(f"right_hand mode but no _rh{suf}.csv data")
        return rh

    # combined / body: start with the model's pose-named body markers
    for name in BODY_MARKERS:
        table[name] = pose_m[:, pose_joints.index(name), :]

    if mode == "combined":
        if lh is not None:
            table.update(lh)
        if rh is not None:
            table.update(rh)

    # Coarse-hand-point rule: for each side, if the detailed hand was NOT
    # emitted, fall its coarse pose points back onto the fingertip markers.
    for coarse, target in COARSE_HAND_FALLBACK.items():
        emitted = (lh if coarse.startswith("LEFT_") else rh)
        if emitted is None and coarse in pose_joints:
            table[target] = pose_m[:, pose_joints.index(coarse), :]

    return table


def write_trc(path, table, times, data_rate):
    """Write an OpenSim TRC. table: {marker: (F,3)}; data_rate Hz."""
    names = list(table)
    F = len(times)
    data = np.concatenate([table[n] for n in names], axis=1)  # (F, 3M)
    assert np.isfinite(data).all(), "non-finite marker data"
    M = len(names)
    hdr = [
        f"PathFileType\t4\t(X/Y/Z)\t{Path(path).name}",
        "DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\t"
        "OrigDataRate\tOrigDataStartFrame\tOrigNumFrames",
        f"{data_rate:.2f}\t{data_rate:.2f}\t{F}\t{M}\tm\t"
        f"{data_rate:.2f}\t1\t{F}",
        "Frame#\tTime\t" + "\t".join(f"{n}\t\t" for n in names),
        "\t\t" + "\t".join(f"X{i}\tY{i}\tZ{i}" for i in range(1, M + 1)),
    ]
    rows = []
    for i, t in enumerate(times):
        vals = "\t".join(f"{v:.6f}" for v in data[i])
        rows.append(f"{i + 1}\t{t:.5f}\t{vals}")
    Path(path).write_text("\n".join(hdr) + "\n" + "\n".join(rows) + "\n")
    return str(path)


# --------------------------------------------------------------------------
# IK XML (TEMPLATE_IK.xml skeleton + generated task set)
# --------------------------------------------------------------------------
# Uniform default marker weight, and a relaxed assembly accuracy: the
# OpenSim 1e-6 default is unreachable for cm-noisy markerless data and
# stalls the optimizer (Ipopt max-iter) even when assembly is sub-mm.
DEFAULT_IK_WEIGHT = 0.6
DEFAULT_IK_ACCURACY = 1e-4

# A body landmark whose mean MediaPipe `visibility` (the CSV :C column)
# falls below this is treated as never reliably seen -- typically a limb
# cropped out of frame. MediaPipe still emits a (hallucinated) world
# position for it, often *stably* wrong, so rigid-CV auto-weights do not
# catch it; it must be zeroed explicitly. Data-justified split: in a
# normally-framed clip every body marker is >=0.85; when the legs are
# cropped the feet/ankles sit at 0.20-0.37 while still-visible knees stay
# ~0.85, so 0.5 cleanly separates cropped vs in-frame.
DEFAULT_VIS_THRESH = 0.5

# Reference "good" segment-length CV: markers on bones noisier than this
# get progressively down-weighted by auto_marker_weights().
DEFAULT_CV0 = 0.04


def _hand_rigid_pairs(prefix):
    """Constant-length bone pairs within one detailed hand."""
    fingers = [["THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP"]]
    for f in ("INDEX_FINGER", "MIDDLE_FINGER", "RING_FINGER"):
        fingers.append([f + s for s in ("_MCP", "_PIP", "_DIP", "_TIP")])
    fingers.append(["PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP"])
    pairs = []
    for chain in fingers:
        pairs.append((prefix + "WRIST", prefix + chain[0]))
        for a, b in zip(chain, chain[1:]):
            pairs.append((prefix + a, prefix + b))
    return pairs


# Marker pairs whose length is ~constant (a rigid bone). Used purely as
# an internal-consistency reliability signal -- it does NOT name "the
# bad marker", it measures every marker's own kinematic stability, so it
# adapts to whatever is unreliable in a given clip (shoulders here,
# something else elsewhere).
RIGID_PAIRS = [
    ("LEFT_SHOULDER", "RIGHT_SHOULDER"), ("LEFT_HIP", "RIGHT_HIP"),
    ("LEFT_SHOULDER", "LEFT_ELBOW"), ("RIGHT_SHOULDER", "RIGHT_ELBOW"),
    ("LEFT_ELBOW", "LEFT_WRIST"), ("RIGHT_ELBOW", "RIGHT_WRIST"),
    ("LEFT_SHOULDER", "LEFT_HIP"), ("RIGHT_SHOULDER", "RIGHT_HIP"),
    ("LEFT_HIP", "LEFT_KNEE"), ("RIGHT_HIP", "RIGHT_KNEE"),
    ("LEFT_KNEE", "LEFT_ANKLE"), ("RIGHT_KNEE", "RIGHT_ANKLE"),
] + _hand_rigid_pairs("LH_") + _hand_rigid_pairs("RH_")


def read_trc(trc_path):
    """(marker_names, (F,M,3) array) from a TRC."""
    lines = [x for x in Path(trc_path).read_text().splitlines() if x != ""]
    names = [t for t in lines[3].split("\t")[2:] if t]
    data = np.array([[float(v) for v in r.split("\t")[2:]]
                     for r in lines[5:]]).reshape(-1, len(names), 3)
    return names, data


def auto_marker_weights(trc_path, base=DEFAULT_IK_WEIGHT, cv0=DEFAULT_CV0,
                        wmin=0.02):
    """Data-driven per-marker IK weights from rigid-segment consistency.

    For every marker, look at the rigid bones it belongs to, take their
    length CV over the clip, and down-weight markers on inconsistent
    bones: w = base / (1 + (cv/cv0)^2), clipped to [wmin, base]. No
    marker is hard-coded as bad -- a clip where the legs (not the
    shoulders) are the unreliable part gets the legs down-weighted
    instead. Returns {marker_name: weight}."""
    names, D = read_trc(trc_path)
    idx = {n: i for i, n in enumerate(names)}
    seg_cv, marker_cvs = {}, {n: [] for n in names}
    for a, b in RIGID_PAIRS:
        if a in idx and b in idx:
            d = np.linalg.norm(D[:, idx[a]] - D[:, idx[b]], axis=1)
            cv = float(d.std() / d.mean()) if d.mean() else 0.0
            seg_cv[(a, b)] = cv
            marker_cvs[a].append(cv)
            marker_cvs[b].append(cv)
    w = {}
    for n in names:
        cvs = marker_cvs[n]
        if not cvs:
            w[n] = base
        else:
            cv = float(np.mean(cvs))
            w[n] = max(wmin, base / (1.0 + (cv / cv0) ** 2))
    return w


def cropped_markers(root, markers=BODY_MARKERS,
                    vis_thresh=DEFAULT_VIS_THRESH):
    """Body markers MediaPipe never reliably saw across the clip --
    mean `visibility` (the _pose.csv :C column) < vis_thresh, i.e. a
    limb cropped out of frame whose emitted world position is a
    (often stably wrong) guess. Returns {marker: mean_visibility}
    rounded, empty if the pose CSV carries no :C columns. Restricted to
    pose body markers: detailed-hand landmarks have structurally-zero
    confidence and are placed by registration, not raw position."""
    root = Path(root)
    pose_p = root.with_name(root.name + "_pose.csv")
    df = pd.read_csv(pose_p)
    out = {}
    for m in markers:
        col = m + ":C"
        if col in df.columns and len(df):
            v = float(df[col].mean())
            if v < vis_thresh:
                out[m] = round(v, 4)
    return out


def ik_meta_path(trc_path):
    """Sidecar metadata path for a TRC: <root><suffix>.ik_meta.json
    (same `.trc`-suffix convention as the generated .IK.xml)."""
    return Path(re.sub(r"\.trc$", ".ik_meta.json", str(trc_path)))


def write_ik_meta(trc_file, model, cropped, vis_thresh=DEFAULT_VIS_THRESH):
    """Write the Stage 2 -> Stage 3 sidecar next to the TRC. Carries the
    cropped-marker decision forward because the TRC has no confidence
    channel and run_ik re-derives weights from the TRC alone. Returns
    the json path."""
    p = ik_meta_path(trc_file)
    p.write_text(json.dumps({
        "trc": Path(trc_file).name,
        "model": str(model),
        "visibility_threshold": vis_thresh,
        "cropped_markers": cropped,            # {name: mean_visibility}
        "zeroed_in_ik": sorted(cropped),       # markers forced to weight 0
    }, indent=2) + "\n")
    return str(p)


def read_ik_meta(trc_path):
    """Load the sidecar for a TRC if present, else None."""
    p = ik_meta_path(trc_path)
    if not p.exists():
        return None
    return json.loads(p.read_text())


def model_markers(model_path):
    """All <Marker> names from an .osim, in file order (regex parse --
    no opensim needed). None if the file can't be read."""
    try:
        txt = Path(model_path).read_text()
    except OSError:
        return None
    return re.findall(r'<Marker name="([^"]+)"', txt) or None


def _ik_taskset(marker_names, present, zero=()):
    """One IKMarkerTask per model marker at the uniform default weight;
    apply=true iff the marker exists in the TRC (`present`), so the
    Vicon-only markers with no markerless data are carried but disabled
    (mirrors the hand-tuned .IK.mod.xml). Markers in `zero` (cropped out
    of frame) are kept/applied but given weight 0 so they cannot pull
    the solve toward a hallucinated position."""
    zero = set(zero)
    objs = []
    for m in marker_names:
        w = 0.0 if m in zero else DEFAULT_IK_WEIGHT
        objs.append(
            f'\t\t\t\t<IKMarkerTask name="{m}">\n'
            f"\t\t\t\t\t<apply>{'true' if m in present else 'false'}</apply>\n"
            f"\t\t\t\t\t<weight>{w}</weight>\n"
            f"\t\t\t\t</IKMarkerTask>"
        )
    return ("\t\t\t<IKTaskSet>\n\t\t\t\t<objects>\n"
            + "\n".join(objs)
            + "\n\t\t\t\t</objects>\n\t\t\t\t<groups />\n\t\t\t</IKTaskSet>")


def write_ik_xml(out_path, trc_file, markers, tstart, tend,
                 model=DEFAULT_MODEL, template=DEFAULT_IK_TEMPLATE,
                 accuracy=DEFAULT_IK_ACCURACY, zero_weight=()):
    """Fill TEMPLATE_IK.xml: a task for every model marker at the uniform
    default weight (apply=false for markers absent from the TRC), plus a
    relaxed <accuracy>. `markers` is the TRC marker list (apply=true
    set); falls back to it alone if the model markers can't be read.
    Markers in `zero_weight` (cropped) get weight 0."""
    present = set(markers)
    zero = set(zero_weight)
    names = model_markers(model) or list(markers)
    txt = Path(template).read_text()
    txt = re.sub(r"<IKTaskSet>.*?</IKTaskSet>",
                 lambda _: _ik_taskset(names, present, zero), txt, flags=re.S)
    txt = Template(txt).substitute(
        osimfile=str(model), tstart=tstart, tend=tend,
        trcfilename=str(trc_file), accuracy=f"{accuracy:.16e}")
    Path(out_path).write_text(txt)
    return str(out_path)


def model_com(model_path=DEFAULT_MODEL):
    """Default-pose model center of mass in the ground frame (m). Uses
    opensim when importable (any model); otherwise falls back to the
    cached value for the default model and errors for any other."""
    try:
        import opensim as osim
    except ImportError:
        if Path(model_path).resolve() == Path(DEFAULT_MODEL).resolve():
            return MODEL_COM_DEFAULT.copy()
        raise RuntimeError(
            "opensim unavailable: cannot compute COM for a non-default "
            "model. Run in the opensim env, or use the default model.")
    m = osim.Model(str(model_path))
    s = m.initSystem()
    c = m.calcMassCenterPosition(s)
    return np.array([c.get(0), c.get(1), c.get(2)])


# Stable torso reference: mean of L/R shoulder + L/R hip. Used instead
# of a whole-marker centroid, which the 42 hand points heavily bias.
TORSO_REF = ("LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_HIP", "RIGHT_HIP")


def torso_center(sources, ref_pair, ref_dist_m, frame=0, z_rescale=False):
    """First-frame torso center (mean of L/R shoulder & L/R hip) in the
    conditioned body frame. Read from the conditioned pose, so it is the
    same anatomical reference in every mode -- including hand-only TRCs,
    whose hands are registered into this same pose frame."""
    pose_xyz, pose_joints = sources["pose"]
    pose_m = condition_pose(pose_xyz, pose_joints, ref_pair, ref_dist_m,
                            z_rescale=z_rescale)
    idx = [pose_joints.index(j) for j in TORSO_REF]
    return pose_m[frame, idx, :].mean(axis=0)


def center_table_on(table, shift):
    """Rigidly translate every marker/frame by `shift` (3,)."""
    for k in table:
        table[k] = table[k] + shift
    return table


# --------------------------------------------------------------------------
# Stage 2 entry point
# --------------------------------------------------------------------------
_MODE_SUFFIX = {"combined": "", "left_hand": ".LH",
                "right_hand": ".RH", "body": ".body"}


def csv_to_trc(root, mode="combined", model=DEFAULT_MODEL,
               ref_segment=("LEFT_SHOULDER", "RIGHT_SHOULDER"),
               ref_dist_m=0.318, out_path=None, write_ik=False,
               center_on_model_com=True, z_rescale=False,
               lowpass_hz=DEFAULT_LOWPASS_HZ,
               vis_thresh=DEFAULT_VIS_THRESH,
               hand_source="roi"):
    """Fuse <root>_{lh,rh,pose}.csv into an OpenSim TRC.

    Defaults assume metric world-landmark CSVs (Stage 1): no z-rescale,
    a 6 Hz low-pass on every trajectory. Set z_rescale=True only for
    legacy normalized-landmark CSVs; lowpass_hz=None disables filtering.

    mode: combined (body + both detailed hands, default) | left_hand |
    right_hand | body. If center_on_model_com, the whole marker cloud is
    rigidly translated so the first-frame torso center (mean of L/R
    shoulder & L/R hip) sits on `model`'s default-pose COM -- a good IK
    assembly seed, robust to the hand-point-heavy marker count. Returns
    the TRC path; if write_ik also writes <root><suffix>.IK.xml targeting
    `model`. Always writes a <root><suffix>.ik_meta.json sidecar
    recording body markers cropped out of frame (mean visibility <
    vis_thresh) so run_ik can zero their IK weight.

    hand_source: "registered" -- the Holistic hands fused by
    similarity registration. "roi" (the default) -- the alternate ROI-cropped
    HandLandmarker hands (needs Stage-1 _{lh,rh}_roi.csv); `.roihands`
    suffix. "combined" -- _roi plus Holistic-recovered rows for
    ROI-missing frames (needs _{lh,rh}_roi.combined.csv from
    combine_hands.py); `.combhands` suffix. Each source writes a
    distinct TRC/IK so the solves can be compared."""
    root = Path(root)
    if hand_source in ("roi", "combined"):
        base = ("lh_roi",) if mode == "left_hand" else \
               ("rh_roi",) if mode == "right_hand" else \
               ("lh_roi", "rh_roi")
        ext = ".combined.csv" if hand_source == "combined" else ".csv"
        missing = [root.with_name(root.name + f"_{k}{ext}") for k in base
                   if not root.with_name(root.name + f"_{k}{ext}").exists()]
        if missing:
            if hand_source == "combined":
                raise FileNotFoundError(
                    f"hand_source='combined' needs {missing}; run "
                    f"src/combine_hands.py on this <root> first "
                    f"(mp2trc --hand-source combined does it for you)")
            raise FileNotFoundError(
                f"hand_source='roi' needs {missing}; re-run Stage 1 with "
                f"roi_hands=True (mp2trc --roi-hands --pose-model heavy)")
    times, sources = load_landmark_frames(root, lowpass_hz=lowpass_hz)
    table = build_marker_table(times, sources, mode, ref_segment, ref_dist_m,
                               z_rescale=z_rescale, hand_source=hand_source)

    if center_on_model_com:
        ref = torso_center(sources, ref_segment, ref_dist_m,
                           z_rescale=z_rescale)
        center_table_on(table, model_com(model) - ref)

    if out_path is None:
        suffix = _MODE_SUFFIX[mode] + (
            ".combhands" if hand_source == "combined"
            else ".roihands" if hand_source == "roi" else "")
        out_path = root.with_name(root.name + suffix + ".trc")
    dt = np.diff(times)
    data_rate = float(1.0 / dt.mean()) if len(dt) else 100.0
    trc = write_trc(out_path, table, times, data_rate)

    cropped = cropped_markers(root, vis_thresh=vis_thresh)
    write_ik_meta(trc, model, cropped, vis_thresh=vis_thresh)

    if write_ik:
        ik = re.sub(r"\.trc$", ".IK.xml", str(out_path))
        write_ik_xml(ik, trc, list(table), float(times[0]), float(times[-1]),
                     model=model, zero_weight=set(cropped))
    return trc


# --------------------------------------------------------------------------
# Stage 1: video -> CSV (cv2/mediapipe imported lazily)
# --------------------------------------------------------------------------
def _ensure_holistic_task():
    """Download the HolisticLandmarker .task bundle if absent."""
    if not HOLISTIC_TASK.exists():
        import urllib.request
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(HOLISTIC_TASK_URL, HOLISTIC_TASK)
    return HOLISTIC_TASK


def _ensure_pose_task(variant):
    """Download the PoseLandmarker .task bundle for a lite/full/heavy
    variant if absent; return its path."""
    if variant not in POSE_MODELS:
        raise ValueError(f"pose_model must be one of {POSE_MODELS} "
                         f"(got {variant!r})")
    p = MODELS_DIR / f"pose_landmarker_{variant}.task"
    if not p.exists():
        import urllib.request
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(POSE_TASK_URL.format(v=variant), p)
    return p


def _ensure_hand_task():
    """Download the standalone HandLandmarker .task bundle if absent."""
    if not HAND_TASK.exists():
        import urllib.request
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(HAND_TASK_URL, HAND_TASK)
    return HAND_TASK


# Per-side pose points that bound a hand (subject's own left/right). The
# ROI is their normalized-pixel bounding box, made square, padded, and
# floored to a fraction of forearm (wrist->elbow) length so a clenched
# fist -- where wrist/index/pinky/thumb collapse onto each other -- still
# gets a crop big enough to contain the fingers.
_ROI_PTS = {
    "left": ("LEFT_WRIST", "LEFT_PINKY", "LEFT_INDEX", "LEFT_THUMB"),
    "right": ("RIGHT_WRIST", "RIGHT_PINKY", "RIGHT_INDEX", "RIGHT_THUMB"),
}
_ROI_WRIST = {"left": "LEFT_WRIST", "right": "RIGHT_WRIST"}
_ROI_ELBOW = {"left": "LEFT_ELBOW", "right": "RIGHT_ELBOW"}
_ROI_INDEX = {"left": "LEFT_INDEX", "right": "RIGHT_INDEX"}
_ROI_SHLDR = ("LEFT_SHOULDER", "RIGHT_SHOULDER")
ROI_PAD = 0.6           # extra half-side, as a fraction of the bbox side
ROI_MIN_FOREARM = 0.55  # ROI half-side floor, as a fraction of forearm px
# Forearm foreshortens hard toward the camera (its projected px length
# can halve mid-stroke) while the hand's apparent size barely changes,
# so a forearm-only floor collapses the ROI and clips the hand. Shoulder
# (biacromial) width is pose-scale but motion-invariant -- a hand spans
# ~0.5x of it -- so floor the half-side to a fraction of shoulder px
# too; whichever floor is larger wins.
ROI_MIN_SHOULDER = 0.42
ROI_MIN_VIS = 0.3       # skip a side whose pose wrist is less visible


def _hand_roi(pose_norm, side, w, h):
    """Square, padded pixel ROI (x0,y0,x1,y1) around `side`'s hand from
    the pose's normalized landmarks, clamped to the frame. None if the
    wrist is too low-visibility to trust (hand likely out of frame)."""
    wi = POSE_LANDMARKS.index(_ROI_WRIST[side])
    wlm = pose_norm[wi]
    if max(getattr(wlm, "visibility", 1.0) or 0.0,
           getattr(wlm, "presence", 0.0) or 0.0) < ROI_MIN_VIS:
        return None
    pts = [np.array([pose_norm[POSE_LANDMARKS.index(n)].x * w,
                     pose_norm[POSE_LANDMARKS.index(n)].y * h])
           for n in _ROI_PTS[side]]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    half = max(max(xs) - min(xs), max(ys) - min(ys)) / 2.0
    elm = pose_norm[POSE_LANDMARKS.index(_ROI_ELBOW[side])]
    forearm = np.hypot((wlm.x - elm.x) * w, (wlm.y - elm.y) * h)
    floors = [half, ROI_MIN_FOREARM * forearm]

    # Pose-scale, motion-invariant floor from shoulder width (skipped
    # only if a shoulder is too occluded to trust).
    s0, s1 = (pose_norm[POSE_LANDMARKS.index(n)] for n in _ROI_SHLDR)
    if min(getattr(s0, "visibility", 1.0) or 0.0,
           getattr(s1, "visibility", 1.0) or 0.0) >= ROI_MIN_VIS:
        floors.append(ROI_MIN_SHOULDER * np.hypot((s0.x - s1.x) * w,
                                                  (s0.y - s1.y) * h))
    half = max(floors) * (1.0 + ROI_PAD)
    if half < 5.0:
        return None

    # Bias the center off the wrist toward the knuckles: the fingers
    # extend ~one knuckle-span further again, so a wrist/palm-centered
    # box clips the fingertips.
    wrist = pts[0]
    knuck = np.mean(pts[1:], axis=0)
    cx, cy = wrist + 1.5 * (knuck - wrist)
    x0 = max(0, int(round(cx - half)))
    y0 = max(0, int(round(cy - half)))
    x1 = min(w, int(round(cx + half)))
    y1 = min(h, int(round(cy + half)))
    if x1 - x0 < 10 or y1 - y0 < 10:
        return None
    return x0, y0, x1, y1


def _unify_hand_world(hand_world, pose_world, side):
    """Map a standalone-hand `hand_world_landmarks` (21,3; origin ~wrist,
    canonical-hand scale) into the pose's metric world frame (hips
    origin, meters), per the wrist-anchored translation with a dynamic
    wrist->index-MCP scale factor s = D_pose / D_hand. Returns (21,3)."""
    hw = np.array([[p.x, p.y, p.z] for p in hand_world], dtype=float)
    pw = np.array([[p.x, p.y, p.z] for p in pose_world], dtype=float)
    p_wrist = pw[POSE_LANDMARKS.index(_ROI_WRIST[side])]
    d_pose = np.linalg.norm(
        p_wrist - pw[POSE_LANDMARKS.index(_ROI_INDEX[side])])
    h_wrist = hw[HAND_LEAF.index("WRIST")]
    d_hand = np.linalg.norm(
        h_wrist - hw[HAND_LEAF.index("INDEX_FINGER_MCP")])
    s = (d_pose / d_hand) if (d_hand > 1e-9 and np.isfinite(d_pose)
                              and d_pose > 0) else 1.0
    return p_wrist + s * (hw - h_wrist)


def _draw(image, lms, w, h, color, glyph="dot"):
    """Landmark overlay for the annotated video. glyph="dot" (small
    filled circle, default) or "ring" (larger hollow circle) so two
    overlaid landmark sets stay visually distinct."""
    import cv2
    for p in lms:
        c = (int(p.x * w), int(p.y * h))
        if glyph == "ring":
            cv2.circle(image, c, 4, color, 1, lineType=cv2.LINE_AA)
        else:
            cv2.circle(image, c, 2, color, -1)


def _merge_audio(src_video, annotated, merged):
    """Mux the original video's audio track(s) onto the (silent)
    annotated video -> <root>_merged.mp4. Best-effort: if ffmpeg is
    absent or the source has no audio, warn and skip without failing
    the pipeline."""
    import shutil
    import subprocess
    ff = shutil.which("ffmpeg")
    if ff is None:
        print("ffmpeg not found; skipping _merged.mp4")
        return None
    cmd = [ff, "-y", "-loglevel", "error",
           "-i", str(annotated), "-i", str(src_video),
           "-map", "0:v:0", "-map", "1:a?",
           "-c:v", "copy", "-c:a", "aac", "-shortest", str(merged)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("ffmpeg audio merge failed; kept _annotated.mp4 only:",
              (r.stderr.strip().splitlines() or [""])[-1])
        return None
    return str(merged)


def video_to_csv(video_path, out_root=None, fps=None,
                 min_detection_confidence=DEFAULT_MIN_DET_CONF,
                 min_tracking_confidence=DEFAULT_MIN_TRK_CONF,
                 annotate=True, show=False, progress=False,
                 mp_options=None, pose_model="heavy", roi_hands=True,
                 debug_roi=True):
    """Run the MediaPipe Tasks HolisticLandmarker over a video, writing
    <root>_lh.csv, _rh.csv, _pose.csv (and optionally
    <root>_annotated.mp4). The CSVs carry the **metric world
    landmarks** (pose_world / *_hand_world): depth-aware, ~real bone
    lengths, so no z-rescale hack is needed downstream (rigid-segment
    length CV ~2-7% vs ~15-25% for the old normalized landmarks). The
    annotated overlay still uses the normalized landmarks (pixel
    coords). Rows are buffered and each CSV written once; timestamps and
    the annotated video use the real capture fps. Left/right hands are
    resolved by HolisticLandmarker with body context. The .task bundle
    is auto-downloaded to models/mediapipe/ on first use.

    min_detection_confidence/min_tracking_confidence are shortcut knobs,
    each broadcasting to two genuine HolisticLandmarkerOptions fields
    (MP_SHORTCUTS). mp_options is an optional {field: value} dict (e.g.
    from parse_mp_config) overriding any individual genuine field at
    higher precedence than the shortcuts.
    pose_model: "heavy" (default) / "lite" / "full" -- run a separate
    PoseLandmarker of that size for the _pose.csv (more accurate 3D
    body) while Holistic still supplies the detailed hands; or
    "holistic" for pose+hands from one Holistic pass. mp_options does
    not apply to the PoseLandmarker path.
    roi_hands=True adds an *alternate* detailed-hand stream: each frame,
    a square padded ROI is cropped around each wrist from the (heavy)
    pose's normalized landmarks and fed to a standalone HandLandmarker,
    whose hand_world_landmarks are mapped into the pose's metric world
    frame (wrist-anchored translation + dynamic wrist->index-MCP scale).
    These land in <root>_lh_roi.csv / _rh_roi.csv (same schema as
    _lh/_rh.csv, but already in the pose world frame) and are overlaid
    on the annotated video as cyan rings for visual comparison against
    the Holistic hands. Requires a separate PoseLandmarker
    (pose_model != "holistic"); the existing _lh/_rh.csv and the TRC
    path are unchanged.
    debug_roi=True (only meaningful with roi_hands) additionally draws
    every proposed ROI crop rectangle on the annotated video -- including
    crops where the HandLandmarker found no hand -- so a missed detection
    is visually distinguishable from a mis-placed or too-small ROI.
    progress=True shows a per-frame tqdm bar (soft dep: silently
    skipped if tqdm is absent); off by default for notebook/library
    use."""
    import csv
    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    if roi_hands and pose_model == "holistic":
        raise ValueError(
            "roi_hands=True requires a separate PoseLandmarker "
            "(pose_model in {lite,full,heavy}); the (heavy) pose is the "
            "ROI source for the alternate cropped-hand stream")

    video_path = Path(video_path)
    root = Path(out_root) if out_root else video_path.with_suffix("")
    p_lh = root.with_name(root.name + "_lh.csv")
    p_rh = root.with_name(root.name + "_rh.csv")
    p_pose = root.with_name(root.name + "_pose.csv")
    p_lh_roi = root.with_name(root.name + "_lh_roi.csv")
    p_rh_roi = root.with_name(root.name + "_rh_roi.csv")
    p_ann = root.with_name(root.name + "_annotated.mp4")
    p_merged = root.with_name(root.name + "_merged.mp4")

    hand_hdr = ["Frame Number", "Time (s)"] + [
        f"{n}:{ax}" for n in HAND_LEAF for ax in "XYZC"]
    pose_hdr = ["Frame Number", "Time (s)"] + [
        f"{n}:{ax}" for n in POSE_LANDMARKS for ax in "XYZC"]

    # defaults < shortcut broadcast < explicit mp_options (per field)
    opt_kwargs = dict(MP_OPTION_DEFAULTS)
    for shortcut, fields in MP_SHORTCUTS.items():
        val = (min_detection_confidence
               if shortcut == "min_detection_confidence"
               else min_tracking_confidence)
        for f in fields:
            opt_kwargs[f] = val
    if mp_options:
        bad = set(mp_options) - set(MP_OPTION_DEFAULTS)
        if bad:
            raise ValueError(f"unknown mp_options {sorted(bad)}; valid: "
                             f"{sorted(MP_OPTION_DEFAULTS)}")
        opt_kwargs.update(mp_options)
    # Forward-safety: drop any field the installed mediapipe doesn't
    # expose rather than crash on a version skew (warn so it's visible).
    valid = set(vision.HolisticLandmarkerOptions.__dataclass_fields__)
    for k in [k for k in opt_kwargs if k not in valid]:
        print(f"warning: HolisticLandmarkerOptions has no {k!r} in "
              f"mediapipe {mp.__version__}; ignoring")
        opt_kwargs.pop(k)
    opts = vision.HolisticLandmarkerOptions(
        base_options=mp_python.BaseOptions(
            model_asset_path=str(_ensure_holistic_task())),
        running_mode=vision.RunningMode.VIDEO,
        **opt_kwargs)

    # Optional separate PoseLandmarker for the _pose.csv stream (lite/
    # full/heavy). Hands still come from HolisticLandmarker below.
    # mp_options is HolisticLandmarker-specific and does NOT apply here;
    # the two confidence shortcuts do.
    pose_opts = None
    if pose_model != "holistic":
        pose_opts = vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=str(_ensure_pose_task(pose_model))),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence)

    # Optional standalone HandLandmarker for the alternate cropped-hand
    # stream. IMAGE mode: it is re-run independently on each per-frame
    # wrist crop (no inter-frame tracking state across crops/sides).
    hand_opts = None
    if roi_hands:
        hand_opts = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=str(_ensure_hand_task())),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence)

    cap = cv2.VideoCapture(str(video_path))
    cap_fps = fps or cap.get(cv2.CAP_PROP_FPS)
    dt = 1.0 / cap_fps
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = (cv2.VideoWriter(str(p_ann), cv2.VideoWriter_fourcc(*"mp4v"),
                           cap_fps, (w, h)) if annotate else None)

    bar = None
    if progress:
        try:
            from tqdm import tqdm
            nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
            bar = tqdm(total=nframes, unit="frame", desc="MediaPipe")
        except ImportError:
            bar = None

    rows = {"lh": [], "rh": [], "pose": [],
            "lh_roi": [], "rh_roi": []}
    last_ts = -1
    import contextlib
    from types import SimpleNamespace

    def _first(seq):
        """PoseLandmarker returns List[List[lm]] (per-pose); Holistic a
        flat List[lm]. Normalize to a flat list (first pose) or []."""
        if not seq:
            return []
        return seq[0] if isinstance(seq[0], (list, tuple)) else seq

    with contextlib.ExitStack() as stack:
        holistic = stack.enter_context(
            vision.HolisticLandmarker.create_from_options(opts))
        pose_lm = (stack.enter_context(
            vision.PoseLandmarker.create_from_options(pose_opts))
            if pose_opts is not None else None)
        hand_lm = (stack.enter_context(
            vision.HandLandmarker.create_from_options(hand_opts))
            if hand_opts is not None else None)
        fid = 0
        while cap.isOpened():
            ok, image = cap.read()
            if not ok:
                break
            fid += 1
            if bar is not None:
                bar.update(1)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            ts = max(int(round((fid - 1) * 1000.0 / cap_fps)), last_ts + 1)
            last_ts = ts
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res = holistic.detect_for_video(mp_img, ts)

            # Pose stream: separate PoseLandmarker if requested, else
            # Holistic. Hands always from Holistic.
            if pose_lm is not None:
                pres = pose_lm.detect_for_video(mp_img, ts)
                pose_world = _first(pres.pose_world_landmarks)
                pose_norm = _first(pres.pose_landmarks)
            else:
                pose_world = res.pose_world_landmarks
                pose_norm = res.pose_landmarks

            # Alternate hands: crop a square ROI around each wrist from
            # the pose and run the standalone HandLandmarker on just that
            # crop; unify its world landmarks into the pose world frame.
            roi_hand = {}
            roi_rects = {}
            if hand_lm is not None and pose_norm and pose_world:
                for side in ("left", "right"):
                    rect = _hand_roi(pose_norm, side, w, h)
                    if rect is None:
                        continue
                    roi_rects[side] = rect
                    x0, y0, x1, y1 = rect
                    crop = np.ascontiguousarray(rgb[y0:y1, x0:x1])
                    hres = hand_lm.detect(mp.Image(
                        image_format=mp.ImageFormat.SRGB, data=crop))
                    if not hres.hand_landmarks:
                        continue
                    sc = (hres.handedness[0][0].score
                          if hres.handedness and hres.handedness[0]
                          else 0.0)
                    roi_hand[side] = {
                        "rect": rect,
                        "norm": hres.hand_landmarks[0],
                        "world": _unify_hand_world(
                            hres.hand_world_landmarks[0], pose_world,
                            side),
                        "score": float(sc),
                    }

            if annotate or show:
                if pose_lm is not None:
                    # Compare both body poses: Holistic (the discarded
                    # one) as dots, the chosen separate PoseLandmarker
                    # as magenta rings.
                    _draw(image, res.pose_landmarks, w, h, (245, 117, 66),
                          glyph="dot")
                    _draw(image, pose_norm, w, h, (255, 0, 255),
                          glyph="ring")
                    legend = f"dot=holistic   ring=pose:{pose_model}"
                else:
                    _draw(image, pose_norm, w, h, (245, 117, 66))
                    legend = "pose:holistic"
                _draw(image, res.left_hand_landmarks, w, h, (66, 245, 117))
                _draw(image, res.right_hand_landmarks, w, h, (66, 117, 245))
                if hand_lm is not None:
                    legend += "   cyan=ROI-hand"
                    if debug_roi:
                        legend += "   box=ROI-crop"
                if debug_roi:
                    # Every crop the pose proposed -- drawn even when the
                    # HandLandmarker found nothing inside it, so a missed
                    # hand is distinguishable from a mis-placed/too-small
                    # ROI.
                    for x0, y0, x1, y1 in roi_rects.values():
                        cv2.rectangle(image, (x0, y0), (x1, y1),
                                      (255, 255, 0), 1)
                for hd in roi_hand.values():
                    x0, y0, x1, y1 = hd["rect"]
                    mapped = [SimpleNamespace(
                        x=(x0 + p.x * (x1 - x0)) / w,
                        y=(y0 + p.y * (y1 - y0)) / h)
                        for p in hd["norm"]]
                    _draw(image, mapped, w, h, (255, 255, 0), glyph="ring")

                # Frame # + per-model landmark counts (33 pose / 21 hand
                # when seen, else 0) so presence/absence of each model is
                # readable frame-by-frame next to the overlay.
                if pose_lm is not None:
                    pc = (f"hol-pose:{len(res.pose_landmarks)}  "
                          f"pose:{pose_model}:{len(pose_norm)}")
                else:
                    pc = f"pose:hol:{len(pose_norm)}"
                counts = (f"f#{fid}  {pc}  "
                          f"Lhand:{len(res.left_hand_landmarks)}  "
                          f"Rhand:{len(res.right_hand_landmarks)}")
                if hand_lm is not None:
                    nl = len(roi_hand["left"]["norm"]) if "left" in roi_hand \
                        else 0
                    nr = len(roi_hand["right"]["norm"]) if "right" in roi_hand \
                        else 0
                    counts += f"  roiL:{nl}  roiR:{nr}"
                for i, txt in enumerate((legend, counts)):
                    cv2.putText(image, txt, (8, 22 + 22 * i),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (255, 255, 255), 1, cv2.LINE_AA)
                if out is not None:
                    out.write(image)
                if show:
                    cv2.imshow("MediaPipe Holistic", image)
                    if cv2.waitKey(5) & 0xFF == 27:
                        break

            # CSVs carry the METRIC world landmarks (depth-aware, ~real
            # bone lengths); the normalized landmarks above are used only
            # to draw the image overlay (they are pixel coords).
            meta = [fid, (fid - 1) * dt]
            for key, lms in (("lh", res.left_hand_world_landmarks),
                             ("rh", res.right_hand_world_landmarks),
                             ("pose", pose_world)):
                if not lms:
                    continue
                flat = np.array([[p.x, p.y, p.z,
                                  getattr(p, "visibility", 0.0) or 0.0]
                                 for p in lms]).flatten()
                rows[key].append(meta + list(flat))

            # Alternate ROI hands, already unified into the pose world
            # frame; C column carries the HandLandmarker handedness score.
            for side, key in (("left", "lh_roi"), ("right", "rh_roi")):
                hd = roi_hand.get(side)
                if hd is None:
                    continue
                flat = np.array([[x, y, z, hd["score"]]
                                 for x, y, z in hd["world"]]).flatten()
                rows[key].append(meta + list(flat))
    cap.release()
    if out is not None:
        out.release()
    if bar is not None:
        bar.close()
    if show:
        cv2.destroyAllWindows()

    out_csvs = [(p_lh, hand_hdr, "lh"), (p_rh, hand_hdr, "rh"),
                (p_pose, pose_hdr, "pose")]
    if roi_hands:
        out_csvs += [(p_lh_roi, hand_hdr, "lh_roi"),
                     (p_rh_roi, hand_hdr, "rh_roi")]
    for path, hdr, key in out_csvs:
        with open(path, "w", newline="") as f:
            wtr = csv.writer(f)
            wtr.writerow(hdr)
            wtr.writerows(rows[key])

    merged = None
    if annotate:
        merged = _merge_audio(video_path, p_ann, p_merged)
    return {"lh": str(p_lh), "rh": str(p_rh), "pose": str(p_pose),
            "lh_roi": str(p_lh_roi) if roi_hands else None,
            "rh_roi": str(p_rh_roi) if roi_hands else None,
            "annotated": str(p_ann) if annotate else None,
            "merged": merged}
