#!/usr/bin/env python3
"""Combine the ROI and Holistic hand-landmark CSVs into one stream.

Post-mp2trc step. mp2trc Stage 1 writes two hand streams per clip:

  <root>_{lh,rh}_roi.csv   ROI-cropped HandLandmarker, ALREADY unified
                           into the pose world frame -- primary, more
                           complete, but still has gaps.
  <root>_{lh,rh}.csv       raw Holistic hand_world_landmarks (hand-
                           centred canonical scale) -- a DIFFERENT,
                           partly-disjoint set of frames.
  <root>_pose.csv          raw pose world landmarks.

A "not found" frame is simply an absent row. Where ROI is missing but
the Holistic hand WAS detected, we recover a real frame instead of
letting the downstream blindly linear-interpolate the ROI gap.

This writes <root>_{lh,rh}_roi.combined.csv = every ROI row verbatim,
plus, for each ROI-missing / Holistic-present frame, the Holistic hand
put into the pose world frame by the SAME transform the ROI stream
used -- `mpipe_pipeline._unify_hand_world` (wrist-anchored to that
frame's pose wrist + dynamic wrist->index scale) -- then
endpoint-residual-blended so it meets the bracketing ROI samples
exactly (removes the per-clip ROI-vs-Holistic detector bias; gaps are
short so a linear residual carry across the gap is sufficient).

Recovered rows carry :C = -1.0 (a provenance sentinel; the hand :C
column is downstream-inert -- only _pose.csv :C is ever read).

Held-out check on runs/Wieniawski2 (drop ROI on tracked frames,
recover, vs blind linear interp): LEFT hand ~+34..44% lower error
across 3/5/8-frame gaps; RIGHT hand ~neutral (its Holistic stream
disagrees with ROI more). Real ROI-failure gaps (fast/occluded motion)
benefit more than this benign synthetic lower bound.

    combine_hands.py runs/clip/clip            # both hands
    combine_hands.py runs/clip                 # dir form (root=dir/dirname)
    combine_hands.py runs/clip/clip --dry-run  # stats only, write nothing

Consumed by csv_to_trc(..., hand_source="combined") /
mp2trc.py --hand-source combined.
"""
import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

import mpipe_pipeline as M

HAND_LEAF = M.HAND_LEAF
POSE_LANDMARKS = M.POSE_LANDMARKS
HAND_HDR = ["Frame Number", "Time (s)"] + [
    f"{n}:{ax}" for n in HAND_LEAF for ax in "XYZC"]
_HXYZ = [f"{n}:{ax}" for n in HAND_LEAF for ax in "XYZ"]
_PXYZ = [f"{n}:{ax}" for n in POSE_LANDMARKS for ax in "XYZ"]
_SIDE_TAG = {"lh": "left", "rh": "right"}


def _pts(arr):
    """(J,3) -> list of objects with .x/.y/.z (what _unify_hand_world wants)."""
    return [SimpleNamespace(x=r[0], y=r[1], z=r[2]) for r in arr]


def _contiguous_runs(frames):
    """sorted [f,...] -> [(start,end), ...] inclusive consecutive runs."""
    runs, s, prev = [], None, None
    for f in frames:
        if s is None:
            s = prev = f
        elif f == prev + 1:
            prev = f
        else:
            runs.append((s, prev))
            s = prev = f
    if s is not None:
        runs.append((s, prev))
    return runs


def combine_hand_csv(roi_csv, hol_csv, pose_csv, out_csv, *, side,
                      min_gap=1, min_copresent=10, recovered_C=-1.0,
                      dry_run=False):
    """Write a combined hand CSV (ROI verbatim + recovered Holistic
    rows). Returns a stats dict. Degrades to a verbatim ROI passthrough
    (so the downstream path never 404s) if the Holistic or pose CSV is
    absent or there are too few co-present frames to anchor."""
    roi_csv, hol_csv = Path(roi_csv), Path(hol_csv)
    pose_csv, out_csv = Path(pose_csv), Path(out_csv)
    tag = _SIDE_TAG[side]

    roi = pd.read_csv(roi_csv).drop_duplicates(
        subset="Frame Number", keep="first").set_index("Frame Number")
    roi = roi.reindex(columns=[c for c in HAND_HDR if c != "Frame Number"])

    st = {"roi_frames": int(roi.shape[0]), "hol_frames": 0,
          "copresent": 0, "recovered": 0, "gaps_filled": 0,
          "gaps_skipped_short": 0, "gaps_unanchored": 0, "reason": ""}

    def passthrough(reason):
        st["reason"] = reason
        if not dry_run:
            roi.reset_index().reindex(columns=HAND_HDR).to_csv(
                out_csv, index=False)
        return st

    if not hol_csv.exists():
        return passthrough("no_holistic_csv (passthrough)")
    if not pose_csv.exists():
        return passthrough("no_pose_csv (passthrough)")

    hol = pd.read_csv(hol_csv).drop_duplicates(
        subset="Frame Number", keep="first").set_index("Frame Number")
    pose = pd.read_csv(pose_csv).drop_duplicates(
        subset="Frame Number", keep="first").set_index("Frame Number")
    st["hol_frames"] = int(hol.shape[0])

    roi_fr, hol_fr, pose_fr = set(roi.index), set(hol.index), set(pose.index)
    copre = sorted(roi_fr & hol_fr & pose_fr)      # for residual anchoring
    st["copresent"] = len(copre)
    recover_fr = sorted((hol_fr & pose_fr) - roi_fr)
    if len(copre) < min_copresent or not recover_fr:
        return passthrough("insufficient_copresent" if len(copre) < min_copresent
                           else "nothing_to_recover")

    copre_arr = np.array(copre)

    def unify(f):
        hw = hol.loc[f, _HXYZ].to_numpy().reshape(len(HAND_LEAF), 3)
        pw = pose.loc[f, _PXYZ].to_numpy().reshape(len(POSE_LANDMARKS), 3)
        return M._unify_hand_world(_pts(hw), _pts(pw), tag)

    def roi_xyz(f):
        return roi.loc[f, _HXYZ].to_numpy().reshape(len(HAND_LEAF), 3)

    def boundary_resid(bf):
        """ROI(bf) - unify(bf); if bf not co-present, the nearest
        co-present frame's residual (best available local estimate)."""
        if bf in roi_fr and bf in hol_fr and bf in pose_fr:
            return roi_xyz(bf) - unify(bf)
        nf = int(copre_arr[np.argmin(np.abs(copre_arr - bf))])
        return roi_xyz(nf) - unify(nf)

    new_rows = []
    for (g0, g1) in _contiguous_runs(recover_fr):
        if (g1 - g0 + 1) < min_gap:
            st["gaps_skipped_short"] += 1
            continue
        L = g0 - 1 if (g0 - 1) in roi_fr else None
        R = g1 + 1 if (g1 + 1) in roi_fr else None
        rL = boundary_resid(L) if L is not None else None
        rR = boundary_resid(R) if R is not None else None
        if L is None and R is None:
            st["gaps_unanchored"] += 1
        st["gaps_filled"] += 1

        for f in range(g0, g1 + 1):
            if f not in hol_fr or f not in pose_fr:
                continue                            # stays absent -> interp_na
            xyz = unify(f)
            if L is not None and R is not None:
                t = (f - L) / (R - L)
                xyz = xyz + (1.0 - t) * rL + t * rR
            elif L is not None:
                xyz = xyz + rL
            elif R is not None:
                xyz = xyz + rR
            if not np.isfinite(xyz).all():
                continue                            # don't poison write_trc
            row = {"Frame Number": int(f),
                   "Time (s)": float(hol.loc[f, "Time (s)"])}
            for j, n in enumerate(HAND_LEAF):
                row[f"{n}:X"], row[f"{n}:Y"], row[f"{n}:Z"] = xyz[j]
                row[f"{n}:C"] = recovered_C
            new_rows.append(row)
            st["recovered"] += 1

    if not dry_run:
        out = roi.reset_index()
        if new_rows:
            out = pd.concat([out, pd.DataFrame(new_rows)], ignore_index=True)
        out = out.sort_values("Frame Number").reindex(columns=HAND_HDR)
        out.to_csv(out_csv, index=False)
    return st


def _resolve_root(arg):
    """Accept a <root> prefix or a run directory (root = dir/dir.name)."""
    p = Path(arg)
    return (p / p.name) if p.is_dir() else p


def combine_root(root, **kw):
    """Combine both hands for a <root>; writes
    <root>_{lh,rh}_roi.combined.csv. Skips a side whose ROI CSV is
    absent. Returns {'lh': stats|None, 'rh': stats|None}."""
    root = _resolve_root(root)
    pose_csv = root.with_name(f"{root.name}_pose.csv")
    out = {}
    for side in ("lh", "rh"):
        roi_csv = root.with_name(f"{root.name}_{side}_roi.csv")
        hol_csv = root.with_name(f"{root.name}_{side}.csv")
        out_csv = root.with_name(f"{root.name}_{side}_roi.combined.csv")
        if not roi_csv.exists():
            out[side] = None
            continue
        out[side] = combine_hand_csv(roi_csv, hol_csv, pose_csv, out_csv,
                                     side=side, **kw)
    return out


def _fmt(side, st):
    if st is None:
        return f"  {side}: skipped (no _{side}_roi.csv)"
    return (f"  {side}: roi={st['roi_frames']} hol={st['hol_frames']} "
            f"copre={st['copresent']} -> recovered {st['recovered']} in "
            f"{st['gaps_filled']} gaps (short-skipped {st['gaps_skipped_short']}"
            f", unanchored {st['gaps_unanchored']})"
            + (f"  [{st['reason']}]" if st["reason"] else ""))


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="combine_hands.py",
        description="Combine ROI + Holistic hand CSVs (post-mp2trc): ROI "
                    "rows verbatim + the Holistic hand (unified via the "
                    "same pose-wrist transform ROI uses, endpoint-blended) "
                    "for ROI-missing frames. Writes "
                    "<root>_{lh,rh}_roi.combined.csv.")
    ap.add_argument("root", help="<root> prefix or the run directory")
    ap.add_argument("--min-gap", type=int, default=1,
                    help="only recover ROI gaps at least this many frames "
                         "long; shorter gaps are left to the downstream "
                         "interpolation (default: 1 = recover all)")
    ap.add_argument("--min-copresent", type=int, default=10,
                    help="min ROI&Holistic&pose co-present frames to trust "
                         "recovery; below this the ROI CSV is passed "
                         "through unchanged (default: 10)")
    ap.add_argument("--recovered-c", type=float, default=-1.0,
                    help="value written to the :C column of recovered rows "
                         "(provenance sentinel; default: -1.0)")
    ap.add_argument("--side", choices=["both", "left", "right"],
                    default="both", help="which hand(s) to process")
    ap.add_argument("--dry-run", action="store_true",
                    help="print stats only; write no files")
    a = ap.parse_args(argv)

    root = _resolve_root(a.root)
    pose_csv = root.with_name(f"{root.name}_pose.csv")
    kw = dict(min_gap=a.min_gap, min_copresent=a.min_copresent,
              recovered_C=a.recovered_c, dry_run=a.dry_run)
    sides = {"both": ("lh", "rh"), "left": ("lh",), "right": ("rh",)}[a.side]

    print(f"combine_hands: {root}{'  (dry-run)' if a.dry_run else ''}")
    any_done = False
    for side in sides:
        roi_csv = root.with_name(f"{root.name}_{side}_roi.csv")
        hol_csv = root.with_name(f"{root.name}_{side}.csv")
        out_csv = root.with_name(f"{root.name}_{side}_roi.combined.csv")
        if not roi_csv.exists():
            print(_fmt(side, None))
            continue
        st = combine_hand_csv(roi_csv, hol_csv, pose_csv, out_csv,
                              side=side, **kw)
        print(_fmt(side, st))
        if not a.dry_run:
            print(f"        -> {out_csv.name}")
        any_done = True
    if not any_done:
        sys.exit("combine_hands: no _roi.csv for the requested side(s)")


if __name__ == "__main__":
    main()
