#!/usr/bin/env python3
"""End-to-end markerless CLI: video -> MediaPipe CSVs + annotated video
-> OpenSim TRC + IK setup, all written into a fresh directory named
after the input video.

    .venv/bin/python src/mp2trc.py clip.mp4
    .venv/bin/python src/mp2trc.py clip.mp4 --outdir runs --mode left_hand

Run with the project .venv (see docs/SETUP.md): needs mediapipe +
opencv + numpy/pandas/scipy/sklearn. This only *generates* the
.trc/.IK.xml; it does not run IK (that's run_ik.py).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mpipe_pipeline as M


def run(video, outdir=None, mode="combined", annotate=True, fps=None,
        write_ik=True, center_on_model_com=True,
        lowpass_hz=M.DEFAULT_LOWPASS_HZ, legacy_normalized=False,
        progress=True, vis_thresh=M.DEFAULT_VIS_THRESH,
        min_detection_confidence=M.DEFAULT_MIN_DET_CONF,
        min_tracking_confidence=M.DEFAULT_MIN_TRK_CONF, mp_options=None,
        pose_model="heavy", roi_hands=True, debug_roi=True,
        hand_source="roi"):
    """Process one video; returns the output directory Path."""
    video = Path(video)
    if not video.exists():
        raise FileNotFoundError(video)
    base = Path(outdir) if outdir else Path.cwd()
    dest = base / video.stem
    dest.mkdir(parents=True, exist_ok=True)
    root = dest / video.stem

    print(f"[1/2] MediaPipe Holistic"
          + (f" + Pose:{pose_model}" if pose_model != "holistic" else "")
          + (" + ROI-hands" if roi_hands else "")
          + f" -> {dest}/")
    csvs = M.video_to_csv(video, out_root=root, fps=fps, annotate=annotate,
                          progress=progress,
                          min_detection_confidence=min_detection_confidence,
                          min_tracking_confidence=min_tracking_confidence,
                          mp_options=mp_options, pose_model=pose_model,
                          roi_hands=roi_hands, debug_roi=debug_roi)
    for k in ("pose", "lh", "rh"):
        print(f"      {Path(csvs[k]).name}")
    if roi_hands:
        for k in ("lh_roi", "rh_roi"):
            print(f"      {Path(csvs[k]).name}")
    if annotate:
        print(f"      {Path(csvs['annotated']).name}")
        if csvs.get("merged"):
            print(f"      {Path(csvs['merged']).name}")

    print(f"[2/2] TRC + IK ({mode}"
          + (", hands=ROI" if hand_source == "roi" else "")
          + ")"
          + ("" if center_on_model_com else " [NOT centered on model COM]"))
    trc = M.csv_to_trc(root, mode=mode, write_ik=write_ik,
                       center_on_model_com=center_on_model_com,
                       lowpass_hz=lowpass_hz, z_rescale=legacy_normalized,
                       vis_thresh=vis_thresh, hand_source=hand_source)
    print(f"      {Path(trc).name}")
    if write_ik:
        print(f"      {Path(trc).name.replace('.trc', '.IK.xml')}")
    print(f"Done -> {dest}/")
    return dest


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="video -> MediaPipe CSVs + annotated mp4 -> "
                    "OpenSim TRC + IK.xml, into a video-named directory")
    ap.add_argument("video", nargs="?", default=None,
                    help="input video file (omit only with "
                         "--dump-mp-config)")
    ap.add_argument("--outdir", default=None,
                    help="base dir for the created folder (default: CWD)")
    ap.add_argument("--mode", default="combined",
                    choices=["combined", "left_hand", "right_hand", "body"],
                    help="TRC marker set (default: combined)")
    ap.add_argument("--no-annotate", action="store_true",
                    help="skip the annotated video")
    ap.add_argument("--no-ik", action="store_true",
                    help="skip writing the .IK.xml")
    ap.add_argument("--fps", type=float, default=None,
                    help="override capture fps")
    ap.add_argument("--no-center-on-model-com", action="store_true",
                    help="disable the default: by default the TRC is "
                         "translated so the first-frame torso center "
                         "(L/R shoulder + L/R hip) sits on the model's "
                         "default-pose COM (needed for reliable IK "
                         "assembly)")
    ap.add_argument("--lowpass", type=float, default=M.DEFAULT_LOWPASS_HZ,
                    help=f"low-pass cutoff Hz (default "
                         f"{M.DEFAULT_LOWPASS_HZ}; 0 disables)")
    ap.add_argument("--legacy-normalized", action="store_true",
                    help="input CSVs are old normalized landmarks: "
                         "re-enable the z-rescale + uniform-scale path")
    ap.add_argument("--no-progress", action="store_true",
                    help="hide the per-frame progress bar")
    ap.add_argument("--vis-thresh", type=float, default=M.DEFAULT_VIS_THRESH,
                    help=f"body markers with mean MediaPipe visibility "
                         f"below this are treated as cropped out of frame "
                         f"and zeroed in IK (default {M.DEFAULT_VIS_THRESH})")
    ap.add_argument("--min-detection-confidence", type=float,
                    default=M.DEFAULT_MIN_DET_CONF,
                    help=f"MediaPipe detection-confidence shortcut "
                         f"(-> pose & face detection; default "
                         f"{M.DEFAULT_MIN_DET_CONF})")
    ap.add_argument("--min-tracking-confidence", type=float,
                    default=M.DEFAULT_MIN_TRK_CONF,
                    help=f"MediaPipe tracking-confidence shortcut "
                         f"(-> pose & hand landmarks; default "
                         f"{M.DEFAULT_MIN_TRK_CONF})")
    ap.add_argument("--pose-model", default="heavy",
                    choices=["holistic", *M.POSE_MODELS],
                    help="body pose backend: a separate PoseLandmarker "
                         "lite/full/heavy (default: heavy -- most accurate "
                         "3D body; hands still from Holistic) or "
                         "'holistic' (pose from the Holistic pass; "
                         "incompatible with the ROI-hand defaults -- also "
                         "pass --no-roi-hands --hand-source registered)")
    ap.add_argument("--roi-hands", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="emit an alternate detailed-hand stream: a "
                         "square crop around each wrist (from the pose) "
                         "is run through a standalone HandLandmarker, "
                         "unified into the pose world frame, written to "
                         "<root>_{lh,rh}_roi.csv and overlaid on the "
                         "annotated video (cyan) for visual comparison "
                         "(default: on; --no-roi-hands disables). "
                         "Requires --pose-model lite/full/heavy")
    ap.add_argument("--debug-roi", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="also draw every proposed ROI crop box on the "
                         "annotated video, including crops where no hand "
                         "was found (distinguishes a missed hand from a "
                         "bad ROI) (default: on; --no-debug-roi disables). "
                         "Requires --roi-hands")
    ap.add_argument("--hand-source", default="roi",
                    choices=["registered", "roi"],
                    help="hands feeding the TRC/IK: 'roi' (default; the "
                         "ROI-cropped HandLandmarker hands, already "
                         "unified into the pose world frame) or "
                         "'registered' (Holistic hands fused by "
                         "similarity registration onto the coarse pose "
                         "points). 'roi' implies --roi-hands and writes a "
                         "separate <root>[.MODE].roihands.trc/.IK.xml so "
                         "the two IK solves can be compared. Requires "
                         "--pose-model lite/full/heavy")
    ap.add_argument("--mp-config", default=None, metavar="YAML",
                    help="YAML overriding individual HolisticLandmarker "
                         "knobs (highest precedence; see --dump-mp-config)")
    ap.add_argument("--dump-mp-config", nargs="?", const="-", metavar="PATH",
                    help="write a template mp-config YAML to PATH (or "
                         "stdout if omitted) and exit")
    a = ap.parse_args(argv)

    if a.dump_mp_config is not None:
        tpl = M.mp_config_template()
        if a.dump_mp_config == "-":
            print(tpl, end="")
        else:
            Path(a.dump_mp_config).write_text(tpl)
            print(f"wrote {a.dump_mp_config}")
        return

    if a.video is None:
        ap.error("video is required (omit it only with --dump-mp-config)")

    if a.hand_source == "roi":
        # The IK TRC needs the Stage-1 _roi CSVs, so force their emission.
        a.roi_hands = True
        if a.pose_model == "holistic":
            ap.error("--hand-source roi requires --pose-model "
                     "lite/full/heavy (the ROI hands derive from the "
                     "separate pose)")
    if a.roi_hands and a.pose_model == "holistic":
        ap.error("--roi-hands requires --pose-model lite/full/heavy "
                 "(the separate pose is the ROI source)")
    if a.debug_roi and not a.roi_hands:
        ap.error("--debug-roi requires --roi-hands (nothing to debug "
                 "without the ROI-hand stream)")

    mp_options = M.parse_mp_config(a.mp_config) if a.mp_config else None
    run(a.video, outdir=a.outdir, mode=a.mode, annotate=not a.no_annotate,
        fps=a.fps, write_ik=not a.no_ik,
        center_on_model_com=not a.no_center_on_model_com,
        lowpass_hz=(a.lowpass or None),
        legacy_normalized=a.legacy_normalized,
        progress=not a.no_progress, vis_thresh=a.vis_thresh,
        min_detection_confidence=a.min_detection_confidence,
        min_tracking_confidence=a.min_tracking_confidence,
        mp_options=mp_options, pose_model=a.pose_model,
        roi_hands=a.roi_hands, debug_roi=a.debug_roi,
        hand_source=a.hand_source)


if __name__ == "__main__":
    main()
