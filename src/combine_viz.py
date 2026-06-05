#!/usr/bin/env python3
"""Combine the musculoskeletal viz .mp4 with the annotated `_merged`
source video, keeping the original audio.

The render from viz_osim.py and the MediaPipe-annotated `_merged.mp4`
from mp2trc.py cover the same clip at the same fps/duration. This
ffmpeg-stitches them into one mp4 (vstack by default = merged on
top, viz below; hstack for side-by-side) and muxes in the audio
(default: from the merged file; fall
back / override with --audio-from). Uses the system `ffmpeg` if
present, else imageio-ffmpeg's bundled binary (same pattern as
viz_osim.py).

    .venv/bin/python src/combine_viz.py \\
        runs/Wieniawski2/Wieniawski2.combhands.viz.mp4

The merged file is auto-detected as `<stem>_merged.mp4` next to the
viz, where <stem> is the source-clip name (everything before the
first `.` in the viz filename, e.g. 'Wieniawski2'). Override with
--merged. Output defaults to `<stem>.combo.mp4` beside the viz.
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pipeline_config as PC


def _ffmpeg_exe():
    """System ffmpeg if available, else imageio-ffmpeg's bundled one."""
    p = shutil.which("ffmpeg")
    if p:
        return p
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as e:
        raise RuntimeError(
            "no ffmpeg found (system PATH nor imageio-ffmpeg). "
            "Install ffmpeg or `pip install imageio-ffmpeg`.") from e


def _source_stem(p):
    """Filename up to the first `.` — the original clip name. E.g.
    'Wieniawski2.combhands.viz.mp4' -> 'Wieniawski2'."""
    return Path(p).name.split(".")[0]


def run(viz, merged=None, out=None, layout="vstack", size=720,
        audio_from=None, no_audio=False):
    """Stitch `viz` + `merged` side-by-side (or vstack) with audio,
    write `out`, return its path."""
    viz = Path(viz).expanduser().resolve()
    if not viz.is_file():
        raise FileNotFoundError(viz)
    if merged is None:
        merged = viz.parent / f"{_source_stem(viz)}_merged.mp4"
    merged = Path(merged).expanduser().resolve()
    if not merged.is_file():
        raise FileNotFoundError(
            f"{merged} (auto-detected from the viz path; pass "
            f"--merged to point elsewhere)")
    if out is None:
        out = viz.parent / f"{_source_stem(viz)}.combo.mp4"
    out = Path(out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    H = int(size)
    if H % 2:
        H -= 1
    if H < 2:
        raise ValueError(f"--size must be >= 2 (got {size})")

    # both streams scaled to a common dimension so the stack filter
    # gets matching sizes; -2 keeps the other dimension even.
    if layout == "hstack":
        fc = (f"[0:v]scale=-2:{H}[a];"
              f"[1:v]scale=-2:{H}[b];"
              f"[a][b]hstack=inputs=2[v]")
    elif layout == "vstack":
        fc = (f"[0:v]scale={H}:-2[a];"
              f"[1:v]scale={H}:-2[b];"
              f"[a][b]vstack=inputs=2[v]")
    else:
        raise ValueError(f"unknown --layout: {layout}")

    # input order: [0]=merged, [1]=viz, optionally [2]=audio source
    inputs = [merged, viz]
    a_idx = None
    if not no_audio:
        af = (Path(audio_from).expanduser().resolve() if audio_from
              else merged)
        if af == merged:
            a_idx = 0
        elif af == viz:
            a_idx = 1
        else:
            if not af.is_file():
                raise FileNotFoundError(af)
            inputs.append(af)
            a_idx = 2

    ff = _ffmpeg_exe()
    cmd = [ff, "-y", "-hide_banner", "-loglevel", "warning"]
    for p in inputs:
        cmd += ["-i", str(p)]
    cmd += ["-filter_complex", fc, "-map", "[v]"]
    if a_idx is not None:
        # `?` makes the audio map conditional -- if the chosen input
        # has no audio stream, the output is silent rather than an error
        cmd += ["-map", f"{a_idx}:a?", "-c:a", "aac", "-b:a", "192k"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
            "-preset", "fast", "-shortest", str(out)]

    print("+", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run(cmd, check=True)
    print(f"merged : {merged}")
    print(f"viz    : {viz}")
    if a_idx is not None:
        print(f"audio  : input {a_idx} ({inputs[a_idx].name}, "
              f"conditional)")
    else:
        print("audio  : (none)")
    print(f"layout : {layout}  size {H}")
    print(f"video  : {out}")
    return str(out)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Combine a viz_osim .mp4 with the mp2trc "
                    "_merged annotated source + audio into one mp4 "
                    "(side-by-side or stacked).")
    ap.add_argument("viz", help="viz_osim render (.mp4)")
    ap.add_argument("--merged", default=None,
                    help="annotated _merged.mp4 (default: "
                         "<stem>_merged.mp4 next to the viz)")
    ap.add_argument("--out", default=None,
                    help="output (default: <stem>.combo.mp4 next to "
                         "the viz)")
    ap.add_argument("--layout", choices=("vstack", "hstack"),
                    default="vstack",
                    help="vstack (merged on top, viz below, default) "
                         "or hstack (side-by-side)")
    ap.add_argument("--size", type=int, default=720,
                    help="matched panel dimension in px (default 720; "
                         "each stream scaled preserving aspect to this "
                         "-- WIDTH for vstack, HEIGHT for hstack)")
    ap.add_argument("--audio-from", default=None, metavar="PATH",
                    help="explicit audio source (default: the merged "
                         "file; use this to pull audio from the original "
                         "clip if _merged has none)")
    ap.add_argument("--no-audio", action="store_true",
                    help="drop audio (silent output)")
    PC.add_args(ap)
    argv = list(sys.argv[1:] if argv is None else argv)
    _cfg = PC.apply(ap, "combine_viz", argv)
    a = ap.parse_args(argv)
    if _cfg:
        print(f"combine_viz: config {_cfg}", flush=True)
    run(a.viz, merged=a.merged, out=a.out, layout=a.layout,
        size=a.size, audio_from=a.audio_from, no_audio=a.no_audio)


if __name__ == "__main__":
    main()
