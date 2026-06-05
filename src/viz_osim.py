#!/usr/bin/env python3
"""Render an OpenSim model + .mot motion to a musculoskeletal video.

    .venv/bin/python src/viz_osim.py \\
        models/combined_body_model/combined_body_model.osim \\
        runs/<name>/<name>.ik.mot

The visual playback step, run after run_ik.py (and optionally on the
same .mot you feed run_id.py). Head-less: VTK off-screen renders the
model's bone meshes (.vtp) and every muscle path, posed per frame from
the .mot coordinates, then imageio/ffmpeg encodes an .mp4 -- no GUI,
no display, no Simbody visualizer window.

Needs `opensim` AND `vtk`; both are in the project .venv (see
docs/SETUP.md), same as run_ik.py / run_id.py.
Geometry meshes are resolved next to the model (`<model_dir>/Geometry`);
pass --geometry to add more search dirs.
"""
import argparse
import math
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mpipe_pipeline as M
import pipeline_config as PC

_MESH_READERS = {".vtp": "vtkXMLPolyDataReader", ".obj": "vtkOBJReader",
                 ".stl": "vtkSTLReader", ".ply": "vtkPLYReader",
                 ".vtk": "vtkPolyDataReader"}

# Minimal 5x7 bitmap font for the timeline panel's small labels
# (group tags, MM:SS ticks, frame number). Only the glyphs that panel
# needs are defined; unknown chars render blank.
_FONT5x7 = {
    "0": ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
    "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    "2": ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
    "3": ["11111", "00010", "00100", "00010", "00001", "10001", "01110"],
    "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
    "5": ["11111", "10000", "11110", "00001", "00001", "10001", "01110"],
    "6": ["00110", "01000", "10000", "11110", "10001", "10001", "01110"],
    "7": ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
    "8": ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
    "9": ["01110", "10001", "10001", "01111", "00001", "00010", "01100"],
    ":": ["00000", "00100", "00100", "00000", "00100", "00100", "00000"],
    "/": ["00001", "00010", "00010", "00100", "01000", "01000", "10000"],
    "-": ["00000", "00000", "00000", "11111", "00000", "00000", "00000"],
    " ": ["00000", "00000", "00000", "00000", "00000", "00000", "00000"],
    "R": ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
    "L": ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
    "U": ["10001", "10001", "10001", "10001", "10001", "10001", "01110"],
    "B": ["11110", "10001", "10001", "11110", "10001", "10001", "11110"],
    "H": ["10001", "10001", "10001", "11111", "10001", "10001", "10001"],
    "F": ["11111", "10000", "11110", "10000", "10000", "10000", "10000"],
}


def _draw_text(img, text, x, y, sc=1, color=(230, 235, 245)):
    """Stamp `text` into an HxWx3 uint8 array at top-left (x,y), each
    font pixel an sc x sc block. Clipped to the array bounds."""
    H, W = img.shape[:2]
    col = list(color)
    cx = x
    for ch in str(text).upper():
        g = _FONT5x7.get(ch)
        if g is not None:
            for ry, rowbits in enumerate(g):
                for rxi, bit in enumerate(rowbits):
                    if bit != "1":
                        continue
                    y0, x0 = y + ry * sc, cx + rxi * sc
                    y1, x1 = min(H, y0 + sc), min(W, x0 + sc)
                    if y0 < H and x0 < W and y1 > 0 and x1 > 0:
                        img[max(0, y0):y1, max(0, x0):x1] = col
        cx += (5 + 1) * sc
    return cx


def _blue_red(v01):
    """Map an array of [0,1] values to the same blue->red ramp as the
    muscle LUT (HSV hue 0.667->0.0, s=v=1). Returns uint8 (...,3)."""
    import numpy as np
    v = np.clip(np.asarray(v01, float), 0.0, 1.0)
    h = (1.0 - v) * (2.0 / 3.0)            # 0->red, 1(low)->blue
    i = np.floor(h * 6.0).astype(int)
    f = h * 6.0 - i
    i = i % 6
    x = 1.0 - np.abs((h * 6.0) % 2.0 - 1.0)
    one = np.ones_like(v)
    zero = np.zeros_like(v)
    r = np.select([i == 0, i == 1, i == 2, i == 3, i == 4, i == 5],
                  [one, x, zero, zero, x, one])
    g = np.select([i == 0, i == 1, i == 2, i == 3, i == 4, i == 5],
                  [x, one, one, x, zero, zero])
    b = np.select([i == 0, i == 1, i == 2, i == 3, i == 4, i == 5],
                  [zero, zero, x, one, one, x])
    return (np.stack([r, g, b], -1) * 255.0 + 0.5).astype("uint8")


def _hand_or_body_group(name):
    """Coordinate name -> timeline band: 'rh','lh','lb' (lower body:
    pelvis + legs) or 'ub' (everything else: trunk/neck/arms)."""
    n = name.lower()
    if n.startswith("rh_"):
        return "rh"
    if n.startswith("lh_"):
        return "lh"
    low = ("pelvis", "hip_", "knee", "ankle", "subtalar", "mtp",
           "femur", "tibia", "calcn", "walker_knee", "patellofemoral")
    if any(k in n for k in low):
        return "lb"
    return "ub"


def viz_out_path(mot):
    """Output .mp4 next to the motion: a trailing `.ik.mot` -> `.viz.mp4`
    (so `<root>.roihands.ik.mot` -> `<root>.roihands.viz.mp4`); any
    other `.mot`/`.sto`/name just gains `.viz.mp4`."""
    mot = Path(mot)
    name = mot.name
    for suf in (".ik.mot", ".mot", ".sto"):
        if name.endswith(suf):
            return mot.with_name(name[: -len(suf)] + ".viz.mp4")
    return mot.with_name(name + ".viz.mp4")


def _geom_search_dirs(model_path, extra):
    md = Path(model_path).resolve().parent
    dirs = [md / "Geometry", md, md.parent / "Geometry",
            M.DEFAULT_MODEL.parent / "Geometry"]
    dirs = [Path(d) for d in (list(extra) + dirs)]
    seen, out = set(), []
    for d in dirs:
        d = d.resolve()
        if d not in seen and d.is_dir():
            seen.add(d)
            out.append(d)
    return out


def _find_geom(fname, search_dirs):
    base = Path(fname).name
    for d in search_dirs:
        p = d / base
        if p.exists():
            return p
    return None


def _vtk_matrix(transform):
    """SimTK Transform (ground) -> vtkMatrix4x4."""
    import vtk
    R, p = transform.R(), transform.p()
    m = vtk.vtkMatrix4x4()
    for i in range(3):
        for j in range(3):
            m.SetElement(i, j, R.get(i, j))
        m.SetElement(i, 3, p.get(i))
    return m


def _id_sto_for(mot):
    """The .id.sto run_id.py would have written for this .mot (same
    suffix convention), for auto-detecting a moment file."""
    name = Path(mot).name
    for suf in (".ik.mot", ".mot", ".sto"):
        if name.endswith(suf):
            return Path(mot).with_name(name[: -len(suf)] + ".id.sto")
    return Path(mot).with_name(name + ".id.sto")


def _so_force_sto_for(mot):
    """The .so_force.sto run_so.py would have written for this .mot
    (same suffix convention), for auto-detecting Static-Optimization
    per-muscle forces -- the true muscle-force colour source."""
    name = Path(mot).name
    for suf in (".ik.mot", ".mot", ".sto"):
        if name.endswith(suf):
            return Path(mot).with_name(name[: -len(suf)] + ".so_force.sto")
    return Path(mot).with_name(name + ".so_force.sto")


# Azimuth (deg) of each named view, orbiting the camera about +Y.
# Tuned for the combined_body_model axes so "front" faces the anterior.
# These views share the requested elevation (default 0 = eye level, a
# level horizontal line of sight); top/bottom set their own.
_VIEW_AZ = {"front": 90, "back": 270, "left": 180, "right": 0,
            "iso": 135}

# Hand-focused panels: token -> (side, wrist, index-MCP, pinky-MCP
# marker names). The MCP markers define the optional orientation
# anchor: WRIST->INDEX_MCP is image-up, WRIST->PINKY_MCP completes the
# palm plane (its normal is the camera axis -> palm faces the viewer).
_HAND_PANELS = {
    "rhand": ("r", "RH_WRIST", "RH_INDEX_FINGER_MCP", "RH_PINKY_MCP",
              "RH_THUMB_CMC"),
    "lhand": ("l", "LH_WRIST", "LH_INDEX_FINGER_MCP", "LH_PINKY_MCP",
              "LH_THUMB_CMC"),
}
_HAND_CAMDIST = 1.0   # camera standoff (m); parallel proj -> size set
                      # by parallel scale, distance only sets clipping
_HAND_WRIST_Y = 0.18  # wrist's height in the panel (0=bottom, 0.5=mid):
                      # anchored low so the hand fills upward, not cropped
_HAND_SLAB = 0.14     # render only +-this (m) of depth around the wrist
                      # so the head/torso (a violin LH sits by the chin)
                      # doesn't occlude the hand


def _resolve_views(views, base_az, base_el):
    """[name,...] -> [(label, azimuth, elevation), ...]. Names:
    front/back/left/right/iso/top/bottom, or an explicit "AZ" / "AZ:EL"
    (degrees). `base_az`/`base_el` shift every view together."""
    out = []
    for v in views:
        v = str(v).strip().lower()
        if not v:
            continue
        if v == "top":
            out.append((v, base_az + 90.0, -89.0))
        elif v == "bottom":
            out.append((v, base_az + 90.0, 89.0))
        elif v in _VIEW_AZ:
            out.append((v, base_az + _VIEW_AZ[v], base_el))
        else:
            try:
                if ":" in v:
                    a, e = v.split(":", 1)
                    out.append((v, base_az + float(a), float(e)))
                else:
                    out.append((v, base_az + float(v), base_el))
            except ValueError:
                raise ValueError(
                    f"unknown view '{v}' -- use front/back/left/right/"
                    f"iso/top/bottom or an explicit AZ[:EL]")
    if not out:
        raise ValueError("no views requested")
    return out


def _grid(n, layout):
    """(cols, rows) tiling for n panels: a single row, or a near-square
    grid."""
    if layout == "grid":
        cols = int(math.ceil(math.sqrt(n)))
        return cols, int(math.ceil(n / cols))
    return n, 1


def _even(v):
    return max(2, (int(v) // 2) * 2)


def run(model, mot, out=None, fps=None, size=None,
        start=None, end=None, stride=1, muscles=True,
        views=("front", "left", "lhand", "rhand"), layout="row",
        azimuth=0.0, elevation=0.0, zoom=0.85, geometry=(),
        moment=None, force=None, moment_norm="coord", colorbar=True,
        hand_track_wrist=True, hand_orient=True, hand_field=0.26,
        timeline=True):
    """Render the posed model frame-by-frame to an .mp4; returns its path.

    `views` are drawn as tiled panels in one frame (one renderer per
    view, all sharing the same actors -- geometry is posed once and
    drawn from each camera). `layout` is 'row' (side by side) or
    'grid'. `azimuth`/`elevation` shift every view together; `size`
    (W,H) is the whole frame (default: 1280x720 for a single view, else
    460x760 per panel tiled).

    A `rhand`/`lhand` view is a hand-focused panel: an orthographic
    camera framing just that hand (`hand_field` metres tall). By
    default it tracks the wrist (`hand_track_wrist` -- the wrist stays
    anchored at panel centre while the arm moves); `hand_orient`
    additionally locks the hand's orientation so the palm faces the
    viewer (WRIST->INDEX_FINGER_MCP is image-up, WRIST->PINKY_MCP
    completes the palm plane) -- isolating pure finger articulation.
    Disable tracking (`hand_track_wrist=False`) for a world-fixed
    close-up where the hand drifts as the arm moves.

    Bone meshes are loaded once and only re-posed per frame (fast); the
    muscle polyline set is rebuilt each frame from the wrapped path.

    Muscles are colored cool->hot from one of two scalar sources
    (preference order):
      * `force` -- a Static Optimization force `.sto` (`.so_force.sto`
        from run_so.py): the TRUE per-muscle force, read straight from
        each muscle's own column. Preferred; auto-detected next to the
        .mot if not passed explicitly.
      * `moment` -- an Inverse Dynamics `.sto` (`.id.sto` from run_id.py):
        each muscle takes the net joint moment of the coordinate it most
        strongly actuates (per-muscle |moment arm| argmax, mapped once at
        a representative frame). This is a *joint-load* projection onto
        muscles, NOT a true muscle force -- a fallback used only when no
        SO force file is available.
    Explicit `force`/`moment` win over auto-detection; with neither and
    nothing found, muscles draw flat red. `moment_norm` 'coord' scales
    each muscle/coordinate to its own clip-peak; 'global' uses one peak
    across all mapped muscles/coordinates.

    `timeline` (default True) adds a thin strip across the bottom: a
    joint-moment heatmap over the whole clip (rows grouped top->bottom
    right-hand / left-hand / upper body / lower body, same blue->red
    scale), an MM:SS time axis, a moving current-frame indicator, and the
    running frame number at the right. It is a joint-moment view, so it
    needs the ID `.sto` -- loaded INDEPENDENTLY of the muscle colouring
    (auto-detected `<root>.id.sto`, or `moment`), so it shows even when
    muscles are coloured by SO force; skipped with a note only if no ID
    `.sto` is found."""
    import numpy as np
    import opensim as osim
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy

    # imageio's ffmpeg plugin; reuse the system ffmpeg if present so no
    # binary is downloaded.
    sysff = shutil.which("ffmpeg")
    if sysff and not os.environ.get("IMAGEIO_FFMPEG_EXE"):
        os.environ["IMAGEIO_FFMPEG_EXE"] = sysff
    import imageio.v2 as imageio

    model_path = Path(model).resolve()
    mot = Path(mot).resolve()
    for p in (model_path, mot):
        if not p.exists():
            raise FileNotFoundError(p)
    out = Path(out).resolve() if out else viz_out_path(mot).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    # Ordered panels: body views (fixed orbit camera) and hand views
    # (per-frame orthographic camera on one hand). Order is preserved
    # for the tiling.
    panels = []
    for v in views:
        vl = str(v).strip().lower()
        if vl in _HAND_PANELS:
            side, wn, im, pm, tcmc = _HAND_PANELS[vl]
            panels.append({"kind": "hand", "label": vl, "side": side,
                           "wrist": wn, "imk": im, "pmk": pm,
                           "tcmc": tcmc, "psign": 0,
                           "az": azimuth + _VIEW_AZ["front"],
                           "el": elevation})
        elif vl:
            lab, az, el = _resolve_views([v], azimuth, elevation)[0]
            panels.append({"kind": "body", "label": lab,
                           "az": az, "el": el})
    if not panels:
        raise ValueError("no views requested")
    cols, grows = _grid(len(panels), layout)
    if size is not None:
        w, h = _even(size[0]), _even(size[1])
    elif len(panels) == 1:
        w, h = 1280, 720
    else:
        w, h = _even(460 * cols), _even(760 * grows)
    search_dirs = _geom_search_dirs(model_path, geometry)

    osim_model = osim.Model(str(model_path))
    state = osim_model.initSystem()

    # ---- motion ----------------------------------------------------
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

    rows = []
    for k in range(sto.getSize()):
        sv = sto.getStateVector(k)
        t = sv.getTime()
        if start is not None and t < start:
            continue
        if end is not None and t > end:
            break
        rows.append((t, sv.getData()))
    rows = rows[::max(1, int(stride))]
    if not rows:
        raise RuntimeError("no frames in the requested time range")
    if fps is None:
        if len(rows) > 1:
            dt = (rows[-1][0] - rows[0][0]) / (len(rows) - 1)
            fps = round(1.0 / dt) if dt > 0 else 30
        else:
            fps = 30
    fps = max(1, int(fps))

    def pose(data):
        for c, di, is_rot in cmap:
            v = data.get(di)
            if is_rot and in_deg:
                v = math.radians(v)
            try:
                c.setValue(state, v, False)
            except Exception:
                pass
        osim_model.realizePosition(state)

    # ---- muscle coloring (optional) --------------------------------
    # Two scalar sources, in preference order:
    #   "force"  -- Static Optimization per-muscle force (.so_force.sto
    #               from run_so.py): the TRUE muscle force, read straight
    #               from each muscle's own column by name.
    #   "moment" -- Inverse Dynamics net joint moment (.id.sto from
    #               run_id.py): a JOINT-LOAD projection (each muscle takes
    #               the moment of the coordinate it most strongly actuates
    #               by |moment arm|), NOT a true muscle force.
    # Explicit --force/--moment win; otherwise auto-detect the SO force
    # file first, then the ID moment file. Either way the per-frame render
    # consumes the same (mus_col -> column, mom_norm -> peak) machinery.
    ms = osim_model.getMuscles() if muscles else None
    musc_list = [ms.get(i) for i in range(ms.getSize())] if ms else []
    color_by_moment = False               # really "colour by a scalar"
    color_src = None                      # "force" | "moment"
    mus_col = []          # per-muscle data index into id_rows, -1 if none
    id_times = id_rows = mom_norm = id_coord_di = None
    if musc_list:
        mom_path = None
        if force is not None:
            mom_path, color_src = Path(force).resolve(), "force"
            if not mom_path.exists():
                raise FileNotFoundError(mom_path)
        elif moment is not None:
            mom_path, color_src = Path(moment).resolve(), "moment"
            if not mom_path.exists():
                raise FileNotFoundError(mom_path)
        elif _so_force_sto_for(mot).exists():
            mom_path, color_src = _so_force_sto_for(mot), "force"
        elif _id_sto_for(mot).exists():
            mom_path, color_src = _id_sto_for(mot), "moment"

        if mom_path is not None:
            color_by_moment = True
            srcsto = osim.Storage(str(mom_path))
            il = srcsto.getColumnLabels()
            lab2di = {il.get(i): i - 1 for i in range(1, il.getSize())}
            tarr, darr = [], []
            for k in range(srcsto.getSize()):
                sv = srcsto.getStateVector(k)
                tarr.append(sv.getTime())
                d = sv.getData()
                darr.append([d.get(j) for j in range(d.getSize())])
            id_times = np.asarray(tarr)
            id_rows = np.asarray(darr)

            if color_src == "force":
                # Direct per-muscle column match -- no moment-arm mapping.
                mus_col = [lab2di.get(mu.getName(), -1) for mu in musc_list]
            else:
                # Coordinate name -> ID data index (labels carry a
                # _moment/_force suffix); map each muscle to the coord it
                # most strongly actuates (max |moment arm|) at a
                # representative frame.
                id_coord_di = {}
                for lab, di in lab2di.items():
                    cn = lab
                    for suf in ("_moment", "_force"):
                        if lab.endswith(suf):
                            cn = lab[: -len(suf)]
                            break
                    id_coord_di[cn] = di
                cand = [(co, id_coord_di[co.getName()])
                        for co in (coords.get(j)
                                   for j in range(coords.getSize()))
                        if co.getName() in id_coord_di]
                pose(rows[len(rows) // 2][1])     # representative frame
                try:
                    from tqdm import tqdm
                    it = tqdm(musc_list, unit="musc",
                              desc="map muscles->joints")
                except ImportError:
                    it = musc_list
                for mu in it:
                    best_di, best_a = -1, 1e-4   # 0.1 mm moment-arm floor
                    for c, di in cand:
                        try:
                            a = abs(mu.computeMomentArm(state, c))
                        except Exception:
                            continue
                        if a > best_a:
                            best_a, best_di = a, di
                    mus_col.append(best_di)

            used = sorted({c for c in mus_col if c >= 0})
            lo, hi = rows[0][0], rows[-1][0]
            sel = (id_times >= lo - 1e-9) & (id_times <= hi + 1e-9)
            sub = id_rows[sel] if sel.any() else id_rows
            if moment_norm == "global":
                g = max((float(np.abs(sub[:, c]).max()) for c in used),
                        default=1.0) or 1.0
                mom_norm = {c: g for c in used}
            else:
                mom_norm = {c: (float(np.abs(sub[:, c]).max()) or 1.0)
                            for c in used}
            n_mapped = sum(1 for c in mus_col if c >= 0)
        elif force is None and moment is None:
            print(f"note: no SO force ({_so_force_sto_for(mot).name}) or "
                  f"ID moment ({_id_sto_for(mot).name}) .sto next to the "
                  f".mot -- muscles drawn flat red. Run src/run_so.py "
                  f"(or run_id.py), or pass --force / --moment.")

    # ---- timeline strip (default on) -------------------------------
    # A thin bottom heatmap of |joint moment| over the whole clip,
    # rows grouped right-hand / left-hand / upper body / lower body,
    # with an MM:SS axis + moving current-frame indicator. Built once
    # (static); per frame only the indicator + frame number change.
    # The strip is a joint-moment view, so it needs the ID .sto. When
    # colouring by moment that data is already loaded; when colouring by
    # SO force (or bones-only) we load the ID .sto here independently so
    # the timeline still shows alongside the force colours -- skipped
    # with a note only if no ID .sto can be found.
    tl_times = tl_rows = tl_coord_di = tl_sub = None
    if timeline:
        if color_src == "moment" and id_coord_di is not None:
            tl_times, tl_rows, tl_coord_di, tl_sub = (
                id_times, id_rows, id_coord_di, sub)
        else:
            tl_path = (Path(moment).resolve() if moment
                       else _id_sto_for(mot))
            if tl_path.exists():
                tsto = osim.Storage(str(tl_path))
                til = tsto.getColumnLabels()
                tl_coord_di = {}
                for i in range(1, til.getSize()):
                    lab = cn = til.get(i)
                    for suf in ("_moment", "_force"):
                        if lab.endswith(suf):
                            cn = lab[: -len(suf)]
                            break
                    tl_coord_di[cn] = i - 1
                tt, td = [], []
                for k in range(tsto.getSize()):
                    sv = tsto.getStateVector(k)
                    tt.append(sv.getTime())
                    d = sv.getData()
                    td.append([d.get(j) for j in range(d.getSize())])
                tl_times, tl_rows = np.asarray(tt), np.asarray(td)
                lo, hi = rows[0][0], rows[-1][0]
                ts = (tl_times >= lo - 1e-9) & (tl_times <= hi + 1e-9)
                tl_sub = tl_rows[ts] if ts.any() else tl_rows

    tl_on = bool(timeline and tl_coord_di is not None)
    tl_px = tl_base = tl_idx = None
    tl_geom = {}
    if timeline and not tl_on:
        print(f"note: --timeline needs an ID joint-moment .sto "
              f"({_id_sto_for(mot).name}); none found -> timeline skipped "
              f"(run src/run_id.py, or pass --moment / --no-timeline).")
    if tl_on:
        N = len(rows)
        tl_idx = [int(np.argmin(np.abs(tl_times - t))) for t, _ in rows]
        groups = [("rh", "RH"), ("lh", "LH"), ("ub", "UB"), ("lb", "LB")]
        gcols = {k: [] for k, _ in groups}
        for j in range(coords.getSize()):
            nm = coords.get(j).getName()
            if nm in tl_coord_di:
                gcols[_hand_or_body_group(nm)].append(tl_coord_di[nm])
        if moment_norm == "global":
            allc = [c for v in gcols.values() for c in v]
            gpk = (float(np.abs(tl_sub[:, allc]).max()) or 1.0) if allc \
                else 1.0
            peak = {c: gpk for c in allc}
        else:
            peak = {c: (float(np.abs(tl_sub[:, c]).max()) or 1.0)
                    for v in gcols.values() for c in v}
        sel_id = tl_rows[np.asarray(tl_idx)]            # (N, ncols)

        Lm, Rm, Tp, sep, Ah, band_h = 34, 8, 3, 1, 18, 16
        tl_px = _even(Tp + 4 * band_h + 3 * sep + Ah)
        hx0, hx1 = Lm, w - Rm
        hw = max(1, hx1 - hx0)
        cidx = (np.arange(hw) * N // max(1, hw)).clip(0, N - 1)
        base = np.zeros((tl_px, w, 3), "uint8")
        base[:] = (26, 28, 33)
        y = Tp
        for gi, (gk, glabel) in enumerate(groups):
            cl = gcols[gk]
            if cl:
                M = np.abs(sel_id[:, cl]).astype(float)        # (N, n)
                M = M / np.array([peak[c] for c in cl])         # norm
                rgb = _blue_red(M.T)                             # (n,N,3)
                n = rgb.shape[0]
                ridx = (np.arange(band_h) * n // band_h).clip(0, n - 1)
                band = rgb[ridx][:, cidx]                        # (bh,hw,3)
            else:
                band = np.full((band_h, hw, 3), 45, "uint8")
            base[y:y + band_h, hx0:hx1] = band
            _draw_text(base, glabel, 3, y + (band_h - 14) // 2, 2,
                       (210, 216, 228))
            y += band_h + sep
        ax_y = Tp + 4 * band_h + 3 * sep
        base[ax_y, hx0:hx1] = (90, 95, 105)                     # axis line
        t0, t1 = rows[0][0], rows[-1][0]
        for k in range(6):
            fx = hx0 + round(k / 5 * (hw - 1))
            base[ax_y:ax_y + 3, max(hx0, fx - 1):fx + 1] = (140, 146, 158)
            sec = int(round((t0 + k / 5 * (t1 - t0)) - t0))
            lab = f"{sec // 60:02d}:{sec % 60:02d}"
            # Skip the leftmost label (00:00) -- the frame counter
            # lives there now; the tick mark still shows the start.
            if k == 0:
                continue
            tx = min(w - 1 - len(lab) * 6, max(0, fx - len(lab) * 3))
            _draw_text(base, lab, tx, ax_y + 5, 1, (170, 176, 188))
        tl_base = base
        tl_geom = dict(N=N, hx0=hx0, hx1=hx1, hw=hw, ax_y=ax_y,
                       y_top=Tp, y_bot=ax_y)

    # ---- scene -----------------------------------------------------
    bg = (0.10, 0.11, 0.13)
    rw = vtk.vtkRenderWindow()
    rw.SetOffScreenRendering(1)
    rw.SetSize(w, h)
    rw.SetNumberOfLayers(2)

    # Bone meshes: reader -> bake scale once -> actor; per-frame only
    # the actor's 4x4 ground matrix changes. Actors are shared by every
    # view's renderer (posed once, drawn from each camera).
    mesh_actors, missing = [], []
    for c in osim_model.getComponentsList():
        if c.getConcreteClassName() != "Mesh":
            continue
        mesh = osim.Mesh.safeDownCast(c)
        gf = _find_geom(mesh.getGeometryFilename(), search_dirs)
        if gf is None:
            missing.append(mesh.getGeometryFilename())
            continue
        rcls = _MESH_READERS.get(gf.suffix.lower())
        if rcls is None:
            missing.append(gf.name)
            continue
        rd = getattr(vtk, rcls)()
        rd.SetFileName(str(gf))
        sf = mesh.get_scale_factors()
        tf = vtk.vtkTransform()
        tf.Scale(sf.get(0), sf.get(1), sf.get(2))
        tpd = vtk.vtkTransformPolyDataFilter()
        tpd.SetTransform(tf)
        tpd.SetInputConnection(rd.GetOutputPort())
        mp = vtk.vtkPolyDataMapper()
        mp.SetInputConnection(tpd.GetOutputPort())
        mp.ScalarVisibilityOff()
        ac = vtk.vtkActor()
        ac.SetMapper(mp)
        pr = ac.GetProperty()
        pr.SetColor(0.93, 0.91, 0.85)
        pr.SetSpecular(0.2)
        pr.SetSpecularPower(15)
        mesh_actors.append((ac, mesh.getFrame()))

    if not mesh_actors:
        raise RuntimeError(
            f"no bone meshes resolved (looked in: "
            f"{', '.join(str(d) for d in search_dirs)}). Pass --geometry "
            f"<dir> pointing at the model's .vtp folder.")

    # Muscle paths: one polydata of polylines, refilled per frame.
    musc_actor, scalar_bar = None, None
    if musc_list:
        mpd = vtk.vtkPolyData()
        mpd.SetPoints(vtk.vtkPoints())
        mpd.SetLines(vtk.vtkCellArray())
        mm = vtk.vtkPolyDataMapper()
        mm.SetInputData(mpd)
        musc_actor = vtk.vtkActor()
        musc_actor.SetMapper(mm)
        mpr = musc_actor.GetProperty()
        mpr.SetLineWidth(2.0)
        mpr.SetLighting(False)
        if color_by_moment:
            lut = vtk.vtkLookupTable()
            lut.SetHueRange(0.667, 0.0)   # blue (low) -> red (high)
            lut.SetSaturationRange(1.0, 1.0)
            lut.SetValueRange(1.0, 1.0)
            lut.SetNumberOfColors(256)
            lut.SetRange(0.0, 1.0)
            lut.Build()
            mm.SetLookupTable(lut)
            mm.SetScalarModeToUseCellData()
            mm.SetColorModeToMapScalars()
            mm.SetScalarRange(0.0, 1.0)
            mm.ScalarVisibilityOn()
            if colorbar:
                scalar_bar = vtk.vtkScalarBarActor()
                scalar_bar.SetLookupTable(lut)
                scalar_bar.SetTitle(
                    ("muscle force" if color_src == "force"
                     else "joint moment") + "\n(frac peak)")
                scalar_bar.SetNumberOfLabels(5)
                scalar_bar.SetMaximumWidthInPixels(70)
                pc = scalar_bar.GetPositionCoordinate()
                pc.SetCoordinateSystemToNormalizedViewport()
                pc.SetValue(0.935, 0.22)
                scalar_bar.SetWidth(0.045)
                scalar_bar.SetHeight(0.56)
                scalar_bar.GetLabelTextProperty().SetColor(1, 1, 1)
                scalar_bar.GetTitleTextProperty().SetColor(1, 1, 1)
        else:
            mm.ScalarVisibilityOff()
            mpr.SetColor(0.80, 0.10, 0.10)

    def update_muscles(t):
        pts = vtk.vtkPoints()
        cells = vtk.vtkCellArray()
        scal = vtk.vtkFloatArray() if color_by_moment else None
        if color_by_moment:
            row = id_rows[int(np.argmin(np.abs(id_times - t)))]
        for idx, mu in enumerate(musc_list):
            try:
                path = mu.getGeometryPath().getCurrentPath(state)
            except Exception:
                continue
            n = path.getSize()
            if n < 2:
                continue
            ids = []
            for j in range(n):
                g = path.get(j).getLocationInGround(state)
                ids.append(pts.InsertNextPoint(g.get(0), g.get(1),
                                               g.get(2)))
            cells.InsertNextCell(len(ids))
            for pid in ids:
                cells.InsertCellPoint(pid)
            if color_by_moment:
                c = mus_col[idx]
                if c >= 0:
                    v = abs(row[c]) / mom_norm.get(c, 1.0)
                    scal.InsertNextValue(
                        0.0 if v < 0.0 else (1.0 if v > 1.0 else v))
                else:
                    scal.InsertNextValue(0.0)
        mpd.SetPoints(pts)
        mpd.SetLines(cells)
        if color_by_moment:
            scal.SetName("load")
            mpd.GetCellData().SetScalars(scal)
        mpd.Modified()

    # One renderer per view, tiled into the window; all share the same
    # actors (cameras differ, the posed geometry does not). Pose frame
    # 0 first so each ResetCamera frames the model.
    pose(rows[0][1])
    for ac, frame in mesh_actors:
        ac.SetUserMatrix(_vtk_matrix(frame.getTransformInGround(state)))
    if musc_actor is not None:
        update_muscles(rows[0][0])

    # Hand-panel camera support (markers -> per-frame orthographic aim).
    hand_panels = [p for p in panels if p["kind"] == "hand"]
    _aim_hand = None
    if hand_panels:
        mset = osim_model.getMarkerSet()
        have = {mset.get(i).getName() for i in range(mset.getSize())}
        need = set()
        for p in hand_panels:
            need |= {p["wrist"], p["imk"], p["pmk"], p["tcmc"]}
        miss = sorted(n for n in need if n not in have)
        if miss:
            raise RuntimeError(
                f"hand view needs markers {miss} on the model. The "
                f"combined model defines RH_/LH_ WRIST / "
                f"INDEX_FINGER_MCP / PINKY_MCP / THUMB_CMC -- use a "
                f"model that has them, or drop the rhand/lhand view.")

        def _mpos(name):
            g = mset.get(name).getLocationInGround(state)
            return np.array([g.get(0), g.get(1), g.get(2)])

        def _aim_hand(cam, d):
            W = _mpos(d["wrist"])
            if hand_orient:
                up = _mpos(d["imk"]) - W
                nu = float(np.linalg.norm(up))
                up = up / nu if nu > 1e-9 else np.array([0., 1., 0.])
                base = np.cross(up, _mpos(d["pmk"]) - W)
                nn = float(np.linalg.norm(base))
                base = (base / nn if nn > 1e-9
                        else np.array([0., 0., 1.]))
                # Palm-outward = the side the thumb base (CMC) is on:
                # it is anatomically palmar and near-rigid to the
                # wrist, so this self-corrects for hand chirality and
                # pose (no L/R hardcode). Decide once (rigid carpus)
                # to avoid per-frame sign flicker.
                if d["psign"] == 0:
                    thumb = _mpos(d["tcmc"]) - W
                    d["psign"] = (1.0 if float(np.dot(base, thumb)) >= 0
                                  else -1.0)
                nrm = base * d["psign"]
                cam.SetFocalPoint(*W)
                cam.SetPosition(*(W + nrm * _HAND_CAMDIST))
                cam.SetViewUp(*up)
            else:
                f = W if hand_track_wrist else d["W0"]
                cam.SetFocalPoint(*f)
                cam.SetPosition(f[0], f[1], f[2] + _HAND_CAMDIST)
                cam.SetViewUp(0, 1, 0)
                cam.Azimuth(d["az"])
                cam.Elevation(d["el"])
            cam.OrthogonalizeViewUp()
            # Anchor the wrist low in the panel: pan focal+camera up
            # along the view-up so the hand fills the frame upward
            # instead of being cropped at the top.
            vu = np.array(cam.GetViewUp(), float)
            sh = vu * ((0.5 - _HAND_WRIST_Y) * hand_field)
            cam.SetFocalPoint(*(np.array(cam.GetFocalPoint(), float) + sh))
            cam.SetPosition(*(np.array(cam.GetPosition(), float) + sh))
            cam.SetParallelScale(max(1e-3, hand_field / 2.0))
            # tight depth slab centred on the wrist (camera is
            # _HAND_CAMDIST from it) -> clip away the body/head
            cam.SetClippingRange(_HAND_CAMDIST - _HAND_SLAB,
                                 _HAND_CAMDIST + _HAND_SLAB)

    # Reserve a right-edge gutter for the shared colorbar so it never
    # overlaps a panel.
    xspan = 0.92 if scalar_bar is not None else 1.0
    body_rens, hand_rens = [], []
    for i, p in enumerate(panels):
        gx, gy = i % cols, i // cols
        # VTK viewport origin is bottom-left; put grid row 0 on top.
        x0, x1 = gx / cols * xspan, (gx + 1) / cols * xspan
        y0, y1 = 1.0 - (gy + 1) / grows, 1.0 - gy / grows
        r = vtk.vtkRenderer()
        r.SetBackground(*bg)
        r.SetViewport(x0, y0, x1, y1)
        r.SetLayer(0)
        for ac, _ in mesh_actors:
            r.AddActor(ac)
        if musc_actor is not None:
            r.AddActor(musc_actor)
        txt = vtk.vtkTextActor()
        txt.SetInput(p["label"])
        txt.GetTextProperty().SetFontSize(
            max(12, int(h / grows * 0.045)))
        txt.GetTextProperty().SetColor(0.85, 0.90, 1.0)
        tpc = txt.GetPositionCoordinate()
        tpc.SetCoordinateSystemToNormalizedViewport()
        tpc.SetValue(0.03, 0.93)
        r.AddActor2D(txt)
        cam = r.GetActiveCamera()
        if p["kind"] == "hand":
            cam.ParallelProjectionOn()
            p["W0"] = _mpos(p["wrist"])     # frame-0 wrist (no-track)
            _aim_hand(cam, p)
            hand_rens.append((cam, p))
        else:
            r.ResetCamera()
            cam.SetViewUp(0, 1, 0)
            cam.Azimuth(p["az"])
            cam.Elevation(p["el"])
            cam.OrthogonalizeViewUp()
            cam.Zoom(zoom)
            r.ResetCameraClippingRange()
            body_rens.append(r)
        rw.AddRenderer(r)

    # Single shared colorbar on a transparent full-window overlay so it
    # is drawn once, not once per panel.
    if scalar_bar is not None:
        ov = vtk.vtkRenderer()
        ov.SetLayer(1)
        ov.InteractiveOff()
        ov.AddActor2D(scalar_bar)
        rw.AddRenderer(ov)

    rw.Render()

    w2i = vtk.vtkWindowToImageFilter()
    w2i.SetInput(rw)
    w2i.SetInputBufferTypeToRGB()
    w2i.ReadFrontBufferOff()

    bar = None
    try:
        from tqdm import tqdm
        bar = tqdm(total=len(rows), unit="frame", desc="viz")
    except ImportError:
        pass

    writer = imageio.get_writer(
        str(out), fps=fps, codec="libx264", quality=7,
        macro_block_size=None, pixelformat="yuv420p")
    try:
        for i, (t, data) in enumerate(rows):
            pose(data)
            for ac, frame in mesh_actors:
                ac.SetUserMatrix(
                    _vtk_matrix(frame.getTransformInGround(state)))
            if musc_actor is not None:
                update_muscles(t)
            for r in body_rens:
                r.ResetCameraClippingRange()
            if hand_rens and (hand_track_wrist or hand_orient):
                for cam, d in hand_rens:
                    _aim_hand(cam, d)
            rw.Render()
            w2i.Modified()
            w2i.Update()
            img = w2i.GetOutput()
            dims = img.GetDimensions()
            arr = vtk_to_numpy(
                img.GetPointData().GetScalars()).reshape(
                    dims[1], dims[0], -1)
            disp = np.flipud(arr)
            if tl_on:
                g = tl_geom
                strip = tl_base.copy()
                frac = i / (g["N"] - 1) if g["N"] > 1 else 0.0
                xi = g["hx0"] + int(round(frac * (g["hw"] - 1)))
                strip[g["y_top"]:g["y_bot"],
                      max(g["hx0"], xi - 1):xi + 1] = (255, 255, 255)
                strip[g["ax_y"]:g["ax_y"] + 4,
                      max(g["hx0"], xi - 1):xi + 1] = (255, 255, 255)
                # Frame counter in the lower-left axis area (the 00:00
                # tick label was skipped to keep this clear); leaves
                # the end-of-clip MM:SS tick at the right visible.
                _draw_text(strip, str(i + 1), 3, g["ax_y"] + 4, 2,
                           (255, 255, 255))
                disp = np.vstack([disp, strip])
            writer.append_data(np.ascontiguousarray(disp))
            if bar is not None:
                bar.update(1)
    finally:
        writer.close()
        if bar is not None:
            bar.close()

    print(f"model : {model_path}")
    print(f"mot   : {mot}")
    print(f"video : {out}")
    print(f"views : {', '.join(p['label'] for p in panels)}  "
          f"({layout}, {cols}x{grows})")
    print(f"frames: {len(rows)} @ {fps} fps  |  size {w}x{h}  |  "
          f"meshes {len(mesh_actors)}  |  muscles "
          f"{len(musc_list) if muscles else 0}")
    if color_by_moment and color_src == "force":
        print(f"muscle color : SO muscle force from {mom_path.name}  |  "
              f"matched {n_mapped}/{len(musc_list)} muscles  |  norm "
              f"{moment_norm}  (true per-muscle force from Static "
              f"Optimization)")
    elif color_by_moment:
        print(f"muscle color : joint moment from {mom_path.name}  |  "
              f"mapped {n_mapped}/{len(musc_list)} muscles  |  norm "
              f"{moment_norm}  (joint-load projection, NOT muscle force "
              f"-- run src/run_so.py for true muscle forces)")
    elif muscles:
        print("muscle color : flat red (no SO force / ID moment .sto)")
    if tl_on:
        print(f"timeline : on  |  strip {tl_px}px (RH/LH/UB/LB joint-"
              f"moment heatmap)  |  out size {w}x{h + tl_px}")
    if missing:
        u = sorted(set(missing))
        print(f"skipped {len(u)} unresolved mesh file(s): "
              f"{', '.join(u[:8])}{' ...' if len(u) > 8 else ''}")
    return str(out)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Render an OpenSim model + .mot to a musculoskeletal "
                    "playback .mp4 (needs opensim+vtk; use the project "
                    ".venv -- see docs/SETUP.md). Run after run_ik.py.")
    ap.add_argument("model", help="OSIM model (.osim)")
    ap.add_argument("mot", help="motion coordinates (.mot/.sto from "
                                "run_ik.py)")
    ap.add_argument("--out", default=None,
                    help="output .mp4 (default: <root>.viz.mp4 next to "
                         "the .mot)")
    ap.add_argument("--fps", type=int, default=None,
                    help="playback fps (default: derived from the .mot "
                         "time step)")
    ap.add_argument("--size", default=None,
                    help="whole-frame size WxH (default: 1280x720 for "
                         "one view, else 460x760 per panel tiled)")
    ap.add_argument("--views", default="front,left,lhand,rhand",
                    help="comma list of panels to tile: front,back,left,"
                         "right,iso,top,bottom, explicit AZ[:EL] deg, or "
                         "rhand/lhand for a hand-focused close-up "
                         "(default: front,left,right,back)")
    ap.add_argument("--layout", choices=("row", "grid"), default="row",
                    help="'row' = side by side (default); 'grid' = "
                         "near-square")
    ap.add_argument("--start", type=float, default=None,
                    help="start time (s); default = mot start")
    ap.add_argument("--end", type=float, default=None,
                    help="end time (s); default = mot end")
    ap.add_argument("--stride", type=int, default=1,
                    help="render every Nth frame (default: 1)")
    ap.add_argument("--bones-only", action="store_true",
                    help="skip muscle paths (faster, bones only)")
    ap.add_argument("--azimuth", type=float, default=0.0,
                    help="extra azimuth (deg) added to every view, to "
                         "spin the whole rig (default: 0)")
    ap.add_argument("--elevation", type=float, default=0.0,
                    help="elevation (deg) for front/back/left/right/iso "
                         "(default: 0 = eye level, level horizontal line "
                         "of sight; top/bottom override it)")
    ap.add_argument("--zoom", type=float, default=0.85,
                    help="camera zoom; <1 widens the view to leave room "
                         "for motion (default: 0.85)")
    ap.add_argument("--geometry", action="append", default=[],
                    metavar="DIR",
                    help="extra geometry search dir, repeatable "
                         "(searched before <model_dir>/Geometry)")
    ap.add_argument("--force", default=None, metavar="STO",
                    help="Static Optimization force .sto to color muscles "
                         "by TRUE per-muscle force (default: auto-detect "
                         "the <root>.so_force.sto run_so.py writes next to "
                         "the .mot). Takes precedence over --moment; each "
                         "muscle is colored cool->hot by its own force "
                         "column")
    ap.add_argument("--moment", default=None, metavar="STO",
                    help="Inverse Dynamics .sto to color muscles by, used "
                         "only when no SO force file is given/found "
                         "(default: auto-detect the <root>.id.sto "
                         "run_id.py writes next to the .mot; if neither "
                         "exists, muscles are flat red). Colors each "
                         "muscle cool->hot by the net moment of the joint "
                         "it most strongly actuates -- a joint-load "
                         "projection, NOT true muscle force")
    ap.add_argument("--moment-norm", choices=("coord", "global"),
                    default="coord",
                    help="color scaling: 'coord' = each muscle/joint to "
                         "its own clip-peak (default); 'global' = one peak "
                         "across all mapped muscles/joints")
    ap.add_argument("--no-colorbar", action="store_true",
                    help="hide the color scale overlay")
    ap.add_argument("--hand-track-wrist",
                    action=argparse.BooleanOptionalAction, default=True,
                    help="rhand/lhand: keep the wrist anchored at the "
                         "panel centre as the arm moves (default: on; "
                         "--no-hand-track-wrist = world-fixed close-up, "
                         "hand drifts)")
    ap.add_argument("--hand-orient", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="rhand/lhand: also lock hand orientation so the "
                         "palm faces the viewer (WRIST->INDEX_FINGER_MCP "
                         "image-up, WRIST->PINKY_MCP completes the palm "
                         "plane) -- isolates finger articulation (default: "
                         "on; --no-hand-orient for a world-fixed close-up)")
    ap.add_argument("--hand-field", type=float, default=0.26,
                    metavar="M",
                    help="rhand/lhand framed height in metres "
                         "(orthographic; default: 0.26)")
    ap.add_argument("--timeline", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="thin bottom strip: a joint-moment heatmap over "
                         "the whole clip (right-hand / left-hand / upper "
                         "body / lower body), an MM:SS axis, a moving "
                         "current-frame indicator and the frame number "
                         "(default: on; needs an ID .sto, loaded "
                         "independently of the muscle colouring). "
                         "--no-timeline disables it.")
    PC.add_args(ap)
    argv = list(sys.argv[1:] if argv is None else argv)
    _cfg = PC.apply(ap, "viz_osim", argv)
    a = ap.parse_args(argv)
    if _cfg:
        print(f"viz_osim: config {_cfg}", flush=True)
    size = None
    if a.size is not None:
        try:
            size = tuple(int(x) for x in a.size.lower().split("x"))
            if len(size) != 2:
                raise ValueError
        except ValueError:
            ap.error("--size must be WxH, e.g. 1280x720")
    views = [v for v in a.views.split(",") if v.strip()]
    run(a.model, a.mot, out=a.out, fps=a.fps, size=size,
        start=a.start, end=a.end, stride=a.stride,
        muscles=not a.bones_only, views=views, layout=a.layout,
        azimuth=a.azimuth, elevation=a.elevation, zoom=a.zoom,
        geometry=a.geometry, moment=a.moment, force=a.force,
        moment_norm=a.moment_norm, colorbar=not a.no_colorbar,
        hand_track_wrist=a.hand_track_wrist, hand_orient=a.hand_orient,
        hand_field=a.hand_field, timeline=a.timeline)


if __name__ == "__main__":
    main()
