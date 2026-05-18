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

_MESH_READERS = {".vtp": "vtkXMLPolyDataReader", ".obj": "vtkOBJReader",
                 ".stl": "vtkSTLReader", ".ply": "vtkPLYReader",
                 ".vtk": "vtkPolyDataReader"}


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


# Azimuth (deg) of each named view, orbiting the camera about +Y.
# Tuned for the combined_body_model axes so "front" faces the anterior.
# These views share the requested elevation (default 0 = eye level, a
# level horizontal line of sight); top/bottom set their own.
_VIEW_AZ = {"front": 90, "back": 270, "left": 180, "right": 0,
            "iso": 135}


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
        views=("front", "left", "right", "back"), layout="row",
        azimuth=0.0, elevation=0.0, zoom=0.85, geometry=(),
        moment=None, moment_norm="coord", colorbar=True):
    """Render the posed model frame-by-frame to an .mp4; returns its path.

    `views` are drawn as tiled panels in one frame (one renderer per
    view, all sharing the same actors -- geometry is posed once and
    drawn from each camera). `layout` is 'row' (side by side) or
    'grid'. `azimuth`/`elevation` shift every view together; `size`
    (W,H) is the whole frame (default: 1280x720 for a single view, else
    460x760 per panel tiled).

    Bone meshes are loaded once and only re-posed per frame (fast); the
    muscle polyline set is rebuilt each frame from the wrapped path.

    If an Inverse Dynamics `.sto` is given (`moment`) or auto-detected
    next to the .mot (the `.id.sto` run_id.py writes), muscles are
    colored cool->hot by the net joint moment of the coordinate each
    one most strongly actuates (per-muscle |moment arm| argmax, mapped
    once at a representative frame). NOTE: this is a *joint-load*
    projection onto muscles, NOT true per-muscle force -- ID gives net
    joint moments, not the muscle-force decomposition (that needs
    Static Optimization; see src/PLAN_muscle_forces.md). `moment_norm`
    'coord' scales each coordinate to its own clip-peak |moment|;
    'global' uses one peak across all mapped coordinates."""
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
    views_r = _resolve_views(views, azimuth, elevation)
    cols, grows = _grid(len(views_r), layout)
    if size is not None:
        w, h = _even(size[0]), _even(size[1])
    elif len(views_r) == 1:
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

    # ---- muscle moment coloring (optional) -------------------------
    # Color each muscle cool->hot by the net joint moment (from an
    # Inverse Dynamics .sto) of the coordinate it most strongly
    # actuates. This is a JOINT-LOAD projection onto muscles, not the
    # muscle-force decomposition (that needs Static Optimization --
    # see src/PLAN_muscle_forces.md).
    ms = osim_model.getMuscles() if muscles else None
    musc_list = [ms.get(i) for i in range(ms.getSize())] if ms else []
    color_by_moment = False
    mus_col = []          # per-muscle ID-data index, -1 if unmapped
    id_times = id_rows = mom_norm = None
    if musc_list:
        if moment is not None:
            mom_path = Path(moment).resolve()
            if not mom_path.exists():
                raise FileNotFoundError(mom_path)
        else:
            mom_path = _id_sto_for(mot)
        if mom_path.exists():
            color_by_moment = True
            idsto = osim.Storage(str(mom_path))
            il = idsto.getColumnLabels()
            id_coord_di = {}              # coord name -> ID data index
            for i in range(1, il.getSize()):
                lab = cn = il.get(i)
                for suf in ("_moment", "_force"):
                    if lab.endswith(suf):
                        cn = lab[: -len(suf)]
                        break
                id_coord_di[cn] = i - 1
            tarr, darr = [], []
            for k in range(idsto.getSize()):
                sv = idsto.getStateVector(k)
                tarr.append(sv.getTime())
                d = sv.getData()
                darr.append([d.get(j) for j in range(d.getSize())])
            id_times = np.asarray(tarr)
            id_rows = np.asarray(darr)
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
                best_di, best_a = -1, 1e-4    # 0.1 mm moment-arm floor
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
            mom_norm = {}
            if moment_norm == "global":
                g = max((float(np.abs(sub[:, c]).max()) for c in used),
                        default=1.0) or 1.0
                mom_norm = {c: g for c in used}
            else:
                mom_norm = {c: (float(np.abs(sub[:, c]).max()) or 1.0)
                            for c in used}
            n_mapped = sum(1 for c in mus_col if c >= 0)
        elif moment is None:
            print(f"note: no ID .sto next to the .mot "
                  f"({_id_sto_for(mot).name}) -- muscles drawn flat "
                  f"red. Run src/run_id.py first, or pass --moment.")

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
                scalar_bar.SetTitle("joint moment\n(frac peak)")
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

    # Reserve a right-edge gutter for the shared colorbar so it never
    # overlaps a panel.
    xspan = 0.92 if scalar_bar is not None else 1.0
    base_rens = []
    for i, (label, vaz, vel) in enumerate(views_r):
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
        txt.SetInput(label)
        txt.GetTextProperty().SetFontSize(
            max(12, int(h / grows * 0.045)))
        txt.GetTextProperty().SetColor(0.85, 0.90, 1.0)
        tpc = txt.GetPositionCoordinate()
        tpc.SetCoordinateSystemToNormalizedViewport()
        tpc.SetValue(0.03, 0.93)
        r.AddActor2D(txt)
        r.ResetCamera()
        cam = r.GetActiveCamera()
        cam.SetViewUp(0, 1, 0)
        cam.Azimuth(vaz)
        cam.Elevation(vel)
        cam.OrthogonalizeViewUp()
        cam.Zoom(zoom)
        r.ResetCameraClippingRange()
        rw.AddRenderer(r)
        base_rens.append(r)

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
        for t, data in rows:
            pose(data)
            for ac, frame in mesh_actors:
                ac.SetUserMatrix(
                    _vtk_matrix(frame.getTransformInGround(state)))
            if musc_actor is not None:
                update_muscles(t)
            for r in base_rens:
                r.ResetCameraClippingRange()
            rw.Render()
            w2i.Modified()
            w2i.Update()
            img = w2i.GetOutput()
            dims = img.GetDimensions()
            arr = vtk_to_numpy(
                img.GetPointData().GetScalars()).reshape(
                    dims[1], dims[0], -1)
            writer.append_data(np.flipud(arr).copy())
            if bar is not None:
                bar.update(1)
    finally:
        writer.close()
        if bar is not None:
            bar.close()

    print(f"model : {model_path}")
    print(f"mot   : {mot}")
    print(f"video : {out}")
    print(f"views : {', '.join(n for n, _, _ in views_r)}  "
          f"({layout}, {cols}x{grows})")
    print(f"frames: {len(rows)} @ {fps} fps  |  size {w}x{h}  |  "
          f"meshes {len(mesh_actors)}  |  muscles "
          f"{len(musc_list) if muscles else 0}")
    if color_by_moment:
        print(f"muscle color : joint moment from {mom_path.name}  |  "
              f"mapped {n_mapped}/{len(musc_list)} muscles  |  norm "
              f"{moment_norm}  (joint-load projection, NOT muscle "
              f"force -- see src/PLAN_muscle_forces.md)")
    elif muscles:
        print("muscle color : flat red (no ID .sto)")
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
    ap.add_argument("--views", default="front,left,right,back",
                    help="comma list of views to tile: front,back,left,"
                         "right,iso,top,bottom or explicit AZ[:EL] deg "
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
    ap.add_argument("--moment", default=None, metavar="STO",
                    help="Inverse Dynamics .sto to color muscles by "
                         "(default: auto-detect the <root>.id.sto "
                         "run_id.py writes next to the .mot; if none, "
                         "muscles are flat red). Colors each muscle "
                         "cool->hot by the net moment of the joint it "
                         "most strongly actuates -- a joint-load "
                         "projection, NOT true muscle force")
    ap.add_argument("--moment-norm", choices=("coord", "global"),
                    default="coord",
                    help="color scaling: 'coord' = each joint to its "
                         "own clip-peak |moment| (default); 'global' = "
                         "one peak across all mapped joints")
    ap.add_argument("--no-colorbar", action="store_true",
                    help="hide the moment colorbar overlay")
    a = ap.parse_args(argv)
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
        geometry=a.geometry, moment=a.moment,
        moment_norm=a.moment_norm, colorbar=not a.no_colorbar)


if __name__ == "__main__":
    main()
