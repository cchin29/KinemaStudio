"""Optional shared pipeline config (``kinemastudio.yaml``).

Every stage script (mp2trc / run_ik / run_id / viz_osim) and the
``run_pipeline.py`` wrapper read the *same* YAML, so re-running one
stage standalone honours exactly the settings a full pipeline run
would. Precedence, low -> high:

    built-in argparse defaults
      <  the stage's section in the YAML
      <  explicit command-line flags

The file is discovered (unless ``--no-config``) from ``--config PATH``,
else ``./kinemastudio.yaml``, else ``<repo-root>/kinemastudio.yaml``.
Schema (every key optional)::

    # top level -- the run_pipeline.py wrapper's knobs; 'model' and
    # 'outdir' are also picked up by any stage that has them, so a
    # standalone run_ik.py uses the same model.
    outdir: runs
    model:  models/combined_body_model/combined_body_model.neck.osim
    no_id:  false
    no_viz: false

    # one block per stage: keys are that script's own --flags (dashes
    # or underscores both work). A scalar -> the flag's value; true /
    # false -> a store_true / BooleanOptionalAction flag; a nested
    # mapping -> a repeatable KEY=VALUE flag (e.g. run_ik marker-weight).
    run_ik:
      marker-weight:
        NOSE: 0.6
        LEFT_EYE: 0.1
"""
import argparse
from pathlib import Path

CONFIG_NAME = "kinemastudio.yaml"
_ROOT = Path(__file__).resolve().parent.parent  # repo root (parent of src/)

WRAPPER_KEYS = {"outdir", "model", "no_id", "no_viz", "scale", "so", "grf",
                "combine"}
STAGE_KEYS = {"mp2trc", "scale_model", "run_ik", "run_grf", "run_id",
              "run_so", "viz_osim", "combine_viz"}
# top-level keys a stage should also inherit if it has the matching dest
_SHARED_WITH_STAGES = {"model", "outdir"}


def add_args(ap):
    """Register --config / --no-config on a stage (or wrapper) parser."""
    ap.add_argument("--config", default=None, metavar="YAML",
                    help=f"pipeline-config YAML (default: ./{CONFIG_NAME} "
                         f"or the repo root); see pipeline_config.py")
    ap.add_argument("--no-config", action="store_true",
                    help=f"ignore the auto-discovered {CONFIG_NAME}")


def discover(explicit=None, no_config=False):
    """Resolve which config file to use, or None. An explicit path that
    does not exist is a hard error (a silent typo would be worse)."""
    if no_config:
        return None
    if explicit is not None:
        p = Path(explicit).expanduser()
        if not p.is_file():
            raise SystemExit(f"config not found: {p}")
        return p
    for p in (Path.cwd() / CONFIG_NAME, _ROOT / CONFIG_NAME):
        if p.is_file():
            return p
    return None


def _normalize(section, where):
    """A stage section -> {argparse_dest: value}. A nested mapping
    becomes the ["KEY=VALUE", ...] list an append-style flag expects
    (its keys stay verbatim -- marker names keep their underscores)."""
    if not isinstance(section, dict):
        raise SystemExit(f"config: '{where}' must be a mapping of "
                         f"flag: value (got {type(section).__name__})")
    out = {}
    for k, v in section.items():
        dest = str(k).replace("-", "_")
        if isinstance(v, dict):
            v = [f"{kk}={vv}" for kk, vv in v.items()]
        out[dest] = v
    return out


def load(stage, explicit=None, no_config=False):
    """Return (defaults: {dest: value}, path | None) for `stage`
    (a STAGE_KEYS name, or None for the run_pipeline wrapper)."""
    path = discover(explicit, no_config)
    if path is None:
        return {}, None
    import yaml
    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict):
        raise SystemExit(f"{path}: top level must be a mapping")
    bad = set(raw) - (WRAPPER_KEYS | STAGE_KEYS)
    if bad:
        raise SystemExit(f"{path}: unknown key(s) {sorted(bad)}. Valid: "
                         f"{sorted(WRAPPER_KEYS | STAGE_KEYS)}")
    if stage is None:                       # run_pipeline.py wrapper
        return {k: raw[k] for k in raw if k in WRAPPER_KEYS}, path
    defaults = {k: raw[k] for k in raw
                if k in WRAPPER_KEYS and k in _SHARED_WITH_STAGES}
    defaults.update(_normalize(raw.get(stage) or {}, stage))
    return defaults, path


def apply(ap, stage, argv):
    """Discover + load the config named by argv's --config/--no-config,
    validate its keys against `ap`, and install them as defaults so an
    explicit command-line flag still wins. Returns the path (or None)."""
    pre = argparse.ArgumentParser(add_help=False)
    add_args(pre)
    known, _ = pre.parse_known_args(argv)
    defaults, path = load(stage, known.config, known.no_config)
    if defaults:
        valid = {a.dest for a in ap._actions}
        # Shared wrapper keys (model/outdir) are injected into every
        # stage; they only *apply* to stages that expose that flag, so
        # silently drop the ones this stage lacks. A key from the
        # stage's OWN section that the stage doesn't accept is a real
        # typo and still errors.
        for k in list(defaults):
            if k in _SHARED_WITH_STAGES and k not in valid:
                del defaults[k]
        unknown = set(defaults) - valid - _SHARED_WITH_STAGES
        if unknown:
            label = stage or "top level"
            raise SystemExit(
                f"{path}: '{label}' has option(s) {sorted(unknown)} that "
                f"{stage or 'run_pipeline'} does not accept (check the "
                f"flag name against --help)")
        # argparse *appends* to an append-action default, so a YAML
        # default would otherwise leak in alongside an explicit CLI
        # value. If the user passed such a flag, let the CLI fully
        # replace the YAML (true "CLI wins"); else keep the YAML list.
        for act in ap._actions:
            if (isinstance(act, argparse._AppendAction)
                    and act.dest in defaults
                    and any(o == t or t.startswith(o + "=")
                            for o in act.option_strings for t in argv)):
                del defaults[act.dest]
        ap.set_defaults(**defaults)
    return path


def template():
    """A ready-to-edit kinemastudio.yaml (mirrors mp2trc's
    --dump-mp-config) -- the head-marker de-weighting is the worked
    example since that is the common fix."""
    rel = (_ROOT / "models" / "combined_body_model"
           / "combined_body_model.neck.osim")
    try:
        rel = rel.relative_to(_ROOT)
    except ValueError:
        pass
    return f"""\
# KinemaStudio pipeline config -- read by run_pipeline.py AND by every
# stage script (mp2trc / run_ik / run_id / viz_osim) when run standalone.
# Auto-loaded from ./{CONFIG_NAME} or the repo root; override with
# --config PATH, ignore with --no-config.
#
# Precedence (low -> high):
#   built-in defaults  <  values in THIS file  <  explicit CLI flags
#   (and, for mp2trc, anything after `--`).
# Every key is optional -- delete a line to keep the built-in default.

# --- shared / run_pipeline.py knobs ---
outdir: runs
model: {rel}
no_id: false      # true => skip Inverse Dynamics
no_viz: false     # true => skip the musculoskeletal render
combine: true     # after the render, stack the annotated source video
                  # above it + mux audio -> <stem>.combo.mp4
                  # (false => skip; auto-skipped when no_viz)
scale: false      # true => scale the model to the subject before IK
                  #         (measurement-based, from the Stage-1 TRC)
so: false         # true => Static Optimization (per-muscle forces)
                  #         after Inverse Dynamics
grf: false        # true => estimate ground reaction forces from the
                  #         kinematics (no force plates) before ID, so ID
                  #         /SO are dynamically consistent (base residual
                  #         ~0). Use the SAME lowpass in run_grf/id/so.

# --- per-stage extra flags (the stage script's own --names) ----------
mp2trc:
  hand-source: combined    # roi | combined | registered
  # pose-model: heavy
scale_model:        # only used when scale: true (or --scale)
  # start: 0.0           # averaging window start (s; default: clip start)
  # end: 5.0             # averaging window end   (s; default: clip end)
  # include-humerus: false   # humerus span is depth-corrupted; keep off
  # mass: -1.0           # final total mass kg (-1 = keep the model's)
  {{}}
run_ik:
  # MediaPipe over-weights the face: the head/upper body bends down to
  # chase noisy eye/ear/mouth points. Keep the nose, de-weight the rest.
  marker-weight:
    NOSE: 0.6
    LEFT_EYE_INNER: 0.1
    LEFT_EYE: 0.1
    LEFT_EYE_OUTER: 0.1
    RIGHT_EYE_INNER: 0.1
    RIGHT_EYE: 0.1
    RIGHT_EYE_OUTER: 0.1
    LEFT_EAR: 0.1
    RIGHT_EAR: 0.1
    MOUTH_LEFT: 0.1
    MOUTH_RIGHT: 0.1
run_grf:            # only used when grf: true (or --grf)
  # NB: with the wrapper, --grf passes one shared --lowpass to
  # run_grf/run_id/run_so (default 6) so their differentiation matches;
  # the per-stage lowpass below is for running a stage standalone.
  # lowpass: 6.0
  # height-thresh: 0.05  # contact-point height over floor (m) => in contact
  # vel-thresh: 0.8      # contact-point horiz speed (m/s) => planted
  {{}}
run_id:
  # lowpass: 6.0         # standalone only; --grf shares one cutoff for you
  {{}}
run_so:             # only used when so: true (or --so)
  # lowpass: 6.0        # filter coords before differentiation (Hz)
  # reserve-force: 1.0  # per-coordinate reserve actuator optimal force
  # activation-exponent: 2.0
  # jobs: 8             # solve frames in N parallel processes (speed-up)
  # stride: 1           # solve every Nth frame only (fast preview)
  {{}}
viz_osim:
  # views: front,left,right,back
  {{}}
"""
