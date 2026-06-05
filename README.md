# KinemaStudio

**Markerless biomechanics from a single video.** KinemaStudio turns an ordinary RGB clip into a driven OpenSim musculoskeletal simulation — pose and hand tracking with Google MediaPipe, fused into an OpenSim-ready marker trajectory, then solved up the dynamics stack: joint kinematics, joint moments, kinematics-estimated ground reaction forces, and individual **muscle forces** (Static Optimization), rendered as a muscle-force video. No motion-capture suit, no markers, no force plates — the ground reactions are recovered from the motion itself.

The walkthrough below is the bundled `demo/Wieniawski2` clip (solo violin performance, ~17 s, 500 frames @ 30 fps) carried through every stage.

---

## Problem & Insight

**The problem.** Muscle-level biomechanics is locked behind a six-figure room — marker-based motion-capture rigs, force plates, marker technicians, and OpenSim specialists. A lab like that studies a few dozen people a year, and the binding constraint isn't the analysis, it's *access to bodies*. That matters most for the parts of the body that move the most: hands. Playing-related musculoskeletal disorders affect **62–93% of musicians**, and **over 80% of professional orchestra players** suffer a wage-losing injury at some point in their career (Kok et al. 2015, 2018) — strain that today goes essentially unmeasured outside a lab.

**The insight / opportunity.** Hand *video* exists at enormous scale; hand *marker data* never will. If muscle-level analysis can be driven from a single ordinary clip, the subject pool jumps from dozens to anyone with a camera — turning a measurement only a few labs can make into something a violinist, a clinician, or a roboticist can run from a phone.

**Why this is original/ambitious.** The published markerless-biomechanics field clusters almost entirely on **gait and the lower body**. A targeted, multi-query PubMed search (June 2026) found **no study that estimates hand muscle force from video**; the one study that combines markerless video with a musculoskeletal model (Auer et al. 2024) reports the hand as its unreliable region (13.7° MAE, "wrist/hand tracking still needs refinement"). KinemaStudio's wedge is exactly that open, hard case: **markerless, single-camera, fine-motor *hand* musculoskeletal analysis** — built as a one-person, AI-leveraged lab. This project grows directly out of the author's published violin-vibrato work (Chin 2025, *ThinkYou?!*), generalizing it from a single-purpose one-hand vibrato analysis jupyter notebook into an integrated full musculoskeletal pipeline.

---

## Execution & Technical Work

**What's built (the substantial artifact).** A working, one-command pipeline — **≈4,800 lines of Python** across `src/` — that takes a single video and produces a driven, rendered OpenSim musculoskeletal simulation. On the validated demo clip it solves and renders **637 muscles** (43 of them in the hand) at a whole-body marker RMS of **0.037 m**. The stages below each run standalone or through `run_pipeline.py`.

### The pipeline

**1 · Input video** — a plain handheld recording, the only input the pipeline needs.
![input clip](docs/assets/01_input.gif)

**2 · MediaPipe tracking (`mp2trc.py`)** — Each frame goes through MediaPipe: a full-body pose plus a detailed 21-landmark HandLandmarker on an ROI cropped around each wrist. Landmarks are unified into one world frame, low-pass filtered, and written as an OpenSim `.trc` marker file with a generated IK setup. A **combine** step (`combine_hands.py`) recovers real hand data for frames the ROI detector dropped by mapping in MediaPipe Holistic's hand (via the same pose-wrist transform) instead of interpolating across the gap.
![mediapipe landmarks](docs/assets/02_mediapipe.gif)

**3 · Kinematics, dynamics & muscle forces (`run_ik.py` → `run_grf.py` → `run_id.py` → `run_so.py`)** — `run_ik.py` solves robust per-frame Inverse Kinematics of the `combined_body_model` against the markers (a non-converging frame holds the last good pose rather than aborting). `run_grf.py` estimates **ground reaction forces straight from the kinematics** (no force plates): the whole-body net wrench is fully determined by motion + inertia, computed from posed frames and attributed to the feet as an OpenSim `ExternalLoads`. `run_id.py` runs Inverse Dynamics for the net joint moments; with the estimated GRF applied, the free-base residual collapses to ≈0. `run_so.py` runs **Static Optimization**, resolving those moments into the individual **muscle forces** (minimizing summed activation² per frame under each muscle's force capacity). Scaling, GRF, and SO are each opt-in (`--scale` / `--grf` / `--so`).
![opensim render](docs/assets/03_opensim.gif)

**4 · Musculoskeletal render (`viz_osim.py`)** — Headless VTK renders the posed skeleton and every muscle path, tiled **front / left / lhand / rhand**, colored by true SO muscle force (blue → red) when present, with an optional whole-clip joint-moment timeline strip.

### Quick start (reproducible)

One Python 3.12 venv runs everything (OpenSim, MediaPipe, VTK) — see **[docs/SETUP.md](docs/SETUP.md)**:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -U pip wheel
.venv/bin/pip install -r requirements.txt
```

Run the whole pipeline on the bundled clip:

```bash
.venv/bin/python run_pipeline.py demo/Wieniawski2.mp4
# full dynamics + muscle forces:
.venv/bin/python run_pipeline.py demo/Wieniawski2.mp4 --scale --grf --so
```

Outputs land in `runs/Wieniawski2/`. Every script takes `--help`; `run_pipeline.py --debug` echoes each stage's exact, copy-pasteable invocation. Reused settings live in an optional `kinemastudio.yaml` that the wrapper and every stage script read identically (precedence: defaults < YAML < command-line).


---

## Evaluation & Evidence

**Validation.**
- **Whole-body marker fit:** 0.0374 m RMS on the validated clip — in the range of published single-camera methods. *Honest scope: this is a whole-body marker fit, not finger-level accuracy, and it is one clip (n=1).*
- **Dynamic consistency:** with kinematics-estimated GRF applied, the free-base vertical residual collapses from ~742 N (≈ one bodyweight) to ~10 N — a concrete, measurable check that the dynamics are self-consistent.
- **Ablation:** the Holistic-combine hand source recovers ~10% of dropped frames per hand at no solve cost (RMS 0.0374→0.0376 m) — i.e. real hand data rather than interpolation.

**A real finding (from the predecessor vibrato study, Chin 2025).** The instrument surfaced a non-obvious result: for sustainable musical technique, what matters isn't how fast a muscle *activates* but how fast it *releases* — the opposite of an athlete — and it localized **excess thumb-muscle tension during position shifts** (the APL/thumb muscle pressing into the neck of the violin). Evidence the pipeline produces interpretable biomechanics, not just pictures.

**Honest limitations.** Markerless capture has no directly measured muscle activity, force plates, or external loads: these are derived from motion based on biomechanical models. GRF and SO are **estimates**: SO leans on per-coordinate reserve actuators where the model is under-actuated (a large reserve flags low confidence), and the double-support GRF split is approximate (the net wrench is exact). **Treat the muscle forces as relative, not validated absolutes.** MediaPipe's shoulder is its noisiest landmark, so `scale_model.py` excludes the humerus from scaling.

**Where it sits vs. prior work.** This is not the first to markerless musculoskeletal: OpenCap (Uhlrich et al. 2023) does two-camera muscle dynamics 25× faster than a physical lab; single-camera (monocular) full-body musculoskeletal exists (Ueno 2024) but is primarily focused on gross motor motion. The defensible, narrower claim is the *hand* case (see Problem & Insight). The "tool → engine" flywheel is proven for the **neck** (as described in He et al. 2026: 1M simulated OpenSim models → a fast surrogate for 72 muscle forces, R>0.95 as reported) — not yet for hands.

---


### Repository layout

```
run_pipeline.py        one-command wrapper for the pipeline stages
src/                   pipeline scripts (mp2trc, combine_hands, scale_model,
                       run_ik/grf/id/so, viz_osim, combine_viz, pipeline_config)
models/                combined_body_model/ (.osim + Geometry/) · mediapipe/ (.task, auto-downloaded)
templates/             IK setup skeleton
docs/                  SETUP.md + asset GIFs
demo/Wieniawski2.mp4   bundled example clip (run output is regenerable)
requirements*.txt      pinned dependencies
```

### The model

`combined_body_model.neck.osim` is the default — a full-body OpenSim model with detailed articulated hands, posed from MediaPipe markers (75.59 kg default subject). It is derived from `combined_body_model.osim` by `scratchpad/build_neck_model.py`, which adds a lumped head–neck joint (a 4.3 kg `head` body on a 3-DoF `neck` joint) and re-parents the 13 MediaPipe head/face markers onto it, so a subject looking down at an instrument no longer forces spurious trunk flexion. Pass `--model` (or set `model:` in `kinemastudio.yaml`) to use a different `.osim`; the original headless model remains as `combined_body_model.osim`.

---

## Process, Integrity & Disclosure

### AI usage disclosure
This project was built with substantial AI assistance, disclosed here per the course AI Policy.
- **Code generation & pair-programming.** The pipeline (~4,800 lines across `src/`) was written with AI coding assistance (Claude Code) used for drafting functions, building the opensim hand models and integrating those with the existing Rajagopal body model, debugging the OpenSim static optimization solve, the GRF/SO math, the VTK render. All generated code was reviewed, run, and validated by me; I'm responsible for its correctness.
- **Debugging & solver tuning.** AI assistance helped diagnose a large number of issues during the design and implementation e.g. the non-converging scale step, marker-weight sweep, the neck-joint fix, SO reserve-actuator behavior; conclusions were confirmed empirically (marker RMS, base residual, rendered output), not taken on the model's word.
- **Research, writing & docs.** Literature/market research, parts of this README, and code comments were drafted/edited with AI assistance. Peer-reviewed citations were verified on PubMed (DOIs confirmed).
- **What AI did *not* do.** It did not operate the lab autonomously (the system is AI-*built*, not AI-*operated* yet pending additional development and testing) and did not generate the validation results (the 0.0374 m RMS, the 637-muscle solve, the residual collapse all come from real pipeline runs on real video).

### Credits & borrowed components (cited with what I changed)
KinemaStudio is built on open-source tools and prior models; this is not original-from-scratch infrastructure:
- **OpenSim** (Delp et al. 2007) — the musculoskeletal simulation engine (IK/ID/SO). Used as a dependency via its Python API.
- **OpenSim Creator** (Kewley, Beesel & Seth 2024) — model authoring/inspection.
- **MediaPipe** (Lugaresi et al. 2019) — pose + 21-landmark hand tracking. Runtime dependency.
- **Base musculoskeletal model** — `combined_body_model.osim` derives from the Rajagopal et al. 2016 full-body gait model for the body, a right hand model based on McFarland et al. 2023, which I mirrored to create a left-hand model (this was a significant modification of an early model from my prior work, Chin 2024). **My substantial changes:** In the current model for this repo, I've grafted the left and right hand models into the Rajagopal body gait model to create a full-body model with detailed bilateral hand musculature, and replaced the rigid neck joint to allow freedom of neck/head movement (in `build_neck_model.py`).  I created Mediapipe marker-to-model mapping for the 2x21 hand landmarks, 16 body landmarks and 11 head landmarks. I augmented MediaPipe's holistic body+head+hand model with customized hand tracking to improve fine-motion tracking/3D coordinate estimation of finger and other hand joints.  I've also tuned a number of joints and muscles in the model to address tracking accuracy and IK/ID simulation convergence issues. 
- **Other libraries:** OpenCV, SciPy/NumPy, VTK, ffmpeg, (as well as ipyvolume for jupyter IK prototyping during model development; not part of this repo).
- **Prior work this builds on:** Chin, C. (2025), *Biomechanical Motion Analysis with Computer Vision: A Feedback System for Improving Violin Vibrato Performance Technique*, *ThinkYou?!: Proceedings of the Bay Honors Consortium*.  This had an early version of the standalone left-hand biomechanical model, which was modified and integrated into the body model of this project.

**Iteration / meaningful progress.** The repo reflects real iterative engineering, not a one-shot script — e.g. the **combined hand source** (recovering ~10% of dropped frames per hand with marker RMS unchanged, 0.0374→0.0376 m), the **kinematics-only GRF** that drops the vertical base residual from a full bodyweight (~742 N) to ~10 N, and the **neck-joint model fix** that cut spurious trunk forward-tilt ~15° (RMS 0.040→0.043 m) after a documented weight-sweep showed the earlier hand/shoulder de-weighting was just compensating for the missing joint.


### Major decisions & limitations
Discussed inline above (Evaluation & Evidence) and in **Notes** below: kinematics-only GRF vs. force plates, SO reserve actuators / under-actuation, double-support GRF approximation, the neck-joint modeling choice, and the whole-body-vs-finger accuracy gap.

### Effort over time / development artifacts
Public repo with commit history for major snapshots of functionality.  
Development took place over several weeks, following the pipeline stages.  Each stage was coded as a separate python script with an overall 'run_pipeline.py' wrapper; additional helper stages were added to address issues as they arose.
- preliminary integrated biomechanical model (at models/combined_body_model/combined_body_model.osim)
- src/mp2trc.py: First-pass mediapipe video-to-3D pose coordinates using Mediapipe's holistic model; iterations to match mediapipe landmarks to model joint locations
- src/run_ik.py: First-pass Opensim inverse kinematics; significant changes to the model to address poor tracking (modify model to add head/neck joint DoF in models/combined_body_model/combined_body_model.neck.osim).  Also added a scaling pipeline stage to adjust model to individually match video subjects' body proportions, src/scale_model.py
- src/viz_osim.py: First-pass visualization utility: render inverse kinematics body motion
- src/run_id.py: First-pass Opensim inverse dynamics (joint moment ~ proxy for muscle forces).  Enhance viz script to add colored force lines for approximated muscle groups.
- src/run_so.py: First-pass Opensim static optimization (compute actual individual muscle forces).  Enhance viz script to add colored individual muscle force lines.  Early SO attempts had poor convergence; debugging led to the root cause of missing external forces (eg. floor normal forces on feet)
- src/run_grf.py: Ground forces estimation stage, added to address missing-forces in prior stage to fix optimization convergence
- src/combine_hands.py: to improve occasional mediapipe hand-mistracking in the Holistic model, I added supplementary custom hand-tracking and hand pose re-extraction.
- src/combine_viz.py: integrated viz musculoskeletal video with the annotated original video for combined playback for analysis

---

## References
- Delp, S. L., et al. (2007). OpenSim: Open-Source Software to Create and Analyze Dynamic Simulations of Movement. *IEEE Trans. Biomed. Eng.* 54(11), 1940–1950.
- Kewley, A., Beesel, J., & Seth, A. (2024). *OpenSim Creator* (v0.5.13). doi:10.5281/zenodo.13133987.
- Lugaresi, C., et al. (2019). MediaPipe: A Framework for Building Perception Pipelines. arXiv:1906.08172.
- Rajagopal, A., et al. (2016). Full-Body Musculoskeletal Model for Muscle-Driven Simulation of Human Gait. *IEEE Trans. Biomed. Eng.* 63(10), 2068–2079. doi:10.1109/TBME.2016.2586891.
- McFarland, D. C., et al. (2023). A Musculoskeletal Model of the Hand and Wrist Capable of Simulating Functional Tasks. *IEEE Trans. Biomed. Eng.* 70(5), 1424–1435.
- Uhlrich, S. D., et al. (2023). OpenCap: Human movement dynamics from smartphone videos. *PLoS Comput. Biol.* 19(10), e1011462. doi:10.1371/journal.pcbi.1011462.
- Auer, S., Süß, F., & Dendorfer, S. (2024). Using Markerless Motion Capture and Musculoskeletal Models: An Evaluation of Joint Kinematics. *Technol. Health Care* 32(5), 3433–3442. doi:10.3233/THC-240202.
- He, Y., Liu, S., & Li, M. (2026). Personalized Cervical Biomechanics from Massive Musculoskeletal Simulation. *Sensors* 26(2), 752. doi:10.3390/s26020752.
- Kok, L. M., et al. (2018). High Prevalence of Playing-Related Musculoskeletal Disorders in Amateur Musicians. *PLOS ONE* 13(2). doi:10.1371/journal.pone.0191772.
- Kok, L. M., et al. (2015). Musculoskeletal Complaints Among Professional Musicians: A Systematic Review. *Int. Arch. Occup. Environ. Health* 89(3), 373–396.
- Ren, L., Jones, R. K., & Howard, D. (2008). Whole-body inverse dynamics over a complete gait cycle based only on measured kinematics. *J. Biomech.* 41(12), 2750–2759.
- Chin, C. (2025). Biomechanical Motion Analysis with Computer Vision: A Feedback System for Improving Violin Vibrato Performance Technique. *ThinkYou?!: Proc. Bay Honors Consortium.*


---

## Notes
- Defaults target the accurate path: separate **heavy** PoseLandmarker, ROI-cropped hands, and the **combined** hand source (ROI + Holistic-recovered frames). Override after `--` (e.g. `-- --hand-source roi` for the ROI-only baseline).
- Markerless capture has no measured muscle activity, force plates, or external loads — GRF and SO are estimates (see Evaluation & Evidence). Treat muscle forces as relative, not validated absolutes.
- The per-foot load during double support is the documented approximation (the net wrench is exact); see `src/run_grf.py` and Ren et al. 2008.
