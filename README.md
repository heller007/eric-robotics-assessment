# Cone, Barrier and Stop Sign Detection + Distance

One detector plus a geometry-only distance module, built to run at 30+ FPS on GPU
and 15+ FPS on CPU. Distance is closed-form from the detector's boxes: no second
network in the realtime path.

**Distance definition:** `Z`, the depth along the camera optical axis, in metres.
That is what nuScenes box centres give in the camera frame, and what the pinhole
model solves for.

## Targets

| Target | Value | Measured on |
| --- | --- | --- |
| End-to-end FPS, GPU | 30+ (p95 under 33 ms) | capture to annotated frame, batch 1 |
| End-to-end FPS, CPU | 15+ | same pipeline, OpenVINO or ONNX Runtime |
| Detection | beat the zero-shot baseline on every class | nuScenes `mini_val` (2 held-out scenes) |
| Distance, cone and barrier | AbsRel <= 0.15, 80%+ within 20%, 0-20 m | matched true positives vs nuScenes 3D ground truth |
| Distance, stop sign | within 15% at 2, 4, 6, 8 m | self-captured, tape-measured photos |
| Optimisation cost | <= 1 mAP50 point (FP16), <= 2 (INT8) | same val split |

## Tasks

The work is split so each task runs and is checked on its own before the next one
starts. Every task has one Kaggle notebook and one set of modules.

| # | Task | Notebook | Modules | Status |
| --- | --- | --- | --- | --- |
| 1 | **Data pipeline** - nuScenes -> YOLO, COCO stop signs, pseudo-labels, merge, sanity checks | `notebooks/task1_data_pipeline.ipynb` | `data/` | done |
| 2 | **Detection** - B0 zero-shot baseline, fine-tune B1 / B1b / B2, per-class eval | `notebooks/task2_detection.ipynb` | `detection/`, `eval/det_eval.py` | done |
| 3 | **Distance** - fit priors, D0 pinhole, D1 ground plane, fusion, correction, eval by range bin | `notebooks/task3_distance.ipynb` | `distance/`, `eval/dist_eval.py` | done |
| 4 | **Edge optimisation** - ONNX / OpenVINO / TensorRT exports, INT8, pruning, benchmarks | `notebooks/task4_optimize.ipynb` | `optimize/`, `eval/benchmark.py` | planned |
| 5 | **Integration** - threaded realtime app, ByteTrack + Kalman on Z, demo video, results tables | `notebooks/task5_integration.ipynb` | `app/`, `tracking/`, `results/` | planned |

### Task 1 - data pipeline (done)

Built locally with `python -m data.prepare`, which converts nuScenes v1.0-mini
(pointed at a local copy, or downloaded as a fallback), adds the COCO stop signs,
pseudo-labels the missing ones, and validates the result.

Builds one 3-class YOLO set: **0 cone, 1 barrier, 2 stop_sign**.

| Source | Classes | Role |
| --- | --- | --- |
| nuScenes v1.0-mini | `movable_object.trafficcone`, `movable_object.barrier` | train + val + distance ground truth |
| COCO 2017 | `stop sign` | train + val for the stop sign class |
| COCO-pretrained YOLO on nuScenes frames | pseudo stop signs | removes the missing-label conflict |

Three details that decide whether the rest works:

- **Split by scene, never by frame.** Keyframes are 0.5 s apart; adjacent frames
  show the same cones, so a frame split leaks the val set into training.
- **Pseudo-label the stop signs.** nuScenes never annotates them, so a naive merge
  teaches the model that every stop sign in a nuScenes frame is background.
- **Keep a sidecar json per image.** `K`, camera height, pitch and per-box `Z`,
  truncation and visibility - Task 3 needs all of it, and recovering it later
  means re-running the devkit.

**A row of barriers is one label, not six.** nuScenes annotates every barrier
segment separately, so a roadside row projects to five or six boxes standing
side by side. In the image that row is one continuous object and a detector
predicts one box for it - the right answer, scored as five misses. Overlap is the wrong test for joining them: a row running
across the view is a line of boxes side by side whose IoU is near zero, while a
row receding from the camera is a stack whose IoU is high. Proximity catches
both. Trained against the raw labels, barrier AP50 was 0.000 with zero recall in
every range bin, on more training instances than cones: YOLO26 is NMS-free and assigns one
prediction per ground-truth box, so the duplicates are contradictory targets
rather than hard ones. The converter merges barrier boxes that touch or sit within
`--merge-gap` pixels of each other into one per visual structure, and keeps the
nearest segment's depth, which is also the
useful quantity ("a barrier starts 9 m ahead"). The merged box's bottom edge is
still that segment's contact point, so the ground-plane distance estimate stays
exact; the pinhole estimate reads slightly tall because the top edge comes from
the far end of the row, and that bias is what fitting `H_eff` absorbs. Every
segment's depth is kept in the sidecar. Cones are never merged - they are
discrete objects.

**The val split is not the devkit's.** `mini_val` (scene-0103, scene-0916)
contains no barrier annotations at all, so barrier AP is unmeasurable on it -
and "beat the zero-shot baseline on every class" then cannot be reported.
Barriers are concentrated in two scenes (989 in scene-0061, 584 in scene-0553,
48 across all the rest), so no split is both class-balanced and 20%. The
hold-out is `scene-0655` + `scene-0553`, set in `configs/paths.yaml`: val gets
218 cones and 584 barriers, train keeps 675 cones, 1037 barriers and all of
scene-0061. That is 24% of cones and 36% of barriers in val - higher than usual,
and the price of being able to measure the class at all.

Boxes are dropped when visibility is below level 2, when no lidar or radar point
hits them, when any cuboid corner is behind the camera, or when the projection is
under 4 px.

### Task 2 - detection (done)

Runs live in `configs/runs.yaml`; `python -m detection.train --run B1` reads one
entry and anything in it can be overridden with `--set epochs=40`.

| Run | Model | Size | Purpose |
| --- | --- | --- | --- |
| B0 | `yolo26n`, COCO weights, no fine-tune | 640 | floor: detects stop signs only |
| B1 | `yolo26n` fine-tuned | 640 | main deployable candidate |
| B1b | `yolo11n` fine-tuned | 640 | other family, for the latency pick |
| B2 | `yolo26s` fine-tuned | 960 | accuracy ceiling and pseudo-label teacher |
| B1bal | B1 + class-balanced sampling | 640 | accuracy lever 4 |
| B1u | B1 continued, backbone unfrozen, lower LR | 640 | accuracy lever 5 |
| B1p2 | extra stride-4 head | 640 | accuracy lever 3, for 10-20 px cones |

**YOLO26 weights are requested, not assumed.** Each run lists its models in
preference order and the trainer falls back to the next entry if the first cannot
be fetched by the installed ultralytics, recording which one it actually used.

**One evaluator for every variant.** `model.val()` cannot score B0: a
COCO-pretrained model predicts 80 classes against this 3-class set, so the
indices do not line up. Evaluating the baseline one way and the fine-tuned runs
another would make the comparison meaningless, so `eval/det_eval.py` implements
COCO-style AP once - greedy highest-IoU matching, one ground-truth box per
prediction, a precision envelope sampled at 101 recall points with step
interpolation - and runs everything through it, remapping zero-shot predictions
from COCO's class order. `--cross-check` additionally prints `model.val()` for a
fine-tuned model as a reference. The AP code is unit-tested against cases whose
answer is known by hand (`tests/test_det_eval.py`).

Beyond mAP it reports, per class, precision / recall / F1 at conf 0.25 (what a
realtime pipeline would actually run at) and **recall split by ground-truth
depth** (0-10, 10-20, 20-40, 40+ m). That last table is the one that shows
whether far cones are being found at all, and it is what justifies spending
compute on input size or a P2 head.

### Task 3 - distance (done)

- **D0, known size (pinhole):** `Z = f_y * H_eff / h_px`, and `Z = f_x * W / w_px`
  for stop signs, whose bottom is often cut off by the pole.
- **D1, ground-plane contact:** `alpha = atan((v_b - c_y) / f_y)`,
  `Z = h_c * cos(alpha) / sin(theta + alpha)`. Error grows with `Z^2`.
- **Fusion** weighted by each estimate's error model, then a per-class linear
  correction fitted on train - projected cuboids are taller than the visible
  object, so the raw pinhole estimate is biased short.
- **D2, Depth Anything V2** as an offline comparison only; it stays out of the
  realtime path.

Evaluated twice: on oracle boxes, which measures the geometry with the detector
out of the picture and therefore works for every class including barriers, and
on predicted boxes matched at IoU 0.5 and scored on true positives only - a miss
is a detection failure, counted in Task 2, and folding it in here would hide
which half is at fault. Reported per class and per range bin for each stage
separately, since D0 and D1 fail in opposite directions.

### Task 4 - edge optimisation (next)

ONNX FP32, OpenVINO FP32/INT8, TensorRT FP16/INT8, an input-size sweep, structured
pruning with DepGraph, and `n` vs `s` backbones - each benchmarked with 20 warmup
and 200 timed runs, reporting mean FPS plus p50 and p95 latency, model-only and
end to end. Unstructured magnitude pruning is included once as a negative result:
it zeroes weights but dense kernels still run them.

### Task 5 - integration

Threaded capture with a depth-1 queue (stale frames dropped, not queued),
detection, vectorised distance, ByteTrack with a 1-D constant-velocity Kalman
filter on `Z`, and the annotated output. Then the results tables, the accuracy vs
speed plot with the 30 FPS line marked, and the demo video.

## Barrier investigation

Barriers would not train: AP50 0.000 on the held-out scene, on more training
instances than cones, while the same pipeline reached 0.71 on cones and 0.85 on
stop signs. Six things were varied to find out why.

| change | barrier AP50 |
| --- | --- |
| raw per-segment labels | 0.000 |
| duplicate containment suppressed | 0.094 |
| rows merged into one box per structure | 0.000 |
| single class, no competition (`B3aabb`) | 0.037 |
| oriented boxes (`B3obb`) | 0.015 |
| **frame-level split, axis-aligned (`B4aabb`)** | **0.789** |
| **frame-level split, oriented (`B4obb`)** | **0.890** |

The last two rows settle it. A frame-level split leaks by construction -
keyframes are 0.5 s apart and show the same barrier - so those numbers are an
upper bound, not a generalisation result. But they answer the question that
mattered: **the detector learns barriers fine** (P 0.91, R 0.85). What it cannot
do is transfer to a scene it has never seen.

**Why that is structural, not fixable here.** nuScenes-mini has barriers in two
substantial scenes, 989 boxes in scene-0061 and 584 in scene-0553, with 48
across all the others. Any scene-level hold-out therefore trains on essentially
one scene and tests on a different one. Cones generalise because they appear in
four scenes; stop signs generalise because they come from COCO, thousands of
images across every condition. The per-scene counts are printed by
`data/sanity_check.py` and the split is recorded in `configs/paths.yaml`.

**What would fix it:** barrier data from more scenes. nuImages v1.0-mini carries
`movable_object.barrier` 2D boxes on different frames and is the obvious source -
though it has no 3D boxes, so it would improve detection while contributing
nothing to distance ground truth.

### Oriented boxes are the better primitive for this class

Visible only once the dataset stops being the bottleneck, which is why the
scene-split comparison missed it:

| | mAP50 | mAP50-95 | inference | GFLOPs |
| --- | --- | --- | --- | --- |
| axis-aligned | 0.789 | 0.553 | 2.1 ms | 6.4 |
| oriented | **0.890** | **0.711** | **1.9 ms** | 6.6 |

+0.10 mAP50 and +0.16 mAP50-95 for no latency cost. The reason is geometric: a
roadside row receding from the camera is a long thin diagonal shape, and an
axis-aligned box around it is roughly **55% object, 45% empty road** (measured on
a synthetic row), which caps IoU even for a correct prediction.

It would also improve distance, which is the part worth carrying forward: the
oriented box's **short side** is the barrier's height in pixels regardless of how
far the row recedes - exactly the inflation that `H_eff` is currently fitted to
absorb - and its lowest **vertex** is a truer ground-contact point than the
bottom edge of an axis-aligned box.

Labels cost nothing to produce: the converter already projects 8 cuboid corners
per barrier, so `cv2.minAreaRect` over a structure's corners gives the rotated
rectangle directly. That is what ruled segmentation out by comparison - masks
would need a second dataset (nuImages), and that dataset has no depth.

Reproduce both with `notebooks/task2b_barrier_obb.ipynb`.

## Running it

Work happens in a local checkout; **git carries the code, not the data**. The
dataset is a published Kaggle dataset and the weights are git-ignored, so each
machine fetches or copies those separately.

### First time on a machine

```bash
git clone https://github.com/heller007/eric-robotics-assessment.git
cd eric-robotics-assessment
```

Install a CUDA build of torch *before* the requirements, or pip will give you the
CPU-only wheel and training will crawl (pick the index URL for your CUDA version
from pytorch.org):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

Then check the machine and build the dataset. Already have nuScenes v1.0-mini
(a Kaggle mirror such as `aadimator/nuscenes-mini`, or your own download)? Point
at it - nothing is downloaded when you do:

```bash
python -m common.doctor                                          # GPU, packages, sources, dataset
python -m data.prepare --nuscenes-root D:/where/you/unpacked/it  # build Task 1
```

The path does not have to be the dataroot itself. A mirror usually leaves a
wrapper folder or two above the `v1.0-mini/` + `samples/` pair, so point at
wherever you unpacked it and the dataroot is found inside. It is then checked for
the metadata tables, the six cameras and a plausible image count before anything
runs - a partial mirror otherwise fails deep inside the devkit, with a confusing
error. The path is remembered in `configs/local.yaml`, so later commands need no
flag.

Without a local copy, downloading is the fallback:

```bash
python -m data.prepare --fetch-nuscenes      # ~3.9 GiB, resumable, then the full build
```

`doctor` is the first thing to run when something looks wrong - it says whether
torch sees a GPU, which optional packages are absent, whether nuScenes and COCO
are present, and whether the dataset has been built. `prepare` runs every Task 1
step in order and skips what is already there, so re-running it is cheap.

Budget 15-25 minutes for the whole build on a GPU machine, plus about 9 GB of
free space if you are downloading. The slow parts are the nuScenes download, the
~350 MB of COCO stop-sign images, and the pseudo-label pass; on CPU the last of
those dominates, so `--no-pseudo` is there if you want to defer it (stop sign AP
suffers - see Task 1 below).

Getting the raw data on its own, without building:

```bash
python -m data.fetch_nuscenes --root D:/kaggle/nuscenes-mini   # register a copy
python -m data.fetch_nuscenes --archive D:/nuscenes-mini.zip   # extract a .zip or .tgz
python -m data.fetch_nuscenes                                  # download it
python -m data.fetch_nuscenes --dest D:/datasets/nuscenes      # download elsewhere
python -m data.fetch_nuscenes --no-keep-archive                # free the archive after
```

The download resumes, so interrupting it costs nothing. Downloading nuScenes
means accepting its [Terms of Use](https://www.nuscenes.org/terms-of-use) (free
for non-commercial use).

**About the published dataset.** An earlier build of this dataset was uploaded as
the Kaggle dataset `pandeyvineet/eric-cones-stops-barriers`, and
`data/fetch_dataset.py` pulls it down through kagglehub, verifies it and wires it
up. That copy is still awaiting compression on Kaggle and cannot be downloaded,
so `kaggle_dataset` in `configs/paths.yaml` is set to null and the local build is
the live path. Put the slug back when it is ready and nothing else needs to
change.

### Working across two machines

| Travels through git | Does not |
| --- | --- |
| everything in `configs/` except `local.yaml` and the generated `data*.yaml` | nuScenes, COCO, and the built dataset |
| all code, notebooks, tests, README | `runs/` and any `.pt` / `.engine` / `.onnx` |
| `results/` tables and plots | `datasets/`, `configs/local.yaml` |

So the loop is: commit and push on one machine, `git pull` on the other, and
`python -m data.prepare --nuscenes-root <where the raw data is>` once per machine. Absolute paths never
get committed - `configs/data.yaml` is generated per machine and git-ignored,
which is why both PCs can keep the data in different places.

To move trained weights across, copy `runs/<run>/weights/best.pt` by hand (or
publish it as its own Kaggle dataset). Nothing in the pipeline assumes they are
in git.

### Task 2 - detection

```bash
python -m eval.det_eval   --run B0                                   # zero-shot floor
python -m detection.train --run B1
python -m eval.det_eval   --weights runs/B1/weights/best.pt --name B1 --cross-check
python -m detection.infer --weights runs/B1/weights/best.pt --source <dataset>/images/val --limit 8
```

Any run in `configs/runs.yaml` works the same way, and `--set key=value` overrides
one knob: `python -m detection.train --run B2 --set epochs=60`. `--device` is
blank by default, meaning CUDA when it is there and CPU when it is not.

### Task 1 - building the dataset

```bash
python -m data.prepare --nuscenes-root D:/where/you/unpacked/it
python -m data.prepare --fetch-nuscenes          # or download it as a fallback
# or step by step
python -m data.fetch_nuscenes --root D:/where/you/unpacked/it
python -m data.nuscenes_to_yolo --out datasets/yolo_3class
python -m data.coco_stopsign    --out datasets/yolo_3class
python -m data.pseudo_label     --out datasets/yolo_3class
python -m data.merge            --out datasets/yolo_3class
python -m data.sanity_check     --out datasets/yolo_3class --draw 20
```

Without `--force`, `data.prepare` finds the dataset that is already there and only
re-validates it.

**Images are linked, not copied.** YOLO locates a label by swapping `/images/` for
`/labels/` in the image path, so all 2,424 frames have to be reachable under
`images/<split>/` - but duplicating 1.3 GB to achieve that is pointless.
`--link-mode auto` (the default) tries a hard link first, which is free and needs
no privileges but cannot cross volumes; then a symbolic link, which crosses
volumes but on Windows needs Developer Mode or an elevated shell; then a copy,
which always works. Force one with `--link-mode copy` if you want a self-contained
dataset directory to move around. The converter reports which was used.

Paths, class order, camera constants, training runs and distance priors all live
in `configs/`, so a run is reproducible from the command line.

### Tests

The parts that are pure logic - the AP implementation, IoU, box matching,
run-config merging, repeat-factor sampling, dataset discovery, image staging -
run without a GPU or a dataset:

```bash
python -m pytest tests -q
```

`tests/test_nuscenes_conversion.py` goes further: it builds a devkit-valid
nuScenes containing a cone at exactly 10 m and a barrier at 15 m in front of a
camera mounted 1.51 m up, runs the real converter over it, and checks the pixel
box and the recorded depth against what the pinhole model requires. A projection
bug does not announce itself - it produces boxes in plausible places - so this is
the check that the camera-frame conventions are right. It needs
`nuscenes-devkit` installed and skips itself when that is absent.

## Repo layout

```
configs/     paths.yaml, classes.yaml, camera.yaml, priors.yaml, runs.yaml
             local.yaml + data.yaml are per-machine and git-ignored
common/      config, dataset discovery, doctor, image staging, label loading, drawing
data/        prepare.py, fetch_nuscenes.py, fetch_dataset.py, nuscenes_to_yolo.py,
             coco_stopsign.py, pseudo_label.py, merge.py, balance.py, sanity_check.py
detection/   train.py, infer.py
distance/    geometry.py (D0, D1, fusion, correction), fit_priors.py
tracking/    kalman_z.py
optimize/    export.py, prune.py
eval/        det_eval.py, dist_eval.py, depth_coverage.py, benchmark.py
app/         realtime.py
tests/       unit tests for the metric and config logic
notebooks/   one per task
results/     csv tables, plots, annotated frames
```

`datasets/` and `runs/` are generated and git-ignored.

## Results

Filled in as the tasks land.

| Variant | mAP50 | mAP50-95 | AP50 cone | AP50 barrier | AP50 stop | Dist AbsRel | Within 20% | GPU FPS (e2e) | CPU FPS (e2e) | Size (MB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B0 zero-shot | | | | | | | | | | |
| B1 FP32 | | | | | | | | | | |
| B1 TensorRT FP16 | | | | | | | | | | |
| B1 TensorRT INT8 | | | | | | | | | | |
| B1 OpenVINO INT8 | | | | | | | | | | |
| B1 pruned 30% | | | | | | | | | | |
| B2 (s, 960) | | | | | | | | | | |

## Known limits

- `mini_val` is two scenes. Numbers from it are noisy; the final model is also
  reported on a swapped train/val fold so the spread is visible.
- Stop signs on nuScenes frames are pseudo-labels, not ground truth.
- Stop sign distance has no nuScenes ground truth and is checked only against
  tape-measured photos.
- Ground-plane distance assumes a flat road and a known mount. On a different
  robot, `camera_height_m` and `pitch_deg` in `configs/camera.yaml` must change;
  the pinhole estimate does not depend on the mount at all.
