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
| 3 | **Distance** - fit priors, D0 pinhole, D1 ground plane, fusion, correction, eval by range bin | `notebooks/task3_distance.ipynb` | `distance/` , `eval/dist_eval.py` | next |
| 4 | **Edge optimisation** - ONNX / OpenVINO / TensorRT exports, INT8, pruning, benchmarks | `notebooks/task4_optimize.ipynb` | `optimize/`, `eval/benchmark.py` | planned |
| 5 | **Integration** - threaded realtime app, ByteTrack + Kalman on Z, demo video, results tables | `notebooks/task5_integration.ipynb` | `app/`, `tracking/`, `results/` | planned |

### Task 1 - data pipeline (done)

Output published as the Kaggle dataset
`pandeyvineet/eric-cones-stops-barriers`; `python -m data.fetch_dataset` pulls it
onto a machine, so nothing downstream needs nuScenes or the devkit.

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

### Task 3 - distance (next)

- **D0, known size (pinhole):** `Z = f_y * H_eff / h_px`, and `Z = f_x * W / w_px`
  for stop signs, whose bottom is often cut off by the pole.
- **D1, ground-plane contact:** `alpha = atan((v_b - c_y) / f_y)`,
  `Z = h_c * cos(alpha) / sin(theta + alpha)`. Error grows with `Z^2`.
- **Fusion** weighted by each estimate's error model, then a per-class linear
  correction fitted on train - projected cuboids are taller than the visible
  object, so the raw pinhole estimate is biased short.
- **D2, Depth Anything V2** as an offline comparison only; it stays out of the
  realtime path.

### Task 4 - edge optimisation

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

Then check the machine and pull the dataset:

```bash
python -m common.doctor          # python, GPU, packages, dataset, weights
python -m data.fetch_dataset     # downloads pandeyvineet/eric-cones-stops-barriers
```

`doctor` is the first thing to run when something looks wrong - it says whether
torch sees a GPU, which optional packages are absent, and whether the dataset has
been located. `fetch_dataset` downloads through kagglehub (which caches), checks
the tree actually contains images, labels **and** the `meta/` sidecars Task 3
needs, writes `configs/data.yaml`, and records the location in
`configs/local.yaml`.

A private Kaggle dataset needs credentials on the machine: `kaggle.json` in
`~/.kaggle/` (Windows: `%USERPROFILE%\.kaggle\`), or `KAGGLE_USERNAME` and
`KAGGLE_KEY` in the environment. Already have a copy on disk? Skip the download
with `python -m data.fetch_dataset --dataset-root D:/path/to/it`.

### Working across two machines

| Travels through git | Does not |
| --- | --- |
| everything in `configs/` except `local.yaml` and the generated `data*.yaml` | the dataset (Kaggle dataset) |
| all code, notebooks, tests, README | `runs/` and any `.pt` / `.engine` / `.onnx` |
| `results/` tables and plots | `datasets/`, `configs/local.yaml` |

So the loop is: commit and push on one machine, `git pull` on the other, and
`python -m data.fetch_dataset` once per machine. Absolute paths never get
committed - `configs/data.yaml` is generated per machine and git-ignored, which is
why both PCs can keep the dataset in different places.

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

### Task 1 - rebuilding the dataset (not normally needed)

Task 1 is done; its output is the Kaggle dataset above. To change the conversion
and rebuild it from the raw sources:

```bash
python -m data.prepare --nuscenes-root /path/to/nuscenes --force
# or step by step
python -m data.nuscenes_to_yolo --nuscenes-root /path/to/nuscenes --out datasets/yolo_3class
python -m data.coco_stopsign --out datasets/yolo_3class
python -m data.pseudo_label  --out datasets/yolo_3class
python -m data.merge         --out datasets/yolo_3class
python -m data.sanity_check  --out datasets/yolo_3class --draw 20
```

Without `--force`, `data.prepare` finds the dataset that is already there and only
re-validates it.

Paths, class order, camera constants, training runs and distance priors all live
in `configs/`, so a run is reproducible from the command line.

### Tests

The parts that are pure logic - the AP implementation, IoU, box matching, run-config
merging, repeat-factor sampling - run without a GPU or a dataset:

```bash
python -m pytest tests -q
```

## Repo layout

```
configs/     paths.yaml, classes.yaml, camera.yaml, priors.yaml, runs.yaml
             local.yaml + data.yaml are per-machine and git-ignored
common/      config, dataset discovery, doctor, label loading, drawing
data/        fetch_dataset.py, prepare.py, nuscenes_to_yolo.py, coco_stopsign.py,
             pseudo_label.py, merge.py, balance.py, sanity_check.py
detection/   train.py, infer.py
distance/    pinhole.py, ground_plane.py, fusion.py, fit_priors.py, depth_model.py
tracking/    kalman_z.py
optimize/    export.py, prune.py
eval/        det_eval.py, dist_eval.py, benchmark.py
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
