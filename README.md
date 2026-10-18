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
| 2 | **Detection** - B0 zero-shot baseline, fine-tune B1 / B1b / B2, per-class eval | `notebooks/task2_detection.ipynb` | `detection/`, `eval/det_eval.py` | next |
| 3 | **Distance** - fit priors, D0 pinhole, D1 ground plane, fusion, correction, eval by range bin | `notebooks/task3_distance.ipynb` | `distance/` , `eval/dist_eval.py` | planned |
| 4 | **Edge optimisation** - ONNX / OpenVINO / TensorRT exports, INT8, pruning, benchmarks | `notebooks/task4_optimize.ipynb` | `optimize/`, `eval/benchmark.py` | planned |
| 5 | **Integration** - threaded realtime app, ByteTrack + Kalman on Z, demo video, results tables | `notebooks/task5_integration.ipynb` | `app/`, `tracking/`, `results/` | planned |

### Task 1 - data pipeline (done)

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

### Task 2 - detection (next)

| Run | Model | Size | Purpose |
| --- | --- | --- | --- |
| B0 | `yolo26n`, COCO weights, no fine-tune | 640 | floor: detects stop signs only |
| B1 | `yolo26n` fine-tuned | 640 | main deployable candidate |
| B1b | `yolo11n` fine-tuned | 640 | other family, for the latency pick |
| B2 | `yolo26s` fine-tuned | 960 | accuracy ceiling and pseudo-label teacher |

### Task 3 - distance

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

### On Kaggle (the intended path)

Open the task's notebook on Kaggle, or create a notebook and paste the first cell:

```python
!git clone --depth 1 https://github.com/heller007/eric-robotics-assessment.git /kaggle/working/eric-robotics-assessment
```

Per notebook: set the accelerator to GPU, turn Internet on, and attach a
**nuScenes v1.0-mini** dataset (plus **COCO 2017** if you want to skip the
stop-sign download). The dataset roots are found by folder layout, so any mirror
slug works.

### Locally

```bash
pip install -r requirements.txt
python -m data.nuscenes_to_yolo --nuscenes-root /path/to/nuscenes --out datasets/yolo_3class
python -m data.coco_stopsign --out datasets/yolo_3class
python -m data.pseudo_label  --out datasets/yolo_3class --device 0
python -m data.merge         --out datasets/yolo_3class
python -m data.sanity_check  --out datasets/yolo_3class --draw 20
```

Paths, class order, camera constants and distance priors all live in `configs/`,
so a run is reproducible from the command line.

## Repo layout

```
configs/     paths.yaml, classes.yaml, camera.yaml, priors.yaml, data.yaml (generated)
common/      config + environment discovery (local vs Kaggle)
data/        nuscenes_to_yolo.py, coco_stopsign.py, pseudo_label.py, merge.py, sanity_check.py
detection/   train.py, infer.py
distance/    pinhole.py, ground_plane.py, fusion.py, fit_priors.py, depth_model.py
tracking/    kalman_z.py
optimize/    export.py, prune.py
eval/        det_eval.py, dist_eval.py, benchmark.py
app/         realtime.py
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
