# Cone, Barrier, Stop Sign Detection + Distance: Implementation Plan

Oct 18, 2026 · @vineet

## Goals and targets

Ship one detector plus a geometry-only distance module that runs at 30+ FPS end to end on GPU and 15+ FPS on CPU, with no more than 1 mAP50 point lost to optimization.

| Target | Value | Measured on |
| --- | --- | --- |
| End-to-end FPS, GPU | 30+ (p95 latency under 33 ms) | Capture to annotated frame, batch 1 |
| End-to-end FPS, CPU | 15+ | Same pipeline, OpenVINO or ONNX Runtime |
| Detection | Beat zero-shot baseline on every class; report AP50 and AP50-95 per class | nuScenes mini\_val (2 scenes) |
| Distance, cone and barrier | AbsRel 0.15 or lower, 80%+ of objects within 20%, range 0 to 20 m | Matched true positives, nuScenes 3D ground truth |
| Distance, stop sign | Within 15% at 2, 4, 6, 8 m | Self-captured, tape-measured photos |
| Optimization cost | 1 mAP50 point or less (FP16), 2 or less (INT8) | Same val split |

**Distance definition:** Z, the depth along the camera optical axis, in meters. This matches nuScenes box centers in the camera frame and the pinhole model.

**Deployment rule:** ship the fastest variant that meets the FPS target while staying inside the accuracy budget above. Accuracy targets are first guesses; revisit them after the first fine-tuned run (B1).

## System architecture

The detector is the only heavy step per frame. Distance is closed-form geometry on its boxes, and calibration constants are fixed per camera.

&#91;embedded content: runtime pipeline · 7 stages\]

Training, export and prior fitting run offline. At runtime, the cost after the detector is a few vectorized operations and a tracker update.

## Data pipeline

Train on nuScenes-mini (cones, barriers, distance ground truth) plus COCO stop signs, unified into one 3-class YOLO set: 0 cone, 1 barrier, 2 stop\_sign.

| Source | Classes used | Images | Role |
| --- | --- | --- | --- |
| [nuScenes v1.0-mini](https://github.com/daetz-coder/TrajectoryPrediction) | movable\_object.trafficcone (1,378 boxes), movable\_object.barrier (2,323 boxes) | 404 samples x 6 cameras = 2,424 | Train + val + distance ground truth |
| COCO 2017 | stop sign | All images containing it | Train + val for stop sign |
| [nuImages v1.0-mini](https://nuscenes.org/tutorials/nuimages_tutorial.html) | trafficcone, barrier | Small mini split | Optional extra detection data |
| Self-captured phone photos | stop sign, cone | 20 to 40 | Distance check at taped distances |

The nuScenes-mini category stats give the physical size priors: cone height 0.78 m (std 0.13), barrier height 1.06 m (std 0.10), barrier width 2.32 m.

### nuScenes to YOLO conversion

1. Load with `NuScenes('v1.0-mini')`; split by scene using the devkit's `mini_train` (8 scenes) and `mini_val` (2 scenes). Never split by frame: adjacent frames leak.
2. For every sample and all 6 cameras, call `get_sample_data`, which returns the image path, boxes already in that camera's frame, and the intrinsic matrix K.
3. Keep boxes of the two classes with visibility level 2 or higher and at least one lidar or radar point.
4. Project the 8 cuboid corners with `view_points(box.corners(), K, normalize=True)`, take the min and max, clip to 1600 x 900, drop boxes under 4 px.
5. Write the YOLO label line. Write a sidecar JSON per image: K, camera height, camera pitch, and per box: class, pixel box, Z = box center depth, truncated flag, visibility level.

### COCO stop signs

Pull only the stop sign images with FiftyOne: `load_zoo_dataset('coco-2017', split='train', label_types=['detections'], classes=['stop sign'])`. Do the same for the validation split. Remap the class to index 2 and drop all other COCO classes.

### Missing-label conflict (fix this, or accuracy caps out)

nuScenes never labels stop signs, so the merged set teaches the model that nuScenes stop signs are background. Fix: run the COCO-pretrained YOLO on all nuScenes images, keep stop sign detections at confidence 0.5 or higher as pseudo-labels, and spot-check 30 by eye. COCO has no cone or barrier class, but those objects are rare in stop sign images, so accept that side.

### Sanity checks before training

- Draw 20 random converted images with boxes and eyeball them.
- Print instance counts per class per split; both val scenes must contain cones and barriers.
- Confirm K differs per camera and Z is positive for every kept box.

## Detection model

Fine-tune Ultralytics YOLO26 and YOLO11 at sizes n and s, then pick by measured latency on your own hardware, not by paper claims.

[YOLO26](https://arxiv.org/abs/2509.25164v2) (released September 2025) removes NMS and DFL and adds small-target-aware label assignment; Ultralytics claims up to 43% faster CPU inference. An [independent benchmark](https://arxiv.org/pdf/2605.24831) found YOLOv8-s still faster on GPU (6.92 ms vs 8.38 ms for YOLO26-s). So benchmark both families before committing.

| Run | Model | Train size | Purpose |
| --- | --- | --- | --- |
| B0 | yolo26n, COCO weights, no fine-tune | 640 | Floor: detects stop sign only |
| B1 | yolo26n fine-tuned | 640 | Main deployable candidate |
| B1b | yolo11n fine-tuned | 640 | Same, other family, for the latency pick |
| B2 | yolo26s fine-tuned | 960 | Accuracy ceiling and pseudo-label teacher |

### Training recipe

```bash
yolo detect train model=yolo26n.pt data=configs/data.yaml imgsz=640 \
  epochs=100 patience=20 batch=-1 cos_lr=True close_mosaic=10 \
  mosaic=1.0 scale=0.5 fliplr=0.5 hsv_v=0.4 freeze=10 project=runs name=B1
```

- `freeze=10` locks the backbone for the first run: mini has only about 2,400 images and overfits fast. Unfreeze in a second short run if val mAP is still rising.
- `batch=-1` picks the largest batch your GPU memory fits.
- No vertical flip: road scenes have a fixed up direction.
- `rect=True` for evaluation keeps the 16:9 frames without padding waste.

### Accuracy levers, cheapest first

1. Pseudo-labels for missing stop signs (Data pipeline section).
2. Larger input size: far cones are 10 to 20 px tall at 640. Compare 640, 800, 960 on the same weights.
3. P2 head (`yolov8-p2.yaml` style config): adds a stride-4 output for tiny objects, at extra compute.
4. Class-balanced sampling: repeat stop sign and barrier images so each class has a similar instance count per epoch.
5. Unfreeze and fine-tune the full network at lower learning rate.
6. Tiled inference (SAHI) for an offline accuracy reference only; it is not realtime.

## Distance estimation

Combine two geometric estimates, known object size and ground-plane contact, weighted by their error models, then correct per class on the train split.

### D0: known object size (pinhole)

A real height H at depth Z projects to h pixels.

```latex
Z = \frac{f_y \, H}{h_{px}} \qquad \text{(stop sign: } Z = \frac{f_x \, W}{w_{px}} \text{)}
```

- Use height for cones and barriers. Barrier width varies with viewing angle; height does not.
- Use width for stop signs: the sign is an octagon, and the pole or other objects often cut the bottom off.
- Relative error equals the box-height error divided by h, so it grows as objects shrink with range.
- The size prior adds a floor: cone height std 0.13 on 0.78 m is about 17%; barrier 0.10 on 1.06 m is about 9%.

**Fit H\_eff, do not hardcode physical sizes.** Boxes made by projecting a 3D cuboid are taller than the visible object, since the cuboid's depth extent adds pixels. On train-split boxes with visibility 4 and no truncation, set H\_eff per class to the median of Z times h\_px divided by f\_y. That absorbs the bias.

### D1: ground-plane contact (cones and barriers only)

The camera sits at height h\_c, pitched down by angle θ. The box bottom row v\_b defines a ray at angle α below the optical axis.

```latex
\alpha = \arctan\frac{v_b - c_y}{f_y}, \qquad Z = \frac{h_c \cos\alpha}{\sin(\theta + \alpha)}
```

- With zero pitch this reduces to Z = f\_y × h\_c / (v\_b − c\_y).
- Error grows with the square of range: a 1 px error in v\_b shifts Z by about Z² / (f\_y × h\_c).
- Reject when θ + α is under 0.5 degrees: the ray is near the horizon and Z blows up.
- Take h\_c from the camera translation z in the nuScenes calibrated\_sensor record, and θ from its rotation. Refine h\_c with a 1-parameter least-squares fit on train boxes.
- On the real robot, h\_c and θ come from its own mount. D0 does not depend on the mount at all.

### Fusion and correction

```latex
\hat{Z} = \frac{Z_0/\sigma_0^2 + Z_1/\sigma_1^2}{1/\sigma_0^2 + 1/\sigma_1^2}, \quad \sigma_0 = Z_0 \, r_{cls}, \quad \sigma_1 = \frac{Z_1^2}{f_y h_c} \, \sigma_v
```

- r\_cls is the class size prior relative std (0.17 cone, 0.09 barrier); σ\_v is about 2 px of box-edge noise.
- Near objects lean on D1, far ones lean on D0, which is what the error models predict.
- Stop signs, truncated boxes, and boxes touching the image bottom use D0 only.
- Last step: fit a per-class linear correction Z = a × Ẑ + b on train. It removes the offset between the box bottom (nearest cuboid edge) and the cuboid center that nuScenes labels as Z.

### Temporal smoothing

Track objects with ByteTrack (`model.track(..., tracker='bytetrack.yaml')`) and run a 1D constant-velocity Kalman filter on Z per track ID. This removes frame-to-frame jitter in the label and costs well under 1 ms.

### D2: learned depth, comparison only

Depth Anything V2 Small (metric outdoor checkpoint), taking the median depth inside each box. It shows where geometry wins or loses. It stays out of the realtime pipeline.

### Evaluation

1. **Oracle boxes first:** run D0, D1 and fusion on ground-truth 2D boxes. This isolates geometry error from detector error.
2. **Predicted boxes:** match predictions to ground truth at IoU 0.5 or higher, evaluate true positives only.
3. Report AbsRel, RMSE, and the share within 10% and 20%, split by class and by range bins 0 to 10, 10 to 20, and 20 to 40 m.
4. Stop signs have no nuScenes ground truth. Photograph a printed or real stop sign at taped 2, 4, 6, 8 m, with a phone calibrated by an OpenCV checkerboard.

## Edge optimization and realtime pipeline

The biggest wins come from the runtime (TensorRT on GPU, OpenVINO on CPU), then precision, then input size; pruning comes last.

| ID | Variant | Device | How |
| --- | --- | --- | --- |
| E0 | PyTorch FP32 | CPU, GPU | `model.predict` baseline |
| E1 | ONNX FP32 | CPU | `format=onnx simplify=True`, ONNX Runtime with intra-op threads = physical cores |
| E2 | OpenVINO FP32 and INT8 | CPU | `format=openvino`, then `int8=True data=configs/data.yaml` (calibrates on val images) |
| E3 | TensorRT FP16 | GPU | `format=engine half=True` |
| E4 | TensorRT INT8 | GPU | `format=engine int8=True data=configs/data.yaml` |
| E5 | Input size sweep 320, 416, 512, 640 | CPU, GPU | Same weights, re-exported per size |
| E6 | Structured pruning 20 to 30% channels | CPU, GPU | torch-pruning (DepGraph), then 20 to 30 epochs of fine-tuning |
| E7 | n vs s backbone | CPU, GPU | Lightweight-backbone comparison the brief asks for |

```bash
yolo export model=runs/B1/weights/best.pt format=engine imgsz=640 half=True
yolo export model=runs/B1/weights/best.pt format=openvino imgsz=640 int8=True data=configs/data.yaml
```

- Unstructured magnitude pruning (`torch.nn.utils.prune`) zeroes weights but dense kernels still run them: no speedup. Include it once as a negative result; it makes the point for you.
- If INT8 costs too much on small cones, keep the detection head in FP16 (mixed precision) or calibrate on more val images.

### Realtime pipeline engineering

- **Threaded capture:** decode frames in a separate thread with a queue of size 1. Stale frames get dropped instead of building latency.
- **Fixed input shape, batch 1:** static engines are faster and timing is stable.
- **NMS:** YOLO26 is NMS-free. For YOLO11, NMS on many boxes eats CPU time; raise `conf` to 0.25 or higher and cap `max_det`.
- **Distance math is vectorized NumPy:** under 0.1 ms per frame for all boxes.
- **Detect every frame on GPU.** On CPU, if under 15 FPS, run the detector every 2nd frame and propagate boxes with Lucas-Kanade optical flow in between (Extra credit section).

### Benchmark protocol

- 20 warmup runs, then 200 timed runs on real val images, batch 1.
- Report mean FPS, p50 and p95 latency, model-only (`result.speed`) and end to end (capture to drawn frame).
- Log CPU model, thread count, GPU model, driver, and power mode. Close other GPU jobs.

## Evaluation protocol and results

All numbers come from nuScenes mini\_val (2 held-out scenes) plus the COCO val stop signs; one row per variant, filled as runs finish.

| Variant | mAP50 | mAP50-95 | AP50 cone | AP50 barrier | AP50 stop | Dist AbsRel | Within 20% | GPU FPS (e2e) | CPU FPS (e2e) | Size (MB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B0 zero-shot |  |  |  |  |  |  |  |  |  |  |
| B1 FP32 |  |  |  |  |  |  |  |  |  |  |
| B1 TensorRT FP16 |  |  |  |  |  |  |  |  |  |  |
| B1 TensorRT INT8 |  |  |  |  |  |  |  |  |  |  |
| B1 OpenVINO INT8 |  |  |  |  |  |  |  |  |  |  |
| B1 pruned 30% |  |  |  |  |  |  |  |  |  |  |
| B2 (s, 960) |  |  |  |  |  |  |  |  |  |  |

- **Distance breakdown:** a second table per class and range bin (0 to 10, 10 to 20, 20 to 40 m) for D0, D1, fusion, fusion + correction, and D2. Run it once on oracle boxes, once on predicted boxes.
- **Accuracy vs speed plot:** mAP50 on y, end-to-end FPS on x, one point per variant, the 30 FPS line marked. It answers the deploy choice at a glance.
- **Before vs after visuals:** the same 4 frames annotated by FP32 and by the deployed variant, side by side.
- **Variance warning:** 2 val scenes is a small sample. Swap the train and val scenes once (another fold) and report the spread for the final model.

## Repo structure and core code

One package with a module per stage; every script reads `configs/`, so runs are reproducible from the command line.

```
configs/      data.yaml, priors.yaml (H_eff, r_cls, correction a and b), camera.yaml
data/         nuscenes_to_yolo.py, coco_stopsign.py, pseudo_label.py, merge.py
detection/    train.py, infer.py
distance/     pinhole.py, ground_plane.py, fusion.py, fit_priors.py, depth_model.py
tracking/     kalman_z.py
optimize/     export.py, prune.py
eval/         det_eval.py, dist_eval.py, benchmark.py
app/          realtime.py (threaded capture, detect, distance, track, draw)
results/      csv tables, plots, annotated frames, demo video
```

### data/nuscenes\_to\_yolo.py (core loop)

```python
import json, numpy as np
from pathlib import Path
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.utils.geometry_utils import view_points, BoxVisibility

CLS = {"movable_object.trafficcone": 0, "movable_object.barrier": 1}
CAMS = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
        "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]
W, H = 1600, 900

def cam_height_pitch(nusc, sd):
    cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
    axis = Quaternion(cs["rotation"]).rotation_matrix @ np.array([0, 0, 1.0])
    return cs["translation"][2], float(np.arcsin(-axis[2]))  # pitch > 0 = looking down

def convert(root, out):
    nusc = NuScenes("v1.0-mini", dataroot=root, verbose=False)
    val_scenes = set(create_splits_scenes()["mini_val"])
    for scene in nusc.scene:
        split = "val" if scene["name"] in val_scenes else "train"
        tok = scene["first_sample_token"]
        while tok:
            sample = nusc.get("sample", tok)
            for cam in CAMS:
                sd = nusc.get("sample_data", sample["data"][cam])
                img, boxes, K = nusc.get_sample_data(sd["token"], box_vis_level=BoxVisibility.ANY)
                h_c, pitch = cam_height_pitch(nusc, sd)
                lines, meta = [], []
                for b in boxes:
                    if b.name not in CLS:
                        continue
                    ann = nusc.get("sample_annotation", b.token)
                    vis = int(ann["visibility_token"])
                    if vis < 2 or ann["num_lidar_pts"] + ann["num_radar_pts"] == 0:
                        continue
                    c3d = b.corners()
                    if c3d[2].min() < 0.1:  # part of the box behind the camera
                        continue
                    uv = view_points(c3d, K, normalize=True)[:2]
                    x1, y1 = uv.min(1); x2, y2 = uv.max(1)
                    trunc = bool(x1 < 0 or y1 < 0 or x2 > W or y2 > H)
                    x1, x2 = np.clip([x1, x2], 0, W); y1, y2 = np.clip([y1, y2], 0, H)
                    if x2 - x1 < 4 or y2 - y1 < 4:
                        continue
                    c = CLS[b.name]
                    lines.append(f"{c} {(x1+x2)/2/W:.6f} {(y1+y2)/2/H:.6f} {(x2-x1)/W:.6f} {(y2-y1)/H:.6f}")
                    meta.append({"cls": c, "box": [float(x1), float(y1), float(x2), float(y2)],
                                 "z": float(b.center[2]), "trunc": trunc, "vis": vis})
                stem = Path(img).stem
                (Path(out) / split / "labels" / f"{stem}.txt").write_text("\n".join(lines))
                (Path(out) / split / "meta" / f"{stem}.json").write_text(json.dumps(
                    {"K": K.tolist(), "h_cam": h_c, "pitch": pitch, "objects": meta}))
                # symlink or copy img into out/split/images/ here
            tok = sample["next"]
```

Create the `images`, `labels` and `meta` folders per split before running.

### distance/fusion.py

```python
import numpy as np

GROUND_CLASSES = {0, 1}  # cone, barrier

def pinhole(box, cls, K, pri):
    x1, y1, x2, y2 = box
    if pri[cls]["dim"] == "h":
        return K[1][1] * pri[cls]["H_eff"] / max(y2 - y1, 1e-3)
    return K[0][0] * pri[cls]["H_eff"] / max(x2 - x1, 1e-3)

def ground_plane(box, K, h_cam, pitch):
    alpha = np.arctan2(box[3] - K[1][2], K[1][1])
    ang = pitch + alpha
    if ang < np.deg2rad(0.5):
        return None
    return h_cam * np.cos(alpha) / np.sin(ang)

def estimate(box, cls, K, h_cam, pitch, pri, img_h, sigma_v=2.0):
    z0 = pinhole(box, cls, K, pri)
    touches_bottom = box[3] >= img_h - 2
    z1 = None if cls not in GROUND_CLASSES or touches_bottom else ground_plane(box, K, h_cam, pitch)
    if z1 is None:
        z = z0
    else:
        v0 = (z0 * pri[cls]["r"]) ** 2
        v1 = (z1 ** 2 / (K[1][1] * h_cam) * sigma_v) ** 2
        z = (z0 / v0 + z1 / v1) / (1 / v0 + 1 / v1)
    return pri[cls]["a"] * z + pri[cls]["b"]
```

`fit_priors.py` fills H\_eff (median of z × h\_px / f\_y over visibility-4, non-truncated train boxes), then fits a and b with `np.polyfit(z_pred, z_gt, 1)` per class. Stop sign uses `dim: w`, H\_eff 0.75 m, a 1, b 0 unless you collect calibration photos.

### eval/benchmark.py (timing core)

```python
import time, numpy as np
from ultralytics import YOLO

def bench(weights, frames, imgsz=640, device=0, warm=20, n=200, post=None):
    m = YOLO(weights)
    for i in range(warm):
        m.predict(frames[i % len(frames)], imgsz=imgsz, device=device, verbose=False)
    e2e, model_ms = [], []
    for i in range(n):
        t0 = time.perf_counter()
        r = m.predict(frames[i % len(frames)], imgsz=imgsz, device=device, verbose=False)[0]
        if post:
            post(r)  # distance + tracking + drawing
        e2e.append((time.perf_counter() - t0) * 1000)
        model_ms.append(sum(r.speed.values()))
    e2e = np.array(e2e)
    return {"fps": 1000 / e2e.mean(), "p50_ms": np.percentile(e2e, 50),
            "p95_ms": np.percentile(e2e, 95), "model_ms": float(np.mean(model_ms))}
```

Pass `device='cpu'` for CPU runs. Exported weights (`.engine`, `_openvino_model/`, `.onnx`) load through the same `YOLO(...)` call, so one script covers every variant.

## 2-day execution plan

Start GPU training by midday on day 1, so it runs while you build the distance module; all work lands in the assignment repo, not a personal one.

**Day 1: data, training, distance**

- [ ] Environment: ultralytics, nuscenes-devkit, fiftyone, openvino, onnxruntime, torch-pruning
- [ ] Download nuScenes v1.0-mini and the COCO stop sign subset
- [ ] Run the converter, sanity-check 20 drawn images and the per-class counts
- [ ] Pseudo-label stop signs on nuScenes images, merge, write data.yaml
- [ ] B0 zero-shot eval
- [ ] Start B1 (yolo26n at 640) and B1b (yolo11n at 640), then B2 (yolo26s at 960) in the background
- [ ] fit\_priors.py: H\_eff, h\_cam refinement, per-class a and b
- [ ] Distance eval on oracle boxes: D0, D1, fusion, by range bin

**Day 2: optimize, evaluate, ship**

- [ ] Distance eval on predicted boxes from B1
- [ ] Exports E1 to E5, prune E6 and fine-tune, benchmark every variant on CPU and GPU
- [ ] Fill the results table, draw the accuracy vs speed plot, pick the deployed variant
- [ ] app/realtime.py with tracking and Kalman smoothing; record a demo video
- [ ] Stop sign calibration photos at 2, 4, 6, 8 m
- [ ] Extra credit if time remains: bird's-eye view first, then optical flow
- [ ] README: setup, derivations, results tables, before vs after frames, known limits

## Extra credit

Bird's-eye view is nearly free once ground-plane calibration exists; optical flow also buys CPU FPS, so do those two before stereo.

### Homography to bird's-eye view

For the ground plane, set the world Z to 0. The full projection then collapses to a 3x3 homography built from K and the first two rotation columns plus the translation.

```latex
\begin{bmatrix} u \\ v \\ 1 \end{bmatrix} \sim K \,[\, r_1 \;\; r_2 \;\; t \,]\begin{bmatrix} X \\ Y \\ 1 \end{bmatrix} = H_g \begin{bmatrix} X \\ Y \\ 1 \end{bmatrix}
```

- Build H\_g from the nuScenes extrinsics, compose with a meters-to-pixels scale, and call `cv2.warpPerspective`.
- Plot each cone and barrier at its box-bottom point mapped through the inverse of H\_g. This cross-checks D1 visually.

### Optical flow tracking

- Track Shi-Tomasi corners inside each box with `cv2.calcOpticalFlowPyrLK`; move the box by the median flow.
- Run the detector every N frames and propagate boxes in between: a direct CPU FPS gain.
- Time to contact from the box scale ratio s = h\_t / h\_(t-1) over frame gap Δt: TTC = Δt / (s − 1). It needs no size prior at all.

### Epipolar geometry: disparity to depth

For a rectified stereo pair with baseline B, a point at depth Z projects to x\_L = f X / Z and x\_R = f (X − B) / Z.

```latex
d = x_L - x_R = \frac{f B}{Z} \quad \Rightarrow \quad Z = \frac{f B}{d}, \qquad \frac{\partial Z}{\partial d} = -\frac{Z^2}{f B}
```

Depth error grows with the square of range, the same law as D1. Write the derivation in the README; nuScenes has no stereo pair, so validate on a KITTI stereo frame if you want numbers.

## Risks and mitigations

The two risks most likely to sink accuracy are missing labels across merged datasets and box-geometry bias in distance; both have cheap fixes above.

| Risk | Effect | Mitigation |
| --- | --- | --- |
| Stop signs unlabeled in nuScenes | Model learns signs as background, stop sign AP drops | Pseudo-label with COCO model at confidence 0.5+ |
| Projected-cuboid boxes taller than the object | Pinhole distance biased short | Fit H\_eff and per-class a, b on train |
| Tiny dataset (about 2,400 images) | Overfitting, noisy val numbers | freeze=10, strong augmentation, second fold swap |
| Far cones under 15 px | Low recall past 25 m | Higher input size or P2 head; report by range bin |
| INT8 hurts small objects | Cone AP falls more than the budget | More calibration images, FP16 head |
| Robot camera mounted lower than the car's (about 1.5 m) | Ground-plane constants wrong | Set h\_c and pitch from the robot mount; D0 is mount-independent |
| No stop sign distance ground truth | Unverified class | Tape-measured photo set |
| 2-day deadline | Optional parts eat the core | Ship the B1 + D0/D1 + TensorRT path first |

## Sources

- [YOLO26: architecture and edge benchmarks (arXiv 2509.25164)](https://arxiv.org/abs/2509.25164v2)
- [YOLO26 vs YOLOv8 independent benchmark (arXiv 2605.24831)](https://arxiv.org/pdf/2605.24831)
- [nuScenes v1.0-mini category stats, devkit output](https://github.com/daetz-coder/TrajectoryPrediction)
- [nuImages tutorial and mini download](https://nuscenes.org/tutorials/nuimages_tutorial.html)
