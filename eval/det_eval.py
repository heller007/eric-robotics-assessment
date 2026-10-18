"""Per-class detection metrics, computed the same way for every variant.

Ultralytics' own `model.val()` cannot score the zero-shot baseline: that model
predicts 80 COCO classes against a 3-class set, so the class indices do not line
up. Rather than evaluate B0 one way and the fine-tuned runs another - which would
make the comparison meaningless - this module implements COCO-style AP once and
runs everything through it, zero-shot included, by remapping predicted classes.

`--cross-check` additionally runs `model.val()` for a fine-tuned model, so the
custom numbers can be compared against the reference implementation.

Usage
  python -m eval.det_eval --run B0                       # zero-shot floor
  python -m eval.det_eval --weights runs/B1/weights/best.pt --name B1
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from common.config import class_names, load_yaml, resolve
from common.env import writable_root
from common.labels import iou_matrix, load_split

IOU_THRS = np.arange(0.5, 0.96, 0.05)
RANGE_BINS = [(0, 10), (10, 20), (20, 40), (40, 1e9)]
RECALL_CONF = 0.25  # the confidence a realtime pipeline would actually run at


# --------------------------------------------------------------------------- AP


def average_precision(tp: np.ndarray, conf: np.ndarray, n_gt: int) -> float:
    """COCO 101-point interpolated AP from a confidence-sorted TP indicator."""
    if n_gt == 0:
        return float("nan")
    if len(tp) == 0:
        return 0.0
    order = np.argsort(-conf)
    tp = tp[order]
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(1 - tp)
    recall = tp_cum / n_gt
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)

    # Precision envelope: the best precision achievable at this recall or higher.
    envelope = np.maximum.accumulate(precision[::-1])[::-1]

    # Sample it at COCO's 101 recall points. The PR curve is a step function, so
    # each point takes the envelope at the first recall that reaches it - linear
    # interpolation between points would invent precision that was never
    # measured. Recall levels the model never reaches contribute zero.
    grid = np.linspace(0, 1, 101)
    idx = np.searchsorted(recall, grid, side="left")
    sampled = np.where(idx < len(envelope), envelope[np.clip(idx, 0, len(envelope) - 1)], 0.0)
    return float(sampled.mean())


def match_class(
    gt_per_image: list[np.ndarray],
    pred_per_image: list[tuple[np.ndarray, np.ndarray]],
    iou_thr: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Greedy highest-IoU matching, one GT per prediction, per image.

    Returns the TP indicator, the confidences, and the ground-truth count.
    """
    tps, confs = [], []
    n_gt = 0
    for gt_boxes, (pred_boxes, pred_conf) in zip(gt_per_image, pred_per_image):
        n_gt += len(gt_boxes)
        if len(pred_boxes) == 0:
            continue
        order = np.argsort(-pred_conf)
        pred_boxes, pred_conf = pred_boxes[order], pred_conf[order]
        ious = iou_matrix(pred_boxes, gt_boxes)
        taken = np.zeros(len(gt_boxes), dtype=bool)
        for i in range(len(pred_boxes)):
            hit = 0.0
            best = -1
            for j in range(len(gt_boxes)):
                if taken[j] or ious[i, j] < iou_thr or ious[i, j] <= hit:
                    continue
                hit, best = ious[i, j], j
            if best >= 0:
                taken[best] = True
                tps.append(1.0)
            else:
                tps.append(0.0)
            confs.append(pred_conf[i])
    return np.asarray(tps), np.asarray(confs), n_gt


def recall_by_range(
    gt_per_image: list[np.ndarray],
    depth_per_image: list[np.ndarray],
    pred_per_image: list[tuple[np.ndarray, np.ndarray]],
    iou_thr: float = 0.5,
    conf_thr: float = RECALL_CONF,
) -> dict:
    """Recall split by ground-truth depth - this is where far cones show up."""
    found = {b: 0 for b in RANGE_BINS}
    total = {b: 0 for b in RANGE_BINS}
    for gt_boxes, depths, (pred_boxes, pred_conf) in zip(
        gt_per_image, depth_per_image, pred_per_image
    ):
        keep = pred_conf >= conf_thr
        pred_boxes = pred_boxes[keep]
        ious = iou_matrix(gt_boxes, pred_boxes)
        hit = ious.max(axis=1) >= iou_thr if ious.size else np.zeros(len(gt_boxes), bool)
        for j, z in enumerate(depths):
            if not np.isfinite(z):
                continue
            for lo, hi in RANGE_BINS:
                if lo <= z < hi:
                    total[(lo, hi)] += 1
                    found[(lo, hi)] += bool(hit[j])
                    break
    return {
        f"{lo}-{hi if hi < 1e9 else 'inf'}m": {
            "gt": total[(lo, hi)],
            "recall": (found[(lo, hi)] / total[(lo, hi)]) if total[(lo, hi)] else None,
        }
        for lo, hi in RANGE_BINS
    }


# -------------------------------------------------------------------- inference


def predict_all(
    weights: str,
    records: list[dict],
    imgsz: int,
    conf: float,
    device,
    class_map: dict[int, int] | None,
    batch: int = 16,
) -> list[dict]:
    """Run the model over every record; returns boxes/classes/confs per image."""
    from ultralytics import YOLO

    model = YOLO(weights)
    wanted = sorted(class_map) if class_map else None
    preds = []
    for start in tqdm(range(0, len(records), batch), desc="predict"):
        chunk = records[start : start + batch]
        results = model.predict(
            [str(r["image"]) for r in chunk],
            imgsz=imgsz,
            conf=conf,
            device=device,
            classes=wanted,
            verbose=False,
        )
        for res in results:
            boxes = res.boxes.xyxy.cpu().numpy() if len(res.boxes) else np.zeros((0, 4))
            classes = (
                res.boxes.cls.cpu().numpy().astype(int)
                if len(res.boxes)
                else np.zeros((0,), int)
            )
            confs = (
                res.boxes.conf.cpu().numpy() if len(res.boxes) else np.zeros((0,))
            )
            if class_map:
                classes = np.array([class_map.get(int(c), -1) for c in classes], dtype=int)
                keep = classes >= 0
                boxes, classes, confs = boxes[keep], classes[keep], confs[keep]
            preds.append({"boxes": boxes, "classes": classes, "confs": confs})
    return preds


# -------------------------------------------------------------------- evaluation


def evaluate(records: list[dict], preds: list[dict], names: list[str]) -> dict:
    out = {"per_class": {}, "n_images": len(records)}
    ap50s, ap5095s = [], []

    for c, name in enumerate(names):
        gt_boxes = [r["boxes"][r["classes"] == c] for r in records]
        gt_depths = [r["depths"][r["classes"] == c] for r in records]
        pr = [
            (p["boxes"][p["classes"] == c], p["confs"][p["classes"] == c]) for p in preds
        ]

        aps = []
        for thr in IOU_THRS:
            tp, conf, n_gt = match_class(gt_boxes, pr, thr)
            aps.append(average_precision(tp, conf, n_gt))
        ap50 = aps[0]
        ap5095 = float(np.nanmean(aps)) if np.any(np.isfinite(aps)) else float("nan")

        # precision / recall / F1 at the confidence a pipeline would deploy with
        tp, conf, n_gt = match_class(gt_boxes, pr, 0.5)
        keep = conf >= RECALL_CONF if len(conf) else np.zeros(0, bool)
        tp_n = float(tp[keep].sum()) if len(tp) else 0.0
        fp_n = float((1 - tp[keep]).sum()) if len(tp) else 0.0
        precision = tp_n / (tp_n + fp_n) if (tp_n + fp_n) else 0.0
        recall = tp_n / n_gt if n_gt else float("nan")
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        )

        out["per_class"][name] = {
            "gt_instances": int(n_gt),
            "predictions": int(sum(len(p[1]) for p in pr)),
            "AP50": ap50,
            "AP50_95": ap5095,
            f"precision@{RECALL_CONF}": precision,
            f"recall@{RECALL_CONF}": recall,
            f"f1@{RECALL_CONF}": f1,
            "recall_by_range": recall_by_range(gt_boxes, gt_depths, pr),
        }
        if np.isfinite(ap50):
            ap50s.append(ap50)
        if np.isfinite(ap5095):
            ap5095s.append(ap5095)

    out["mAP50"] = float(np.mean(ap50s)) if ap50s else float("nan")
    out["mAP50_95"] = float(np.mean(ap5095s)) if ap5095s else float("nan")
    return out


def cross_check(weights: str, data: str, imgsz: int, device) -> dict:
    """Ultralytics' own validator, for comparison. Fine-tuned models only."""
    from ultralytics import YOLO

    res = YOLO(weights).val(
        data=data, imgsz=imgsz, device=device, rect=True, verbose=False
    )
    box = res.box
    return {
        "mAP50": float(box.map50),
        "mAP50_95": float(box.map),
        "per_class_AP50": {
            name: float(box.ap50[i]) for i, name in enumerate(class_names())
        }
        if getattr(box, "ap50", None) is not None and len(box.ap50)
        else {},
    }


def report(result: dict, title: str) -> str:
    lines = [f"\n{title}", "-" * len(title)]
    lines.append(f"mAP50 {result['mAP50']:.4f}   mAP50-95 {result['mAP50_95']:.4f}")
    for name, m in result["per_class"].items():
        lines.append(
            f"  {name:<10} AP50 {m['AP50']:.4f}  AP50-95 {m['AP50_95']:.4f}  "
            f"P {m[f'precision@{RECALL_CONF}']:.3f}  R {m[f'recall@{RECALL_CONF}']:.3f}  "
            f"(gt {m['gt_instances']})"
        )
        bins = "  ".join(
            f"{k} R={v['recall']:.2f} (n={v['gt']})"
            for k, v in m["recall_by_range"].items()
            if v["recall"] is not None
        )
        if bins:
            lines.append(f"             by range: {bins}")
    return "\n".join(lines)


def append_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = path.exists()
    with open(path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row))
        if not existing:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", default=None, help="a .pt/.engine/.onnx, or omit with --run")
    ap.add_argument("--run", default=None, help="an entry in configs/runs.yaml, e.g. B0")
    ap.add_argument("--name", default=None, help="label for the results row")
    ap.add_argument("--out", default=None, help="dataset root")
    ap.add_argument("--split", default="val")
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--conf", type=float, default=0.001, help="low, so AP sees the tail")
    ap.add_argument("--device", default=0)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--cross-check", action="store_true", help="also run model.val()")
    ap.add_argument("--results", default="results")
    args = ap.parse_args()

    paths = load_yaml("paths.yaml")
    runs = load_yaml("runs.yaml")
    out = resolve(args.out) if args.out else writable_root() / paths["out_root"]
    names = class_names()

    class_map = None
    imgsz = args.imgsz or 640
    weights = args.weights
    name = args.name

    if args.run:
        spec = {**runs["defaults"], **runs["runs"][args.run]}
        imgsz = args.imgsz or spec["imgsz"]
        name = name or args.run
        if spec.get("zero_shot"):
            coco_index = load_yaml("classes.yaml")["coco_pretrained_index"]
            class_map = {coco_index: names.index("stop_sign")}
            print(
                f"zero-shot: COCO index {coco_index} -> stop_sign; cone and barrier "
                "have no COCO class, so their AP is 0 by construction"
            )
        if weights is None:
            from detection.train import resolve_weights

            weights = resolve_weights(spec["models"])
    if weights is None:
        ap.error("pass --weights or --run")
    name = name or Path(weights).stem

    records = load_split(out, args.split)
    print(f"{len(records)} {args.split} images from {out}")
    preds = predict_all(
        weights, records, imgsz, args.conf, args.device, class_map, args.batch
    )
    result = evaluate(records, preds, names)
    result.update({"variant": name, "weights": str(weights), "imgsz": imgsz, "split": args.split})
    print(report(result, f"{name}  ({Path(str(weights)).name}, imgsz {imgsz})"))

    if args.cross_check:
        ref = cross_check(str(weights), str(resolve("configs/data.yaml")), imgsz, args.device)
        result["ultralytics_val"] = ref
        print(
            f"\nultralytics model.val(): mAP50 {ref['mAP50']:.4f}  "
            f"mAP50-95 {ref['mAP50_95']:.4f}"
        )

    results_dir = resolve(args.results)
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / f"det_eval_{name}.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    row = {
        "variant": name,
        "weights": Path(str(weights)).name,
        "imgsz": imgsz,
        "mAP50": round(result["mAP50"], 4),
        "mAP50_95": round(result["mAP50_95"], 4),
    }
    for cls in names:
        row[f"AP50_{cls}"] = round(result["per_class"][cls]["AP50"], 4)
    append_csv(results_dir / "detection.csv", row)
    print(f"\nwrote {results_dir / f'det_eval_{name}.json'} and appended detection.csv")


if __name__ == "__main__":
    main()
