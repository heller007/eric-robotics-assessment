"""Score the distance estimates against nuScenes depth.

Oracle boxes first (`--boxes oracle`): run the geometry on ground-truth boxes, so
what you measure is the geometry rather than the detector. Then predicted boxes
(`--boxes pred`), matched to ground truth at IoU 0.5 and scored on true
positives only - a missed object is a detection failure, already counted in
Task 2, and folding it in here would hide which part is wrong.

Reported per class and per range bin, for each stage separately, because they
fail differently: AbsRel, RMSE, and the share within 10% and 20%.

  python -m eval.dist_eval --boxes oracle
  python -m eval.dist_eval --boxes pred --weights runs/B1/weights/best.pt
"""
from __future__ import annotations

import argparse
import csv
import json

import numpy as np
from tqdm import tqdm

from common.cli import friendly
from common.config import class_names, load_yaml, resolve
from common.env import dataset_root_for_cli
from common.labels import iou_matrix, load_split
from distance.geometry import estimate

RANGE_BINS = [(0, 10), (10, 20), (20, 40), (40, 1e9)]
STAGES = ("d0", "d1", "fused", "corrected")


def metrics(pred: np.ndarray, true: np.ndarray) -> dict:
    if len(pred) == 0:
        return {"n": 0}
    err = pred - true
    rel = np.abs(err) / true
    return {
        "n": int(len(pred)),
        "AbsRel": float(rel.mean()),
        "RMSE": float(np.sqrt((err**2).mean())),
        "within10": float((rel <= 0.10).mean()),
        "within20": float((rel <= 0.20).mean()),
        "bias": float(err.mean()),  # signed, so a short read is visible
    }


def collect(records, priors, names, preds=None, iou_thr=0.5) -> dict[str, dict[str, list]]:
    """-> {class: {stage: [(estimated, true), ...]}}"""
    out = {n: {s: [] for s in STAGES} for n in names}
    for i, rec in enumerate(records):
        meta = rec["meta"]
        if not meta:
            continue  # COCO frame: no camera model, no depth
        gt_boxes, gt_cls, gt_z = rec["boxes"], rec["classes"], rec["depths"]

        if preds is None:
            boxes, classes, truths = gt_boxes, gt_cls, gt_z
        else:
            # match predictions to ground truth; score the true positives
            p = preds[i]
            if len(p["boxes"]) == 0 or len(gt_boxes) == 0:
                continue
            ious = iou_matrix(p["boxes"], gt_boxes)
            boxes, classes, truths = [], [], []
            taken = set()
            order = np.argsort(-p["confs"])
            for pi in order:
                row = ious[pi]
                best, best_iou = -1, iou_thr
                for gi in range(len(gt_boxes)):
                    if gi in taken or gt_cls[gi] != p["classes"][pi]:
                        continue
                    if row[gi] >= best_iou:
                        best, best_iou = gi, row[gi]
                if best < 0:
                    continue
                taken.add(best)
                boxes.append(p["boxes"][pi])
                classes.append(p["classes"][pi])
                truths.append(gt_z[best])
            if not boxes:
                continue

        for box, cls, true_z in zip(boxes, classes, truths):
            if not np.isfinite(true_z) or true_z <= 0:
                continue  # a pseudo-labelled stop sign has no depth
            name = names[int(cls)]
            got = estimate(
                box,
                name,
                meta["K"],
                meta["h_cam"],
                meta["pitch"],
                priors,
                img_height=meta["height"],
            )
            for stage in STAGES:
                if got[stage] is None:
                    continue
                out[name][stage].append((float(got[stage]), float(true_z)))
    return out


def summarise(collected: dict) -> dict:
    result = {}
    for name, stages in collected.items():
        entry = {}
        for stage, pairs in stages.items():
            if not pairs:
                continue
            pred = np.array([p for p, _ in pairs])
            true = np.array([t for _, t in pairs])
            entry[stage] = {"all": metrics(pred, true), "by_range": {}}
            for lo, hi in RANGE_BINS:
                keep = (true >= lo) & (true < hi)
                label = f"{lo}-{hi if hi < 1e9 else 'inf'}m"
                entry[stage]["by_range"][label] = metrics(pred[keep], true[keep])
        if entry:
            result[name] = entry
    return result


def report(result: dict, title: str) -> str:
    lines = [f"\n{title}", "-" * len(title)]
    for name, stages in result.items():
        lines.append(f"\n{name}")
        lines.append(
            f"  {'stage':<10}{'n':>6}{'AbsRel':>9}{'RMSE':>8}{'<10%':>7}{'<20%':>7}{'bias':>8}"
        )
        for stage, data in stages.items():
            m = data["all"]
            lines.append(
                f"  {stage:<10}{m['n']:>6}{m['AbsRel']:>9.3f}{m['RMSE']:>8.2f}"
                f"{m['within10']:>7.2f}{m['within20']:>7.2f}{m['bias']:>8.2f}"
            )
        best = stages.get("corrected") or next(iter(stages.values()))
        for label, m in best["by_range"].items():
            if m["n"]:
                lines.append(
                    f"     {label:<10}{m['n']:>5}  AbsRel {m['AbsRel']:.3f}  "
                    f"<20% {m['within20']:.2f}"
                )
    return "\n".join(lines)


def predict_all(weights, records, imgsz, conf, device, batch=16):
    from ultralytics import YOLO

    model = YOLO(weights)
    preds = []
    for start in tqdm(range(0, len(records), batch), desc="predict"):
        chunk = records[start : start + batch]
        for res in model.predict(
            [str(r["image"]) for r in chunk],
            imgsz=imgsz,
            conf=conf,
            device=device,
            verbose=False,
        ):
            if len(res.boxes):
                preds.append(
                    {
                        "boxes": res.boxes.xyxy.cpu().numpy(),
                        "classes": res.boxes.cls.cpu().numpy().astype(int),
                        "confs": res.boxes.conf.cpu().numpy(),
                    }
                )
            else:
                preds.append(
                    {
                        "boxes": np.zeros((0, 4)),
                        "classes": np.zeros((0,), int),
                        "confs": np.zeros((0,)),
                    }
                )
    return preds


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--boxes", choices=["oracle", "pred"], default="oracle")
    ap.add_argument("--weights", default=None, help="required for --boxes pred")
    ap.add_argument("--out", "--dataset-root", dest="out", default=None)
    ap.add_argument("--split", default="val")
    ap.add_argument("--priors", default="priors.yaml")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default="", help="blank = auto")
    ap.add_argument("--name", default=None)
    ap.add_argument("--results", default="results")
    args = ap.parse_args()

    root = dataset_root_for_cli(args.out)
    names = class_names()
    priors = load_yaml(args.priors)
    records = load_split(root, args.split)
    print(f"{len(records)} {args.split} frames from {root}")

    preds = None
    if args.boxes == "pred":
        if not args.weights:
            ap.error("--boxes pred needs --weights")
        preds = predict_all(args.weights, records, args.imgsz, args.conf, args.device)

    result = summarise(collect(records, priors, names, preds))
    name = args.name or args.boxes
    print(report(result, f"distance, {args.boxes} boxes, {args.split}"))

    dest = resolve(args.results)
    dest.mkdir(parents=True, exist_ok=True)
    (dest / f"dist_eval_{name}.json").write_text(
        json.dumps({"boxes": args.boxes, "split": args.split, "per_class": result}, indent=2),
        encoding="utf-8",
    )

    rows = []
    for cls, stages in result.items():
        for stage, data in stages.items():
            rows.append({"variant": name, "class": cls, "stage": stage, **data["all"]})
    if rows:
        path = dest / "distance.csv"
        exists = path.exists()
        with open(path, "a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            if not exists:
                writer.writeheader()
            writer.writerows(rows)
    print(f"\nwrote {dest / f'dist_eval_{name}.json'} and appended distance.csv")


if __name__ == "__main__":
    friendly(main)()
