"""Reading the converted dataset: YOLO label files, sidecar meta, box geometry.

Shared by the evaluators and the distance module so there is one definition of
"the ground truth for this image".
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

IMG_EXT = (".jpg", ".jpeg", ".png")


def image_size(path: Path) -> tuple[int, int]:
    """(width, height) from the file header - no full decode."""
    from PIL import Image

    with Image.open(path) as im:
        return im.size


def list_images(dirpath: Path) -> list[Path]:
    return sorted(p for p in Path(dirpath).glob("*") if p.suffix.lower() in IMG_EXT)


def load_yolo_labels(label_path: Path, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """-> boxes (N,4) xyxy in pixels, classes (N,)."""
    if not Path(label_path).exists():
        return np.zeros((0, 4)), np.zeros((0,), dtype=int)
    boxes, classes = [], []
    for line in Path(label_path).read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        cls, cx, cy, bw, bh = int(parts[0]), *(float(v) for v in parts[1:])
        boxes.append(
            [
                (cx - bw / 2) * width,
                (cy - bh / 2) * height,
                (cx + bw / 2) * width,
                (cy + bh / 2) * height,
            ]
        )
        classes.append(cls)
    if not boxes:
        return np.zeros((0, 4)), np.zeros((0,), dtype=int)
    return np.asarray(boxes, dtype=float), np.asarray(classes, dtype=int)


def load_meta(meta_path: Path) -> dict | None:
    p = Path(meta_path)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def depth_for(meta_objects: list[dict], cls: int, box, tol_px: float = 4.0) -> float:
    """Ground-truth Z of the meta box whose centre is nearest `box`, else NaN.

    A label line is a rounded, normalised copy of a meta box, so the centres agree
    to within a pixel or two; nothing matches for a pseudo-labelled stop sign,
    which is the intended answer since nuScenes gives it no depth.
    """
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    best, best_d = float("nan"), tol_px
    for obj in meta_objects or []:
        if obj["cls"] != cls:
            continue
        ox = (obj["box"][0] + obj["box"][2]) / 2
        oy = (obj["box"][1] + obj["box"][3]) / 2
        d = abs(ox - cx) + abs(oy - cy)
        if d < best_d:
            best, best_d = float(obj["z"]), d
    return best


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU of every box in a (N,4) against every box in b (M,4), both xyxy."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


def load_split(out: Path, split: str, with_depth: bool = True) -> list[dict]:
    """Every val/train image with its boxes, classes, depths and camera meta."""
    out = Path(out)
    records = []
    for img in list_images(out / "images" / split):
        width, height = image_size(img)
        boxes, classes = load_yolo_labels(
            out / "labels" / split / f"{img.stem}.txt", width, height
        )
        meta = load_meta(out / "meta" / split / f"{img.stem}.json") if with_depth else None
        depths = np.array(
            [depth_for(meta["objects"], c, b) for c, b in zip(classes, boxes)]
            if meta
            else [float("nan")] * len(classes)
        )
        records.append(
            {
                "image": img,
                "width": width,
                "height": height,
                "boxes": boxes,
                "classes": classes,
                "depths": depths,
                "meta": meta,
            }
        )
    return records
