"""Pseudo-label the stop signs nuScenes never annotates.

Merging nuScenes (cones, barriers) with COCO (stop signs) teaches the model that
every stop sign standing in a nuScenes frame is background, which caps stop-sign
AP. Fix: run a COCO-pretrained YOLO over the nuScenes images and keep its
stop-sign detections above a confidence threshold as labels.

Only nuScenes images are touched; they are the ones with a meta/<split>/*.json
sidecar. Re-running is safe: existing class-2 lines in a nuScenes label file can
only have come from a previous pass, so they are stripped first.

Usage
  python -m data.pseudo_label --conf 0.5 --review 30
"""
from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm

from common.config import load_yaml, resolve
from common.env import writable_root

STOP_SIGN_CLASS = 2


def nuscenes_images(out: Path, split: str) -> list[Path]:
    meta_dir = out / "meta" / split
    img_dir = out / "images" / split
    if not meta_dir.is_dir():
        return []
    found = []
    for meta in sorted(meta_dir.glob("*.json")):
        for ext in (".jpg", ".png", ".jpeg"):
            img = img_dir / (meta.stem + ext)
            if img.exists():
                found.append(img)
                break
    return found


def strip_class(label_path: Path, cls: int) -> list[str]:
    if not label_path.exists():
        return []
    kept = [
        ln
        for ln in label_path.read_text(encoding="utf-8").splitlines()
        if ln.strip() and int(ln.split()[0]) != cls
    ]
    return kept


def run(
    out: Path,
    weights: str,
    conf: float,
    imgsz: int,
    device: str | int,
    splits: list[str],
    review: int,
    review_dir: Path,
) -> dict:
    from ultralytics import YOLO

    coco_index = load_yaml("classes.yaml")["coco_pretrained_index"]
    model = YOLO(weights)
    summary = {}
    review_crops: list[tuple[Path, list[float]]] = []

    for split in splits:
        images = nuscenes_images(out, split)
        if not images:
            continue
        added_boxes = 0
        added_images = 0
        for img in tqdm(images, desc=f"pseudo-label {split}"):
            res = model.predict(
                str(img),
                imgsz=imgsz,
                conf=conf,
                device=device,
                classes=[coco_index],
                verbose=False,
            )[0]
            label_path = out / "labels" / split / f"{img.stem}.txt"
            lines = strip_class(label_path, STOP_SIGN_CLASS)

            h, w = res.orig_shape
            new = 0
            for box in res.boxes:
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
                if (x2 - x1) < 4 or (y2 - y1) < 4:
                    continue
                lines.append(
                    f"{STOP_SIGN_CLASS} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
                    f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}"
                )
                new += 1
                if len(review_crops) < review:
                    review_crops.append((img, [x1, y1, x2, y2]))
            label_path.write_text("\n".join(lines), encoding="utf-8")
            added_boxes += new
            added_images += 1 if new else 0

        summary[split] = {"images_with_signs": added_images, "boxes": added_boxes}
        print(f"{split}: {summary[split]}")

    if review_crops:
        save_review_sheet(review_crops, review_dir)
    return summary


def save_review_sheet(crops: list[tuple[Path, list[float]]], review_dir: Path) -> None:
    """A contact sheet to eyeball: pseudo-labels are only as good as this check."""
    import cv2
    import numpy as np

    review_dir.mkdir(parents=True, exist_ok=True)
    tiles = []
    for img_path, (x1, y1, x2, y2) in crops:
        im = cv2.imread(str(img_path))
        if im is None:
            continue
        pad = 20
        h, w = im.shape[:2]
        crop = im[
            max(0, int(y1) - pad) : min(h, int(y2) + pad),
            max(0, int(x1) - pad) : min(w, int(x2) + pad),
        ]
        if crop.size == 0:
            continue
        tiles.append(cv2.resize(crop, (128, 128)))

    if not tiles:
        return
    cols = 6
    rows = (len(tiles) + cols - 1) // cols
    sheet = np.zeros((rows * 128, cols * 128, 3), dtype=np.uint8)
    for i, tile in enumerate(tiles):
        r, c = divmod(i, cols)
        sheet[r * 128 : (r + 1) * 128, c * 128 : (c + 1) * 128] = tile
    path = review_dir / "pseudo_label_review.jpg"
    cv2.imwrite(str(path), sheet)
    print(f"wrote {path} - open it and confirm these are really stop signs")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=None)
    ap.add_argument("--weights", default="yolo11n.pt", help="any COCO-pretrained YOLO")
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--device", default=0)
    ap.add_argument("--splits", nargs="*", default=["train", "val"])
    ap.add_argument("--review", type=int, default=30, help="crops in the contact sheet")
    ap.add_argument("--review-dir", default="results")
    args = ap.parse_args()

    paths = load_yaml("paths.yaml")
    out = resolve(args.out) if args.out else writable_root() / paths["out_root"]
    run(
        out,
        args.weights,
        args.conf,
        args.imgsz,
        args.device,
        args.splits,
        args.review,
        resolve(args.review_dir),
    )


if __name__ == "__main__":
    main()
