"""Accuracy lever 4: class-balanced sampling by repeating images.

The merged set is lopsided - barriers outnumber cones, and stop signs come from a
different source entirely - so a plain epoch shows the rare classes far less
often. Repeat-factor sampling (the LVIS recipe) repeats an image as often as its
rarest class needs:

    f_c = fraction of train images containing class c
    r_c = max(1, sqrt(t / f_c))
    repeat(image) = round(max over the classes in that image of r_c)

Writes a train list file plus a dataset yaml pointing at it, leaving the original
dataset untouched. Train with it via  --run B1bal.

Usage
  python -m data.balance --out datasets/yolo_3class --threshold 0.5
"""
from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path

from common.cli import friendly
from common.config import REPO_ROOT, class_names, resolve, save_yaml
from common.env import dataset_root_for_cli
from common.labels import list_images


def image_classes(out: Path, split: str) -> dict[Path, set[int]]:
    per_image = {}
    for img in list_images(out / "images" / split):
        lbl = out / "labels" / split / f"{img.stem}.txt"
        classes = set()
        if lbl.exists():
            for line in lbl.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    classes.add(int(line.split()[0]))
        per_image[img] = classes
    return per_image


def instance_counts(out: Path, split: str) -> Counter:
    counts = Counter()
    for lbl in (out / "labels" / split).glob("*.txt"):
        for line in lbl.read_text(encoding="utf-8").splitlines():
            if line.strip():
                counts[int(line.split()[0])] += 1
    return counts


def repeat_factors(
    per_image: dict[Path, set[int]], threshold: float
) -> tuple[dict[Path, int], dict[int, float], Counter]:
    n_images = len(per_image)
    present = Counter()
    for classes in per_image.values():
        for c in classes:
            present[c] += 1

    r_c = {
        c: max(1.0, math.sqrt(threshold / (present[c] / n_images)))
        for c in present
        if present[c] > 0
    }
    repeats = {}
    for img, classes in per_image.items():
        if not classes:
            repeats[img] = 1  # background frames stay at their natural rate
            continue
        repeats[img] = max(1, round(max(r_c[c] for c in classes)))
    return repeats, r_c, present


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", "--dataset-root", dest="out", default=None, help="dataset root")
    ap.add_argument("--split", default="train")
    ap.add_argument("--threshold", type=float, default=0.5, help="target class frequency t")
    ap.add_argument("--list-name", default="train_balanced.txt")
    ap.add_argument(
        "--list-dir",
        default="datasets",
        help="where the repeated-image list is written; never the dataset itself, "
        "which may be a read-only cache",
    )
    ap.add_argument("--data-yaml", default=str(REPO_ROOT / "configs" / "data_balanced.yaml"))
    args = ap.parse_args()

    out = dataset_root_for_cli(args.out)
    names = class_names()

    per_image = image_classes(out, args.split)
    if not per_image:
        raise SystemExit(f"no images under {out / 'images' / args.split}")
    repeats, r_c, present = repeat_factors(per_image, args.threshold)
    before = instance_counts(out, args.split)

    after = Counter()
    lines = []
    for img, times in repeats.items():
        lines.extend([str(img.resolve()).replace("\\", "/")] * times)
        lbl = out / "labels" / args.split / f"{img.stem}.txt"
        if lbl.exists():
            for line in lbl.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    after[int(line.split()[0])] += times

    list_dir = resolve(args.list_dir)
    list_dir.mkdir(parents=True, exist_ok=True)
    list_path = list_dir / args.list_name
    list_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"{len(per_image)} images -> {len(lines)} entries per epoch")
    for c, name in enumerate(names):
        print(
            f"  {name:<10} in {present.get(c, 0):>5} images  repeat x{r_c.get(c, 1):.2f}  "
            f"instances {before.get(c, 0):>6} -> {after.get(c, 0):>6}"
        )

    # absolute everywhere: the list lives in the repo, the images in the dataset
    cfg = {
        "path": str(Path(out).resolve()).replace("\\", "/"),
        "train": str(list_path.resolve()).replace("\\", "/"),
        "val": str((Path(out) / "images" / "val").resolve()).replace("\\", "/"),
        "names": {i: n for i, n in enumerate(names)},
    }
    dest = save_yaml(cfg, Path(args.data_yaml))
    print(f"\nwrote {list_path}\nwrote {dest}\nnext: python -m detection.train --run B1bal")


if __name__ == "__main__":
    friendly(main)()
