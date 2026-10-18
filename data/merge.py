"""Finalise the merged 3-class set: verify pairing, count, write data.yaml.

The converter and the COCO exporter both write straight into the same tree, so
"merging" is really validation plus the dataset yaml Ultralytics will read.

Usage
  python -m data.merge --out datasets/yolo_3class
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from common.config import REPO_ROOT, class_names, load_yaml, resolve, save_yaml
from common.env import writable_root

IMG_EXT = (".jpg", ".jpeg", ".png")


def scan_split(out: Path, split: str) -> dict:
    img_dir = out / "images" / split
    lbl_dir = out / "labels" / split
    images = sorted(p for p in img_dir.glob("*") if p.suffix.lower() in IMG_EXT)

    counts = Counter()
    backgrounds = 0
    missing_labels = []
    bad_lines = []
    for img in images:
        lbl = lbl_dir / f"{img.stem}.txt"
        if not lbl.exists():
            missing_labels.append(img.name)
            continue
        lines = [ln for ln in lbl.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if not lines:
            backgrounds += 1
        for ln in lines:
            parts = ln.split()
            if len(parts) != 5:
                bad_lines.append(f"{lbl.name}: {ln}")
                continue
            cls = int(parts[0])
            vals = [float(v) for v in parts[1:]]
            if not all(0.0 <= v <= 1.0 for v in vals) or vals[2] <= 0 or vals[3] <= 0:
                bad_lines.append(f"{lbl.name}: {ln}")
                continue
            counts[cls] += 1

    orphan_labels = [
        p.name
        for p in lbl_dir.glob("*.txt")
        if not any((img_dir / f"{p.stem}{e}").exists() for e in IMG_EXT)
    ]
    return {
        "images": len(images),
        "backgrounds": backgrounds,
        "counts": dict(counts),
        "missing_labels": missing_labels,
        "orphan_labels": orphan_labels,
        "bad_lines": bad_lines,
    }


def write_data_yaml(out: Path, names: list[str], dest: Path) -> Path:
    cfg = {
        "path": str(out).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "names": {i: n for i, n in enumerate(names)},
    }
    return save_yaml(cfg, dest)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=None)
    ap.add_argument("--data-yaml", default=str(REPO_ROOT / "configs" / "data.yaml"))
    ap.add_argument(
        "--strict",
        action="store_true",
        help="exit non-zero if a split is missing a class or a label is malformed",
    )
    args = ap.parse_args()

    paths = load_yaml("paths.yaml")
    out = resolve(args.out) if args.out else writable_root() / paths["out_root"]
    names = class_names()

    report = {}
    problems = []
    for split in ("train", "val"):
        info = scan_split(out, split)
        report[split] = info
        present = ", ".join(
            f"{names[c]}={info['counts'].get(c, 0)}" for c in range(len(names))
        )
        print(
            f"{split}: {info['images']} images "
            f"({info['backgrounds']} with no object) | {present}"
        )
        if info["missing_labels"]:
            problems.append(
                f"{split}: {len(info['missing_labels'])} images without a label file "
                f"(first: {info['missing_labels'][0]})"
            )
        if info["orphan_labels"]:
            problems.append(
                f"{split}: {len(info['orphan_labels'])} label files without an image"
            )
        if info["bad_lines"]:
            problems.append(
                f"{split}: {len(info['bad_lines'])} malformed label lines "
                f"(first: {info['bad_lines'][0]})"
            )
        for c in range(len(names)):
            if info["counts"].get(c, 0) == 0:
                problems.append(f"{split}: no {names[c]} instances")

    dest = write_data_yaml(out, names, Path(args.data_yaml))
    print(f"wrote {dest}")
    (out / "merge_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    if problems:
        print("\nproblems:")
        for p in problems:
            print(f"  - {p}")
        if args.strict:
            raise SystemExit(1)
    else:
        print("\nno problems found")


if __name__ == "__main__":
    main()
