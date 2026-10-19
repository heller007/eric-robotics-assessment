"""The full 3-class set with oriented labels, for a like-for-like OBB variant.

Measured on barriers alone, oriented boxes gained +0.10 mAP50 and +0.16
mAP50-95 at no latency cost (1.9 ms against 2.1 ms). This builds the same three
classes with oriented labels so the whole pipeline can be compared.

Where each class's orientation comes from:

  cone, barrier   minAreaRect over the projected 8 cuboid corners. Barriers are
                  merged into one structure per row first, which is where the
                  gain comes from; cones are left discrete and will come out
                  near axis-aligned, which is the honest answer for an upright
                  object rather than a defect.
  stop_sign       axis-aligned quads converted from the labels Task 1 already
                  wrote - both the COCO boxes and the pseudo-labels on nuScenes
                  frames. COCO gives no orientation, and an octagon fills ~82%
                  of its bounding box, so there is nothing to recover.

One caveat worth stating in any write-up: an OBB variant has **no zero-shot
baseline**. B0 works because a COCO-pretrained model knows `stop sign`; there is
no COCO-pretrained OBB equivalent, since DOTA has no such class. So this scores
against the axis-aligned runs, not against a floor.

  python -m data.obb_dataset
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from tqdm import tqdm

from common.cli import friendly
from common.config import CONFIG_DIR, class_names, load_paths, load_yaml, resolve, save_yaml
from common.env import dataset_root_for_cli, find_nuscenes_root, writable_root
from common.staging import stage_image
from data.barrier_obb import group_structures, obb_line
from data.nuscenes_to_yolo import (
    CAMS,
    IMG_H,
    IMG_W,
    MERGE_CLASSES,
    MERGE_GAP,
    MIN_VISIBILITY,
    camera_height_pitch,
)

STOP_SIGN = 2


def quad_from_yolo(line: str) -> str | None:
    """An axis-aligned YOLO line as an oriented-box quad, clockwise."""
    parts = line.split()
    if len(parts) != 5:
        return None
    cls, cx, cy, bw, bh = int(parts[0]), *(float(v) for v in parts[1:])
    x1, x2 = cx - bw / 2, cx + bw / 2
    y1, y2 = cy - bh / 2, cy + bh / 2
    pts = [x1, y1, x2, y1, x2, y2, x1, y2]
    return f"{cls} " + " ".join(f"{min(max(v, 0.0), 1.0):.6f}" for v in pts)


def stop_sign_quads(source: Path, split: str, stem: str) -> list[str]:
    """Class-2 lines Task 1 wrote for this frame, as quads."""
    label = source / "labels" / split / f"{stem}.txt"
    if not label.exists():
        return []
    out = []
    for line in label.read_text(encoding="utf-8").splitlines():
        if not line.strip() or int(line.split()[0]) != STOP_SIGN:
            continue
        quad = quad_from_yolo(line)
        if quad:
            out.append(quad)
    return out


def build(dataroot: Path, source: Path, out: Path, val_scenes, gap: float) -> dict:
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils.geometry_utils import BoxVisibility, view_points
    from nuscenes.utils.splits import create_splits_scenes

    cls_map = load_yaml("classes.yaml")["nuscenes_map"]
    val_scenes = set(val_scenes or create_splits_scenes()["mini_val"])
    for split in ("train", "val"):
        for sub in ("images", "labels", "meta"):
            (out / sub / split).mkdir(parents=True, exist_ok=True)

    nusc = NuScenes("v1.0-mini", dataroot=str(dataroot), verbose=False)
    stats = Counter()
    seen_stems: set[str] = set()

    for scene in tqdm(nusc.scene, desc="nuScenes"):
        split = "val" if scene["name"] in val_scenes else "train"
        token = scene["first_sample_token"]
        while token:
            sample = nusc.get("sample", token)
            for cam in CAMS:
                if cam not in sample["data"]:
                    continue
                sd_token = sample["data"][cam]
                sd = nusc.get("sample_data", sd_token)
                img_path, boxes, K = nusc.get_sample_data(
                    sd_token, box_vis_level=BoxVisibility.ANY
                )
                h_cam, pitch = camera_height_pitch(nusc, sd)

                per_class: dict[int, list[dict]] = {}
                for box in boxes:
                    if box.name not in cls_map:
                        continue
                    ann = nusc.get("sample_annotation", box.token)
                    if int(ann["visibility_token"]) < MIN_VISIBILITY:
                        continue
                    if ann["num_lidar_pts"] + ann["num_radar_pts"] == 0:
                        continue
                    c3d = box.corners()
                    if c3d[2].min() < 0.1:
                        continue
                    uv = view_points(c3d, K, normalize=True)[:2]
                    per_class.setdefault(cls_map[box.name], []).append(
                        {
                            "corners": uv,
                            "z": float(box.center[2]),
                            "box": [
                                float(uv[0].min()), float(uv[1].min()),
                                float(uv[0].max()), float(uv[1].max()),
                            ],
                        }
                    )

                lines, meta_objs = [], []
                for cls, items in per_class.items():
                    # only barriers form a continuous structure; cones stay discrete
                    structures = (
                        group_structures(items, gap)
                        if cls in MERGE_CLASSES
                        else [{**it, "n_merged": 1} for it in items]
                    )
                    for s in structures:
                        quad = obb_line(s["corners"])
                        if quad is None:
                            stats["too_small"] += 1
                            continue
                        lines.append(f"{cls}{quad[1:]}")  # obb_line hardcodes class 0
                        meta_objs.append(
                            {
                                "cls": cls, "box": s["box"], "z": s["z"],
                                "n_merged": s["n_merged"], "trunc": False, "vis": 4,
                            }
                        )
                        stats[f"{split}_cls{cls}"] += 1

                src = Path(img_path)
                lines += stop_sign_quads(source, split, src.stem)
                stats[f"{split}_cls2"] += len(lines) - len(meta_objs)

                stage_image(src, out / "images" / split / src.name, "auto")
                (out / "labels" / split / f"{src.stem}.txt").write_text(
                    "\n".join(lines), encoding="utf-8"
                )
                (out / "meta" / split / f"{src.stem}.json").write_text(
                    json.dumps(
                        {
                            "image": src.name, "camera": cam, "scene": scene["name"],
                            "width": IMG_W, "height": IMG_H,
                            "K": np.asarray(K).tolist(), "h_cam": h_cam, "pitch": pitch,
                            "objects": meta_objs,
                        }
                    ),
                    encoding="utf-8",
                )
                seen_stems.add(src.stem)
                stats[f"{split}_images"] += 1
            token = sample["next"]

    # the COCO stop-sign frames: no meta sidecar in the source, so no camera model
    for split in ("train", "val"):
        for img in tqdm(
            sorted((source / "images" / split).glob("*")), desc=f"COCO {split}"
        ):
            if img.stem in seen_stems or img.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            quads = stop_sign_quads(source, split, img.stem)
            if not quads:
                continue
            stage_image(img, out / "images" / split / img.name, "auto")
            (out / "labels" / split / f"{img.stem}.txt").write_text(
                "\n".join(quads), encoding="utf-8"
            )
            stats[f"{split}_images"] += 1
            stats[f"{split}_cls2"] += len(quads)
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nuscenes-root", default=None)
    ap.add_argument("--source", default=None, help="the Task 1 dataset, for stop signs")
    ap.add_argument("--out", default=None)
    ap.add_argument("--val-scenes", nargs="*", default=None)
    ap.add_argument("--merge-gap", type=float, default=MERGE_GAP)
    args = ap.parse_args()

    paths = load_paths()
    root = find_nuscenes_root(args.nuscenes_root or paths.get("nuscenes_root"))
    source = dataset_root_for_cli(args.source)
    out = resolve(args.out) if args.out else writable_root() / "datasets" / "yolo_3class_obb"
    val_scenes = args.val_scenes or paths.get("val_scenes")
    names = class_names()

    print(f"nuScenes : {root}")
    print(f"source   : {source}  (stop-sign labels come from here)")
    print(f"output   : {out}")
    print(f"val      : {', '.join(val_scenes)}")
    stats = build(root, source, out, val_scenes, args.merge_gap)

    for split in ("train", "val"):
        counts = ", ".join(f"{names[c]}={stats[f'{split}_cls{c}']}" for c in range(3))
        print(f"{split}: {stats[f'{split}_images']} images | {counts}")
    if stats["too_small"]:
        print(f"dropped {stats['too_small']} structures as too small")

    cfg = {
        "path": str(out.resolve()).replace("\\", "/"),
        "train": str((out / "images" / "train").resolve()).replace("\\", "/"),
        "val": str((out / "images" / "val").resolve()).replace("\\", "/"),
        "names": {i: n for i, n in enumerate(names)},
    }
    dest = save_yaml(cfg, CONFIG_DIR / "data_obb.yaml")
    print(f"wrote {dest}")
    print("\nnext: python -m detection.train --run B5obb")


if __name__ == "__main__":
    friendly(main)()
