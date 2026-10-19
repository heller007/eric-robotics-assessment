"""Barrier-only datasets in both box types, for a controlled comparison.

A roadside barrier row receding from the camera is a long, thin, diagonal shape.
An axis-aligned box around it is mostly empty road, so even a correct prediction
scores poor IoU - which is a plausible reason barriers keep failing. A rotated
rectangle wraps it tightly.

The labels come free: the converter already projects 8 cuboid corners per
barrier, so `cv2.minAreaRect` over the corners of a structure gives the tight
rotated rectangle directly. No extra dataset, unlike segmentation.

Writes two datasets off the same frames, same filters, same split, differing
only in the label geometry, so the comparison attributes any gain to the box
type rather than to the data:

  <out>_aabb/   class cx cy w h                      (axis-aligned control)
  <out>_obb/    class x1 y1 x2 y2 x3 y3 x4 y4        (oriented, Ultralytics OBB)

  python -m data.barrier_obb
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from pyquaternion import Quaternion
from tqdm import tqdm

from common.cli import friendly
from common.config import CONFIG_DIR, load_paths, resolve, save_yaml
from common.env import find_nuscenes_root, writable_root
from common.staging import stage_image
from data.nuscenes_to_yolo import (
    CAMS,
    IMG_H,
    IMG_W,
    MIN_BOX_PX,
    MIN_VISIBILITY,
    MERGE_GAP,
    camera_height_pitch,
    touches,
)

BARRIER = "movable_object.barrier"


def group_structures(items: list[dict], gap: float) -> list[dict]:
    """Chain barriers that touch into one structure, keeping all their corners."""
    groups: list[list[dict]] = []
    for item in items:
        hit = None
        for group in groups:
            if any(touches(item["box"], other["box"], gap) for other in group):
                hit = group
                break
        if hit is None:
            groups.append([item])
        else:
            hit.append(item)

    merged = []
    for group in groups:
        corners = np.concatenate([g["corners"] for g in group], axis=1)
        nearest = min(group, key=lambda g: g["z"])
        merged.append(
            {
                "corners": corners,
                "z": nearest["z"],
                "n_merged": len(group),
                "box": [
                    float(corners[0].min()),
                    float(corners[1].min()),
                    float(corners[0].max()),
                    float(corners[1].max()),
                ],
            }
        )
    return merged


def aabb_line(box: list[float]) -> str | None:
    x1, y1, x2, y2 = box
    x1, x2 = np.clip([x1, x2], 0, IMG_W)
    y1, y2 = np.clip([y1, y2], 0, IMG_H)
    if (x2 - x1) < MIN_BOX_PX or (y2 - y1) < MIN_BOX_PX:
        return None
    return (
        f"0 {(x1 + x2) / 2 / IMG_W:.6f} {(y1 + y2) / 2 / IMG_H:.6f} "
        f"{(x2 - x1) / IMG_W:.6f} {(y2 - y1) / IMG_H:.6f}"
    )


def obb_line(corners: np.ndarray) -> str | None:
    """Tight rotated rectangle around the projected corners, as 4 points."""
    import cv2

    points = corners.T.astype(np.float32)  # N x 2
    (_, _), (w, h), _ = cv2.minAreaRect(points)
    if min(w, h) < MIN_BOX_PX:
        return None
    quad = cv2.boxPoints(cv2.minAreaRect(points))  # 4 x 2, in order
    quad[:, 0] = np.clip(quad[:, 0], 0, IMG_W) / IMG_W
    quad[:, 1] = np.clip(quad[:, 1], 0, IMG_H) / IMG_H
    return "0 " + " ".join(f"{v:.6f}" for v in quad.reshape(-1))


def build(
    dataroot: Path,
    out_aabb: Path,
    out_obb: Path,
    val_scenes,
    gap: float,
    split_mode: str = "scene",
    frame_stride: int = 5,
) -> dict:
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils.geometry_utils import BoxVisibility, view_points
    from nuscenes.utils.splits import create_splits_scenes

    val_scenes = set(val_scenes or create_splits_scenes()["mini_val"])
    for root in (out_aabb, out_obb):
        for split in ("train", "val"):
            for sub in ("images", "labels", "meta"):
                (root / sub / split).mkdir(parents=True, exist_ok=True)

    nusc = NuScenes("v1.0-mini", dataroot=str(dataroot), verbose=False)
    stats = Counter()

    for scene in tqdm(nusc.scene, desc="scenes"):
        scene_split = "val" if scene["name"] in val_scenes else "train"
        token = scene["first_sample_token"]
        frame = 0
        while token:
            sample = nusc.get("sample", token)
            # scene mode holds whole scenes out, which is the honest protocol but
            # on this dataset means testing barriers on a scene with no analogue
            # in training. frame mode interleaves instead: it leaks, because
            # keyframes 0.5 s apart show the same objects, so it cannot support a
            # generalisation claim - it answers only whether the detector can
            # learn the class at all.
            if split_mode == "frame":
                split = "val" if frame % frame_stride == 0 else "train"
            else:
                split = scene_split
            frame += 1
            for cam in CAMS:
                if cam not in sample["data"]:
                    continue
                sd_token = sample["data"][cam]
                sd = nusc.get("sample_data", sd_token)
                img_path, boxes, K = nusc.get_sample_data(
                    sd_token, box_vis_level=BoxVisibility.ANY
                )
                h_cam, pitch = camera_height_pitch(nusc, sd)

                items = []
                for box in boxes:
                    if box.name != BARRIER:
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
                    items.append(
                        {
                            "corners": uv,
                            "z": float(box.center[2]),
                            "box": [
                                float(uv[0].min()),
                                float(uv[1].min()),
                                float(uv[0].max()),
                                float(uv[1].max()),
                            ],
                        }
                    )

                structures = group_structures(items, gap) if items else []
                aabb_lines, obb_lines, meta_objs = [], [], []
                for s in structures:
                    a = aabb_line(s["box"])
                    o = obb_line(s["corners"])
                    if a is None or o is None:
                        stats["too_small"] += 1
                        continue
                    aabb_lines.append(a)
                    obb_lines.append(o)
                    meta_objs.append(
                        {"cls": 0, "box": s["box"], "z": s["z"], "n_merged": s["n_merged"],
                         "trunc": False, "vis": 4}
                    )
                    stats[f"{split}_structures"] += 1
                    stats[f"{split}_segments"] += s["n_merged"]

                src = Path(img_path)
                meta = json.dumps(
                    {
                        "image": src.name, "camera": cam, "scene": scene["name"],
                        "width": IMG_W, "height": IMG_H,
                        "K": np.asarray(K).tolist(), "h_cam": h_cam, "pitch": pitch,
                        "objects": meta_objs,
                    }
                )
                for root, lines in ((out_aabb, aabb_lines), (out_obb, obb_lines)):
                    stage_image(src, root / "images" / split / src.name, "auto")
                    (root / "labels" / split / f"{src.stem}.txt").write_text(
                        "\n".join(lines), encoding="utf-8"
                    )
                    (root / "meta" / split / f"{src.stem}.json").write_text(
                        meta, encoding="utf-8"
                    )
                stats[f"{split}_images"] += 1
            token = sample["next"]
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nuscenes-root", default=None)
    ap.add_argument("--out", default=None, help="base path; _aabb and _obb are appended")
    ap.add_argument("--val-scenes", nargs="*", default=None)
    ap.add_argument("--merge-gap", type=float, default=MERGE_GAP)
    ap.add_argument(
        "--split-mode",
        choices=["scene", "frame"],
        default="scene",
        help="scene: hold whole scenes out (honest). frame: every Nth frame, "
        "which leaks and proves only that the class is learnable.",
    )
    ap.add_argument("--frame-stride", type=int, default=5)
    ap.add_argument("--tag", default="", help="suffix for the dataset and config names")
    args = ap.parse_args()

    paths = load_paths()
    root = find_nuscenes_root(args.nuscenes_root or paths.get("nuscenes_root"))
    tag = f"_{args.tag}" if args.tag else ""
    base = resolve(args.out) if args.out else writable_root() / "datasets" / f"barrier{tag}"
    out_aabb, out_obb = Path(f"{base}_aabb"), Path(f"{base}_obb")
    val_scenes = args.val_scenes or paths.get("val_scenes")

    print(f"nuScenes : {root}")
    if args.split_mode == "frame":
        print(f"val      : every {args.frame_stride}th frame of every scene (leaks)")
    else:
        print(f"val      : {', '.join(val_scenes)}")
    stats = build(
        root, out_aabb, out_obb, val_scenes, args.merge_gap,
        args.split_mode, args.frame_stride,
    )

    for split in ("train", "val"):
        print(
            f"{split}: {stats[f'{split}_images']} images, "
            f"{stats[f'{split}_structures']} barrier structures "
            f"from {stats[f'{split}_segments']} annotated segments"
        )
    if stats["too_small"]:
        print(f"dropped {stats['too_small']} structures under {MIN_BOX_PX} px")

    for root_dir, kind in ((out_aabb, "aabb"), (out_obb, "obb")):
        cfg = {
            "path": str(root_dir.resolve()).replace("\\", "/"),
            "train": str((root_dir / "images" / "train").resolve()).replace("\\", "/"),
            "val": str((root_dir / "images" / "val").resolve()).replace("\\", "/"),
            "names": {0: "barrier"},
        }
        dest = save_yaml(cfg, CONFIG_DIR / f"barrier{tag}_{kind}.yaml")
        print(f"wrote {dest}")

    print("\nnext: train B3aabb and B3obb from the barrier notebook")


if __name__ == "__main__":
    friendly(main)()
