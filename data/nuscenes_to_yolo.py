"""nuScenes v1.0-mini -> YOLO labels + a per-image sidecar json for distance.

Split is by scene (devkit mini_train / mini_val). Splitting by frame would leak:
adjacent keyframes are 0.5 s apart and show the same cones.

Each image gets
  labels/<split>/<stem>.txt   YOLO lines: cls cx cy w h   (normalised)
  meta/<split>/<stem>.json    K, camera height, pitch, and per box the ground
                              truth depth Z, truncation flag, visibility level

Usage
  python -m data.nuscenes_to_yolo --out datasets/yolo_3class
"""
from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
from pyquaternion import Quaternion
from tqdm import tqdm

from common.cli import friendly
from common.config import load_yaml, resolve
from common.env import find_nuscenes_root, writable_root

CAMS = [
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]
IMG_W, IMG_H = 1600, 900
MIN_BOX_PX = 4
MIN_VISIBILITY = 2  # devkit levels 1..4; 2 means at least 40% visible


def camera_height_pitch(nusc, sample_data) -> tuple[float, float]:
    """Height above ground (m) and downward pitch (rad) of this camera.

    The calibrated_sensor rotation maps camera axes into the ego frame, whose
    origin sits on the ground under the rear axle and whose z points up. The
    optical axis is +z in the camera frame, so its ego-frame z component is
    negative when the camera looks down.
    """
    cs = nusc.get("calibrated_sensor", sample_data["calibrated_sensor_token"])
    axis = Quaternion(cs["rotation"]).rotation_matrix @ np.array([0.0, 0.0, 1.0])
    pitch = float(np.arcsin(np.clip(-axis[2], -1.0, 1.0)))
    return float(cs["translation"][2]), pitch


def link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        dst.symlink_to(src)
    except (OSError, NotImplementedError):
        shutil.copy2(src, dst)


def make_dirs(out: Path) -> None:
    for split in ("train", "val"):
        for sub in ("images", "labels", "meta"):
            (out / sub / split).mkdir(parents=True, exist_ok=True)


def convert(
    dataroot: Path,
    out: Path,
    version: str = "v1.0-mini",
    cams: list[str] | None = None,
    max_scenes: int | None = None,
) -> dict:
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils.geometry_utils import BoxVisibility, view_points
    from nuscenes.utils.splits import create_splits_scenes

    cams = cams or CAMS
    cls_map = load_yaml("classes.yaml")["nuscenes_map"]
    make_dirs(out)

    nusc = NuScenes(version=version, dataroot=str(dataroot), verbose=False)
    val_scenes = set(create_splits_scenes()["mini_val"])

    stats = {
        "train": Counter(),
        "val": Counter(),
        "images": Counter(),
        "scenes": {"train": [], "val": []},
        "dropped": Counter(),
    }

    scenes = nusc.scene[:max_scenes] if max_scenes else nusc.scene
    for scene in tqdm(scenes, desc="scenes"):
        split = "val" if scene["name"] in val_scenes else "train"
        stats["scenes"][split].append(scene["name"])
        token = scene["first_sample_token"]
        while token:
            sample = nusc.get("sample", token)
            for cam in cams:
                sd_token = sample["data"][cam]
                sd = nusc.get("sample_data", sd_token)
                img_path, boxes, K = nusc.get_sample_data(
                    sd_token, box_vis_level=BoxVisibility.ANY
                )
                h_cam, pitch = camera_height_pitch(nusc, sd)

                lines, meta_objs = [], []
                for box in boxes:
                    if box.name not in cls_map:
                        continue
                    ann = nusc.get("sample_annotation", box.token)
                    vis = int(ann["visibility_token"])
                    if vis < MIN_VISIBILITY:
                        stats["dropped"]["low_visibility"] += 1
                        continue
                    if ann["num_lidar_pts"] + ann["num_radar_pts"] == 0:
                        stats["dropped"]["no_lidar_radar"] += 1
                        continue

                    corners = box.corners()  # 3 x 8, camera frame
                    if corners[2].min() < 0.1:
                        stats["dropped"]["behind_camera"] += 1
                        continue

                    uv = view_points(corners, K, normalize=True)[:2]
                    x1, y1 = uv.min(axis=1)
                    x2, y2 = uv.max(axis=1)
                    truncated = bool(x1 < 0 or y1 < 0 or x2 > IMG_W or y2 > IMG_H)
                    x1, x2 = np.clip([x1, x2], 0, IMG_W)
                    y1, y2 = np.clip([y1, y2], 0, IMG_H)
                    if (x2 - x1) < MIN_BOX_PX or (y2 - y1) < MIN_BOX_PX:
                        stats["dropped"]["too_small"] += 1
                        continue

                    cls = cls_map[box.name]
                    lines.append(
                        f"{cls} {(x1 + x2) / 2 / IMG_W:.6f} {(y1 + y2) / 2 / IMG_H:.6f} "
                        f"{(x2 - x1) / IMG_W:.6f} {(y2 - y1) / IMG_H:.6f}"
                    )
                    meta_objs.append(
                        {
                            "cls": int(cls),
                            "box": [float(x1), float(y1), float(x2), float(y2)],
                            "z": float(box.center[2]),  # depth along optical axis
                            "dims_whl": [float(v) for v in box.wlh],
                            "trunc": truncated,
                            "vis": vis,
                        }
                    )
                    stats[split][cls] += 1

                src = Path(img_path)
                stem = src.stem
                link_or_copy(src, out / "images" / split / src.name)
                (out / "labels" / split / f"{stem}.txt").write_text(
                    "\n".join(lines), encoding="utf-8"
                )
                (out / "meta" / split / f"{stem}.json").write_text(
                    json.dumps(
                        {
                            "image": src.name,
                            "camera": cam,
                            "scene": scene["name"],
                            "width": IMG_W,
                            "height": IMG_H,
                            "K": np.asarray(K).tolist(),
                            "h_cam": h_cam,
                            "pitch": pitch,
                            "objects": meta_objs,
                        }
                    ),
                    encoding="utf-8",
                )
                stats["images"][split] += 1
            token = sample["next"]

    return stats


def summarise(stats: dict) -> str:
    names = load_yaml("classes.yaml")["names"]
    out = []
    for split in ("train", "val"):
        counts = ", ".join(
            f"{names[c]}={stats[split][c]}" for c in sorted(names) if c in stats[split]
        )
        out.append(
            f"{split}: {stats['images'][split]} images, {counts or 'no boxes'}\n"
            f"        scenes: {', '.join(stats['scenes'][split])}"
        )
    out.append("dropped: " + ", ".join(f"{k}={v}" for k, v in stats["dropped"].items()))
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nuscenes-root", default=None)
    ap.add_argument("--version", default="v1.0-mini")
    ap.add_argument("--out", default=None, help="default: configs/paths.yaml out_root")
    ap.add_argument("--cams", nargs="*", default=None, help="default: all 6")
    ap.add_argument("--max-scenes", type=int, default=None, help="smoke test")
    args = ap.parse_args()

    paths = load_yaml("paths.yaml")
    root = find_nuscenes_root(args.nuscenes_root or paths.get("nuscenes_root"))
    out = resolve(args.out) if args.out else writable_root() / paths["out_root"]

    print(f"nuScenes root : {root}")
    print(f"output        : {out}")
    stats = convert(root, out, args.version, args.cams, args.max_scenes)
    print(summarise(stats))

    report = out / "nuscenes_stats.json"
    report.write_text(
        json.dumps(
            {
                "images": dict(stats["images"]),
                "train_boxes": {str(k): v for k, v in stats["train"].items()},
                "val_boxes": {str(k): v for k, v in stats["val"].items()},
                "dropped": dict(stats["dropped"]),
                "scenes": stats["scenes"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {report}")


if __name__ == "__main__":
    friendly(main)()
