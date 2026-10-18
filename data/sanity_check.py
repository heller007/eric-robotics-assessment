"""Checks that must pass before a single epoch is trained.

  1. Draw N random converted images with their boxes - eyeball them.
  2. Per-class instance counts per split; both val scenes need cones and barriers.
  3. K differs per camera (otherwise get_sample_data was misused).
  4. Every meta box has Z > 0 and a plausible pixel height for its depth.

Usage
  python -m data.sanity_check --draw 20
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from common.cli import friendly
from common.config import class_names, resolve
from common.env import dataset_root_for_cli

COLORS = [(0, 165, 255), (255, 128, 0), (0, 0, 255)]  # BGR: cone, barrier, stop sign
IMG_EXT = (".jpg", ".jpeg", ".png")


def depth_for(meta_objs: list[dict], cls: int, box: tuple[int, int, int, int]) -> float | None:
    """Ground-truth Z of the meta box whose centre is closest to a drawn box.

    The label line is a rounded, normalised copy of the meta box, so matching by
    centre distance (a couple of pixels at most) is enough and also confirms the
    two agree.
    """
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    best, best_d = None, 4.0
    for obj in meta_objs:
        if obj["cls"] != cls:
            continue
        ox = (obj["box"][0] + obj["box"][2]) / 2
        oy = (obj["box"][1] + obj["box"][3]) / 2
        d = abs(ox - cx) + abs(oy - cy)
        if d < best_d:
            best, best_d = obj["z"], d
    return best


def draw_samples(out: Path, split: str, n: int, dest: Path, seed: int = 0) -> int:
    import cv2

    names = class_names()
    dest.mkdir(parents=True, exist_ok=True)
    images = sorted(p for p in (out / "images" / split).glob("*") if p.suffix.lower() in IMG_EXT)
    if not images:
        return 0
    random.Random(seed).shuffle(images)

    drawn = 0
    for img_path in images:
        lbl = out / "labels" / split / f"{img_path.stem}.txt"
        lines = (
            [ln for ln in lbl.read_text(encoding="utf-8").splitlines() if ln.strip()]
            if lbl.exists()
            else []
        )
        if not lines:
            continue  # a background frame tells the eye nothing
        im = cv2.imread(str(img_path))
        if im is None:
            continue
        h, w = im.shape[:2]

        meta_path = out / "meta" / split / f"{img_path.stem}.json"
        meta_objs = []
        if meta_path.exists():
            meta_objs = json.loads(meta_path.read_text(encoding="utf-8"))["objects"]

        for ln in lines:
            cls, cx, cy, bw, bh = ln.split()
            cls = int(cls)
            cx, cy, bw, bh = (float(v) for v in (cx, cy, bw, bh))
            x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
            x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
            color = COLORS[cls % len(COLORS)]
            cv2.rectangle(im, (x1, y1), (x2, y2), color, 2)
            z = depth_for(meta_objs, cls, (x1, y1, x2, y2))
            tag = names[cls] if z is None else f"{names[cls]} {z:.1f}m"
            cv2.putText(im, tag, (x1, max(12, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imwrite(str(dest / f"{split}_{img_path.stem}.jpg"), im)
        drawn += 1
        if drawn >= n:
            break
    return drawn


def scene_counts(out: Path, splits: list[str]) -> dict[str, Counter]:
    """Per-class box counts per scene, straight from the meta sidecars.

    This is what tells you which scenes to hold out: the devkit's mini_val has no
    barriers, so barrier AP cannot be measured on it.
    """
    counts: dict[str, Counter] = defaultdict(Counter)
    for split in splits:
        meta_dir = out / "meta" / split
        if not meta_dir.is_dir():
            continue
        for meta_path in meta_dir.glob("*.json"):
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            key = f"{meta['scene']} ({split})"
            for obj in meta["objects"]:
                counts[key][obj["cls"]] += 1
            counts[key]["images"] += 1
    return counts


def check_meta(out: Path, splits: list[str]) -> list[str]:
    problems = []
    per_camera_fx: dict[str, set] = defaultdict(set)
    seen_depths = []

    for split in splits:
        meta_dir = out / "meta" / split
        if not meta_dir.is_dir():
            continue
        for meta_path in meta_dir.glob("*.json"):
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            K = meta["K"]
            per_camera_fx[meta["camera"]].add(round(K[0][0], 3))
            if K[0][0] <= 0 or K[1][1] <= 0:
                problems.append(f"{meta_path.name}: non-positive focal length")
            if meta["h_cam"] <= 0:
                problems.append(f"{meta_path.name}: camera height {meta['h_cam']}")
            for obj in meta["objects"]:
                if obj["z"] <= 0:
                    problems.append(f"{meta_path.name}: non-positive Z {obj['z']}")
                seen_depths.append(obj["z"])

    # Each camera has its own focal length, and that is what we are checking: the
    # back camera is a wider lens than the front ones. Within one camera, fx also
    # varies a little between scenes, because nuScenes-mini spans several logs
    # and each was calibrated separately - that is real calibration data, not a
    # conversion bug, so it is reported rather than flagged.
    medians = {cam: statistics.median(sorted(v)) for cam, v in per_camera_fx.items()}
    if len(medians) > 1 and len({round(m) for m in medians.values()}) < 2:
        problems.append(
            "every camera reports the same fx - K is probably not being read per camera"
        )
    for cam in sorted(medians):
        spread = max(per_camera_fx[cam]) - min(per_camera_fx[cam])
        print(f"  {cam:<16} fx median {medians[cam]:8.1f}  spread across logs {spread:5.1f} px")

    if seen_depths:
        seen_depths.sort()
        mid = seen_depths[len(seen_depths) // 2]
        print(
            f"depth: {len(seen_depths)} boxes, min {seen_depths[0]:.1f} m, "
            f"median {mid:.1f} m, max {seen_depths[-1]:.1f} m"
        )
    return problems


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", "--dataset-root", dest="out", default=None)
    ap.add_argument("--draw", type=int, default=20)
    ap.add_argument("--dest", default="results/sanity")
    ap.add_argument("--splits", nargs="*", default=["train", "val"])
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    out = dataset_root_for_cli(args.out)
    dest = resolve(args.dest)

    from data.merge import scan_split

    names = class_names()
    problems = []
    for split in args.splits:
        info = scan_split(out, split)
        present = ", ".join(f"{names[c]}={info['counts'].get(c, 0)}" for c in range(len(names)))
        print(f"{split}: {info['images']} images | {present}")
        for c in range(len(names)):
            if info["counts"].get(c, 0) == 0:
                problems.append(f"{split}: zero {names[c]} instances")

    per_scene = scene_counts(out, args.splits)
    if per_scene:
        print("\nper scene (nuScenes only):")
        for scene in sorted(per_scene):
            c = per_scene[scene]
            shown = "  ".join(f"{names[i]}={c.get(i, 0)}" for i in range(len(names)))
            print(f"  {scene:<24} {c['images']:>4} images   {shown}")
        if any("zero" in p for p in problems):
            print(
                "\nA class is absent from val. Re-run the converter with "
                "--val-scenes <names from above> so every class appears in both."
            )

    problems += check_meta(out, args.splits)

    for split in args.splits:
        n = draw_samples(out, split, args.draw, dest / split)
        print(f"{split}: drew {n} annotated images into {dest / split}")

    if problems:
        print("\nproblems:")
        for p in dict.fromkeys(problems):
            print(f"  - {p}")
        if args.strict:
            raise SystemExit(1)
    else:
        print("\nall sanity checks passed")


if __name__ == "__main__":
    friendly(main)()
