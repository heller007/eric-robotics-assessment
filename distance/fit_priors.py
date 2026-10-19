"""Fit the distance constants on the train split and write configs/priors.yaml.

Two things get fitted, both for the same reason: the labels are boxes projected
from 3D cuboids, not outlines of what the camera sees.

  H_eff   the effective height, from median(Z * h_px / f_y). A cuboid's depth
          extent adds pixels, so the box is taller than the object and the
          physical height would read short. For barriers the box also spans a
          merged row, which makes it taller still - fitting absorbs both.
  a, b    a per-class linear correction on the fused estimate, which removes
          what is left: the offset between the box bottom (the nearest cuboid
          edge) and the cuboid centre that nuScenes calls Z.

Stop signs have no nuScenes ground truth, so they keep the configured width and
an identity correction until tape-measured photos exist.

  python -m distance.fit_priors
"""
from __future__ import annotations

import argparse
import statistics

import numpy as np

from common.cli import friendly
from common.config import CONFIG_DIR, class_names, load_yaml, resolve, save_yaml
from common.env import dataset_root_for_cli
from common.labels import load_split
from distance.geometry import estimate

MIN_VISIBILITY = 4  # fit on the cleanest boxes only
MIN_SAMPLES = 20


def gather(root, split: str, names: list[str]) -> dict[str, list[dict]]:
    """Boxes worth fitting on, per class: fully visible, untruncated, with depth."""
    per_class: dict[str, list[dict]] = {n: [] for n in names}
    for rec in load_split(root, split):
        meta = rec["meta"]
        if not meta:
            continue  # a COCO frame; no camera model and no depth
        for obj in meta["objects"]:
            if obj.get("trunc") or obj.get("vis", 0) < MIN_VISIBILITY:
                continue
            if not np.isfinite(obj["z"]) or obj["z"] <= 0:
                continue
            per_class[names[obj["cls"]]].append(
                {
                    "box": obj["box"],
                    "z": obj["z"],
                    "K": meta["K"],
                    "h_cam": meta["h_cam"],
                    "pitch": meta["pitch"],
                    "height": meta["height"],
                }
            )
    return per_class


def fit_h_eff(samples: list[dict], dim: str) -> float | None:
    """Median of Z * size_px / focal - the height the pinhole model needs."""
    values = []
    for s in samples:
        x1, y1, x2, y2 = s["box"]
        if dim == "w":
            px, focal = x2 - x1, s["K"][0][0]
        else:
            px, focal = y2 - y1, s["K"][1][1]
        if px > 1:
            values.append(s["z"] * px / focal)
    return statistics.median(values) if len(values) >= MIN_SAMPLES else None


def fit_correction(samples: list[dict], priors: dict, name: str) -> tuple[float, float] | None:
    """Least-squares a, b mapping the fused estimate onto the true depth."""
    pred, true = [], []
    for s in samples:
        out = estimate(
            s["box"],
            name,
            s["K"],
            s["h_cam"],
            s["pitch"],
            priors,
            img_height=s["height"],
            apply_correction=False,
        )
        pred.append(out["fused"])
        true.append(s["z"])
    if len(pred) < MIN_SAMPLES:
        return None
    a, b = np.polyfit(np.asarray(pred), np.asarray(true), 1)
    return float(a), float(b)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", "--dataset-root", dest="out", default=None)
    ap.add_argument("--split", default="train")
    ap.add_argument("--priors", default=str(CONFIG_DIR / "priors.yaml"))
    ap.add_argument("--dest", default=None, help="where to write; default overwrites --priors")
    args = ap.parse_args()

    root = dataset_root_for_cli(args.out)
    names = class_names()
    priors = load_yaml(args.priors)
    samples = gather(root, args.split, names)

    print(f"fitting on {args.split} from {root}")
    for name in names:
        prior = priors["classes"][name]
        rows = samples[name]
        print(f"\n{name}: {len(rows)} boxes with visibility {MIN_VISIBILITY} and no truncation")
        if not rows:
            print("  no ground truth - keeping the configured values")
            continue

        h_eff = fit_h_eff(rows, prior.get("dim", "h"))
        if h_eff is None:
            print(f"  too few samples to fit H_eff, keeping {prior['H_eff']}")
        else:
            print(f"  H_eff {prior['H_eff']:.3f} -> {h_eff:.3f} m")
            prior["H_eff"] = round(h_eff, 4)

        fitted = fit_correction(rows, priors, name)
        if fitted is None:
            print("  too few samples to fit the correction")
        else:
            a, b = fitted
            print(f"  correction a {prior.get('a', 1.0):.3f} -> {a:.4f}, "
                  f"b {prior.get('b', 0.0):.3f} -> {b:.4f}")
            prior["a"], prior["b"] = round(a, 4), round(b, 4)

        # what the fit achieved on its own data, as a sanity figure only
        errors = []
        for s in rows:
            out = estimate(
                s["box"], name, s["K"], s["h_cam"], s["pitch"], priors,
                img_height=s["height"],
            )
            errors.append(abs(out["corrected"] - s["z"]) / s["z"])
        if errors:
            print(f"  train AbsRel after fitting: {float(np.mean(errors)):.3f}")

    dest = resolve(args.dest or args.priors)
    save_yaml(priors, dest)
    print(f"\nwrote {dest}")
    print("next: python -m eval.dist_eval --boxes oracle")


if __name__ == "__main__":
    friendly(main)()
