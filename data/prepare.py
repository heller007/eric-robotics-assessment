"""Build the Task 1 dataset in one command.

This is the live path: the published copy on Kaggle is still awaiting
compression, so the dataset gets built locally from nuScenes v1.0-mini plus the
COCO stop signs.

It skips the conversion when the dataset is already there, so re-running is
cheap. Each step is invoked exactly as a person would from the shell, so this
file also serves as the canonical order of operations.

Usage
  python -m data.prepare --nuscenes-root D:/kaggle/nuscenes-mini   # a copy you have
  python -m data.prepare                      # build whatever is missing
  python -m data.prepare --fetch-nuscenes     # download the raw data if none is found
  python -m data.prepare --force              # rebuild from scratch
  python -m data.prepare --no-pseudo          # skip the GPU step
  python -m data.prepare --link-mode copy     # duplicate images instead of linking

Roughly 15-25 minutes end to end with a GPU, most of it the COCO download and
the pseudo-label pass; on CPU the pseudo-label pass dominates.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from common.cli import friendly
from common.config import REPO_ROOT, load_paths, resolve, set_local
from common.env import (
    find_coco_root,
    find_dataset_root,
    find_nuscenes_root,
    verify_nuscenes,
    writable_root,
)
from common.staging import LINK_ORDER


def step(label: str, cmd: list[str]) -> None:
    print(f"\n=== {label} ===")
    print("$ " + " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if result.returncode != 0:
        raise SystemExit(f"{label} failed with exit code {result.returncode}")


def already_built(out: Path) -> bool:
    labels = out / "labels" / "train"
    return labels.is_dir() and any(labels.glob("*.txt"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", "--dataset-root", dest="out", default=None)
    ap.add_argument("--nuscenes-root", default=None)
    ap.add_argument("--coco-root", default=None)
    ap.add_argument("--force", action="store_true", help="rebuild even if present")
    ap.add_argument("--no-pseudo", action="store_true", help="skip pseudo-labelling")
    ap.add_argument("--no-coco", action="store_true", help="skip the stop sign source")
    ap.add_argument("--device", default="", help="blank = auto (CUDA if present)")
    ap.add_argument("--pseudo-weights", default="yolo11n.pt")
    ap.add_argument("--pseudo-batch", type=int, default=16)
    ap.add_argument("--draw", type=int, default=12)
    ap.add_argument(
        "--link-mode",
        choices=list(LINK_ORDER),
        default="auto",
        help="how source images reach images/<split>/; auto tries hardlink, then "
        "symlink, then copy",
    )
    ap.add_argument(
        "--fetch-nuscenes",
        action="store_true",
        help="as a last resort, download v1.0-mini (~3.9 GiB) when no local copy "
        "is found. A --nuscenes-root is always preferred.",
    )
    ap.add_argument("--cams", nargs="*", default=None, help="restrict to these cameras")
    ap.add_argument(
        "--no-save",
        action="store_true",
        help="do not record --nuscenes-root in configs/local.yaml",
    )
    args = ap.parse_args()

    paths = load_paths()
    py = [sys.executable, "-m"]

    # Task 1 may already be done - built here earlier, or downloaded with
    # data.fetch_dataset. Only fall through to rebuilding when it is not.
    existing = None
    if not args.force:
        try:
            existing = find_dataset_root(
                args.out or paths.get("dataset_root"),
                kaggle_slug=None,  # fetching is data.fetch_dataset's job, not this one
                allow_download=False,
                local_default=writable_root() / paths["out_root"],
            )
        except FileNotFoundError:
            existing = None

    out = existing or (resolve(args.out) if args.out else writable_root() / paths["out_root"])

    if existing:
        print(f"dataset already present at {out} - validating only (use --force to rebuild)")
    else:
        if args.fetch_nuscenes:
            try:
                find_nuscenes_root(args.nuscenes_root or paths.get("nuscenes_root"))
            except FileNotFoundError:
                step("download nuScenes v1.0-mini", py + ["data.fetch_nuscenes"])
                paths = load_paths()  # fetch_nuscenes records the root it extracted to

        nuscenes = find_nuscenes_root(args.nuscenes_root or paths.get("nuscenes_root"))
        print(f"\nbuilding {out}")
        print(f"  nuScenes  : {nuscenes}")
        print(f"  staging   : {args.link_mode}")
        print(
            "  expect roughly: conversion a few minutes, the COCO stop signs a few "
            "more (~350 MB over HTTP unless --coco-root is set),\n"
            "  and the pseudo-label pass 2-5 minutes on a GPU or up to an hour on CPU"
        )

        step(
            "nuScenes -> YOLO",
            py
            + [
                "data.nuscenes_to_yolo",
                "--nuscenes-root",
                str(nuscenes),
                "--out",
                str(out),
                "--link-mode",
                args.link_mode,
            ]
            + (["--cams", *args.cams] if args.cams else []),
        )
        if not args.no_coco:
            coco = find_coco_root(args.coco_root or paths.get("coco_root"))
            cmd = py + ["data.coco_stopsign", "--out", str(out), "--link-mode", args.link_mode]
            if coco:
                cmd += ["--coco-root", str(coco)]
            step("COCO stop signs", cmd)
        if not args.no_pseudo:
            step(
                "pseudo-label stop signs on nuScenes frames",
                py
                + [
                    "data.pseudo_label",
                    "--out",
                    str(out),
                    "--weights",
                    args.pseudo_weights,
                    "--conf",
                    "0.5",
                    "--batch",
                    str(args.pseudo_batch),
                    "--device",
                    str(args.device),
                ],
            )

    step("merge + data.yaml", py + ["data.merge", "--out", str(out)])
    step(
        "sanity checks",
        py + ["data.sanity_check", "--out", str(out), "--draw", str(args.draw)],
    )
    print(f"\ndataset ready at {out}")
    print(f"data.yaml at {resolve('configs/data.yaml')}")


if __name__ == "__main__":
    friendly(main)()
