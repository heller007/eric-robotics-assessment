"""Get an already-built Task 1 dataset onto this machine and point the repo at it.

Downloads (or locates) a prepared copy, checks what it actually contains, writes
configs/data.yaml with absolute paths, and remembers the location in
configs/local.yaml so later commands need no flags.

NOT the current path: the published copy
(pandeyvineet/eric-cones-stops-barriers) is still awaiting compression on Kaggle
and cannot be downloaded, so `kaggle_dataset` in configs/paths.yaml is null and
the dataset is built locally with `python -m data.prepare --fetch-nuscenes`
instead. Put the slug back once it is ready and this works again - nothing else
needs to change.

Usage
  python -m data.fetch_dataset                          # download the published set
  python -m data.fetch_dataset --dataset-root D:/eric   # use a copy already on disk
  python -m data.fetch_dataset --no-save                # do not touch configs/local.yaml

Credentials: a private Kaggle dataset needs kaggle.json in ~/.kaggle/ (on Windows
%USERPROFILE%\\.kaggle\\), or KAGGLE_USERNAME and KAGGLE_KEY in the environment.
"""
from __future__ import annotations

import argparse
import json

from common.cli import friendly
from common.config import REPO_ROOT, class_names, load_paths, resolve, set_local
from common.env import dataset_health, find_dataset_root, writable_root
from data.merge import scan_split, write_data_yaml


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-root", default=None, help="a copy already on disk")
    ap.add_argument("--kaggle-dataset", default=None, help="override the slug")
    ap.add_argument("--no-download", action="store_true", help="fail instead of fetching")
    ap.add_argument("--no-save", action="store_true", help="do not write configs/local.yaml")
    ap.add_argument("--data-yaml", default=str(REPO_ROOT / "configs" / "data.yaml"))
    ap.add_argument("--results", default="results")
    ap.add_argument("--strict", action="store_true", help="exit non-zero on any problem")
    args = ap.parse_args()

    paths = load_paths()
    root = find_dataset_root(
        args.dataset_root or paths.get("dataset_root"),
        args.kaggle_dataset or paths.get("kaggle_dataset"),
        allow_download=not args.no_download,
        local_default=writable_root() / paths["out_root"],
    )
    print(f"dataset root: {root}")

    health = dataset_health(root)
    for split, counts in health["splits"].items():
        print(
            f"  {split}: {counts['images']} images, {counts['labels']} labels, "
            f"{counts['meta']} meta sidecars"
        )

    names = class_names()
    report = {}
    for split in ("train", "val"):
        info = scan_split(root, split)
        report[split] = info
        present = ", ".join(
            f"{names[c]}={info['counts'].get(c, 0)}" for c in range(len(names))
        )
        print(f"  {split}: {present}")
        for c in range(len(names)):
            if info["counts"].get(c, 0) == 0:
                health["problems"].append(f"{split}: no {names[c]} instances")

    dest = write_data_yaml(root, names, resolve(args.data_yaml))
    print(f"\nwrote {dest}")

    results_dir = resolve(args.results)
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "dataset_health.json").write_text(
        json.dumps({"health": health, "counts": report}, indent=2), encoding="utf-8"
    )

    if not args.no_save:
        local = set_local("dataset_root", root)
        print(f"remembered the location in {local} (git-ignored, this machine only)")

    if health["problems"]:
        print("\nproblems:")
        for p in dict.fromkeys(health["problems"]):
            print(f"  - {p}")
        if args.strict:
            raise SystemExit(1)
    else:
        print("\ndataset looks complete")
        print("next: python -m eval.det_eval --run B0")


if __name__ == "__main__":
    friendly(main)()
