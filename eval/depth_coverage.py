"""What depth range each scene actually covers, per class.

The distance targets are stated per range bin, so the split decides which bins
can be reported at all. Reads the meta sidecars only - no re-conversion.

  python -m eval.depth_coverage
"""
from __future__ import annotations

import argparse
import collections
import json

from common.cli import friendly
from common.config import class_names
from common.env import dataset_root_for_cli


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", "--dataset-root", dest="out", default=None)
    args = ap.parse_args()

    root = dataset_root_for_cli(args.out)
    names = class_names()
    rows = collections.defaultdict(list)
    for split in ("train", "val"):
        meta_dir = root / "meta" / split
        if not meta_dir.is_dir():
            continue
        for path in meta_dir.glob("*.json"):
            meta = json.loads(path.read_text(encoding="utf-8"))
            for obj in meta["objects"]:
                rows[(meta["scene"], split, obj["cls"])].append(obj["z"])

    print(f"{'scene':<14}{'split':<7}{'class':<10}{'n':>5}{'min':>7}{'<10m':>6}{'<20m':>6}")
    for (scene, split, cls), depths in sorted(rows.items()):
        depths.sort()
        under10 = sum(1 for z in depths if z < 10)
        under20 = sum(1 for z in depths if z < 20)
        label = names[cls] if cls < len(names) else str(cls)
        print(
            f"{scene:<14}{split:<7}{label:<10}{len(depths):>5}"
            f"{depths[0]:>7.1f}{under10:>6}{under20:>6}"
        )


if __name__ == "__main__":
    friendly(main)()
