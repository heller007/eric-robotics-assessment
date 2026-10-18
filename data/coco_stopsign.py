"""COCO 2017 stop signs -> the same YOLO tree, as class 2.

Two sources, tried in order:
  1. a local COCO root (a Kaggle dataset, or your own download) with
     annotations/instances_{train,val}2017.json and the image folders;
  2. plain HTTP: the annotation zip, then only the stop-sign images from
     images.cocodataset.org. About 2,000 train + 75 val images, ~350 MB.

FiftyOne does the same job but drags in a MongoDB server, which is a poor fit
for a Kaggle session, so this module speaks to COCO directly.

Usage
  python -m data.coco_stopsign --out datasets/yolo_3class
"""
from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from tqdm import tqdm

from common.config import load_yaml, resolve
from common.env import find_coco_root, writable_root

ANN_ZIP = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
IMG_URL = "http://images.cocodataset.org/{split}/{file_name}"
STOP_SIGN_CLASS = 2  # index in our unified 3-class set
SPLIT_MAP = {"train2017": "train", "val2017": "val"}


def _load_annotations(coco_root: Path | None, split: str, cache: Path) -> dict:
    """instances_<split>.json, from disk if we have it, else from the zip."""
    if coco_root is not None:
        local = coco_root / "annotations" / f"instances_{split}.json"
        if local.exists():
            return json.loads(local.read_text(encoding="utf-8"))

    cache.mkdir(parents=True, exist_ok=True)
    cached = cache / f"instances_{split}.json"
    if not cached.exists():
        zip_path = cache / "annotations_trainval2017.zip"
        if not zip_path.exists():
            print(f"downloading {ANN_ZIP} (~241 MB, once)")
            with requests.get(ANN_ZIP, stream=True, timeout=120) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0))
                with open(zip_path, "wb") as fh, tqdm(
                    total=total, unit="B", unit_scale=True, desc="annotations"
                ) as bar:
                    for chunk in r.iter_content(1 << 20):
                        fh.write(chunk)
                        bar.update(len(chunk))
        with zipfile.ZipFile(zip_path) as zf:
            for name in zf.namelist():
                if name.endswith(f"instances_{split}.json"):
                    cached.write_bytes(zf.read(name))
                    break
    return json.loads(cached.read_text(encoding="utf-8"))


def _stop_sign_items(ann: dict, category_name: str) -> dict[int, dict]:
    cat_ids = {c["id"] for c in ann["categories"] if c["name"] == category_name}
    if not cat_ids:
        raise ValueError(f"category {category_name!r} not in this annotation file")

    keep: dict[int, dict] = {}
    for a in ann["annotations"]:
        if a["category_id"] not in cat_ids or a.get("iscrowd", 0):
            continue
        keep.setdefault(a["image_id"], {"boxes": []})["boxes"].append(a["bbox"])

    by_id = {img["id"]: img for img in ann["images"]}
    for image_id, rec in keep.items():
        rec["info"] = by_id[image_id]
    return keep


def _yolo_lines(rec: dict) -> list[str]:
    w, h = rec["info"]["width"], rec["info"]["height"]
    lines = []
    for x, y, bw, bh in rec["boxes"]:
        if bw < 2 or bh < 2:
            continue
        lines.append(
            f"{STOP_SIGN_CLASS} {(x + bw / 2) / w:.6f} {(y + bh / 2) / h:.6f} "
            f"{bw / w:.6f} {bh / h:.6f}"
        )
    return lines


def _fetch_image(file_name: str, split: str, dst: Path, session: requests.Session) -> bool:
    if dst.exists() and dst.stat().st_size > 0:
        return True
    url = IMG_URL.format(split=split, file_name=file_name)
    try:
        r = session.get(url, timeout=60)
        r.raise_for_status()
    except requests.RequestException:
        return False
    dst.write_bytes(r.content)
    return True


def export_split(
    split: str,
    out: Path,
    coco_root: Path | None,
    cache: Path,
    category_name: str,
    workers: int = 16,
    limit: int | None = None,
) -> Counter:
    target = SPLIT_MAP[split]
    for sub in ("images", "labels"):
        (out / sub / target).mkdir(parents=True, exist_ok=True)

    ann = _load_annotations(coco_root, split, cache)
    items = _stop_sign_items(ann, category_name)
    if limit:
        items = dict(list(items.items())[:limit])
    print(f"{split}: {len(items)} images contain {category_name}")

    local_images = coco_root / split if coco_root is not None else None
    have_local = local_images is not None and local_images.is_dir()

    stats = Counter()
    pending: list[tuple[str, Path]] = []
    for rec in items.values():
        file_name = rec["info"]["file_name"]
        dst = out / "images" / target / file_name
        if have_local:
            src = local_images / file_name
            if src.exists():
                if not dst.exists():
                    try:
                        dst.symlink_to(src)
                    except (OSError, NotImplementedError):
                        shutil.copy2(src, dst)
                stats["linked"] += 1
            else:
                pending.append((file_name, dst))
        else:
            pending.append((file_name, dst))

    if pending:
        session = requests.Session()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_fetch_image, fn, split, dst, session): fn
                for fn, dst in pending
            }
            for fut in tqdm(
                as_completed(futures), total=len(futures), desc=f"{split} images"
            ):
                stats["downloaded" if fut.result() else "failed"] += 1

    for rec in items.values():
        file_name = rec["info"]["file_name"]
        img = out / "images" / target / file_name
        if not img.exists():
            continue
        lines = _yolo_lines(rec)
        (out / "labels" / target / f"{Path(file_name).stem}.txt").write_text(
            "\n".join(lines), encoding="utf-8"
        )
        stats["images"] += 1
        stats["boxes"] += len(lines)
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--coco-root", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--cache", default=None, help="where the annotation zip is kept")
    ap.add_argument("--splits", nargs="*", default=["train2017", "val2017"])
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None, help="smoke test")
    args = ap.parse_args()

    paths = load_yaml("paths.yaml")
    classes = load_yaml("classes.yaml")
    out = resolve(args.out) if args.out else writable_root() / paths["out_root"]
    cache = resolve(args.cache) if args.cache else writable_root() / "datasets" / "coco_cache"
    coco_root = find_coco_root(args.coco_root or paths.get("coco_root"))

    print(f"coco root : {coco_root or 'none, downloading over HTTP'}")
    print(f"output    : {out}")

    totals = Counter()
    for split in args.splits:
        stats = export_split(
            split,
            out,
            coco_root,
            cache,
            classes["coco_category_name"],
            args.workers,
            args.limit,
        )
        print(f"{split}: {dict(stats)}")
        totals.update(stats)
    print(f"total: {dict(totals)}")


if __name__ == "__main__":
    main()
