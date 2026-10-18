"""Where things live.

Local runs read configs/paths.yaml. On Kaggle the dataset slugs differ per
mirror, so the roots are discovered by looking for the folder layout instead of
hardcoding a slug.
"""
from __future__ import annotations

import os
from pathlib import Path

from common.config import REPO_ROOT, resolve

KAGGLE_INPUT = Path("/kaggle/input")
KAGGLE_WORKING = Path("/kaggle/working")


def on_kaggle() -> bool:
    return KAGGLE_WORKING.exists() and KAGGLE_INPUT.exists()


def writable_root() -> Path:
    """Where generated datasets and runs go. /kaggle/input is read-only."""
    if on_kaggle():
        return KAGGLE_WORKING
    return REPO_ROOT


def _iter_candidates(max_depth: int = 3):
    """Every directory under /kaggle/input down to max_depth, shallowest first."""
    if not KAGGLE_INPUT.exists():
        return
    frontier = [KAGGLE_INPUT]
    for _ in range(max_depth):
        nxt = []
        for d in frontier:
            try:
                children = [c for c in d.iterdir() if c.is_dir()]
            except (PermissionError, OSError):
                continue
            for c in children:
                yield c
                nxt.append(c)
        frontier = nxt


def _looks_like_nuscenes(d: Path) -> bool:
    return (d / "samples").is_dir() and any(
        (d / v).is_dir() for v in ("v1.0-mini", "v1.0-trainval", "v1.0-test")
    )


def _looks_like_coco(d: Path) -> bool:
    return (d / "annotations").is_dir() or (
        (d / "train2017").is_dir() and (d / "val2017").is_dir()
    )


def find_nuscenes_root(explicit: str | Path | None = None) -> Path:
    """Folder that directly contains v1.0-mini/ and samples/."""
    for cand in (explicit, os.environ.get("NUSCENES_ROOT")):
        if cand:
            p = resolve(cand)
            if _looks_like_nuscenes(p):
                return p
            raise FileNotFoundError(
                f"{p} does not look like a nuScenes root "
                "(expected v1.0-mini/ and samples/ inside it)"
            )
    for d in _iter_candidates():
        if _looks_like_nuscenes(d):
            return d
    raise FileNotFoundError(
        "No nuScenes root found. Pass --nuscenes-root, set NUSCENES_ROOT, or "
        "attach a nuScenes v1.0-mini dataset to the Kaggle notebook."
    )


def find_coco_root(explicit: str | Path | None = None) -> Path | None:
    """Optional: returns None so the caller can fall back to HTTP download."""
    for cand in (explicit, os.environ.get("COCO_ROOT")):
        if cand:
            p = resolve(cand)
            if _looks_like_coco(p):
                return p
            raise FileNotFoundError(f"{p} does not look like a COCO 2017 root")
    for d in _iter_candidates():
        if _looks_like_coco(d):
            return d
    return None


def describe() -> str:
    lines = [f"kaggle={on_kaggle()}  writable_root={writable_root()}"]
    if KAGGLE_INPUT.exists():
        for d in sorted(p.name for p in KAGGLE_INPUT.iterdir() if p.is_dir()):
            lines.append(f"  input: {d}")
    return "\n".join(lines)
