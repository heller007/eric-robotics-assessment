"""Putting source images where YOLO expects them, without duplicating gigabytes.

YOLO finds a label by swapping `/images/` for `/labels/` in the image path, so
every frame has to be reachable under images/<split>/. The nuScenes frames alone
are about 1.3 GB and the COCO stop signs another 300 MB, and copying them is
pointless when the originals are already on the disk.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

LINK_ORDER = {
    "auto": ("hardlink", "symlink", "copy"),
    "hardlink": ("hardlink",),
    "symlink": ("symlink",),
    "copy": ("copy",),
}


def stage_image(src: Path, dst: Path, mode: str = "auto") -> str:
    """Make `src` available at `dst`; returns how it was done.

    Preference order, and why: a hard link costs nothing and needs no
    privileges, but only works within one volume - so nuScenes on C: with the
    dataset on D: falls through. A symbolic link crosses volumes but on Windows
    needs Developer Mode or an elevated shell. A copy always works.
    """
    src, dst = Path(src), Path(dst)
    if dst.exists():
        return "present"
    if mode not in LINK_ORDER:
        raise ValueError(f"unknown link mode {mode!r}; pick one of {list(LINK_ORDER)}")

    dst.parent.mkdir(parents=True, exist_ok=True)
    last: Exception | None = None
    for attempt in LINK_ORDER[mode]:
        try:
            if attempt == "hardlink":
                os.link(src, dst)
            elif attempt == "symlink":
                dst.symlink_to(src)
            else:
                shutil.copy2(src, dst)
            return attempt
        except (OSError, NotImplementedError) as exc:
            last = exc
            continue
    raise RuntimeError(f"could not stage {src} as {dst} using mode {mode!r}: {last}")
