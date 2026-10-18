"""Where things live.

Local runs read configs/paths.yaml. On Kaggle the dataset slugs differ per
mirror, so the roots are discovered by looking for the folder layout instead of
hardcoding a slug.
"""
from __future__ import annotations

import os
from pathlib import Path

from common.config import REPO_ROOT, load_paths, resolve

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


def looks_like_prepared_dataset(d: Path) -> bool:
    """A Task 1 output: parallel images/ and labels/ trees with a train split."""
    d = Path(d)
    return (d / "images" / "train").is_dir() and (d / "labels" / "train").is_dir()


def descend(root: Path, predicate, max_depth: int = 3) -> Path | None:
    """`root` itself, or the shallowest directory under it that satisfies predicate.

    A dataset downloaded as an archive usually unpacks with one or two wrapper
    folders, and which one depends on how it was zipped - so find the tree rather
    than require the caller to know.
    """
    root = Path(root)
    if not root.is_dir():
        return None
    if predicate(root):
        return root
    frontier = [root]
    for _ in range(max_depth):
        nxt = []
        for d in frontier:
            try:
                children = sorted(c for c in d.iterdir() if c.is_dir())
            except (PermissionError, OSError):
                continue
            for c in children:
                if predicate(c):
                    return c
                nxt.append(c)
        frontier = nxt
    return None


def find_nuscenes_root(
    explicit: str | Path | None = None, local_default: str | Path | None = None
) -> Path:
    """Folder that directly contains v1.0-mini/ and samples/.

    A path given here does not have to be the dataroot itself: a Kaggle mirror or
    a hand-extracted zip usually leaves one or two wrapper folders above it, and
    which ones depends on how it was packed. So point at wherever you unpacked it
    and the dataroot is found inside.
    """
    for cand in (explicit, os.environ.get("NUSCENES_ROOT"), local_default):
        if not cand:
            continue
        p = resolve(cand)
        found = descend(p, _looks_like_nuscenes)
        if found:
            return found
        if cand is not local_default:
            # an explicit path that holds nothing is a mistake worth reporting,
            # while a default that holds nothing is just not set up yet
            raise FileNotFoundError(
                f"{p} does not contain a nuScenes dataroot.\n"
                "Expected a folder with v1.0-mini/ and samples/ inside it, at or "
                "below the path given."
                + (f"\nIt holds: {_brief_listing(p)}" if p.is_dir() else "\nIt does not exist.")
            )

    for d in _iter_candidates():
        if _looks_like_nuscenes(d):
            return d

    raise FileNotFoundError(
        "No nuScenes v1.0-mini found. Either point at a copy you already have\n"
        "  python -m data.fetch_nuscenes --root D:/path/where/you/unpacked/it\n"
        "or download it (~3.9 GiB, resumable)\n"
        "  python -m data.fetch_nuscenes\n"
        "A path can also be set once in configs/local.yaml as nuscenes_root, or "
        "given per command with --nuscenes-root / NUSCENES_ROOT."
    )


def _brief_listing(d: Path, limit: int = 8) -> str:
    try:
        names = sorted(p.name + ("/" if p.is_dir() else "") for p in d.iterdir())
    except (PermissionError, OSError):
        return "unreadable"
    if not names:
        return "nothing"
    shown = ", ".join(names[:limit])
    return shown + (f", ... ({len(names)} entries)" if len(names) > limit else "")


EXPECTED_CAMERAS = [
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]


def verify_nuscenes(root: Path, version: str = "v1.0-mini") -> dict:
    """A quick look at a copy: which cameras it has and how many images."""
    root = Path(root)
    samples = root / "samples"
    cameras = (
        sorted(p.name for p in samples.iterdir() if p.is_dir() and p.name.startswith("CAM"))
        if samples.is_dir()
        else []
    )
    n_images = sum(1 for _ in samples.rglob("*.jpg")) if samples.is_dir() else 0
    problems = []
    if not (root / version).is_dir():
        problems.append(f"no {version}/ folder - wrong split?")
    if not n_images:
        problems.append("samples/ holds no .jpg images")
    return {
        "root": str(root),
        "cameras": cameras,
        "keyframe_images": n_images,
        "problems": problems,
    }


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


def download_kaggle_dataset(slug: str) -> Path:
    """Fetch a Kaggle dataset through kagglehub and return its local path.

    kagglehub caches, so calling this again is a no-op once the dataset is on the
    machine.
    """
    try:
        import kagglehub
    except ImportError as exc:
        raise RuntimeError(
            "kagglehub is not installed. Either `pip install kagglehub`, or point "
            "--dataset-root at a copy you downloaded by hand."
        ) from exc

    try:
        return Path(kagglehub.dataset_download(slug))
    except Exception as exc:
        raise RuntimeError(
            f"could not download {slug!r}: {type(exc).__name__}: {exc}\n"
            "A private dataset needs Kaggle credentials on this machine: put\n"
            "kaggle.json in ~/.kaggle/ (Windows: %USERPROFILE%\\.kaggle\\), or set\n"
            "KAGGLE_USERNAME and KAGGLE_KEY."
        ) from exc


def find_dataset_root(
    explicit: str | Path | None = None,
    kaggle_slug: str | None = None,
    allow_download: bool = True,
    local_default: str | Path | None = None,
) -> Path:
    """Locate the prepared Task 1 dataset.

    Order: an explicit path, then ERIC_DATASET_ROOT, then anything attached under
    /kaggle/input, then a locally built copy, and only then the published Kaggle
    dataset - downloading is the slowest option, so it is the last one tried.
    """
    for cand in (explicit, os.environ.get("ERIC_DATASET_ROOT")):
        if cand:
            found = descend(resolve(cand), looks_like_prepared_dataset)
            if found:
                return found
            raise FileNotFoundError(
                f"{resolve(cand)} has no images/train + labels/train inside it"
            )

    if KAGGLE_INPUT.exists():
        for d in _iter_candidates():
            if looks_like_prepared_dataset(d):
                return d

    if local_default:
        found = descend(resolve(local_default), looks_like_prepared_dataset)
        if found:
            return found

    if kaggle_slug and allow_download:
        print(f"fetching {kaggle_slug} through kagglehub ...")
        found = descend(download_kaggle_dataset(kaggle_slug), looks_like_prepared_dataset)
        if found:
            return found
        raise FileNotFoundError(
            f"{kaggle_slug} downloaded but contains no images/train + labels/train"
        )

    raise FileNotFoundError(
        "No prepared dataset found. Either\n"
        "  python -m data.prepare --fetch-nuscenes   (build it - this is the live path), or\n"
        "  python -m data.fetch_dataset              (download a published copy, once\n"
        "                                             kaggle_dataset is set again), or\n"
        "  pass --dataset-root /path/to/it"
    )


def dataset_root_for_cli(explicit: str | Path | None = None, allow_download: bool = True) -> Path:
    """What every script's --dataset-root flag resolves to.

    allow_download is False for the scripts that *write* into the dataset: pulling
    a fresh copy only to modify it inside a cache directory is never what was
    meant.
    """
    paths = load_paths()
    return find_dataset_root(
        explicit or paths.get("dataset_root"),
        paths.get("kaggle_dataset") if allow_download else None,
        allow_download=allow_download,
        local_default=writable_root() / paths["out_root"],
    )


def dataset_health(root: Path) -> dict:
    """What a downloaded dataset actually contains - checked, not assumed.

    Zipping a tree of symlinks can silently produce either real files or dangling
    links, and the meta/ sidecars are easy to leave out of an upload even though
    Task 3 cannot compute a single distance error without them.
    """
    root = Path(root)
    info: dict = {"root": str(root), "splits": {}, "problems": []}
    for split in ("train", "val"):
        images = root / "images" / split
        labels = root / "labels" / split
        meta = root / "meta" / split
        n_images = len(list(images.glob("*"))) if images.is_dir() else 0
        n_labels = len(list(labels.glob("*.txt"))) if labels.is_dir() else 0
        n_meta = len(list(meta.glob("*.json"))) if meta.is_dir() else 0
        info["splits"][split] = {"images": n_images, "labels": n_labels, "meta": n_meta}
        if n_images == 0:
            info["problems"].append(f"{split}: no images")
        if n_labels == 0:
            info["problems"].append(f"{split}: no label files")
        if n_meta == 0:
            info["problems"].append(
                f"{split}: no meta/ sidecars - Task 3 needs K, camera height, "
                "pitch and ground-truth depth from them"
            )
        # a dangling symlink survives a directory listing but cannot be opened
        for probe in list(images.glob("*"))[:5]:
            if not probe.exists():
                info["problems"].append(f"{split}: {probe.name} is a broken link")
                break
    return info


def describe() -> str:
    lines = [f"kaggle={on_kaggle()}  writable_root={writable_root()}"]
    if KAGGLE_INPUT.exists():
        for d in sorted(p.name for p in KAGGLE_INPUT.iterdir() if p.is_dir()):
            lines.append(f"  input: {d}")
    return "\n".join(lines)
