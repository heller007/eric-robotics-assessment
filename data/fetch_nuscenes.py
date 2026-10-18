"""Download and extract nuScenes v1.0-mini locally, then remember where it went.

The archive is about 4.2 GB and extracts to roughly the same again, so budget
~9 GB free while it runs and ~4.5 GB afterwards if the archive is deleted.

The download resumes: interrupt it and run the same command again and it picks up
where it stopped rather than starting over.

Usage
  python -m data.fetch_nuscenes                        # download + extract + remember
  python -m data.fetch_nuscenes --dest D:/datasets/nuscenes
  python -m data.fetch_nuscenes --archive D:/v1.0-mini.tgz   # extract one you have
  python -m data.fetch_nuscenes --no-keep-archive      # delete the .tgz after extracting

By downloading you accept the nuScenes Terms of Use: https://www.nuscenes.org/terms-of-use
The data is free for non-commercial use, which is what this assessment is.
"""
from __future__ import annotations

import argparse
import shutil
import tarfile
from pathlib import Path

import requests
from tqdm import tqdm

from common.cli import friendly
from common.config import load_paths, resolve, set_local
from common.env import descend

MIRRORS = [
    "https://www.nuscenes.org/data/v1.0-mini.tgz",
    "https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-mini.tgz",
]
MIN_FREE_GB = 9.0


def looks_like_nuscenes(d: Path) -> bool:
    return (d / "samples").is_dir() and (d / "v1.0-mini").is_dir()


def free_gb(path: Path) -> float:
    probe = path
    while not probe.exists():
        probe = probe.parent
    return shutil.disk_usage(probe).free / 1024**3


# Content-Encoding has to be off for a resumable binary download: with gzip on,
# Content-Length counts compressed bytes while the bytes we write are
# decompressed, so the size check and every byte offset in a Range header would
# refer to different things.
NO_ENCODING = {"Accept-Encoding": "identity"}


def remote_size(url: str) -> int | None:
    try:
        r = requests.head(url, allow_redirects=True, timeout=30, headers=NO_ENCODING)
        r.raise_for_status()
        return int(r.headers["content-length"])
    except (requests.RequestException, KeyError, ValueError):
        return None


def supports_resume(url: str) -> bool:
    try:
        r = requests.head(url, allow_redirects=True, timeout=30, headers=NO_ENCODING)
        return r.headers.get("accept-ranges", "").lower() == "bytes"
    except requests.RequestException:
        return False


def download(url: str, dest: Path) -> Path:
    """Resumable download. Partial data is kept in a .part file until complete."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    expected = remote_size(url)
    have = part.stat().st_size if part.exists() else 0

    if expected and have == expected:
        part.rename(dest)
        return dest
    if dest.exists() and expected and dest.stat().st_size == expected:
        print(f"{dest.name} is already complete")
        return dest

    headers = dict(NO_ENCODING)
    if have:
        headers["Range"] = f"bytes={have}-"
    with requests.get(url, headers=headers, stream=True, timeout=120) as r:
        if r.status_code == 416:  # the range is past the end: already complete
            part.rename(dest)
            return dest
        r.raise_for_status()
        resuming = r.status_code == 206
        if have and not resuming:
            print("server ignored the resume request - starting over")
            have = 0
        total = have + int(r.headers.get("content-length", 0))
        if have:
            shown = (
                f"{have / 1024**3:.2f} GB" if have >= 1024**3 else f"{have / 1024**2:.1f} MB"
            )
            print(f"resuming at {shown} of {total / 1024**3:.2f} GB")
        with open(part, "ab" if have else "wb") as fh, tqdm(
            total=total or None,
            initial=have,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=dest.name,
        ) as bar:
            for chunk in r.iter_content(1 << 22):
                fh.write(chunk)
                bar.update(len(chunk))

    if expected and part.stat().st_size != expected:
        raise RuntimeError(
            f"downloaded {part.stat().st_size} bytes but the server reported "
            f"{expected}. Run the command again to resume."
        )
    part.replace(dest)
    return dest


def extract(archive: Path, dest: Path) -> Path:
    """Extract into its own directory, then locate the dataroot inside it."""
    dest.mkdir(parents=True, exist_ok=True)
    if looks_like_nuscenes(dest):
        print(f"{dest} already holds an extracted dataset - skipping extraction")
        return dest

    print(f"extracting {archive.name} -> {dest}")
    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        with tqdm(total=len(members), unit="file", desc="extract") as bar:
            for member in members:
                # the data filter refuses absolute paths, .. traversal and
                # anything that is not a plain file or directory
                try:
                    tar.extract(member, path=dest, filter="data")
                except TypeError:  # Python without extraction filters
                    if member.name.startswith(("/", "..")) or ".." in Path(member.name).parts:
                        raise RuntimeError(f"refusing unsafe archive entry {member.name!r}")
                    tar.extract(member, path=dest)
                bar.update(1)

    found = descend(dest, looks_like_nuscenes, max_depth=3)
    if found is None:
        raise RuntimeError(
            f"extracted to {dest} but found no v1.0-mini/ + samples/ inside it"
        )
    return found


def summarise(root: Path) -> None:
    tables = sorted(p.name for p in (root / "v1.0-mini").glob("*.json"))
    cameras = sorted(p.name for p in (root / "samples").iterdir() if p.is_dir())
    n_images = sum(1 for _ in (root / "samples").rglob("*.jpg"))
    print(f"\ndataroot : {root}")
    print(f"tables   : {len(tables)} json files")
    print(f"sensors  : {len(cameras)} ({', '.join(c for c in cameras if c.startswith('CAM'))})")
    print(f"images   : {n_images} keyframe images under samples/")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dest", default=None, help="where to extract (default: datasets/nuscenes)")
    ap.add_argument("--archive", default=None, help="a .tgz already on disk")
    ap.add_argument("--url", default=None, help="override the download URL")
    ap.add_argument(
        "--no-keep-archive",
        action="store_true",
        help="delete the .tgz once extracted, saving ~4.2 GB",
    )
    ap.add_argument("--no-save", action="store_true", help="do not write configs/local.yaml")
    ap.add_argument("--skip-space-check", action="store_true")
    args = ap.parse_args()

    paths = load_paths()
    dest = resolve(args.dest or paths.get("nuscenes_dest") or "datasets/nuscenes")

    if looks_like_nuscenes(dest) or descend(dest, looks_like_nuscenes, max_depth=3):
        root = descend(dest, looks_like_nuscenes, max_depth=3)
        print(f"nuScenes is already extracted at {root}")
    else:
        if args.archive:
            archive = resolve(args.archive)
            if not archive.exists():
                raise FileNotFoundError(f"{archive} does not exist")
        else:
            available = free_gb(dest)
            if available < MIN_FREE_GB and not args.skip_space_check:
                raise RuntimeError(
                    f"{available:.1f} GB free where the archive would go, and the "
                    f"download plus extraction needs about {MIN_FREE_GB:.0f} GB. "
                    "Use --dest to point at a bigger drive, or "
                    "--skip-space-check if you know better."
                )
            urls = [args.url] if args.url else MIRRORS
            if not supports_resume(urls[0]):
                print(
                    "note: this host does not advertise byte ranges, so an "
                    "interrupted download may have to start over"
                )
            archive = None
            for i, url in enumerate(urls):
                try:
                    print(f"downloading from {url}")
                    archive = download(url, dest.parent / "v1.0-mini.tgz")
                    break
                except (requests.RequestException, RuntimeError) as exc:
                    print(f"  failed: {exc}")
                    if i == len(urls) - 1:
                        raise
            assert archive is not None

        root = extract(archive, dest)
        if args.no_keep_archive and not args.archive:
            archive.unlink(missing_ok=True)
            print(f"deleted {archive.name}")

    summarise(root)

    if not args.no_save:
        local = set_local("nuscenes_root", root)
        print(f"\nremembered nuscenes_root in {local}")
    print("\nnext: python -m data.prepare")


if __name__ == "__main__":
    friendly(main)()
