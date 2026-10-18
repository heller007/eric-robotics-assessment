"""Point the repo at nuScenes v1.0-mini, downloading it only if it has to.

Preference order: a path you give, then one already in configs/local.yaml or
already extracted, then an archive on disk, and only as a last resort the ~3.9
GiB download. Whichever it is, the copy is verified and its location recorded in
configs/local.yaml so no later command needs a flag.

Usage
  python -m data.fetch_nuscenes --root D:/kaggle/nuscenes-mini   # a copy you have
  python -m data.fetch_nuscenes --archive D:/nuscenes-mini.zip   # .zip or .tgz
  python -m data.fetch_nuscenes                                  # download it
  python -m data.fetch_nuscenes --dest D:/datasets/nuscenes      # download elsewhere
  python -m data.fetch_nuscenes --no-keep-archive                # free the archive after

--root does not have to be the dataroot: a Kaggle mirror such as
aadimator/nuscenes-mini, or a zip extracted by hand, usually leaves a wrapper
folder or two above the v1.0-mini/ + samples/ pair, so point at wherever you
unpacked it. The copy is then checked for the metadata tables, the six cameras
and a plausible image count, because a partial mirror otherwise fails deep
inside the devkit.

Downloading means accepting the nuScenes Terms of Use:
https://www.nuscenes.org/terms-of-use - free for non-commercial use, which is
what this assessment is.
"""
from __future__ import annotations

import argparse
import shutil
import tarfile
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

from common.cli import friendly
from common.config import load_paths, resolve, set_local
from common.env import descend, verify_nuscenes

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


def _safe_name(name: str) -> bool:
    return not name.startswith(("/", "\\")) and ".." not in Path(name).parts


def extract(archive: Path, dest: Path) -> Path:
    """Extract a .tgz/.tar.gz or .zip, then locate the dataroot inside it.

    Kaggle serves .zip while nuScenes serves .tgz, and either may be what is on
    the disk, so both are handled.
    """
    dest.mkdir(parents=True, exist_ok=True)
    already = descend(dest, looks_like_nuscenes, max_depth=3)
    if already:
        print(f"{already} already holds an extracted dataset - skipping extraction")
        return already

    suffixes = "".join(archive.suffixes[-2:]).lower()
    print(f"extracting {archive.name} -> {dest}")
    if archive.suffix.lower() == ".zip":
        with zipfile.ZipFile(archive) as zf:
            names = zf.namelist()
            with tqdm(total=len(names), unit="file", desc="extract") as bar:
                for name in names:
                    if not _safe_name(name):
                        raise RuntimeError(f"refusing unsafe archive entry {name!r}")
                    zf.extract(name, path=dest)
                    bar.update(1)
    elif suffixes.endswith((".tgz", ".tar.gz", ".tar")):
        with tarfile.open(archive) as tar:
            members = tar.getmembers()
            with tqdm(total=len(members), unit="file", desc="extract") as bar:
                for member in members:
                    # the data filter refuses absolute paths, .. traversal and
                    # anything that is not a plain file or directory
                    try:
                        tar.extract(member, path=dest, filter="data")
                    except TypeError:  # Python without extraction filters
                        if not _safe_name(member.name):
                            raise RuntimeError(
                                f"refusing unsafe archive entry {member.name!r}"
                            )
                        tar.extract(member, path=dest)
                    bar.update(1)
    else:
        raise RuntimeError(
            f"{archive.name}: expected a .tgz, .tar.gz or .zip. If it is already "
            "extracted, pass --root <that folder> instead of --archive."
        )

    found = descend(dest, looks_like_nuscenes, max_depth=3)
    if found is None:
        raise RuntimeError(
            f"extracted to {dest} but found no v1.0-mini/ + samples/ inside it"
        )
    return found


def report(root: Path) -> list[str]:
    """Print what this copy holds and return whatever is wrong with it."""
    info = verify_nuscenes(root)
    print(f"\ndataroot : {root}")
    print(f"cameras  : {len(info['cameras'])} ({', '.join(info['cameras'])})")
    print(f"images   : {info['keyframe_images']} keyframe images under samples/")
    for problem in info["problems"]:
        print(f"problem  : {problem}")
    return info["problems"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--root",
        "--nuscenes-root",
        dest="root",
        default=None,
        help="a copy already on disk - the dataroot, or any folder above it. "
        "Nothing is downloaded when this is given.",
    )
    ap.add_argument("--dest", default=None, help="where to extract (default: datasets/nuscenes)")
    ap.add_argument("--archive", default=None, help="a .tgz or .zip already on disk")
    ap.add_argument("--url", default=None, help="override the download URL")
    ap.add_argument(
        "--no-keep-archive",
        action="store_true",
        help="delete the downloaded archive once extracted, saving ~3.9 GiB",
    )
    ap.add_argument("--no-save", action="store_true", help="do not write configs/local.yaml")
    ap.add_argument("--skip-space-check", action="store_true")
    ap.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="record the copy even if verification finds problems",
    )
    args = ap.parse_args()

    paths = load_paths()
    dest = resolve(args.dest or paths.get("nuscenes_dest") or "datasets/nuscenes")

    # Order of preference: a path you gave, one already configured or extracted,
    # an archive on disk, and only then the 3.9 GiB download.
    configured = None if args.root else paths.get("nuscenes_root")
    given = args.root or configured
    found = descend(resolve(given), looks_like_nuscenes, max_depth=3) if given else None

    if given and not found:
        raise FileNotFoundError(
            f"{resolve(given)} does not contain a nuScenes dataroot.\n"
            "Expected a folder with v1.0-mini/ and samples/ inside it, at or below "
            "the path given. If you have the archive rather than an extracted "
            "copy, use --archive instead."
        )

    if found:
        root = found
        where = "configured" if configured and not args.root else "given"
        print(f"using the {where} copy at {root}")
    elif descend(dest, looks_like_nuscenes, max_depth=3):
        root = descend(dest, looks_like_nuscenes, max_depth=3)
        print(f"nuScenes is already extracted at {root}")
    else:
        if args.archive:
            archive = resolve(args.archive)
            if not archive.exists():
                raise FileNotFoundError(f"{archive} does not exist")
        else:
            print(
                "no local copy found - falling back to downloading v1.0-mini "
                "(~3.9 GiB). Pass --root <folder> if you already have it."
            )
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

    problems = report(root)
    if problems and not args.allow_incomplete:
        raise RuntimeError(
            "this copy is not usable as it stands (see above). Point at a "
            "different one with --root, or pass --allow-incomplete to record it "
            "anyway and restrict the converter with --cams."
        )

    if not args.no_save:
        local = set_local("nuscenes_root", root)
        print(f"\nremembered nuscenes_root in {local}")
    print("\nnext: python -m data.prepare")


if __name__ == "__main__":
    friendly(main)()
