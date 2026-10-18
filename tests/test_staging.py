"""Image staging: the step that decides whether the build costs 1.6 GB of
duplicated files or nothing at all.

    python tests/test_staging.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.staging import LINK_ORDER, stage_image  # noqa: E402


def make_src(tmp: Path, name: str = "a.jpg", body: bytes = b"pretend jpeg") -> Path:
    src = tmp / name
    src.write_bytes(body)
    return src


def test_auto_prefers_a_link_over_a_copy():
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        src = make_src(tmp)
        used = stage_image(src, tmp / "staged.jpg")
        # within one volume a hard link always works, so nothing should be copied
        assert used == "hardlink", used
        assert (tmp / "staged.jpg").read_bytes() == src.read_bytes()


def test_existing_destination_is_left_alone():
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        src = make_src(tmp)
        dst = tmp / "staged.jpg"
        dst.write_bytes(b"already here")
        assert stage_image(src, dst) == "present"
        assert dst.read_bytes() == b"already here"


def test_copy_mode_makes_an_independent_file():
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        src = make_src(tmp)
        dst = tmp / "copied.jpg"
        assert stage_image(src, dst, "copy") == "copy"
        src.write_bytes(b"changed after staging")
        assert dst.read_bytes() == b"pretend jpeg"


def test_missing_parent_directory_is_created():
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        src = make_src(tmp)
        dst = tmp / "images" / "train" / "a.jpg"
        assert stage_image(src, dst) in LINK_ORDER["auto"]
        assert dst.exists()


def test_a_missing_source_fails_loudly():
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        try:
            stage_image(tmp / "nope.jpg", tmp / "out.jpg", "copy")
        except RuntimeError as exc:
            assert "could not stage" in str(exc)
            return
        raise AssertionError("staging a file that does not exist should fail")


def test_unknown_mode_is_rejected():
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        try:
            stage_image(make_src(tmp), tmp / "out.jpg", "teleport")
        except ValueError as exc:
            assert "teleport" in str(exc)
            return
        raise AssertionError("an unknown link mode should be refused")


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"ok   {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL {name}: {exc}")
    raise SystemExit(1 if failures else 0)
