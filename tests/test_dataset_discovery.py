"""Finding the dataset and layering per-machine config.

Both PCs keep the dataset somewhere different, so this is the logic that decides
what every script operates on.

    python tests/test_dataset_discovery.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.config import load_paths, save_yaml  # noqa: E402
from common.env import (  # noqa: E402
    dataset_health,
    descend,
    find_dataset_root,
    looks_like_prepared_dataset,
)


def make_dataset(root: Path, meta: bool = True, splits=("train", "val")) -> Path:
    for split in splits:
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
        (root / "images" / split / "a.jpg").write_bytes(b"not really a jpeg")
        (root / "labels" / split / "a.txt").write_text("0 0.5 0.5 0.1 0.1")
        if meta:
            (root / "meta" / split).mkdir(parents=True, exist_ok=True)
            (root / "meta" / split / "a.json").write_text("{}")
    return root


def test_looks_like_prepared_dataset():
    with tempfile.TemporaryDirectory() as tmp:
        root = make_dataset(Path(tmp) / "ds")
        assert looks_like_prepared_dataset(root)
        assert not looks_like_prepared_dataset(Path(tmp))
        # labels alone is not a dataset
        bare = Path(tmp) / "bare"
        (bare / "images" / "train").mkdir(parents=True)
        assert not looks_like_prepared_dataset(bare)


def test_descend_finds_a_wrapped_tree():
    with tempfile.TemporaryDirectory() as tmp:
        deep = Path(tmp) / "download" / "versions" / "1" / "yolo_3class"
        make_dataset(deep)
        found = descend(Path(tmp), looks_like_prepared_dataset, max_depth=5)
        assert found == deep
        # too shallow a search finds nothing rather than the wrong thing
        assert descend(Path(tmp), looks_like_prepared_dataset, max_depth=2) is None


def test_descend_accepts_the_root_itself():
    with tempfile.TemporaryDirectory() as tmp:
        root = make_dataset(Path(tmp) / "ds")
        assert descend(root, looks_like_prepared_dataset) == root


def test_find_dataset_root_prefers_explicit_over_env():
    with tempfile.TemporaryDirectory() as tmp:
        wanted = make_dataset(Path(tmp) / "wanted")
        other = make_dataset(Path(tmp) / "other")
        os.environ["ERIC_DATASET_ROOT"] = str(other)
        try:
            assert find_dataset_root(wanted) == wanted
            assert find_dataset_root() == other
        finally:
            del os.environ["ERIC_DATASET_ROOT"]


def test_find_dataset_root_explains_a_wrong_path():
    with tempfile.TemporaryDirectory() as tmp:
        try:
            find_dataset_root(tmp)
        except FileNotFoundError as exc:
            assert "images/train" in str(exc)
            return
        raise AssertionError("a directory with no dataset in it should be refused")


def test_find_dataset_root_never_downloads_when_told_not_to():
    with tempfile.TemporaryDirectory() as tmp:
        try:
            find_dataset_root(
                None, kaggle_slug="someone/something", allow_download=False,
                local_default=Path(tmp),
            )
        except FileNotFoundError as exc:
            assert "fetch_dataset" in str(exc)
            return
        raise AssertionError("nothing should have been found")


def test_dataset_health_flags_missing_meta():
    with tempfile.TemporaryDirectory() as tmp:
        root = make_dataset(Path(tmp) / "ds", meta=False)
        health = dataset_health(root)
        assert health["splits"]["train"]["images"] == 1
        assert health["splits"]["train"]["meta"] == 0
        assert any("meta/" in p for p in health["problems"])
        # Task 3 is the reason it matters, so say so
        assert any("Task 3" in p for p in health["problems"])


def test_dataset_health_is_quiet_on_a_complete_set():
    with tempfile.TemporaryDirectory() as tmp:
        root = make_dataset(Path(tmp) / "ds")
        assert dataset_health(root)["problems"] == []


def test_local_config_overrides_defaults_but_not_with_null():
    with tempfile.TemporaryDirectory() as tmp:
        local = Path(tmp) / "local.yaml"
        save_yaml({"dataset_root": "D:/elsewhere", "kaggle_dataset": None}, local)
        cfg = load_paths(local)
        assert cfg["dataset_root"] == "D:/elsewhere"
        # a null in the local file must not blank out a working default
        assert cfg["kaggle_dataset"] == load_paths(Path(tmp) / "absent.yaml")["kaggle_dataset"]
        assert cfg["out_root"]  # untouched keys survive


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
