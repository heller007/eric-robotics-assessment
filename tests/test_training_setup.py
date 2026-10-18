"""Run-config merging and class-balanced sampling, neither of which needs a GPU.

    python tests/test_training_setup.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.config import load_yaml  # noqa: E402
from data.balance import repeat_factors  # noqa: E402
from detection.train import build_args, coerce  # noqa: E402


def test_coerce_types():
    assert coerce("10") == 10
    assert coerce("0.5") == 0.5
    assert coerce("True") is True
    assert coerce("none") is None
    assert coerce("auto") == "auto"


def test_build_args_merges_defaults_and_overrides():
    runs = load_yaml("runs.yaml")
    spec, args = build_args(runs, "B1", ["epochs=5", "batch=4"])
    assert args["epochs"] == 5 and args["batch"] == 4
    assert args["freeze"] == 10          # from defaults
    assert args["flipud"] == 0.0         # no vertical flip on road scenes
    assert "models" not in args          # script-level keys never reach ultralytics
    assert "purpose" not in args
    assert spec["models"][0].startswith("yolo")


def test_build_args_per_run_overrides_default():
    runs = load_yaml("runs.yaml")
    _, b2 = build_args(runs, "B2", [])
    assert b2["imgsz"] == 960
    assert b2["batch"] == 8              # 960 px at batch=-1 tends to OOM
    _, b1u = build_args(runs, "B1u", [])
    assert b1u["freeze"] == 0 and b1u["lr0"] == 0.002


def test_build_args_rejects_unknown_run():
    runs = load_yaml("runs.yaml")
    try:
        build_args(runs, "nope", [])
    except SystemExit:
        return
    raise AssertionError("an unknown run should be refused")


def test_repeat_factors_boosts_the_rare_class():
    # stop signs in 1 image out of 10, cones in 9
    per_image = {Path(f"a{i}.jpg"): {0} for i in range(9)}
    per_image[Path("rare.jpg")] = {2}
    repeats, r_c, present = repeat_factors(per_image, threshold=0.5)
    assert present[0] == 9 and present[2] == 1
    assert r_c[0] == 1.0                       # already frequent, never downsampled
    assert r_c[2] > 2.0                        # sqrt(0.5 / 0.1)
    assert repeats[Path("rare.jpg")] == round(r_c[2])
    assert repeats[Path("a0.jpg")] == 1


def test_repeat_factors_leaves_background_frames_alone():
    per_image = {Path("bg.jpg"): set(), Path("cone.jpg"): {0}}
    repeats, _, _ = repeat_factors(per_image, threshold=0.5)
    assert repeats[Path("bg.jpg")] == 1


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
