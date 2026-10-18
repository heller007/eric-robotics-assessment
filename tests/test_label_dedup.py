"""Duplicate-label suppression - the thing that made barriers trainable.

    python tests/test_label_dedup.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from data.nuscenes_to_yolo import DEDUP_COVER, covered_fraction, suppress_covered

    HAVE_DEPS = True
except ImportError:  # pyquaternion / devkit absent
    HAVE_DEPS = False


def box(cls, x1, y1, x2, y2, z=10.0):
    return {"cls": cls, "box": [x1, y1, x2, y2], "z": z}


def test_covered_fraction():
    assert covered_fraction([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert covered_fraction([0, 0, 10, 10], [5, 0, 15, 10]) == 0.5
    assert covered_fraction([0, 0, 10, 10], [100, 100, 110, 110]) == 0.0
    # a small box fully inside a big one is entirely covered
    assert covered_fraction([2, 2, 4, 4], [0, 0, 10, 10]) == 1.0


def test_a_row_of_barriers_collapses_to_the_nearest():
    row = [
        box(1, 100, 400, 600, 520, z=9.0),
        box(1, 150, 410, 560, 505, z=11.0),
        box(1, 200, 415, 520, 500, z=13.0),
    ]
    kept, dropped = suppress_covered(row, DEDUP_COVER)
    assert len(kept) == 1 and dropped == 2
    # the largest box is the nearest segment, and its depth is a real measurement
    assert kept[0]["z"] == 9.0


def test_separate_objects_are_kept():
    objs = [
        box(1, 100, 400, 600, 520, z=9.0),
        box(1, 1200, 430, 1260, 470, z=30.0),
    ]
    kept, dropped = suppress_covered(objs, DEDUP_COVER)
    assert dropped == 0 and len(kept) == 2


def test_suppression_is_per_class():
    """A cone standing in front of a barrier must survive."""
    objs = [box(1, 100, 400, 600, 520), box(0, 160, 415, 200, 500)]
    kept, dropped = suppress_covered(objs, DEDUP_COVER)
    assert dropped == 0
    assert {o["cls"] for o in kept} == {0, 1}


def test_partial_overlap_is_not_a_duplicate():
    objs = [box(1, 0, 0, 100, 100), box(1, 50, 0, 150, 100)]  # half covered
    assert suppress_covered(objs, DEDUP_COVER)[1] == 0


def test_zero_cover_disables_it():
    row = [box(1, 100, 400, 600, 520), box(1, 150, 410, 560, 505)]
    assert suppress_covered(row, 0.0) == (row, 0)


if __name__ == "__main__":
    if not HAVE_DEPS:
        print("skipped: pyquaternion not installed")
        raise SystemExit(0)
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
