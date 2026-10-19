"""Duplicate-label suppression - the thing that made barriers trainable.

    python tests/test_label_dedup.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from data.nuscenes_to_yolo import (
        DEDUP_COVER,
        MERGE_CLASSES,
        MERGE_GAP,
        covered_fraction,
        merge_rows,
        suppress_covered,
    )

    HAVE_DEPS = True
except ImportError:  # pyquaternion / devkit absent
    HAVE_DEPS = False


def box(cls, x1, y1, x2, y2, z=10.0):
    return {
        "cls": cls,
        "box": [x1, y1, x2, y2],
        "z": z,
        "dims_whl": [2.32, 0.3, 1.06],
        "trunc": False,
        "vis": 4,
    }


def receding_row(n=6):
    """A roadside row: side by side, each partly covering its neighbour."""
    return [
        box(1, 100 + 70 * i, 500 - 9 * i, 280 + 70 * i, 640 - 20 * i, z=9.0 + 3 * i)
        for i in range(n)
    ]


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


def test_a_row_merges_into_one_box():
    row = receding_row(6)
    merged, dropped = merge_rows(row, MERGE_GAP, MERGE_CLASSES)
    assert len(merged) == 1 and dropped == 5
    # chained through neighbours: the ends do not overlap each other directly
    assert merge_rows(row[:1] + row[-1:], MERGE_GAP, MERGE_CLASSES)[1] == 0


def test_merged_box_keeps_the_nearest_depth_and_contact_point():
    row = receding_row(6)
    merged = merge_rows(row, MERGE_GAP, MERGE_CLASSES)[0][0]
    assert merged["z"] == min(o["z"] for o in row)
    # the bottom edge is still the nearest segment's contact point, so the
    # ground-plane distance estimate stays exact
    assert merged["box"][3] == max(o["box"][3] for o in row)
    # per-segment depths survive for checking the geometry object by object
    assert merged["z_segments"] == sorted(round(o["z"], 2) for o in row)
    assert merged["n_merged"] == 6


def test_a_barrier_across_the_road_stays_separate():
    objs = receding_row(3) + [box(1, 1300, 440, 1360, 480, z=35.0)]
    merged, _ = merge_rows(objs, MERGE_GAP, MERGE_CLASSES)
    assert len(merged) == 2


def test_cones_are_never_merged():
    """Cones are discrete objects; only barriers form a continuous structure."""
    cones = [box(0, 100, 500, 200, 640, z=9.0), box(0, 150, 505, 250, 635, z=10.0)]
    assert merge_rows(cones, MERGE_GAP, MERGE_CLASSES) == (cones, 0)


def test_a_row_across_the_view_merges_even_without_overlap():
    """Side by side with small gaps - the case IoU-based merging missed."""
    row = [box(1, 100 + 210 * i, 500, 290 + 210 * i, 610, z=12.0 + i) for i in range(5)]
    assert all(
        row[i]["box"][2] < row[i + 1]["box"][0] for i in range(4)
    ), "these must not overlap at all"
    merged, dropped = merge_rows(row, MERGE_GAP, MERGE_CLASSES)
    assert len(merged) == 1 and dropped == 4
    assert merged[0]["box"] == [100, 500, 1130, 610]


def test_a_wide_gap_is_not_one_structure():
    objs = [box(1, 100, 500, 290, 610, z=12.0), box(1, 900, 500, 1090, 610, z=13.0)]
    assert merge_rows(objs, MERGE_GAP, MERGE_CLASSES)[1] == 0


def test_vertical_separation_is_not_one_structure():
    """Same column, far apart vertically - not a row."""
    objs = [box(1, 100, 100, 290, 200, z=30.0), box(1, 100, 500, 290, 610, z=10.0)]
    assert merge_rows(objs, MERGE_GAP, MERGE_CLASSES)[1] == 0


def test_negative_gap_disables_merging():
    row = receding_row(4)
    assert merge_rows(row, -1, MERGE_CLASSES) == (row, 0)


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
