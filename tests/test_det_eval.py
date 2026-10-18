"""The AP implementation is the one number every variant is compared on, so it
gets checked against cases whose answer is known by hand.

    python -m pytest tests -q      (or just: python tests/test_det_eval.py)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.labels import iou_matrix  # noqa: E402
from eval.det_eval import average_precision, evaluate, match_class, recall_by_range  # noqa: E402

NAMES = ["cone", "barrier", "stop_sign"]


def box(x, y, w=40, h=60):
    return [x, y, x + w, y + h]


def test_iou_matrix():
    a = np.array([box(0, 0, 10, 10)])
    b = np.array([box(0, 0, 10, 10), box(5, 0, 10, 10), box(100, 100, 10, 10)])
    iou = iou_matrix(a, b)
    assert np.isclose(iou[0, 0], 1.0)
    assert np.isclose(iou[0, 1], 50 / 150)  # 50 overlap, 150 union
    assert np.isclose(iou[0, 2], 0.0)
    assert iou_matrix(np.zeros((0, 4)), b).shape == (0, 3)


def test_average_precision_perfect_and_empty():
    tp = np.ones(5)
    conf = np.linspace(0.9, 0.5, 5)
    assert np.isclose(average_precision(tp, conf, n_gt=5), 1.0)
    # half the ground truth missed -> recall caps at 0.5, precision stays 1
    assert np.isclose(average_precision(tp, conf, n_gt=10), 0.5, atol=0.02)
    # no predictions at all
    assert average_precision(np.zeros(0), np.zeros(0), n_gt=4) == 0.0
    # no ground truth -> undefined, not zero
    assert np.isnan(average_precision(tp, conf, n_gt=0))


def test_average_precision_ranks_confidence():
    # one TP, one FP. AP is higher when the TP is the more confident of the two.
    good = average_precision(np.array([1.0, 0.0]), np.array([0.9, 0.2]), n_gt=1)
    bad = average_precision(np.array([0.0, 1.0]), np.array([0.9, 0.2]), n_gt=1)
    assert good > bad
    assert np.isclose(good, 1.0)


def test_match_class_one_gt_per_prediction():
    gt = [np.array([box(0, 0), box(200, 0)])]
    # two predictions on the same object: the second must count as a false positive
    preds = [(np.array([box(0, 0), box(2, 2), box(200, 0)]), np.array([0.9, 0.8, 0.7]))]
    tp, conf, n_gt = match_class(gt, preds, iou_thr=0.5)
    assert n_gt == 2
    assert tp.tolist() == [1.0, 0.0, 1.0]
    assert conf.tolist() == [0.9, 0.8, 0.7]


def test_evaluate_perfect_predictions():
    records = [
        {
            "boxes": np.array([box(10, 10), box(300, 50)]),
            "classes": np.array([0, 1]),
            "depths": np.array([8.0, 25.0]),
        },
        {
            "boxes": np.array([box(100, 100)]),
            "classes": np.array([2]),
            "depths": np.array([np.nan]),
        },
    ]
    preds = [
        {
            "boxes": r["boxes"].copy(),
            "classes": r["classes"].copy(),
            "confs": np.full(len(r["classes"]), 0.9),
        }
        for r in records
    ]
    result = evaluate(records, preds, NAMES)
    assert np.isclose(result["mAP50"], 1.0)
    assert np.isclose(result["mAP50_95"], 1.0)
    for name in NAMES:
        assert np.isclose(result["per_class"][name]["recall@0.25"], 1.0)

    # depth bins: the cone at 8 m and the barrier at 25 m land in different bins
    cone_bins = result["per_class"]["cone"]["recall_by_range"]
    assert cone_bins["0-10m"]["gt"] == 1
    assert cone_bins["10-20m"]["gt"] == 0
    assert result["per_class"]["barrier"]["recall_by_range"]["20-40m"]["gt"] == 1
    # a pseudo-labelled stop sign has no depth, so it is counted in no bin
    assert all(
        v["gt"] == 0 for v in result["per_class"]["stop_sign"]["recall_by_range"].values()
    )


def test_evaluate_zero_shot_shape():
    """A model that only ever predicts stop signs: AP 0 on the other classes."""
    records = [
        {
            "boxes": np.array([box(10, 10), box(100, 100)]),
            "classes": np.array([0, 2]),
            "depths": np.array([5.0, np.nan]),
        }
    ]
    preds = [{"boxes": np.array([box(100, 100)]), "classes": np.array([2]), "confs": np.array([0.8])}]
    result = evaluate(records, preds, NAMES)
    assert np.isclose(result["per_class"]["stop_sign"]["AP50"], 1.0)
    assert result["per_class"]["cone"]["AP50"] == 0.0
    assert np.isnan(result["per_class"]["barrier"]["AP50"])  # no barrier in this set
    # the mean skips the undefined class rather than treating it as zero
    assert np.isclose(result["mAP50"], 0.5)


def test_recall_by_range_respects_conf():
    gt = [np.array([box(0, 0)])]
    depths = [np.array([5.0])]
    low = [(np.array([box(0, 0)]), np.array([0.1]))]
    high = [(np.array([box(0, 0)]), np.array([0.9]))]
    assert recall_by_range(gt, depths, low)["0-10m"]["recall"] == 0.0
    assert recall_by_range(gt, depths, high)["0-10m"]["recall"] == 1.0


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
