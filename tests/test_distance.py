"""Distance geometry, checked against boxes built from a known depth.

Both estimators must return exactly the depth that produced the box, otherwise a
sign or a convention is wrong - and a convention error here looks like plausible
numbers, not like a crash.

    python tests/test_distance.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from distance.geometry import estimate, fuse, ground_plane, pinhole  # noqa: E402

FX = FY = 1266.4
CX, CY = 800.0, 450.0
K = [[FX, 0, CX], [0, FY, CY], [0, 0, 1]]
H_CAM = 1.51
CONE_H = 0.78
IMG_H = 900

PRIORS = {
    "sigma_v_px": 2.0,
    "classes": {
        "cone": {"dim": "h", "H_eff": CONE_H, "r": 0.17, "use_ground_plane": True, "a": 1.0, "b": 0.0},
        "barrier": {"dim": "h", "H_eff": 1.06, "r": 0.09, "use_ground_plane": True, "a": 1.0, "b": 0.0},
        "stop_sign": {"dim": "w", "H_eff": 0.75, "r": 0.10, "use_ground_plane": False, "a": 1.0, "b": 0.0},
    },
}


def cone_box(z, height=CONE_H, h_cam=H_CAM):
    """The box a cone of this height standing at depth z would project to."""
    h_px = FY * height / z
    v_bottom = CY + FY * h_cam / z          # where the ground is at that depth
    return [700.0, v_bottom - h_px, 740.0, v_bottom]


def test_pinhole_recovers_the_depth():
    for z in (3.0, 10.0, 25.0, 40.0):
        got = pinhole(cone_box(z), PRIORS["classes"]["cone"], K)
        assert abs(got - z) < 1e-6, (z, got)


def test_ground_plane_recovers_the_depth():
    for z in (3.0, 10.0, 25.0, 40.0):
        got = ground_plane(cone_box(z), K, H_CAM, pitch=0.0)
        assert abs(got - z) < 1e-6, (z, got)


def test_pitch_moves_the_estimate_the_right_way():
    """For a fixed box bottom, pitching the camera down means a nearer object.

    The ray leaves the lens further below horizontal, so it meets the ground
    sooner. Getting this sign backwards would read every distance long.
    """
    box = cone_box(12.0)
    flat = ground_plane(box, K, H_CAM, pitch=0.0)
    pitched = ground_plane(box, K, H_CAM, pitch=math.radians(5.0))
    assert pitched < flat
    # and a camera pitched up reaches further
    assert ground_plane(box, K, H_CAM, pitch=math.radians(-2.0)) > flat


def test_ground_plane_refuses_the_horizon():
    """Above the horizon sin(t + a) -> 0 and Z diverges, so it must decline."""
    box = [700.0, 400.0, 740.0, CY - 5]  # bottom edge above the principal point
    assert ground_plane(box, K, H_CAM, pitch=0.0) is None


def test_fusion_lies_between_the_two_estimates():
    prior = PRIORS["classes"]["cone"]
    fused = fuse(12.0, 10.0, prior, K, H_CAM)
    assert 10.0 <= fused <= 12.0


def test_fusion_falls_back_when_the_ground_plane_declines():
    prior = PRIORS["classes"]["cone"]
    assert fuse(17.5, None, prior, K, H_CAM) == 17.5


def test_fusion_prefers_the_ground_plane_at_short_range():
    """Its error grows with Z^2, so it is the sharper estimate up close."""
    prior = PRIORS["classes"]["cone"]
    near = fuse(6.0, 5.0, prior, K, H_CAM)      # D0 says 6, D1 says 5
    assert abs(near - 5.0) < abs(near - 6.0)


def test_stop_sign_uses_width_and_no_ground_plane():
    """A sign is on a pole: its box bottom is not a contact point."""
    z = 8.0
    w_px = FX * 0.75 / z
    box = [800 - w_px / 2, 300.0, 800 + w_px / 2, 300.0 + w_px]
    out = estimate(box, "stop_sign", K, H_CAM, 0.0, PRIORS, img_height=IMG_H)
    assert out["d1"] is None
    assert abs(out["d0"] - z) < 1e-6
    assert abs(out["fused"] - z) < 1e-6


def test_a_box_clipped_at_the_frame_bottom_drops_the_ground_plane():
    box = [700.0, 700.0, 740.0, IMG_H - 1]
    out = estimate(box, "cone", K, H_CAM, 0.0, PRIORS, img_height=IMG_H)
    assert out["d1"] is None, "no visible contact point, so D1 must not be used"


def test_the_linear_correction_is_applied_last():
    priors = {
        "sigma_v_px": 2.0,
        "classes": {
            "cone": {**PRIORS["classes"]["cone"], "a": 1.1, "b": -0.5},
        },
    }
    box = cone_box(10.0)
    raw = estimate(box, "cone", K, H_CAM, 0.0, priors, img_height=IMG_H, apply_correction=False)
    done = estimate(box, "cone", K, H_CAM, 0.0, priors, img_height=IMG_H)
    assert abs(done["corrected"] - (1.1 * raw["fused"] - 0.5)) < 1e-9


def test_a_taller_prior_reads_further_away():
    """H_eff scales the pinhole estimate linearly - the fitted-bias mechanism."""
    box = cone_box(10.0)
    small = pinhole(box, {"dim": "h", "H_eff": 0.78}, K)
    large = pinhole(box, {"dim": "h", "H_eff": 0.90}, K)
    assert large > small
    assert abs(large / small - 0.90 / 0.78) < 1e-9


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
