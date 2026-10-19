"""Distance from a 2D box, by geometry alone. No second network in the loop.

Two independent estimates and a weighted combination:

  D0  known size   Z = f_y * H_eff / h_px        (f_x * W_eff / w_px for a sign)
  D1  ground plane Z = h_c * cos(a) / sin(t + a),  a = atan((v_b - c_y) / f_y)

They fail in opposite directions, which is why both are worth having. D0's
relative error is the box-height error divided by the height, so it degrades
slowly with range but is floored by how much the real object size varies. D1 is
exact for a known mount on flat ground up close, but its error grows with Z^2,
so a single pixel of box-bottom noise ruins it far away.

`Z` throughout is depth along the optical axis, in metres - the same quantity
nuScenes stores as a box centre in the camera frame.
"""
from __future__ import annotations

import math

GROUND_CLASSES = ("cone", "barrier")  # a stop sign is on a pole, not the ground


def pinhole(box, prior: dict, K) -> float:
    """D0. Uses the box height, or its width when the prior says `w`.

    Height is right for cones and barriers: a barrier's apparent width changes
    with viewing angle while its height does not. Width is right for a stop
    sign, because the pole and whatever stands in front of it keep cutting the
    bottom of the box off, and an octagon is as wide as it is tall.
    """
    x1, y1, x2, y2 = box
    if prior.get("dim", "h") == "w":
        return float(K[0][0]) * prior["H_eff"] / max(x2 - x1, 1e-3)
    return float(K[1][1]) * prior["H_eff"] / max(y2 - y1, 1e-3)


def ground_plane(box, K, h_cam: float, pitch: float, guard_deg: float = 0.5) -> float | None:
    """D1. Where the bottom of the box meets the ground.

    Returns None when the ray is too close to the horizon: there sin(t + a)
    approaches zero and Z diverges, so a nominally valid answer would be
    arbitrarily wrong.
    """
    v_bottom = box[3]
    alpha = math.atan2(v_bottom - float(K[1][2]), float(K[1][1]))
    angle = pitch + alpha
    if angle < math.radians(guard_deg):
        return None
    return h_cam * math.cos(alpha) / math.sin(angle)


def fuse(z0: float, z1: float | None, prior: dict, K, h_cam: float, sigma_v: float = 2.0):
    """Inverse-variance weighted mean of the two estimates.

    sigma_0 = Z_0 * r      the size prior's relative spread
    sigma_1 = Z_1^2 / (f_y * h_c) * sigma_v    one pixel of bottom-edge noise,
                                               propagated through D1

    Near objects end up leaning on D1 and far ones on D0, which is what those
    two error models say should happen - no tuning required.
    """
    if z1 is None:
        return z0
    var0 = (z0 * prior["r"]) ** 2
    var1 = (z1**2 / (float(K[1][1]) * h_cam) * sigma_v) ** 2
    if var0 <= 0 or var1 <= 0:
        return z0
    return (z0 / var0 + z1 / var1) / (1.0 / var0 + 1.0 / var1)


def estimate(
    box,
    cls_name: str,
    K,
    h_cam: float,
    pitch: float,
    priors: dict,
    img_height: int | None = None,
    sigma_v: float | None = None,
    guard_deg: float = 0.5,
    apply_correction: bool = True,
) -> dict:
    """Every stage for one box, so the evaluation can score them separately."""
    prior = priors["classes"][cls_name]
    sigma_v = priors.get("sigma_v_px", 2.0) if sigma_v is None else sigma_v

    z0 = pinhole(box, prior, K)

    # D1 needs the bottom edge to be the object's contact point with the ground.
    # A box clipped by the frame edge has no such edge, and a class on a pole
    # never had one.
    touches_bottom = img_height is not None and box[3] >= img_height - 2
    z1 = (
        ground_plane(box, K, h_cam, pitch, guard_deg)
        if prior.get("use_ground_plane") and not touches_bottom
        else None
    )

    fused = fuse(z0, z1, prior, K, h_cam, sigma_v)
    corrected = prior.get("a", 1.0) * fused + prior.get("b", 0.0) if apply_correction else fused
    return {"d0": z0, "d1": z1, "fused": fused, "corrected": max(corrected, 0.0)}
