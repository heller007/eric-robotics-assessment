"""One drawing routine, shared by detection/infer.py and app/realtime.py.

Kept free of model objects so it takes plain arrays: the realtime loop passes
TensorRT output, the eval scripts pass ground truth, and both look the same.
"""
from __future__ import annotations

import numpy as np

# BGR, one per class index: cone orange, barrier blue, stop sign red
CLASS_COLORS = [(0, 165, 255), (255, 150, 0), (0, 0, 220)]
GRAY = (160, 160, 160)


def color_for(cls: int) -> tuple[int, int, int]:
    return CLASS_COLORS[cls % len(CLASS_COLORS)] if cls >= 0 else GRAY


def draw_detections(
    img: np.ndarray,
    boxes: np.ndarray,
    classes: np.ndarray,
    names: list[str],
    confs: np.ndarray | None = None,
    distances: np.ndarray | None = None,
    track_ids: np.ndarray | None = None,
    thickness: int = 2,
) -> np.ndarray:
    """Draw xyxy boxes in place and return the image.

    distances, when given, are metres along the optical axis; NaN is skipped so a
    rejected geometric estimate shows as a box with no range rather than a lie.
    """
    import cv2

    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    classes = np.asarray(classes, dtype=int).reshape(-1)
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        cls = int(classes[i])
        color = color_for(cls)
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(img, p1, p2, color, thickness)

        parts = [names[cls] if 0 <= cls < len(names) else str(cls)]
        if track_ids is not None and not _isnan(track_ids[i]):
            parts.append(f"#{int(track_ids[i])}")
        if confs is not None:
            parts.append(f"{float(confs[i]):.2f}")
        if distances is not None and not _isnan(distances[i]):
            parts.append(f"{float(distances[i]):.1f}m")
        label = " ".join(parts)

        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ty = max(th + base, p1[1])
        cv2.rectangle(img, (p1[0], ty - th - base), (p1[0] + tw + 4, ty), color, -1)
        cv2.putText(
            img,
            label,
            (p1[0] + 2, ty - base + 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return img


def draw_hud(img: np.ndarray, lines: list[str]) -> np.ndarray:
    """Top-left overlay for FPS and variant name."""
    import cv2

    for i, text in enumerate(lines):
        y = 24 + i * 22
        cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(
            img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA
        )
    return img


def _isnan(v) -> bool:
    try:
        return bool(np.isnan(v))
    except (TypeError, ValueError):
        return v is None
