"""Run a trained detector over images, a folder or a video and save annotations.

Any exported variant loads through the same call, so this works for .pt,
.engine, .onnx and an _openvino_model/ folder.

  python -m detection.infer --weights runs/B1/weights/best.pt --source datasets/yolo_3class/images/val --limit 8
  python -m detection.infer --weights runs/B1/weights/best.pt --source clip.mp4 --save-video results/B1_clip.mp4
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from common.cli import friendly
from common.config import class_names, resolve
from common.draw import draw_detections, draw_hud
from common.labels import list_images

VIDEO_EXT = (".mp4", ".avi", ".mov", ".mkv")


def annotate(result, names: list[str], img=None):
    """Draw one ultralytics Result onto its own image (or a provided frame)."""
    frame = result.orig_img.copy() if img is None else img
    boxes = result.boxes
    if len(boxes) == 0:
        return frame
    return draw_detections(
        frame,
        boxes.xyxy.cpu().numpy(),
        boxes.cls.cpu().numpy().astype(int),
        names,
        confs=boxes.conf.cpu().numpy(),
    )


def run_images(model, sources: list[Path], dest: Path, names, imgsz, conf, device) -> int:
    import cv2

    dest.mkdir(parents=True, exist_ok=True)
    written = 0
    for path in sources:
        res = model.predict(str(path), imgsz=imgsz, conf=conf, device=device, verbose=False)[0]
        frame = annotate(res, names)
        ms = sum(res.speed.values())
        draw_hud(frame, [f"{Path(path).name}", f"{ms:.1f} ms  {len(res.boxes)} det"])
        cv2.imwrite(str(dest / f"{Path(path).stem}_pred.jpg"), frame)
        written += 1
    return written


def run_video(model, source, dest: Path, names, imgsz, conf, device) -> int:
    import cv2

    cap = cv2.VideoCapture(0 if source == "webcam" else str(source))
    if not cap.isOpened():
        raise SystemExit(f"could not open {source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    dest.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(dest), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    n = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            res = model.predict(frame, imgsz=imgsz, conf=conf, device=device, verbose=False)[0]
            out = annotate(res, names, frame)
            ms = sum(res.speed.values())
            draw_hud(out, [f"frame {n}", f"{ms:.1f} ms model"])
            writer.write(out)
            n += 1
    finally:
        cap.release()
        writer.release()
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--source", required=True, help="image, folder, video, or 'webcam'")
    ap.add_argument("--dest", default="results/predictions")
    ap.add_argument("--save-video", default=None)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default="", help="blank = auto (CUDA if present)")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.weights)
    names = class_names()
    source = args.source
    path = Path(source)

    if source == "webcam" or path.suffix.lower() in VIDEO_EXT:
        dest = resolve(args.save_video or "results/infer.mp4")
        n = run_video(model, source, dest, names, args.imgsz, args.conf, args.device)
        print(f"wrote {n} frames to {dest}")
        return

    sources = list_images(path) if path.is_dir() else [path]
    if args.limit:
        rng = np.random.default_rng(0)
        idx = rng.permutation(len(sources))[: args.limit]
        sources = [sources[i] for i in sorted(idx)]
    n = run_images(model, sources, resolve(args.dest), names, args.imgsz, args.conf, args.device)
    print(f"wrote {n} annotated images to {resolve(args.dest)}")


if __name__ == "__main__":
    friendly(main)()
