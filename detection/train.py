"""Fine-tune one run from configs/runs.yaml.

  python -m detection.train --run B1
  python -m detection.train --run B2 --set epochs=60 --set batch=4

Why the knobs in configs/runs.yaml are what they are:
  freeze=10     nuScenes-mini is ~2,400 images and overfits the backbone fast;
                B1u unfreezes at a lower LR as a second stage.
  batch=-1      let ultralytics pick the largest batch the GPU fits.
  flipud=0      road scenes have a fixed up direction.
  close_mosaic  the last 10 epochs train on un-mosaicked images, which is what
                inference actually sees.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from common.config import load_yaml, resolve

# keys that configure this script rather than ultralytics
META_KEYS = {"purpose", "models", "finetune", "zero_shot", "init_from", "load_from"}


def resolve_weights(candidates: list[str] | str) -> str:
    """First checkpoint that actually loads.

    YOLO26 weights only exist for recent ultralytics releases, so a run asks for
    them first and names a fallback. Whichever one loads is what gets reported -
    never assume the preferred entry was used.
    """
    from ultralytics import YOLO

    if isinstance(candidates, str):
        candidates = [candidates]
    errors = []
    for name in candidates:
        try:
            YOLO(name)
            if name != candidates[0]:
                print(f"note: {candidates[0]} unavailable, using {name}")
            return name
        except Exception as exc:  # network, unknown model name, bad checkpoint
            errors.append(f"  {name}: {type(exc).__name__}: {exc}")
    raise RuntimeError("none of the candidate weights could be loaded:\n" + "\n".join(errors))


def coerce(value: str):
    low = value.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("none", "null"):
        return None
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            pass
    return value


def build_args(runs: dict, run: str, overrides: list[str]) -> tuple[dict, dict]:
    if run not in runs["runs"]:
        raise SystemExit(f"unknown run {run!r}; known: {', '.join(runs['runs'])}")
    spec = {**runs["defaults"], **runs["runs"][run]}
    for item in overrides or []:
        key, _, value = item.partition("=")
        spec[key.strip()] = coerce(value.strip())
    train_args = {k: v for k, v in spec.items() if k not in META_KEYS}
    return spec, train_args


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, help="B1, B1b, B2, B1u, B1bal ...")
    ap.add_argument("--set", action="append", default=[], metavar="key=value")
    ap.add_argument("--device", default=None)
    ap.add_argument("--project", default=None)
    ap.add_argument("--results", default="results")
    args = ap.parse_args()

    runs = load_yaml("runs.yaml")
    spec, train_args = build_args(runs, args.run, args.set)

    if spec.get("finetune") is False:
        raise SystemExit(
            f"{args.run} is a no-fine-tune baseline. Score it directly:\n"
            f"  python -m eval.det_eval --run {args.run}"
        )

    data_path = resolve(train_args["data"])
    if not Path(data_path).exists():
        raise SystemExit(
            f"{data_path} is missing - run Task 1 first:\n"
            "  python -m data.merge --out <dataset root>"
        )
    train_args["data"] = str(data_path)
    train_args["name"] = args.run
    if args.device is not None:
        train_args["device"] = args.device
    if args.project is not None:
        train_args["project"] = args.project
    train_args["project"] = str(resolve(train_args["project"]))

    start = spec.get("init_from")
    if start:
        start = resolve(start)
        if not Path(start).exists():
            raise SystemExit(f"{args.run} starts from {start}, which does not exist yet")
        weights = str(start)
        print(f"continuing from {weights}")
    else:
        weights = resolve_weights(spec["models"])

    print(f"run {args.run}: {spec.get('purpose', '')}")
    print(f"weights {weights}")
    print("args " + " ".join(f"{k}={v}" for k, v in sorted(train_args.items())))

    from ultralytics import YOLO

    model = YOLO(weights)
    if spec.get("load_from"):
        # Building from a .yaml gives random weights. .load() transfers every
        # tensor whose name and shape still match, which is how a P2 run reuses
        # the pretrained backbone it does share with the stock architecture; the
        # new stride-4 head necessarily starts cold.
        print(f"transferring matching weights from {spec['load_from']}")
        model = model.load(spec["load_from"])
    results = model.train(**train_args)

    save_dir = Path(getattr(results, "save_dir", train_args["project"]))
    best = save_dir / "weights" / "best.pt"
    print(f"\nbest weights: {best}")

    record = {
        "run": args.run,
        "purpose": spec.get("purpose"),
        "weights_used": weights,
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in train_args.items()},
        "save_dir": str(save_dir),
        "best": str(best),
    }
    dest = resolve(args.results)
    dest.mkdir(parents=True, exist_ok=True)
    (dest / f"train_{args.run}.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"wrote {dest / f'train_{args.run}.json'}")
    print(f"\nnext: python -m eval.det_eval --weights {best} --name {args.run}")


if __name__ == "__main__":
    main()
