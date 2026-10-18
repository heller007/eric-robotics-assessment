"""What this machine can actually do. Run it first on a new PC.

    python -m common.doctor

Reports the interpreter, the GPU situation, the packages each task needs, and
whether the prepared dataset has been located - so a missing CUDA build or an
unfetched dataset is found here rather than forty minutes into a training run.
"""
from __future__ import annotations

import importlib
import platform
import sys

OK = "ok  "
NO = "MISSING"
WARN = "warn"


def line(status: str, label: str, detail: str = "") -> None:
    print(f"  [{status:<7}] {label}" + (f"  {detail}" if detail else ""))


def check_python() -> None:
    print("interpreter")
    line(OK, platform.python_version(), sys.executable)
    line(OK, f"{platform.system()} {platform.release()}", platform.machine())


def check_torch() -> bool:
    print("\ntorch / GPU")
    try:
        torch = importlib.import_module("torch")
    except ImportError:
        line(NO, "torch", "pip install torch (see the README for the CUDA wheel)")
        return False

    line(OK, f"torch {torch.__version__}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            line(
                OK,
                f"cuda:{i} {props.name}",
                f"{props.total_memory / 1024**3:.1f} GB, capability "
                f"{props.major}.{props.minor}",
            )
        line(OK, f"cuda runtime {torch.version.cuda}")
        return True

    cpu_only = torch.version.cuda is None
    line(
        WARN,
        "no CUDA device visible",
        "this torch build is CPU-only - training will be very slow"
        if cpu_only
        else "torch has CUDA support but no device was found (driver?)",
    )
    return False


def check_packages() -> None:
    print("\npackages")
    needed = {
        "ultralytics": "Task 2 training and inference",
        "cv2": "drawing and video IO",
        "numpy": "everything",
        "yaml": "configs",
        "PIL": "image headers",
        "tqdm": "progress bars",
    }
    optional = {
        "kagglehub": "downloading the prepared dataset",
        "nuscenes": "only needed to rebuild Task 1 from scratch",
        "scipy": "Task 3 fitting",
        "pandas": "results tables",
        "matplotlib": "plots",
        "onnxruntime": "Task 4 CPU runtime",
        "openvino": "Task 4 CPU runtime",
        "tensorrt": "Task 4 GPU runtime",
    }
    for mod, why in needed.items():
        try:
            m = importlib.import_module(mod)
            line(OK, mod, getattr(m, "__version__", ""))
        except ImportError:
            line(NO, mod, why)
    for mod, why in optional.items():
        try:
            m = importlib.import_module(mod)
            line(OK, mod, getattr(m, "__version__", ""))
        except ImportError:
            line(WARN, mod, f"optional - {why}")


def check_dataset() -> None:
    print("\ndataset")
    from common.config import CONFIG_DIR, LOCAL_CONFIG, load_paths
    from common.env import dataset_health, find_dataset_root, writable_root

    paths = load_paths()
    line(
        OK if LOCAL_CONFIG.exists() else WARN,
        "configs/local.yaml",
        str(LOCAL_CONFIG) if LOCAL_CONFIG.exists() else "not written yet",
    )
    line(
        OK if (CONFIG_DIR / "data.yaml").exists() else WARN,
        "configs/data.yaml",
        "written by data.fetch_dataset or data.merge",
    )
    try:
        root = find_dataset_root(
            paths.get("dataset_root"),
            paths.get("kaggle_dataset"),
            allow_download=False,  # a doctor never downloads 2 GB
            local_default=writable_root() / paths["out_root"],
        )
    except FileNotFoundError:
        line(
            NO,
            "prepared dataset",
            f"not on this machine - run: python -m data.fetch_dataset "
            f"(slug {paths.get('kaggle_dataset')})",
        )
        return

    line(OK, "prepared dataset", str(root))
    health = dataset_health(root)
    for split, counts in health["splits"].items():
        line(
            OK if counts["images"] else NO,
            f"{split} split",
            f"{counts['images']} images, {counts['labels']} labels, "
            f"{counts['meta']} meta sidecars",
        )
    for problem in dict.fromkeys(health["problems"]):
        line(WARN, "dataset", problem)


def check_weights() -> None:
    print("\ntrained weights")
    from common.config import load_paths, resolve

    runs_root = resolve(load_paths().get("runs_root", "runs"))
    if not runs_root.is_dir():
        line(WARN, "runs/", "nothing trained on this machine yet")
        return
    found = sorted(runs_root.glob("*/weights/best.pt"))
    if not found:
        line(WARN, "runs/", f"{runs_root} exists but holds no best.pt")
    for w in found:
        line(OK, w.parent.parent.name, f"{w}  ({w.stat().st_size / 1024**2:.1f} MB)")


def main() -> None:
    check_python()
    has_gpu = check_torch()
    check_packages()
    check_dataset()
    check_weights()
    print(
        "\nsummary: "
        + ("GPU training available." if has_gpu else "No GPU - expect CPU speeds.")
        + " Fix anything marked MISSING before starting a task."
    )


if __name__ == "__main__":
    main()
