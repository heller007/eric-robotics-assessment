"""Config loading. Every script takes --config and reads configs/ through here."""
from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "configs"


def load_yaml(path: str | Path) -> dict:
    path = Path(path)
    if not path.is_absolute() and not path.exists():
        path = CONFIG_DIR / path
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def save_yaml(obj: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(obj, fh, sort_keys=False)
    return path


def resolve(path: str | Path | None) -> Path | None:
    """Relative paths in configs are relative to the repo root, not the cwd."""
    if path is None:
        return None
    p = Path(path)
    return p if p.is_absolute() else (REPO_ROOT / p)


def class_names(classes_cfg: dict | None = None) -> list[str]:
    cfg = classes_cfg or load_yaml("classes.yaml")
    names = cfg["names"]
    return [names[i] for i in sorted(names)]
