"""Config loading.

Two layers, because the repo is shared between machines over git:

  configs/paths.yaml   committed defaults, identical on every machine
  configs/local.yaml   this machine only, git-ignored, overrides the above

So a dataset that lives in a kagglehub cache on one PC and on a drive on another
is recorded per machine instead of being committed and then conflicting.
"""
from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "configs"
LOCAL_CONFIG = CONFIG_DIR / "local.yaml"


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


def load_paths(local: str | Path | None = None) -> dict:
    """configs/paths.yaml with configs/local.yaml layered on top.

    A null in local.yaml is treated as "not set" rather than as an override, so a
    half-filled local file cannot blank out a working default.
    """
    cfg = load_yaml("paths.yaml")
    local_path = Path(local) if local is not None else LOCAL_CONFIG
    if local_path.exists():
        for key, value in (load_yaml(local_path) or {}).items():
            if value is not None:
                cfg[key] = value
    return cfg


def set_local(key: str, value) -> Path:
    """Remember a per-machine setting in configs/local.yaml."""
    current = load_yaml(LOCAL_CONFIG) if LOCAL_CONFIG.exists() else {}
    current[key] = str(value) if isinstance(value, Path) else value
    return save_yaml(current, LOCAL_CONFIG)


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
