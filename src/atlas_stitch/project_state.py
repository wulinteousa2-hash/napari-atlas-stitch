from __future__ import annotations

import json
from pathlib import Path

from .models import AtlasProject


def save_atlas_project(project: AtlasProject, path: str) -> None:
    """
    Serialize an AtlasProject to disk in JSON format.
    """
    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = project.to_dict()
    try:
        destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"Failed to save atlas project to {destination}: {exc}") from exc


def load_atlas_project(path: str) -> AtlasProject:
    """
    Load an AtlasProject previously saved with :func:`save_atlas_project`.
    """
    source = Path(path).expanduser()
    if not source.exists():
        raise FileNotFoundError(f"Atlas project not found: {source}")
    text = source.read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid atlas project JSON at {source}: {exc}") from exc
    try:
        return AtlasProject.from_dict(payload)
    except Exception as exc:
        raise ValueError(f"Atlas project at {source} is malformed: {exc}") from exc
