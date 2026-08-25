"""Configuration loading with explicit release gates."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class ConfigurationError(ValueError):
    """Raised when a reproducibility-critical setting is absent."""


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ConfigurationError(f"Expected a mapping in {path}")
    return config


def get_required(config: dict[str, Any], dotted_key: str) -> Any:
    value: Any = config
    for key in dotted_key.split("."):
        if not isinstance(value, dict) or key not in value:
            raise ConfigurationError(f"Missing required configuration: {dotted_key}")
        value = value[key]
    if value is None or value == "" or value == []:
        raise ConfigurationError(f"Unresolved configuration: {dotted_key}")
    return value


def unresolved_fields(config: dict[str, Any], prefix: str = "") -> list[str]:
    unresolved: list[str] = []
    for key, value in config.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            unresolved.extend(unresolved_fields(value, name))
        elif value is None or value == []:
            unresolved.append(name)
    return unresolved
