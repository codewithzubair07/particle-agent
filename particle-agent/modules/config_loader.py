"""Configuration loader for Particle.

This module provides a robust, centralized configuration loading flow that:
- Reads configuration values from ``config.yaml``
- Loads optional environment variables from ``.env``
- Applies sane defaults when values are absent
- Exposes cached configuration via ``get_config()``
- Logs loaded settings with secrets masked
"""

from __future__ import annotations

import copy
import logging
import os
import threading
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


logger = logging.getLogger("particle.config")


class ConfigNamespace(dict):
    """Dictionary-like config object with recursive attribute access."""

    def __getattr__(self, name: str) -> Any:
        """Return a key as an attribute when available."""
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        """Allow attribute assignment to map to dictionary keys."""
        self[name] = value


_DEFAULT_CONFIG: dict[str, Any] = {
    "app": {
        "name": "Particle",
        "environment": "development",
        "timezone": "UTC",
        "polling_interval_minutes": 15,
    },
    "paths": {
        "base_dir": ".",
        "context_dir": "context",
        "data_dir": "data",
        "logs_dir": "logs",
        "task_db": "data/tasks.db",
        "memory_db": "data/memory.db",
        "chroma_dir": "data/chroma",
        "log_file": "logs/particle.log",
    },
    "llm": {
        "provider_order": ["gemini-2.0-flash-exp", "gemini-1.5-flash", "openrouter"],
        "openrouter_models": [
            "meta-llama/llama-3.1-8b-instruct:free",
            "nvidia/nemotron-3-super-120b-a12b:free",
            "mistralai/mistral-7b-instruct:free",
        ],
        "max_retries_per_provider": 3,
        "request_timeout_seconds": 45,
    },
    "telegram": {
        "enabled": True,
        "default_status": "available",
    },
    "email": {
        "imap_host": "imap.gmail.com",
        "imap_port": 993,
        "smtp_host": "smtp.gmail.com",
        "smtp_port": 587,
        "digest_hour": 9,
    },
    "calendar": {
        "enabled": True,
        "reminder_minutes": 15,
    },
    "voice": {
        "enabled": True,
        "voice_engine": "kokoro",
    },
    "browser": {
        "enabled": False,
        "pinchtab_url": "http://localhost:9867",
        "max_sources_per_research": 3,
    },
    "logging": {
        "level": "INFO",
    },
}

_ENV_BINDINGS: dict[str, tuple[list[str], Any]] = {
    "APP_ENVIRONMENT": (["app", "environment"], "development"),
    "TELEGRAM_BOT_TOKEN": (["telegram", "bot_token"], ""),
    "TELEGRAM_HOME_ID": (["telegram", "home_id"], ""),
    "EMAIL_ADDRESS": (["email", "address"], ""),
    "EMAIL_PASSWORD": (["email", "password"], ""),
    "GEMINI_API_KEY": (["llm", "gemini_api_key"], ""),
    "HF_TOKEN": (["llm", "hf_token"], ""),
    "MISTRAL_API_KEY": (["llm", "mistral_api_key"], ""),
    "OPENROUTER_API_KEY": (["llm", "openrouter_api_key"], ""),
    "GOOGLE_CALENDAR_CREDENTIALS": (["calendar", "credentials"], ""),
    "ELEVENLABS_API_KEY": (["voice", "elevenlabs_api_key"], ""),
    "ELEVENLABS_VOICE_ID": (["voice", "elevenlabs_voice_id"], ""),
}

_SENSITIVE_MARKERS = ("key", "token", "secret", "password", "credentials")

_CONFIG_CACHE: ConfigNamespace | None = None
_CONFIG_LOCK = threading.Lock()


def _deep_merge(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    """Merge ``updates`` into ``base`` recursively and return ``base``."""
    for key, value in updates.items():
        if isinstance(base.get(key), dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def _set_nested(config: dict[str, Any], path: list[str], value: Any) -> None:
    """Set a value in a nested dictionary path, creating nodes when missing."""
    node = config
    for key in path[:-1]:
        node = node.setdefault(key, {})
    node[path[-1]] = value


def _to_namespace(value: Any) -> Any:
    """Recursively convert dictionaries into :class:`ConfigNamespace` objects."""
    if isinstance(value, dict):
        return ConfigNamespace({k: _to_namespace(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_to_namespace(item) for item in value]
    return value


def _mask_value(key: str, value: Any) -> Any:
    """Mask sensitive values before writing them to logs."""
    lowered_key = key.lower()
    if any(marker in lowered_key for marker in _SENSITIVE_MARKERS):
        if value in (None, ""):
            return "<empty>"
        return "***"
    return value


def _log_config(config: ConfigNamespace) -> None:
    """Log all loaded settings using masked values for sensitive keys."""
    stack: list[tuple[str, Any]] = [("", config)]
    while stack:
        prefix, value = stack.pop()
        if isinstance(value, dict):
            for child_key, child_value in value.items():
                next_prefix = f"{prefix}.{child_key}" if prefix else child_key
                stack.append((next_prefix, child_value))
            continue
        logger.info("config.%s=%s", prefix, _mask_value(prefix, value))


def _load_yaml_config(config_path: Path) -> dict[str, Any]:
    """Load YAML configuration from disk or return an empty mapping if absent."""
    if not config_path.exists():
        logger.warning("Configuration file %s not found. Falling back to defaults.", config_path)
        return {}

    try:
        with config_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
    except yaml.YAMLError as exc:
        logger.error("Invalid YAML in %s", config_path)
        raise ValueError(f"Invalid YAML in {config_path}") from exc
    except OSError as exc:
        logger.error("Failed reading configuration file: %s", config_path)
        raise RuntimeError(f"Failed to read configuration file: {config_path}") from exc

    if not isinstance(raw, dict):
        raise ValueError(f"Configuration file must contain a top-level mapping: {config_path}")

    return raw


def _resolve_paths(config: dict[str, Any], project_root: Path) -> None:
    """Normalize configured filesystem paths to absolute values."""
    paths = config.setdefault("paths", {})
    for key, path_value in list(paths.items()):
        if not isinstance(path_value, str) or not path_value:
            continue
        candidate = Path(path_value)
        if not candidate.is_absolute():
            candidate = (project_root / candidate).resolve()
        paths[key] = str(candidate)


def _load_env_file(env_path: Path) -> None:
    """Load environment values from ``.env`` without failing on missing files."""
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=False)
        logger.info("Loaded environment variables from %s", env_path)
    else:
        logger.warning("Environment file %s not found. Continuing with process environment only.", env_path)


def _apply_env_overrides(config: dict[str, Any]) -> None:
    """Apply environment values to the merged configuration map."""
    for env_name, (config_path, default) in _ENV_BINDINGS.items():
        value = os.getenv(env_name, default)
        _set_nested(config, config_path, value)


def get_config(
    config_path: str | Path | None = None,
    env_path: str | Path | None = None,
    *,
    force_reload: bool = False,
) -> ConfigNamespace:
    """Return cached application configuration.

    Args:
        config_path: Optional explicit path to ``config.yaml``.
        env_path: Optional explicit path to ``.env``.
        force_reload: When ``True``, bypass cache and reload from disk.

    Returns:
        Loaded configuration object with both dict-style and attribute access.
    """
    global _CONFIG_CACHE

    with _CONFIG_LOCK:
        if _CONFIG_CACHE is not None and not force_reload:
            return _CONFIG_CACHE

        project_root = Path(__file__).resolve().parent.parent
        resolved_config_path = Path(config_path) if config_path else project_root / "config.yaml"
        resolved_env_path = Path(env_path) if env_path else project_root / ".env"

        _load_env_file(resolved_env_path)
        yaml_config = _load_yaml_config(resolved_config_path)

        merged = _deep_merge(copy.deepcopy(_DEFAULT_CONFIG), yaml_config)
        _apply_env_overrides(merged)
        _resolve_paths(merged, project_root)

        config = _to_namespace(merged)
        _log_config(config)

        _CONFIG_CACHE = config
        return config
