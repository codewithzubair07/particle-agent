"""Particle — main entry point.

Initialises the logging stack, validates configuration, prints a startup
banner, indexes context files, and hands off to the orchestrator.

Usage:
    python main.py          # run full agent
    python main.py --validate  # config-check only (exits 0 on success)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging setup (must happen before any module imports)
# ---------------------------------------------------------------------------

def _setup_logging(log_file: str, level: str = "INFO") -> None:
    """Configure root logger to write to both stdout and a rotating file."""
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    numeric_level = getattr(logging, level.upper(), logging.INFO)
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    try:
        from logging.handlers import RotatingFileHandler

        handlers.append(
            RotatingFileHandler(log_path, maxBytes=10 * 1024 * 1024, backupCount=5)
        )
    except OSError as exc:
        print(f"[WARNING] Cannot open log file {log_path}: {exc}", file=sys.stderr)

    logging.basicConfig(level=numeric_level, format=fmt, handlers=handlers)


# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------

_BANNER = r"""
 ██████╗  █████╗ ██████╗ ████████╗██╗ ██████╗██╗     ███████╗
 ██╔══██╗██╔══██╗██╔══██╗╚══██╔══╝██║██╔════╝██║     ██╔════╝
 ██████╔╝███████║██████╔╝   ██║   ██║██║     ██║     █████╗  
 ██╔═══╝ ██╔══██║██╔══██╗   ██║   ██║██║     ██║     ██╔══╝  
 ██║     ██║  ██║██║  ██║   ██║   ██║╚██████╗███████╗███████╗
 ╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝   ╚═╝ ╚═════╝╚══════╝╚══════╝
         Personal AI Chief of Staff — 24/7 Autonomous Agent
"""


logger = logging.getLogger("particle.main")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Particle — autonomous personal AI assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate configuration and exit without starting the agent.",
    )
    parser.add_argument(
        "--log-level",
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Override log level from config.",
    )
    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    args = _build_parser().parse_args()

    # Load config first to get paths/log level
    from modules.config_loader import get_config

    try:
        cfg = get_config()
    except Exception as exc:
        # Can't use logger yet — use print
        print(f"[FATAL] Configuration load failed: {exc}", file=sys.stderr)
        return 1

    log_level = args.log_level or getattr(cfg.logging, "level", "INFO")
    _setup_logging(cfg.paths.log_file, log_level)

    print(_BANNER)
    logger.info("Particle starting up (env=%s, log_level=%s)", cfg.app.environment, log_level)

    if args.validate:
        logger.info("Configuration validation passed — exiting")
        return 0

    # Ensure required directories exist
    for dir_key in ("data_dir", "logs_dir", "context_dir"):
        Path(getattr(cfg.paths, dir_key, ".")).mkdir(parents=True, exist_ok=True)

    # Start the orchestrator
    from orchestrator import Orchestrator

    orchestrator = Orchestrator()
    try:
        orchestrator.run()
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received — shutting down")
    except Exception as exc:
        logger.exception("Orchestrator terminated with error: %s", exc)
        return 1

    logger.info("Particle shut down cleanly")
    return 0


if __name__ == "__main__":
    sys.exit(main())
