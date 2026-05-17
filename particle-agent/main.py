"""Particle bootstrap entrypoint.

This entrypoint initializes logging, loads application configuration,
and exits successfully once the runtime environment is validated.
"""

from __future__ import annotations

import logging
import sys

from modules.config_loader import get_config


logger = logging.getLogger("particle.main")


def main() -> int:
    """Load configuration and exit with an appropriate process status code."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    try:
        config = get_config()
        logger.info("Particle bootstrap completed for environment: %s", config.app.environment)
        return 0
    except Exception:
        logger.exception("Particle bootstrap failed during configuration loading")
        return 1


if __name__ == "__main__":
    sys.exit(main())
