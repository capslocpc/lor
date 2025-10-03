"""
Logging configuration using Loguru.

Logs are output to both console and a rotating file.
"""

from loguru import logger

from pathlib import Path


# Ensure logs directory exists
log_dir = Path(__file__).resolve().parents[2] / "logs"
log_dir.mkdir(parents=True, exist_ok=True)

# Clear existing handlers (avoid duplicate logs)
logger.remove()

# Console logging (off by default, can be enabled via settings)
logger.remove()  # remove default stderr sink

# File logging (rotates automatically)
logger.add(
    log_dir / "app.log",
    rotation="1 MB",
    retention="10 days",
    compression="zip",
    level="DEBUG"
)


def get_logger():
    """
    Get the configured logger instance.

    Returns:
        logger: Configured Loguru logger instance.
    """
    return logger
