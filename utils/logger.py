import logging
import os
import sys
from datetime import datetime


def get_logger(name: str, log_level: str = "INFO", log_file: str = None) -> logging.Logger:
    """Create and configure a logger."""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    if not logger.handlers:
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        if log_file is not None:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger


def get_run_logger(run_name: str = None) -> logging.Logger:
    """Get a logger with a timestamped run name."""
    if run_name is None:
        run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    return get_logger(run_name)


if __name__ == "__main__":
    log = get_logger("swarm_shield")
    log.info("Logger initialized successfully.")
