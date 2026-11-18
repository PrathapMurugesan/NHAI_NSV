"""
Logger module for the DLM Extraction Pipeline.

Provides centralized, thread-safe logging configured with both console and file handlers.
Replaces Databricks print/display statements with standardized logging.
"""

import logging
import os
import sys
from typing import Optional, Union


DEFAULT_LOG_FORMAT = "%(asctime)s [%(levelname)s] [%(name)s:%(lineno)d] %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logger(
    name: str = "dlm_pipeline",
    log_level: Union[int, str] = logging.INFO,
    log_file: Optional[str] = None,
    log_format: str = DEFAULT_LOG_FORMAT,
    date_format: str = DEFAULT_DATE_FORMAT,
    level: Optional[Union[int, str]] = None,
) -> logging.Logger:
    """
    Configure and return a standard logger instance.

    Args:
        name: Name of the logger (typically module __name__ or 'dlm_pipeline').
        log_level: Logging level (e.g., logging.INFO, logging.DEBUG).
        log_file: Optional path to a file where logs should be written.
        log_format: String format for log messages.
        date_format: Date format for timestamps in log messages.
        level: Alias for log_level.

    Returns:
        logging.Logger: Configured logger instance.
    """
    if level is not None:
        log_level = level

    if isinstance(log_level, str):
        log_level = getattr(logging, log_level.upper(), logging.INFO)
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Avoid adding multiple duplicate handlers if logger was already initialized
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(fmt=log_format, datefmt=date_format)

    # Console Handler (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler (if log_file path is specified)
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger


# Default application-wide logger
logger = setup_logger("dlm_pipeline")


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return configured logger instance."""
    if name:
        return logging.getLogger(name)
    return logger
