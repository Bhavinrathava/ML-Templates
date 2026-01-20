"""
Logging utilities for NLP Templates.

This module provides a common logger configuration for the entire NLP Templates package.
It supports console and file logging with customizable levels and formats.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any


class Logger:
    """
    A simple wrapper around Python's logging module for consistent logging across the package.

    Features:
    - Console and file logging
    - Customizable log levels and formats
    - Singleton pattern to ensure single logger instance
    - Colored console output (INFO=blue, WARNING=yellow, ERROR=red)
    """

    _loggers: Dict[str, logging.Logger] = {}

    # Color codes for terminal output
    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[34m",  # Blue
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
        "RESET": "\033[0m",  # Reset
    }

    DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

    @classmethod
    def get_logger(
        cls,
        name: str = "nlp_templates",
        log_level: str = "INFO",
        log_file: Optional[str] = None,
        console_output: bool = True,
        colored_output: bool = True,
        log_format: Optional[str] = None,
    ) -> logging.Logger:
        """
        Get or create a logger with the specified configuration.

        Args:
            name (str): Logger name (typically module name). Defaults to "nlp_templates"
            log_level (str): Logging level ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
            log_file (str, optional): Path to log file. If None, only console logging
            console_output (bool): Whether to log to console. Defaults to True
            colored_output (bool): Whether to use colored console output. Defaults to True
            log_format (str, optional): Custom log format. Uses default if None

        Returns:
            logging.Logger: Configured logger instance

        Examples:
            >>> logger = Logger.get_logger("my_module")
            >>> logger.info("This is an info message")
            >>>
            >>> logger = Logger.get_logger(
            ...     name="data_processing",
            ...     log_level="DEBUG",
            ...     log_file="logs/data_processing.log",
            ...     colored_output=True
            ... )
        """
        # Return existing logger if already configured
        if name in cls._loggers:
            return cls._loggers[name]

        # Create logger
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, log_level.upper()))

        # Clear any existing handlers to avoid duplicates
        logger.handlers.clear()

        # Use provided format or default
        fmt = log_format or cls.DEFAULT_FORMAT

        # Add console handler if requested
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(getattr(logging, log_level.upper()))

            if colored_output and sys.stdout.isatty():
                formatter = cls._get_colored_formatter(fmt)
            else:
                formatter = logging.Formatter(
                    fmt, datefmt=cls.DEFAULT_DATE_FORMAT
                )

            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

        # Add file handler if log_file specified
        if log_file:
            file_path = Path(log_file)
            file_path.parent.mkdir(parents=True, exist_ok=True)

            file_handler = logging.FileHandler(file_path)
            file_handler.setLevel(getattr(logging, log_level.upper()))
            formatter = logging.Formatter(fmt, datefmt=cls.DEFAULT_DATE_FORMAT)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        # Store logger for future use
        cls._loggers[name] = logger

        return logger

    @classmethod
    def _get_colored_formatter(cls, fmt: str) -> logging.Formatter:
        """
        Create a formatter with colored log levels.

        Args:
            fmt (str): Log format string

        Returns:
            logging.Formatter: Formatter with color support
        """

        class ColoredFormatter(logging.Formatter):
            def format(self, record):
                # Color the level name
                level_color = cls.COLORS.get(
                    record.levelname, cls.COLORS["RESET"]
                )
                record.levelname = (
                    f"{level_color}{record.levelname}{cls.COLORS['RESET']}"
                )
                return super().format(record)

        return ColoredFormatter(fmt, datefmt=cls.DEFAULT_DATE_FORMAT)

    @classmethod
    def setup_root_logger(
        cls,
        log_level: str = "INFO",
        log_file: Optional[str] = None,
        console_output: bool = True,
    ) -> logging.Logger:
        """
        Setup the root logger for the entire package.

        Call this once at application startup to configure logging for all modules.

        Args:
            log_level (str): Logging level for the root logger
            log_file (str, optional): Path to root log file
            console_output (bool): Whether to log to console

        Returns:
            logging.Logger: Root logger instance

        Example:
            >>> Logger.setup_root_logger(log_level="DEBUG", log_file="logs/app.log")
            >>> logger = Logger.get_logger("my_module")  # Will inherit root config
        """
        return cls.get_logger(
            name="nlp_templates",
            log_level=log_level,
            log_file=log_file,
            console_output=console_output,
            colored_output=True,
        )

    @classmethod
    def disable_logger(cls, name: str = "nlp_templates") -> None:
        """
        Disable a logger by setting its level to CRITICAL.

        Args:
            name (str): Name of the logger to disable
        """
        if name in cls._loggers:
            cls._loggers[name].setLevel(logging.CRITICAL)

    @classmethod
    def clear_loggers(cls) -> None:
        """Clear all cached loggers."""
        for logger in cls._loggers.values():
            logger.handlers.clear()
        cls._loggers.clear()


def get_logger(
    name: str = "nlp_templates",
    log_level: str = "INFO",
    log_file: Optional[str] = None,
) -> logging.Logger:
    """
    Convenience function to get a logger.

    This is a shorthand for Logger.get_logger() with common defaults.

    Args:
        name (str): Logger name (typically __name__ from the calling module)
        log_level (str): Logging level ("DEBUG", "INFO", "WARNING", "ERROR")
        log_file (str, optional): Path to log file

    Returns:
        logging.Logger: Configured logger instance

    Example:
        >>> # In your module
        >>> from nlp_templates.utils.logging_utils import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.info("Module initialized")
    """
    return Logger.get_logger(
        name=name,
        log_level=log_level,
        log_file=log_file,
        console_output=True,
        colored_output=True,
    )


# Module-level logger for this package
_module_logger = get_logger(__name__)


__all__ = ["Logger", "get_logger"]
