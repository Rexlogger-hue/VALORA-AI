"""
services/shared/logger.py

Centralized logging module for Valora AI.

Provides a single `get_logger(name)` factory that produces loggers with:
    - Colored console output (TTY-aware, safe on Windows and Linux).
    - Daily rotating general log files.
    - Dedicated rotating error-level log files.
    - Optional dedicated debug-level log files.
    - Optional structured JSON log formatting.

All loggers created through this module share a common log directory
and rotation policy, and are safe to reuse as module-level singletons
across the codebase (e.g. `services/tts/*`, `services/api/*`).
"""

from __future__ import annotations

import json
import logging
import sys
import threading
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Optional

_LOG_DIR = Path("logs")
_MAX_BYTES = 10 * 1024 * 1024  # 10 MB per rotated file
_BACKUP_COUNT = 5
_DAILY_BACKUP_COUNT = 14

_DEFAULT_CONSOLE_LEVEL = logging.INFO
_DEFAULT_FILE_LEVEL = logging.DEBUG

_configured_loggers: Dict[str, logging.Logger] = {}
_lock = threading.Lock()

_JSON_LOGGING_ENABLED = False


class _ColorCodes:
    """ANSI color codes used for colored console log output."""

    RESET = "\033[0m"
    GREY = "\033[38;5;245m"
    BLUE = "\033[34m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    BOLD_RED = "\033[1;31m"


class ColoredConsoleFormatter(logging.Formatter):
    """
    Console log formatter that colorizes output by log level.

    Falls back to plain (uncolored) formatting automatically when the
    output stream is not a TTY (e.g. when redirected to a file or
    piped), keeping log files and non-interactive output clean.
    """

    _LEVEL_COLORS = {
        logging.DEBUG: _ColorCodes.GREY,
        logging.INFO: _ColorCodes.BLUE,
        logging.WARNING: _ColorCodes.YELLOW,
        logging.ERROR: _ColorCodes.RED,
        logging.CRITICAL: _ColorCodes.BOLD_RED,
    }

    _BASE_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

    def __init__(self, use_color: bool = True) -> None:
        """
        Initialize the colored console formatter.

        Args:
            use_color: Whether ANSI color codes should be applied.
        """
        super().__init__(fmt=self._BASE_FORMAT, datefmt="%Y-%m-%d %H:%M:%S")
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        """
        Format a log record, optionally wrapping it in ANSI color codes
        based on its severity level.

        Args:
            record: The log record to format.

        Returns:
            str: The formatted (optionally colorized) log line.
        """
        message = super().format(record)
        if not self.use_color:
            return message

        color = self._LEVEL_COLORS.get(record.levelno, "")
        if not color:
            return message

        return f"{color}{message}{_ColorCodes.RESET}"


class JSONFormatter(logging.Formatter):
    """
    Structured JSON log formatter, suitable for ingestion by log
    aggregation systems (e.g. ELK, CloudWatch, Datadog).
    """

    def format(self, record: logging.LogRecord) -> str:
        """
        Format a log record as a single-line JSON object.

        Args:
            record: The log record to format.

        Returns:
            str: A JSON-encoded string representing the log record.
        """
        payload: Dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "thread": record.threadName,
        }

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=False)


def _is_tty_stream(stream: Any) -> bool:
    """
    Determine whether a stream supports ANSI color output.

    Args:
        stream: The output stream to check (e.g. `sys.stdout`).

    Returns:
        bool: True if the stream is a TTY and supports color.
    """
    try:
        return bool(hasattr(stream, "isatty") and stream.isatty())
    except Exception:
        return False


def _ensure_log_directory(directory: Path) -> Path:
    """
    Ensure the given log directory exists, creating it if necessary.

    Args:
        directory: The directory path to ensure exists.

    Returns:
        Path: The resolved, guaranteed-to-exist directory path.

    Raises:
        RuntimeError: If the directory cannot be created.
    """
    resolved = directory.resolve()
    try:
        resolved.mkdir(parents=True, exist_ok=True)
        return resolved
    except OSError as exc:
        raise RuntimeError(f"Failed to create log directory '{resolved}': {exc}") from exc


def configure_logging(
    log_dir: Optional[Path] = None,
    console_level: int = _DEFAULT_CONSOLE_LEVEL,
    file_level: int = _DEFAULT_FILE_LEVEL,
    json_logging: bool = False,
    enable_debug_log: bool = True,
) -> None:
    """
    Configure global logging defaults used by `get_logger`.

    This should typically be called once at application startup. If not
    called explicitly, `get_logger` will lazily configure itself using
    module-level defaults.

    Args:
        log_dir: Directory in which log files are written. Defaults to
            `./logs` relative to the current working directory.
        console_level: Minimum severity level emitted to the console.
        file_level: Minimum severity level emitted to the general
            rotating/daily log files.
        json_logging: Whether file logs should be formatted as
            structured JSON instead of plain text.
        enable_debug_log: Whether a dedicated debug-level log file
            should be created in addition to the general log.
    """
    global _LOG_DIR, _DEFAULT_CONSOLE_LEVEL, _DEFAULT_FILE_LEVEL, _JSON_LOGGING_ENABLED

    with _lock:
        _LOG_DIR = _ensure_log_directory(log_dir or _LOG_DIR)
        _DEFAULT_CONSOLE_LEVEL = console_level
        _DEFAULT_FILE_LEVEL = file_level
        _JSON_LOGGING_ENABLED = json_logging
        _configured_loggers.clear()

    logging.getLogger("valora.shared.logger").info(
        "Logging configured (log_dir=%s, console_level=%s, file_level=%s, json_logging=%s).",
        _LOG_DIR,
        logging.getLevelName(console_level),
        logging.getLevelName(file_level),
        json_logging,
    )

    if enable_debug_log:
        pass  # Debug log files are created per-logger in get_logger().


def _build_file_formatter() -> logging.Formatter:
    """
    Build the formatter used for file-based log handlers, honoring the
    globally configured JSON logging preference.

    Returns:
        logging.Formatter: A JSON or plain-text formatter instance.
    """
    if _JSON_LOGGING_ENABLED:
        return JSONFormatter()
    return logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _add_console_handler(logger: logging.Logger) -> None:
    """
    Attach a colored console (stdout) handler to the given logger.

    Args:
        logger: The logger to attach the handler to.
    """
    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setLevel(_DEFAULT_CONSOLE_LEVEL)
    console_handler.setFormatter(
        ColoredConsoleFormatter(use_color=_is_tty_stream(sys.stdout))
    )
    logger.addHandler(console_handler)


def _add_daily_rotating_handler(logger: logging.Logger, safe_name: str) -> None:
    """
    Attach a daily rotating file handler capturing all configured file
    levels for the given logger.

    Args:
        logger: The logger to attach the handler to.
        safe_name: Filesystem-safe logger name used in the filename.
    """
    daily_path = _LOG_DIR / f"{safe_name}.daily.log"
    daily_handler = TimedRotatingFileHandler(
        filename=str(daily_path),
        when="midnight",
        interval=1,
        backupCount=_DAILY_BACKUP_COUNT,
        encoding="utf-8",
        utc=True,
    )
    daily_handler.setLevel(_DEFAULT_FILE_LEVEL)
    daily_handler.setFormatter(_build_file_formatter())
    daily_handler.suffix = "%Y-%m-%d"
    logger.addHandler(daily_handler)


def _add_size_rotating_handler(logger: logging.Logger, safe_name: str) -> None:
    """
    Attach a size-based rotating file handler capturing all configured
    file levels for the given logger.

    Args:
        logger: The logger to attach the handler to.
        safe_name: Filesystem-safe logger name used in the filename.
    """
    rotating_path = _LOG_DIR / f"{safe_name}.log"
    rotating_handler = RotatingFileHandler(
        filename=str(rotating_path),
        maxBytes=_MAX_BYTES,
        backupCount=_BACKUP_COUNT,
        encoding="utf-8",
    )
    rotating_handler.setLevel(_DEFAULT_FILE_LEVEL)
    rotating_handler.setFormatter(_build_file_formatter())
    logger.addHandler(rotating_handler)


def _add_error_handler(logger: logging.Logger, safe_name: str) -> None:
    """
    Attach a rotating file handler capturing only ERROR and above for
    the given logger.

    Args:
        logger: The logger to attach the handler to.
        safe_name: Filesystem-safe logger name used in the filename.
    """
    error_path = _LOG_DIR / f"{safe_name}.error.log"
    error_handler = RotatingFileHandler(
        filename=str(error_path),
        maxBytes=_MAX_BYTES,
        backupCount=_BACKUP_COUNT,
        encoding="utf-8",
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(_build_file_formatter())
    logger.addHandler(error_handler)


def _add_debug_handler(logger: logging.Logger, safe_name: str) -> None:
    """
    Attach a rotating file handler capturing only DEBUG-level records
    for the given logger.

    Args:
        logger: The logger to attach the handler to.
        safe_name: Filesystem-safe logger name used in the filename.
    """
    debug_path = _LOG_DIR / f"{safe_name}.debug.log"
    debug_handler = RotatingFileHandler(
        filename=str(debug_path),
        maxBytes=_MAX_BYTES,
        backupCount=_BACKUP_COUNT,
        encoding="utf-8",
    )
    debug_handler.setLevel(logging.DEBUG)
    debug_handler.addFilter(lambda record: record.levelno == logging.DEBUG)
    debug_handler.setFormatter(_build_file_formatter())
    logger.addHandler(debug_handler)


def get_logger(
    name: str,
    level: int = logging.DEBUG,
    enable_console: bool = True,
    enable_daily_log: bool = True,
    enable_rotating_log: bool = True,
    enable_error_log: bool = True,
    enable_debug_log: bool = True,
) -> logging.Logger:
    """
    Retrieve (or lazily construct) a fully configured logger.

    Loggers are cached by name, so repeated calls with the same `name`
    return the same configured instance rather than duplicating
    handlers. Safe to call from any module without prior explicit
    initialization; `configure_logging` may optionally be called once
    at startup to customize global defaults (log directory, levels,
    JSON formatting) before any loggers are created.

    Args:
        name: Logger name, typically the calling module's `__name__`
            or a dotted service identifier (e.g. 'valora.tts.model').
        level: Overall minimum severity level for the logger itself.
        enable_console: Whether to attach a colored console handler.
        enable_daily_log: Whether to attach a daily rotating file handler.
        enable_rotating_log: Whether to attach a size-based rotating
            file handler.
        enable_error_log: Whether to attach a dedicated error-level
            rotating file handler.
        enable_debug_log: Whether to attach a dedicated debug-level
            rotating file handler.

    Returns:
        logging.Logger: A fully configured logger instance.

    Raises:
        RuntimeError: If the log directory cannot be created or a
            handler fails to initialize.
    """
    with _lock:
        if name in _configured_loggers:
            return _configured_loggers[name]

        try:
            _ensure_log_directory(_LOG_DIR)

            logger = logging.getLogger(name)
            logger.setLevel(level)
            logger.propagate = False

            if logger.handlers:
                logger.handlers.clear()

            safe_name = name.replace(".", "_").replace("/", "_").replace("\\", "_")

            if enable_console:
                _add_console_handler(logger)
            if enable_daily_log:
                _add_daily_rotating_handler(logger, safe_name)
            if enable_rotating_log:
                _add_size_rotating_handler(logger, safe_name)
            if enable_error_log:
                _add_error_handler(logger, safe_name)
            if enable_debug_log:
                _add_debug_handler(logger, safe_name)

            _configured_loggers[name] = logger
            return logger

        except Exception as exc:
            raise RuntimeError(f"Failed to configure logger '{name}': {exc}") from exc


__all__ = [
    "get_logger",
    "configure_logging",
    "ColoredConsoleFormatter",
    "JSONFormatter",
]