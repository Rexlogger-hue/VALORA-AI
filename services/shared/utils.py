"""
services/shared/utils.py

General-purpose shared utility functions for Valora AI, used across
the `services.tts` and `services.api` packages.

Provides identifier generation, hashing, timing/retry decorators,
filesystem helpers, lightweight audio helpers, GPU diagnostics, JSON
helpers, and environment variable helpers.
"""

from __future__ import annotations

import functools
import hashlib
import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union

import torch

logger = logging.getLogger("valora.shared.utils")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

F = TypeVar("F", bound=Callable[..., Any])
JSONType = Union[Dict[str, Any], List[Any], str, int, float, bool, None]


# ==================================================================== #
# Identifier generation
# ==================================================================== #


def generate_uuid() -> str:
    """
    Generate a random UUID4 string.

    Returns:
        str: A UUID4 string, e.g. 'a1b2c3d4-e5f6-7890-abcd-ef1234567890'.
    """
    return str(uuid.uuid4())


def generate_uuid_hex(length: Optional[int] = None) -> str:
    """
    Generate a compact hexadecimal UUID4 identifier.

    Args:
        length: Optional number of leading hex characters to keep.
            If None, the full 32-character hex string is returned.

    Returns:
        str: A hexadecimal UUID4 string, optionally truncated.

    Raises:
        ValueError: If `length` is provided but not in range [1, 32].
    """
    hex_value = uuid.uuid4().hex
    if length is None:
        return hex_value
    if not (1 <= length <= 32):
        raise ValueError(f"length must be between 1 and 32, got {length}.")
    return hex_value[:length]


def generate_request_id(prefix: str = "req") -> str:
    """
    Generate a timestamp-prefixed unique request identifier.

    Args:
        prefix: A short prefix identifying the ID's purpose.

    Returns:
        str: An identifier in the form '{prefix}_{timestamp_ms}_{uuid8}'.
    """
    timestamp_ms = int(time.time() * 1000)
    return f"{prefix}_{timestamp_ms}_{generate_uuid_hex(8)}"


# ==================================================================== #
# Hashing
# ==================================================================== #


def sha256_hash_bytes(data: bytes) -> str:
    """
    Compute the SHA-256 hex digest of raw bytes.

    Args:
        data: Bytes to hash.

    Returns:
        str: Hexadecimal SHA-256 digest.
    """
    return hashlib.sha256(data).hexdigest()


def sha256_hash_text(text: str, encoding: str = "utf-8") -> str:
    """
    Compute the SHA-256 hex digest of a text string.

    Args:
        text: Input string to hash.
        encoding: Text encoding used prior to hashing.

    Returns:
        str: Hexadecimal SHA-256 digest.
    """
    return hashlib.sha256(text.encode(encoding)).hexdigest()


def sha256_hash_file(path: Union[str, Path], chunk_size: int = 65536) -> str:
    """
    Compute the SHA-256 hex digest of a file's contents, streaming the
    file in fixed-size chunks to bound memory usage.

    Args:
        path: Path to the file to hash.
        chunk_size: Number of bytes read per iteration.

    Returns:
        str: Hexadecimal SHA-256 digest of the file contents.

    Raises:
        FileNotFoundError: If the file does not exist.
        OSError: If the file cannot be read.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")

    hasher = hashlib.sha256()
    with file_path.open("rb") as file_obj:
        while chunk := file_obj.read(chunk_size):
            hasher.update(chunk)

    return hasher.hexdigest()


# ==================================================================== #
# Timing and retry decorators
# ==================================================================== #


def timer(func: F) -> F:
    """
    Decorator that logs the execution time of the wrapped function at
    INFO level and returns the original result unmodified.

    Args:
        func: The function to wrap.

    Returns:
        Callable: The wrapped function.
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            elapsed = time.perf_counter() - start
            logger.info("%s executed in %.4fs.", func.__qualname__, elapsed)

    return wrapper  # type: ignore[return-value]


def timed_result(func: F) -> Callable[..., Tuple[Any, float]]:
    """
    Decorator that returns a tuple of (result, elapsed_seconds) instead
    of just the function's result, for callers that need the timing
    value programmatically.

    Args:
        func: The function to wrap.

    Returns:
        Callable: A wrapped function returning `(result, elapsed_seconds)`.
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Tuple[Any, float]:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        return result, elapsed

    return wrapper


def retry(
    max_attempts: int = 3,
    delay_seconds: float = 0.5,
    backoff_factor: float = 2.0,
    exceptions: Tuple[type, ...] = (Exception,),
) -> Callable[[F], F]:
    """
    Decorator factory that retries a function call on failure using
    exponential backoff.

    Args:
        max_attempts: Maximum number of attempts, including the first.
        delay_seconds: Initial delay between attempts, in seconds.
        backoff_factor: Multiplier applied to the delay after each
            failed attempt.
        exceptions: Tuple of exception types that trigger a retry.
            Other exceptions propagate immediately without retrying.

    Returns:
        Callable: A decorator applying retry logic to the target function.

    Raises:
        ValueError: If `max_attempts` is less than 1.
    """
    if max_attempts < 1:
        raise ValueError(f"max_attempts must be >= 1, got {max_attempts}.")

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            attempt = 1
            current_delay = delay_seconds
            last_exception: Optional[BaseException] = None

            while attempt <= max_attempts:
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    last_exception = exc
                    if attempt == max_attempts:
                        logger.error(
                            "%s failed after %d attempt(s): %s",
                            func.__qualname__,
                            attempt,
                            exc,
                        )
                        raise
                    logger.warning(
                        "%s failed on attempt %d/%d: %s. Retrying in %.2fs...",
                        func.__qualname__,
                        attempt,
                        max_attempts,
                        exc,
                        current_delay,
                    )
                    time.sleep(current_delay)
                    current_delay *= backoff_factor
                    attempt += 1

            if last_exception is not None:
                raise last_exception
            raise RuntimeError("retry decorator reached an unreachable state.")

        return wrapper  # type: ignore[return-value]

    return decorator


# ==================================================================== #
# Directory and file utilities
# ==================================================================== #


def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it (and any parents) if needed.

    Args:
        path: Directory path to ensure exists.

    Returns:
        Path: The resolved, guaranteed-to-exist directory path.

    Raises:
        RuntimeError: If the directory cannot be created.
    """
    resolved = Path(path).resolve()
    try:
        resolved.mkdir(parents=True, exist_ok=True)
        return resolved
    except OSError as exc:
        raise RuntimeError(f"Failed to create directory '{resolved}': {exc}") from exc


def safe_delete(path: Union[str, Path]) -> bool:
    """
    Delete a file if it exists, logging and swallowing any errors
    rather than raising.

    Args:
        path: Path to the file to delete.

    Returns:
        bool: True if the file was deleted, False otherwise.
    """
    file_path = Path(path)
    try:
        if file_path.exists() and file_path.is_file():
            file_path.unlink()
            return True
        return False
    except OSError as exc:
        logger.warning("Failed to delete file '%s': %s", file_path, exc)
        return False


def get_file_size_bytes(path: Union[str, Path]) -> int:
    """
    Get the size of a file in bytes.

    Args:
        path: Path to the file.

    Returns:
        int: File size in bytes.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")
    return file_path.stat().st_size


def list_files(directory: Union[str, Path], pattern: str = "*") -> List[Path]:
    """
    List files in a directory matching a glob pattern, non-recursively.

    Args:
        directory: Directory to search.
        pattern: Glob pattern to match against (default: all files).

    Returns:
        List[Path]: Sorted list of matching file paths.

    Raises:
        FileNotFoundError: If the directory does not exist.
    """
    dir_path = Path(directory)
    if not dir_path.is_dir():
        raise FileNotFoundError(f"Directory not found: {dir_path}")
    return sorted(p for p in dir_path.glob(pattern) if p.is_file())


def generate_unique_filename(prefix: str = "valora", extension: str = "bin") -> str:
    """
    Generate a unique, collision-resistant filename.

    Args:
        prefix: Prefix identifying the file's purpose.
        extension: File extension, without a leading dot.

    Returns:
        str: A filename in the form '{prefix}_{timestamp_ms}_{uuid12}.{ext}'.
    """
    timestamp_ms = int(time.time() * 1000)
    unique_id = generate_uuid_hex(12)
    clean_extension = extension.lstrip(".")
    return f"{prefix}_{timestamp_ms}_{unique_id}.{clean_extension}"


# ==================================================================== #
# Audio utilities
# ==================================================================== #


def bytes_to_megabytes(size_bytes: int) -> float:
    """
    Convert a byte count to megabytes, rounded to two decimal places.

    Args:
        size_bytes: Size in bytes.

    Returns:
        float: Size in megabytes.
    """
    return round(size_bytes / (1024 ** 2), 2)


def estimate_audio_bytes(duration_seconds: float, sample_rate: int, bytes_per_sample: int = 2) -> int:
    """
    Estimate the raw PCM byte size of an audio clip given its duration,
    sample rate, and sample width (mono audio assumed).

    Args:
        duration_seconds: Duration of the audio in seconds.
        sample_rate: Sample rate in Hz.
        bytes_per_sample: Bytes per sample (2 for 16-bit PCM).

    Returns:
        int: Estimated size in bytes.

    Raises:
        ValueError: If any input is non-positive.
    """
    if duration_seconds <= 0 or sample_rate <= 0 or bytes_per_sample <= 0:
        raise ValueError("duration_seconds, sample_rate, and bytes_per_sample must be positive.")
    return int(duration_seconds * sample_rate * bytes_per_sample)


def seconds_to_hms(seconds: float) -> str:
    """
    Convert a duration in seconds to a 'HH:MM:SS' formatted string.

    Args:
        seconds: Duration in seconds.

    Returns:
        str: Formatted duration string.
    """
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


# ==================================================================== #
# GPU utilities
# ==================================================================== #


def get_device() -> str:
    """
    Determine the best available compute device.

    Returns:
        str: "cuda" if a CUDA-capable GPU is available, otherwise "cpu".
    """
    try:
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        logger.warning("CUDA availability check failed; defaulting to CPU.", exc_info=True)
        return "cpu"


def get_gpu_summary() -> Dict[str, Any]:
    """
    Collect a concise summary of available GPU hardware.

    Returns:
        Dict[str, Any]: GPU availability, device count, and per-device
            name/memory information. Minimal dict if CUDA is unavailable.
    """
    summary: Dict[str, Any] = {"cuda_available": torch.cuda.is_available()}

    if not summary["cuda_available"]:
        return summary

    try:
        device_count = torch.cuda.device_count()
        summary["device_count"] = device_count
        summary["devices"] = [
            {
                "index": index,
                "name": torch.cuda.get_device_properties(index).name,
                "total_memory_mb": bytes_to_megabytes(
                    torch.cuda.get_device_properties(index).total_memory
                ),
            }
            for index in range(device_count)
        ]
    except Exception as exc:
        logger.warning("Failed to collect GPU summary: %s", exc)
        summary["error"] = str(exc)

    return summary


def clear_gpu_memory() -> None:
    """
    Release cached (but unallocated) CUDA memory back to the driver.
    Safe to call in CPU-only environments (no-op).
    """
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception as exc:
            logger.warning("Failed to clear GPU memory: %s", exc)


# ==================================================================== #
# JSON helpers
# ==================================================================== #


def to_json(data: Any, indent: Optional[int] = None) -> str:
    """
    Serialize a Python object to a JSON string.

    Args:
        data: The object to serialize. Must be JSON-serializable.
        indent: Optional indentation level for pretty-printing.

    Returns:
        str: The JSON-encoded string.

    Raises:
        TypeError: If `data` contains non-JSON-serializable objects.
    """
    try:
        return json.dumps(data, indent=indent, ensure_ascii=False, default=str)
    except TypeError as exc:
        raise TypeError(f"Failed to serialize object to JSON: {exc}") from exc


def from_json(text: str) -> JSONType:
    """
    Deserialize a JSON string into a Python object.

    Args:
        text: The JSON string to parse.

    Returns:
        JSONType: The deserialized Python object.

    Raises:
        ValueError: If `text` is not valid JSON.
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse JSON: {exc}") from exc


def read_json_file(path: Union[str, Path]) -> JSONType:
    """
    Read and parse a JSON file from disk.

    Args:
        path: Path to the JSON file.

    Returns:
        JSONType: The deserialized Python object.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file does not contain valid JSON.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"JSON file not found: {file_path}")

    try:
        return json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in file '{file_path}': {exc}") from exc


def write_json_file(path: Union[str, Path], data: Any, indent: int = 2) -> Path:
    """
    Serialize an object to JSON and write it to disk, creating parent
    directories as needed.

    Args:
        path: Destination file path.
        data: The object to serialize. Must be JSON-serializable.
        indent: Indentation level used for pretty-printing.

    Returns:
        Path: The resolved path the JSON was written to.

    Raises:
        TypeError: If `data` contains non-JSON-serializable objects.
        OSError: If the file cannot be written.
    """
    file_path = Path(path).resolve()
    ensure_directory(file_path.parent)

    try:
        file_path.write_text(to_json(data, indent=indent), encoding="utf-8")
        return file_path
    except OSError as exc:
        raise OSError(f"Failed to write JSON file '{file_path}': {exc}") from exc


# ==================================================================== #
# Environment helpers
# ==================================================================== #


def get_env(key: str, default: Optional[str] = None) -> Optional[str]:
    """
    Read a string environment variable.

    Args:
        key: Environment variable name.
        default: Value to return if the variable is unset.

    Returns:
        Optional[str]: The environment variable's value, or `default`.
    """
    return os.environ.get(key, default)


def get_env_bool(key: str, default: bool = False) -> bool:
    """
    Read a boolean environment variable, interpreting common truthy
    string values.

    Args:
        key: Environment variable name.
        default: Value to return if the variable is unset.

    Returns:
        bool: The parsed boolean value.
    """
    raw = os.environ.get(key)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on", "y")


def get_env_int(key: str, default: int = 0) -> int:
    """
    Read an integer environment variable.

    Args:
        key: Environment variable name.
        default: Value to return if the variable is unset or invalid.

    Returns:
        int: The parsed integer value, or `default` on failure.
    """
    raw = os.environ.get(key)
    if raw is None:
        return default
    try:
        return int(raw.strip())
    except ValueError:
        logger.warning("Invalid integer for env var '%s': %r. Using default %d.", key, raw, default)
        return default


def get_env_float(key: str, default: float = 0.0) -> float:
    """
    Read a floating-point environment variable.

    Args:
        key: Environment variable name.
        default: Value to return if the variable is unset or invalid.

    Returns:
        float: The parsed float value, or `default` on failure.
    """
    raw = os.environ.get(key)
    if raw is None:
        return default
    try:
        return float(raw.strip())
    except ValueError:
        logger.warning("Invalid float for env var '%s': %r. Using default %f.", key, raw, default)
        return default


def is_production() -> bool:
    """
    Determine whether the application is running in a production
    environment, based on the 'ENV' environment variable.

    Returns:
        bool: True if 'ENV' is set to 'production' (case-insensitive).
    """
    return get_env("ENV", "production").strip().lower() == "production"


__all__ = [
    "generate_uuid",
    "generate_uuid_hex",
    "generate_request_id",
    "sha256_hash_bytes",
    "sha256_hash_text",
    "sha256_hash_file",
    "timer",
    "timed_result",
    "retry",
    "ensure_directory",
    "safe_delete",
    "get_file_size_bytes",
    "list_files",
    "generate_unique_filename",
    "bytes_to_megabytes",
    "estimate_audio_bytes",
    "seconds_to_hms",
    "get_device",
    "get_gpu_summary",
    "clear_gpu_memory",
    "to_json",
    "from_json",
    "read_json_file",
    "write_json_file",
    "get_env",
    "get_env_bool",
    "get_env_int",
    "get_env_float",
    "is_production",
]