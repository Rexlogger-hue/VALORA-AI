"""
services/tts/utils.py

Shared utility functions for the Valora AI Text-to-Speech pipeline.

This module provides low-level helpers consumed by `services/tts/model.py`,
`services/tts/config.py`, `services/tts/preprocess.py`, and
`services/tts/inference.py`, covering audio inspection/normalization,
file and cache management, GPU diagnostics, and cross-cutting decorators
for timing and retry logic.
"""

from __future__ import annotations

import functools
import hashlib
import logging
import tempfile
import time
import uuid
import wave
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, TypeVar

import numpy as np
import torch

from services.tts.config import settings

logger = logging.getLogger("valora.tts.utils")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(settings.LOG_FORMAT)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(getattr(logging, settings.LOG_LEVEL, logging.INFO))

F = TypeVar("F", bound=Callable[..., Any])


# ==================================================================== #
# Audio utilities
# ==================================================================== #


def calculate_audio_duration(audio: np.ndarray, sample_rate: int) -> float:
    """
    Calculate the duration of a waveform in seconds.

    Args:
        audio: 1-D (or flattenable) numpy array of audio samples.
        sample_rate: Sample rate of the audio, in Hz.

    Returns:
        float: Duration of the audio in seconds.

    Raises:
        ValueError: If `sample_rate` is not positive or `audio` is empty.
    """
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}.")

    flattened = np.asarray(audio).reshape(-1)
    if flattened.size == 0:
        raise ValueError("Cannot calculate duration of empty audio array.")

    return float(flattened.size) / float(sample_rate)


def validate_wav_bytes(wav_bytes: bytes) -> Dict[str, Any]:
    """
    Validate that a bytes object is a well-formed WAV file and extract
    its basic properties.

    Args:
        wav_bytes: Raw WAV file content.

    Returns:
        Dict[str, Any]: Metadata with keys "channels", "sample_width",
            "frame_rate", "frame_count", and "duration_seconds".

    Raises:
        ValueError: If `wav_bytes` is empty or not a valid WAV file.
    """
    if not wav_bytes:
        raise ValueError("wav_bytes is empty.")

    import io

    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            frame_rate = wav_file.getframerate()
            frame_count = wav_file.getnframes()

            if frame_rate <= 0 or frame_count <= 0:
                raise ValueError("WAV file has invalid frame rate or frame count.")

            return {
                "channels": channels,
                "sample_width": sample_width,
                "frame_rate": frame_rate,
                "frame_count": frame_count,
                "duration_seconds": frame_count / float(frame_rate),
            }
    except wave.Error as exc:
        raise ValueError(f"Invalid WAV data: {exc}") from exc
    except Exception as exc:
        raise ValueError(f"Failed to validate WAV data: {exc}") from exc


def normalize_audio(audio: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
    """
    Normalize an audio waveform so its peak absolute amplitude matches
    `target_peak`, preserving silence for all-zero input.

    Args:
        audio: 1-D numpy array of audio samples.
        target_peak: Desired peak absolute amplitude, in range (0, 1].

    Returns:
        np.ndarray: Normalized float32 waveform.

    Raises:
        ValueError: If `target_peak` is not in the range (0, 1].
    """
    if not (0.0 < target_peak <= 1.0):
        raise ValueError(f"target_peak must be in (0, 1], got {target_peak}.")

    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    peak = float(np.max(np.abs(audio))) if audio.size > 0 else 0.0

    if peak == 0.0:
        return audio

    scale = target_peak / peak
    return (audio * scale).astype(np.float32)


def prevent_clipping(audio: np.ndarray, ceiling: float = 0.99) -> np.ndarray:
    """
    Clip and rescale an audio waveform to ensure no sample exceeds
    `ceiling` in absolute value, avoiding harsh digital clipping.

    Args:
        audio: 1-D numpy array of audio samples.
        ceiling: Maximum permitted absolute sample value, in (0, 1].

    Returns:
        np.ndarray: Clipping-safe float32 waveform.

    Raises:
        ValueError: If `ceiling` is not in the range (0, 1].
    """
    if not (0.0 < ceiling <= 1.0):
        raise ValueError(f"ceiling must be in (0, 1], got {ceiling}.")

    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    peak = float(np.max(np.abs(audio))) if audio.size > 0 else 0.0

    if peak > ceiling:
        audio = audio * (ceiling / peak)

    return np.clip(audio, -ceiling, ceiling).astype(np.float32)


# ==================================================================== #
# Hashing and filenames
# ==================================================================== #


def sha256_hash(data: bytes) -> str:
    """
    Compute the SHA-256 hex digest of a bytes object.

    Args:
        data: Raw bytes to hash.

    Returns:
        str: Hexadecimal SHA-256 digest.
    """
    return hashlib.sha256(data).hexdigest()


def sha256_hash_text(text: str, encoding: str = "utf-8") -> str:
    """
    Compute the SHA-256 hex digest of a text string.

    Args:
        text: Input string to hash.
        encoding: Text encoding used before hashing.

    Returns:
        str: Hexadecimal SHA-256 digest.
    """
    return hashlib.sha256(text.encode(encoding)).hexdigest()


def generate_unique_filename(prefix: str = "tts", extension: str = "wav") -> str:
    """
    Generate a unique, collision-resistant filename.

    Args:
        prefix: Prefix identifying the file's purpose (e.g. "tts").
        extension: File extension, without a leading dot.

    Returns:
        str: A filename in the form "{prefix}_{timestamp}_{uuid4hex}.{ext}".
    """
    timestamp = int(time.time() * 1000)
    unique_id = uuid.uuid4().hex[:12]
    clean_extension = extension.lstrip(".")
    return f"{prefix}_{timestamp}_{unique_id}.{clean_extension}"


# ==================================================================== #
# File and directory management
# ==================================================================== #


def create_temp_file(
    suffix: str = ".wav", directory: Optional[Path] = None
) -> Path:
    """
    Create an empty temporary file and return its path.

    Args:
        suffix: File suffix/extension, including the leading dot.
        directory: Directory to create the file in. Defaults to
            `settings.TEMP_DIR`.

    Returns:
        Path: Path to the newly created temporary file.

    Raises:
        OSError: If the temporary file cannot be created.
    """
    target_dir = directory or settings.TEMP_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    try:
        fd, raw_path = tempfile.mkstemp(suffix=suffix, dir=str(target_dir))
        import os

        os.close(fd)
        return Path(raw_path)
    except OSError as exc:
        logger.error("Failed to create temporary file in '%s': %s", target_dir, exc)
        raise


def safe_delete_file(path: Path) -> bool:
    """
    Delete a file if it exists, swallowing and logging any errors
    rather than raising.

    Args:
        path: Path to the file to delete.

    Returns:
        bool: True if the file was deleted, False if it did not exist
            or deletion failed.
    """
    try:
        file_path = Path(path)
        if file_path.exists() and file_path.is_file():
            file_path.unlink()
            return True
        return False
    except OSError as exc:
        logger.warning("Failed to delete file '%s': %s", path, exc)
        return False


def ensure_output_directory(directory: Optional[Path] = None) -> Path:
    """
    Ensure an output directory exists, creating it (and parents) if
    necessary.

    Args:
        directory: Directory to ensure exists. Defaults to
            `settings.OUTPUT_DIR`.

    Returns:
        Path: The resolved, guaranteed-to-exist directory path.

    Raises:
        RuntimeError: If the directory cannot be created.
    """
    target_dir = (directory or settings.OUTPUT_DIR).resolve()
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        return target_dir
    except OSError as exc:
        raise RuntimeError(f"Failed to create output directory '{target_dir}': {exc}") from exc


def get_model_cache_path(model_name: str) -> Path:
    """
    Compute a filesystem-safe cache path for a given model identifier
    under `settings.MODEL_CACHE_DIR`.

    Args:
        model_name: Model identifier (e.g. "suno/bark").

    Returns:
        Path: Directory path where the model's cache should live.
    """
    safe_name = model_name.replace("/", "__").replace(" ", "_")
    cache_path = settings.MODEL_CACHE_DIR / safe_name
    cache_path.mkdir(parents=True, exist_ok=True)
    return cache_path


def is_model_cached(model_name: str) -> bool:
    """
    Determine whether a model's cache directory exists and is non-empty.

    Args:
        model_name: Model identifier (e.g. "suno/bark").

    Returns:
        bool: True if a non-empty cache directory exists for the model.
    """
    cache_path = get_model_cache_path(model_name)
    return cache_path.exists() and any(cache_path.iterdir())


# ==================================================================== #
# GPU diagnostics
# ==================================================================== #


def get_gpu_info() -> Dict[str, Any]:
    """
    Collect diagnostic information about the available GPU(s).

    Returns:
        Dict[str, Any]: GPU availability, device count, names, and
            compute capability. Returns a minimal dict when CUDA is
            unavailable.
    """
    info: Dict[str, Any] = {"cuda_available": torch.cuda.is_available()}

    if not info["cuda_available"]:
        return info

    try:
        device_count = torch.cuda.device_count()
        info["device_count"] = device_count
        info["devices"] = []

        for index in range(device_count):
            properties = torch.cuda.get_device_properties(index)
            info["devices"].append(
                {
                    "index": index,
                    "name": properties.name,
                    "total_memory_mb": round(properties.total_memory / (1024 ** 2), 2),
                    "multi_processor_count": properties.multi_processor_count,
                    "compute_capability": f"{properties.major}.{properties.minor}",
                }
            )
    except Exception as exc:
        logger.warning("Failed to collect GPU info: %s", exc)
        info["error"] = str(exc)

    return info


def get_cuda_memory_usage(device_index: int = 0) -> Dict[str, float]:
    """
    Retrieve current CUDA memory usage statistics for a given device.

    Args:
        device_index: Index of the CUDA device to query.

    Returns:
        Dict[str, float]: Memory usage in megabytes, with keys
            "allocated_mb", "reserved_mb", and "max_allocated_mb". All
            values are 0.0 if CUDA is unavailable.
    """
    if not torch.cuda.is_available():
        return {"allocated_mb": 0.0, "reserved_mb": 0.0, "max_allocated_mb": 0.0}

    try:
        allocated = torch.cuda.memory_allocated(device_index) / (1024 ** 2)
        reserved = torch.cuda.memory_reserved(device_index) / (1024 ** 2)
        max_allocated = torch.cuda.max_memory_allocated(device_index) / (1024 ** 2)

        return {
            "allocated_mb": round(allocated, 2),
            "reserved_mb": round(reserved, 2),
            "max_allocated_mb": round(max_allocated, 2),
        }
    except Exception as exc:
        logger.warning("Failed to query CUDA memory usage: %s", exc)
        return {"allocated_mb": 0.0, "reserved_mb": 0.0, "max_allocated_mb": 0.0}


def clear_cuda_cache() -> None:
    """
    Release cached (but unallocated) CUDA memory back to the driver.
    Safe to call on CPU-only environments (no-op).
    """
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception as exc:
            logger.warning("Failed to clear CUDA cache: %s", exc)


# ==================================================================== #
# Decorators
# ==================================================================== #


def timed(func: F) -> F:
    """
    Decorator that logs the execution time of the wrapped function at
    INFO level.

    Args:
        func: The function to wrap.

    Returns:
        Callable: The wrapped function.
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.monotonic()
        try:
            return func(*args, **kwargs)
        finally:
            elapsed = time.monotonic() - start
            logger.info("%s completed in %.3fs.", func.__qualname__, elapsed)

    return wrapper  # type: ignore[return-value]


def retry(
    max_attempts: int = 3,
    delay_seconds: float = 0.5,
    backoff_factor: float = 2.0,
    exceptions: Tuple[type, ...] = (Exception,),
) -> Callable[[F], F]:
    """
    Decorator factory that retries a function call on failure, using
    exponential backoff between attempts.

    Args:
        max_attempts: Maximum number of attempts, including the first.
        delay_seconds: Initial delay between attempts, in seconds.
        backoff_factor: Multiplier applied to the delay after each
            failed attempt.
        exceptions: Tuple of exception types that should trigger a
            retry. Other exceptions propagate immediately.

    Returns:
        Callable: A decorator that wraps the target function with
            retry logic.

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
# Logging helpers
# ==================================================================== #


def get_logger(name: str) -> logging.Logger:
    """
    Retrieve a module-level logger configured consistently with the
    Valora AI TTS logging format and level.

    Args:
        name: Logger name, typically `__name__` of the caller.

    Returns:
        logging.Logger: A configured logger instance.
    """
    module_logger = logging.getLogger(name)

    if not module_logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(settings.LOG_FORMAT)
        handler.setFormatter(formatter)
        module_logger.addHandler(handler)

    module_logger.setLevel(getattr(logging, settings.LOG_LEVEL, logging.INFO))
    module_logger.propagate = False

    return module_logger


def log_exception(logger_instance: logging.Logger, message: str, exc: BaseException) -> None:
    """
    Log an exception with full traceback context and a custom message.

    Args:
        logger_instance: The logger to emit the message on.
        message: Human-readable context message describing the failure.
        exc: The exception instance that was raised.
    """
    logger_instance.error("%s: %s", message, exc, exc_info=True)