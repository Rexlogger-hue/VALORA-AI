"""
services/api/dependencies.py

FastAPI dependency injection providers for the Valora AI API.

Centralizes singleton access to the TTS inference service, text
preprocessor, application settings, and module logger, ensuring the
underlying TTS model is loaded exactly once per process and shared
safely across concurrent requests.
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

from services.tts.config import Settings, settings
from services.tts.inference import TTSInferenceError, TTSInferenceService, get_tts_service as _get_tts_service_singleton
from services.tts.preprocess import TextPreprocessor
from services.tts.utils import get_logger as _get_utils_logger

_dependencies_logger = _get_utils_logger("valora.api.dependencies")

_preprocessor_instance: Optional[TextPreprocessor] = None
_preprocessor_lock = threading.Lock()


def get_settings() -> Settings:
    """
    Provide the process-wide application settings instance.

    Returns:
        Settings: The shared, validated `Settings` object loaded from
            environment variables and `.env`.
    """
    return settings


def get_tts_service() -> TTSInferenceService:
    """
    Provide the process-wide `TTSInferenceService` singleton, ensuring
    the underlying TTS model is loaded only once per process.

    This delegates to `services.tts.inference.get_tts_service`, which
    performs thread-safe, lazy singleton construction. The model itself
    is warmed up explicitly during application startup (see
    `services/api/main.py`); this dependency does not trigger a load on
    its own.

    Returns:
        TTSInferenceService: The shared inference service instance.

    Raises:
        RuntimeError: If the inference service singleton cannot be
            constructed.
    """
    try:
        return _get_tts_service_singleton()
    except Exception as exc:
        _dependencies_logger.exception("Failed to obtain TTSInferenceService singleton.")
        raise RuntimeError(f"Failed to obtain TTS inference service: {exc}") from exc


def get_inference_service() -> TTSInferenceService:
    """
    Alias dependency for `get_tts_service`, provided for readability in
    route signatures that refer to the "inference service" explicitly.

    Returns:
        TTSInferenceService: The shared inference service instance.

    Raises:
        RuntimeError: If the inference service singleton cannot be
            constructed.
    """
    return get_tts_service()


def get_text_preprocessor() -> TextPreprocessor:
    """
    Provide the process-wide `TextPreprocessor` singleton, constructing
    it on first access using limits sourced from application settings.

    Returns:
        TextPreprocessor: The shared text preprocessor instance.

    Raises:
        RuntimeError: If the preprocessor cannot be constructed.
    """
    global _preprocessor_instance

    if _preprocessor_instance is None:
        with _preprocessor_lock:
            if _preprocessor_instance is None:
                try:
                    _preprocessor_instance = TextPreprocessor(
                        max_text_length=settings.TTS_MAX_TEXT_LENGTH,
                        max_chunk_chars=settings.TTS_MAX_CHUNK_CHARS,
                    )
                    _dependencies_logger.info(
                        "TextPreprocessor singleton initialized "
                        "(max_text_length=%d, max_chunk_chars=%d).",
                        settings.TTS_MAX_TEXT_LENGTH,
                        settings.TTS_MAX_CHUNK_CHARS,
                    )
                except Exception as exc:
                    _dependencies_logger.exception("Failed to initialize TextPreprocessor singleton.")
                    raise RuntimeError(f"Failed to initialize text preprocessor: {exc}") from exc

    return _preprocessor_instance


def get_logger(name: str = "valora.api") -> logging.Logger:
    """
    Provide a consistently configured logger for use within API route
    handlers and other request-scoped components.

    Args:
        name: Logger name, typically identifying the calling module
            (e.g. 'valora.api.routes.tts').

    Returns:
        logging.Logger: A configured logger instance.
    """
    return _get_utils_logger(name)


__all__ = [
    "get_settings",
    "get_tts_service",
    "get_inference_service",
    "get_text_preprocessor",
    "get_logger",
]