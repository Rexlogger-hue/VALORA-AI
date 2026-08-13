"""
services/tts/config.py

Centralized, environment-driven configuration for the Valora AI
Text-to-Speech pipeline.

This module defines a single `Settings` object (Pydantic v2, via
pydantic-settings) that is consumed by `services/tts/model.py` and
`services/tts/inference.py`. Values are automatically loaded from a
`.env` file (if present) and may be overridden via environment
variables. Device selection (CUDA vs CPU) is auto-detected at import
time using `torch.cuda.is_available()`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Literal, Optional

import torch
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger("valora.tts.config")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


def _detect_device() -> str:
    """
    Detect the best available compute device.

    Returns:
        str: "cuda" if a CUDA-capable GPU is available, otherwise "cpu".
    """
    try:
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        logger.warning("CUDA availability check failed; defaulting to CPU.", exc_info=True)
        return "cpu"


class Settings(BaseSettings):
    """
    Application configuration for the Valora AI TTS service.

    Values are loaded, in order of precedence, from:
        1. Explicit environment variables.
        2. A `.env` file located at the project root.
        3. The default values declared below.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # -------------------------------------------------------------- #
    # MODEL SETTINGS
    # -------------------------------------------------------------- #
    TTS_MODEL_NAME: str = Field(
        default="suno/bark",
        description="HuggingFace model identifier for the TTS model.",
    )
    TTS_DEFAULT_VOICE_PRESET: str = Field(
        default="v2/en_speaker_6",
        description="Default speaker/voice preset used for synthesis.",
    )

    # -------------------------------------------------------------- #
    # DEVICE SETTINGS
    # -------------------------------------------------------------- #
    TTS_DEVICE: str = Field(
        default_factory=_detect_device,
        description="Compute device used for inference ('cuda' or 'cpu').",
    )
    TTS_USE_FP16: bool = Field(
        default=True,
        description="Use half-precision (fp16) weights when running on CUDA.",
    )

    # -------------------------------------------------------------- #
    # GPU SETTINGS
    # -------------------------------------------------------------- #
    GPU_MEMORY_FRACTION: float = Field(
        default=0.9,
        ge=0.05,
        le=1.0,
        description="Fraction of GPU memory PyTorch is allowed to reserve.",
    )
    ENABLE_TF32: bool = Field(
        default=True,
        description="Enable TF32 matmul/cuDNN kernels on Ampere+ GPUs.",
    )
    ENABLE_CUDNN_BENCHMARK: bool = Field(
        default=True,
        description="Enable cuDNN autotuner for consistent input shapes.",
    )

    # -------------------------------------------------------------- #
    # TEXT LIMITS
    # -------------------------------------------------------------- #
    TTS_MAX_TEXT_LENGTH: int = Field(
        default=5000,
        gt=0,
        description="Maximum allowed length of input text, in characters.",
    )
    TTS_MAX_CHUNK_CHARS: int = Field(
        default=250,
        gt=0,
        description="Maximum number of characters per synthesis chunk.",
    )
    TTS_MAX_NEW_TOKENS: Optional[int] = Field(
        default=None,
        gt=0,
        description="Optional cap on generated audio tokens per chunk.",
    )

    # -------------------------------------------------------------- #
    # INFERENCE
    # -------------------------------------------------------------- #
    TTS_INFERENCE_WORKERS: int = Field(
        default=1,
        gt=0,
        description="Number of worker threads used for async synthesis.",
    )
    ENABLE_ASYNC: bool = Field(
        default=True,
        description="Whether asynchronous synthesis endpoints are enabled.",
    )

    # -------------------------------------------------------------- #
    # DIRECTORIES
    # -------------------------------------------------------------- #
    PROJECT_ROOT: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parents[2],
        description="Root directory of the project.",
    )
    CACHE_DIR: Path = Field(
        default=Path(".cache"),
        description="General-purpose cache directory.",
    )
    MODEL_CACHE_DIR: Path = Field(
        default=Path(".cache/models"),
        description="Directory for cached HuggingFace model weights.",
    )
    OUTPUT_DIR: Path = Field(
        default=Path("outputs/tts"),
        description="Directory for generated audio output files.",
    )
    LOG_DIR: Path = Field(
        default=Path("logs"),
        description="Directory for log files.",
    )
    TEMP_DIR: Path = Field(
        default=Path("tmp/tts"),
        description="Directory for transient/temporary files.",
    )

    # -------------------------------------------------------------- #
    # LOGGING
    # -------------------------------------------------------------- #
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Logging verbosity level.",
    )
    LOG_FORMAT: str = Field(
        default="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        description="Logging message format string.",
    )

    # -------------------------------------------------------------- #
    # ENVIRONMENT
    # -------------------------------------------------------------- #
    DEBUG: bool = Field(
        default=False,
        description="Whether debug mode is enabled.",
    )
    ENV: Literal["development", "staging", "production", "test"] = Field(
        default="production",
        description="Deployment environment name.",
    )
    VERSION: str = Field(
        default="1.0.0",
        description="Application/service version string.",
    )

    @field_validator("TTS_DEVICE")
    @classmethod
    def _validate_device(cls, value: str) -> str:
        """
        Validate the configured device string and fall back to a safe
        value if CUDA was requested but is not actually available.
        """
        normalized = value.strip().lower()
        if normalized not in ("cuda", "cpu"):
            raise ValueError(f"TTS_DEVICE must be 'cuda' or 'cpu', got: {value!r}")

        if normalized == "cuda" and not torch.cuda.is_available():
            logger.warning(
                "TTS_DEVICE was set to 'cuda' but no CUDA device is available; "
                "falling back to 'cpu'."
            )
            return "cpu"

        return normalized

    @model_validator(mode="after")
    def _finalize(self) -> "Settings":
        """
        Post-process settings: disable fp16 on CPU, resolve directory
        paths, create directories, and apply GPU-related global torch
        settings.
        """
        if self.TTS_DEVICE != "cuda" and self.TTS_USE_FP16:
            logger.info("Disabling fp16 because TTS_DEVICE is not 'cuda'.")
            object.__setattr__(self, "TTS_USE_FP16", False)

        self._resolve_directories()
        self._create_directories()
        self._apply_gpu_settings()

        return self

    def _resolve_directories(self) -> None:
        """Resolve all configured directory paths against PROJECT_ROOT."""
        root = self.PROJECT_ROOT.resolve()
        object.__setattr__(self, "PROJECT_ROOT", root)

        for attr in ("CACHE_DIR", "MODEL_CACHE_DIR", "OUTPUT_DIR", "LOG_DIR", "TEMP_DIR"):
            path = getattr(self, attr)
            if not path.is_absolute():
                path = (root / path).resolve()
            object.__setattr__(self, attr, path)

    def _create_directories(self) -> None:
        """
        Create all required directories if they do not already exist.

        Raises:
            RuntimeError: If a required directory cannot be created.
        """
        for attr in ("CACHE_DIR", "MODEL_CACHE_DIR", "OUTPUT_DIR", "LOG_DIR", "TEMP_DIR"):
            path: Path = getattr(self, attr)
            try:
                path.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                raise RuntimeError(
                    f"Failed to create required directory '{path}' for '{attr}': {exc}"
                ) from exc

    def _apply_gpu_settings(self) -> None:
        """
        Apply GPU-related global PyTorch configuration when running on
        CUDA. Safe no-op on CPU-only environments.
        """
        if self.TTS_DEVICE != "cuda":
            return

        try:
            torch.backends.cuda.matmul.allow_tf32 = self.ENABLE_TF32
            torch.backends.cudnn.allow_tf32 = self.ENABLE_TF32
            torch.backends.cudnn.benchmark = self.ENABLE_CUDNN_BENCHMARK

            if torch.cuda.is_available():
                torch.cuda.set_per_process_memory_fraction(
                    self.GPU_MEMORY_FRACTION, device=0
                )
        except Exception:
            logger.warning("Failed to apply one or more GPU settings.", exc_info=True)

    def is_gpu_enabled(self) -> bool:
        """
        Determine whether the service is configured to use a GPU.

        Returns:
            bool: True if `TTS_DEVICE` is 'cuda' and CUDA is available.
        """
        return self.TTS_DEVICE == "cuda" and torch.cuda.is_available()

    def device_name(self) -> str:
        """
        Get a human-readable name for the active compute device.

        Returns:
            str: The CUDA device name if GPU-enabled, otherwise "CPU".
        """
        if self.is_gpu_enabled():
            try:
                return torch.cuda.get_device_name(0)
            except Exception:
                logger.warning("Failed to query CUDA device name.", exc_info=True)
                return "CUDA (unknown device)"
        return "CPU"

    def health_summary(self) -> Dict[str, Any]:
        """
        Produce a structured summary of the current configuration and
        runtime environment.

        Returns:
            Dict[str, Any]: Summary of model, device, limits, and
                environment configuration.
        """
        cuda_available = torch.cuda.is_available()
        return {
            "environment": self.ENV,
            "version": self.VERSION,
            "debug": self.DEBUG,
            "model": {
                "name": self.TTS_MODEL_NAME,
                "default_voice_preset": self.TTS_DEFAULT_VOICE_PRESET,
            },
            "device": {
                "configured": self.TTS_DEVICE,
                "gpu_enabled": self.is_gpu_enabled(),
                "device_name": self.device_name(),
                "cuda_available": cuda_available,
                "fp16": self.TTS_USE_FP16,
                "tf32_enabled": self.ENABLE_TF32,
                "cudnn_benchmark": self.ENABLE_CUDNN_BENCHMARK,
                "gpu_memory_fraction": self.GPU_MEMORY_FRACTION,
            },
            "text_limits": {
                "max_text_length": self.TTS_MAX_TEXT_LENGTH,
                "max_chunk_chars": self.TTS_MAX_CHUNK_CHARS,
                "max_new_tokens": self.TTS_MAX_NEW_TOKENS,
            },
            "inference": {
                "workers": self.TTS_INFERENCE_WORKERS,
                "async_enabled": self.ENABLE_ASYNC,
            },
            "directories": {
                "project_root": str(self.PROJECT_ROOT),
                "cache_dir": str(self.CACHE_DIR),
                "model_cache_dir": str(self.MODEL_CACHE_DIR),
                "output_dir": str(self.OUTPUT_DIR),
                "log_dir": str(self.LOG_DIR),
                "temp_dir": str(self.TEMP_DIR),
            },
            "logging": {
                "level": self.LOG_LEVEL,
                "format": self.LOG_FORMAT,
            },
        }


settings = Settings()