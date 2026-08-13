"""
services/stt/config.py

Configuration for the Valora AI Speech-to-Text (STT) service, using
faster-whisper for local, CPU/GPU-flexible transcription.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import torch
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger("valora.stt.config")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


class STTSettings(BaseSettings):
    """Configuration for the STT service, loaded from environment/.env."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    STT_MODEL_SIZE: Literal["tiny", "base", "small", "medium", "large-v3"] = Field(
        default="small",
        description="faster-whisper model size. 'small' is a good accuracy/speed balance for CPU.",
    )
    STT_DEVICE: str = Field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu",
        description="Compute device for STT inference.",
    )
    STT_COMPUTE_TYPE: str = Field(
        default_factory=lambda: "float16" if torch.cuda.is_available() else "int8",
        description="CTranslate2 compute type (float16 on GPU, int8 on CPU for speed).",
    )
    STT_MODEL_CACHE_DIR: Path = Field(
        default=Path(".cache/stt_models"),
        description="Directory for cached Whisper model weights.",
    )

    def resolve(self) -> "STTSettings":
        """Resolve and create the model cache directory."""
        self.STT_MODEL_CACHE_DIR = self.STT_MODEL_CACHE_DIR.resolve()
        self.STT_MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        return self


stt_settings = STTSettings().resolve()
