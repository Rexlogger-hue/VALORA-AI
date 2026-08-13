"""
services/stt/model.py

Speech-to-Text (STT) model wrapper for Valora AI, using faster-whisper
(a CTranslate2-based reimplementation of OpenAI Whisper) for fast,
accurate, multilingual transcription -- including Hindi and other
Indian languages.
"""

from __future__ import annotations

import io
import logging
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from faster_whisper import WhisperModel

from services.stt.config import stt_settings

logger = logging.getLogger("valora.stt.model")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


class STTModelError(Exception):
    """Base exception for STT errors."""


class STTModelLoadError(STTModelError):
    """Raised when the STT model fails to load."""


class STTTranscriptionError(STTModelError):
    """Raised when transcription fails."""


@dataclass(frozen=True)
class TranscriptionSegment:
    """A single timed segment of transcribed speech."""

    start: float
    end: float
    text: str


@dataclass(frozen=True)
class TranscriptionResult:
    """Full transcription result."""

    text: str
    language: str
    language_probability: float
    segments: List[TranscriptionSegment] = field(default_factory=list)


class SpeechToTextModel:
    """
    Reusable wrapper around faster-whisper for speech-to-text.

    Example:
        stt = SpeechToTextModel()
        stt.load()
        result = stt.transcribe(audio_bytes)
        print(result.text)
    """

    def __init__(self, model_size: Optional[str] = None) -> None:
        self.model_size: str = model_size or stt_settings.STT_MODEL_SIZE
        self.device: str = stt_settings.STT_DEVICE
        self.compute_type: str = stt_settings.STT_COMPUTE_TYPE

        self._model: Optional[WhisperModel] = None
        self._is_loaded: bool = False
        self._load_lock = threading.Lock()

        logger.info(
            "SpeechToTextModel initialized (model_size=%s, device=%s, compute_type=%s)",
            self.model_size, self.device, self.compute_type,
        )

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    def load(self) -> None:
        """Load the Whisper model into memory. Idempotent and thread-safe."""
        if self._is_loaded:
            return

        with self._load_lock:
            if self._is_loaded:
                return

            logger.info("Loading STT model 'whisper-%s' onto device '%s'...", self.model_size, self.device)
            try:
                self._model = WhisperModel(
                    self.model_size,
                    device=self.device,
                    compute_type=self.compute_type,
                    download_root=str(stt_settings.STT_MODEL_CACHE_DIR),
                )
                self._is_loaded = True
                logger.info("STT model 'whisper-%s' loaded successfully.", self.model_size)
            except Exception as exc:
                self._model = None
                self._is_loaded = False
                logger.exception("Failed to load STT model.")
                raise STTModelLoadError(f"Failed to load STT model: {exc}") from exc

    def unload(self) -> None:
        """Release the model from memory."""
        with self._load_lock:
            self._model = None
            self._is_loaded = False
            logger.info("STT model unloaded.")

    def transcribe(
        self,
        audio_bytes: bytes,
        language: Optional[str] = None,
    ) -> TranscriptionResult:
        """
        Transcribe speech audio to text.

        Args:
            audio_bytes: Raw audio file content (wav/mp3/etc -- faster-whisper
                uses ffmpeg internally to decode most common formats).
            language: Optional ISO language code (e.g. 'en', 'hi') to force
                a specific language. If None, language is auto-detected.

        Returns:
            TranscriptionResult: Full text, detected language, and timed segments.

        Raises:
            STTModelLoadError: If the model is not loaded and auto-load fails.
            STTTranscriptionError: If transcription fails.
        """
        if not audio_bytes:
            raise ValueError("audio_bytes must not be empty.")

        if not self._is_loaded:
            self.load()

        assert self._model is not None

        try:
            # faster-whisper reads from a file path; write to a temp file.
            with tempfile.NamedTemporaryFile(suffix=".audio", delete=False) as tmp_file:
                tmp_file.write(audio_bytes)
                tmp_path = Path(tmp_file.name)

            try:
                segments_iter, info = self._model.transcribe(
                    str(tmp_path),
                    language=language,
                    beam_size=5,
                    vad_filter=True,
                )

                segments: List[TranscriptionSegment] = []
                full_text_parts: List[str] = []

                for segment in segments_iter:
                    segments.append(
                        TranscriptionSegment(start=segment.start, end=segment.end, text=segment.text.strip())
                    )
                    full_text_parts.append(segment.text.strip())

                full_text = " ".join(full_text_parts).strip()

                logger.info(
                    "Transcribed %d segment(s), language=%s (p=%.2f).",
                    len(segments), info.language, info.language_probability,
                )

                return TranscriptionResult(
                    text=full_text,
                    language=info.language,
                    language_probability=info.language_probability,
                    segments=segments,
                )
            finally:
                tmp_path.unlink(missing_ok=True)

        except Exception as exc:
            logger.exception("Transcription failed.")
            raise STTTranscriptionError(f"Transcription failed: {exc}") from exc

    def health_check(self) -> dict:
        return {
            "model_size": self.model_size,
            "is_loaded": self._is_loaded,
            "device": self.device,
            "compute_type": self.compute_type,
        }