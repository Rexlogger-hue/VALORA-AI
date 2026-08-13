"""
services/tts/inference.py

Inference-layer service for the Valora AI Text-to-Speech pipeline.

This module wraps `TextToSpeechModel` (services/tts/model.py) with:
    - Configuration-driven initialization (services/tts/config.py).
    - Text preprocessing and safe chunking for long inputs.
    - A process-wide singleton accessor for use in FastAPI dependencies.
    - Synchronous and async synthesis APIs (async via a thread executor,
      since the underlying model call is CPU/GPU-bound, not I/O-bound).
    - Structured request/response objects ready to be (de)serialized by
      FastAPI/Pydantic route handlers.

Intended usage inside a FastAPI app:

    from services.tts.inference import get_tts_service, SynthesisRequest

    @app.on_event("startup")
    async def startup() -> None:
        get_tts_service().warm_up()

    @app.post("/tts")
    async def synthesize(request: SynthesisRequest):
        service = get_tts_service()
        result = await service.synthesize_async(request)
        return Response(content=result.audio_bytes, media_type="audio/wav")
"""

from __future__ import annotations

import asyncio
import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from services.tts.config import settings
from services.tts.model import (
    SynthesisResult,
    TextToSpeechModel,
    TTSDeviceMismatchError,
    TTSModelError,
    TTSModelLoadError,
    TTSSynthesisError,
)

logger = logging.getLogger("valora.tts.inference")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


class TTSInferenceError(Exception):
    """Raised when the inference service fails to process a request."""


@dataclass
class SynthesisRequest:
    """
    Represents a single text-to-speech inference request.

    Attributes:
        text: Input text to synthesize. Required, non-empty.
        voice_preset: Optional voice/speaker preset override.
        max_new_tokens: Optional cap on generated audio tokens per chunk.
        language: Optional ISO language hint (for logging/metrics only;
            the underlying model infers language from `voice_preset`).
        age: Optional target apparent age (1-100) for pitch shifting.
        pitch_semitones: Optional direct pitch shift in semitones.
    """

    text: str
    voice_preset: Optional[str] = None
    max_new_tokens: Optional[int] = None
    language: Optional[str] = None
    age: Optional[int] = None
    pitch_semitones: Optional[float] = None
    
@dataclass
class SynthesisResponse:
    """
    Represents the result of a text-to-speech inference request.

    Attributes:
        audio_bytes: WAV-encoded audio content.
        sample_rate: Sample rate (Hz) of the generated audio.
        duration_seconds: Total duration of the generated audio.
        chunk_count: Number of text chunks synthesized and concatenated.
        latency_seconds: Wall-clock time taken to produce the result.
        voice_preset: The voice preset actually used for synthesis.
    """

    audio_bytes: bytes
    sample_rate: int
    duration_seconds: float
    chunk_count: int
    latency_seconds: float
    voice_preset: str = field(default="")


class TTSInferenceService:
    """
    High-level inference service that orchestrates text preprocessing,
    chunking, model synthesis, and audio assembly for the Valora AI
    TTS pipeline.

    This class is safe to use as a process-wide singleton: model loading
    is guarded by a lock, and synthesis calls are serialized per-instance
    via an internal lock to avoid GPU contention across threads.
    """

    _SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?。！？])\s+")

    def __init__(
        self,
        model: Optional[TextToSpeechModel] = None,
        max_chars_per_chunk: Optional[int] = None,
        max_workers: Optional[int] = None,
    ) -> None:
        """
        Initialize the inference service.

        Args:
            model: Optional pre-constructed `TextToSpeechModel`. If None,
                one is built from `services.tts.config.settings`.
            max_chars_per_chunk: Maximum characters per synthesis chunk.
                Defaults to `settings.TTS_MAX_CHUNK_CHARS`.
            max_workers: Number of worker threads for async synthesis.
                Defaults to `settings.TTS_INFERENCE_WORKERS`.
        """
        self._model: TextToSpeechModel = model or TextToSpeechModel(
            model_name=settings.TTS_MODEL_NAME,
            device=settings.TTS_DEVICE,
            use_fp16=settings.TTS_USE_FP16,
        )
        self._default_voice_preset: str = settings.TTS_DEFAULT_VOICE_PRESET
        self._max_new_tokens: Optional[int] = getattr(
            settings, "TTS_MAX_NEW_TOKENS", None
        )
        self._max_chars_per_chunk: int = max_chars_per_chunk or getattr(
            settings, "TTS_MAX_CHUNK_CHARS", 250
        )
        self._max_text_length: int = getattr(
            settings, "TTS_MAX_TEXT_LENGTH", 5000
        )

        worker_count = max_workers or getattr(settings, "TTS_INFERENCE_WORKERS", 1)
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, worker_count), thread_name_prefix="tts-inference"
        )
        self._synthesis_lock = threading.Lock()

        logger.info(
            "TTSInferenceService initialized (chunk_chars=%d, max_text_length=%d, workers=%d, device=%s).",
            self._max_chars_per_chunk,
            self._max_text_length,
            worker_count,
            self._model.device,
        )

    def warm_up(self) -> None:
        """
        Eagerly load the underlying model. Should be called during
        application startup to avoid latency on the first request.

        Raises:
            TTSInferenceError: If the model fails to load.
        """
        try:
            logger.info("Warming up TTS inference service...")
            self._model.load()
            logger.info(
                "TTS inference service warm-up complete (device=%s).",
                self._model.device,
            )
        except TTSModelLoadError as exc:
            logger.exception("TTS inference service warm-up failed.")
            raise TTSInferenceError(f"Warm-up failed: {exc}") from exc

    def health_check(self) -> dict:
        """
        Return the health/status of the underlying model, suitable for
        exposing via a FastAPI health endpoint.
        """
        return self._model.health_check()

    def _validate_request(self, request: SynthesisRequest) -> str:
        """
        Validate and normalize the text of a synthesis request.

        Args:
            request: The incoming synthesis request.

        Returns:
            str: Cleaned, whitespace-normalized text.

        Raises:
            TTSInferenceError: If the request is invalid.
        """
        if not isinstance(request.text, str) or not request.text.strip():
            raise TTSInferenceError("Request text must be a non-empty string.")

        cleaned = re.sub(r"\s+", " ", request.text).strip()

        if len(cleaned) > self._max_text_length:
            raise TTSInferenceError(
                f"Input text exceeds maximum length of "
                f"{self._max_text_length} characters (got {len(cleaned)})."
            )

        return cleaned

    def _chunk_text(self, text: str) -> List[str]:
        """
        Split text into synthesis-friendly chunks along sentence
        boundaries, respecting `max_chars_per_chunk`.

        Args:
            text: Cleaned input text.

        Returns:
            List[str]: Ordered list of non-empty text chunks.
        """
        if len(text) <= self._max_chars_per_chunk:
            return [text]

        sentences = self._SENTENCE_SPLIT_PATTERN.split(text)
        chunks: List[str] = []
        current = ""

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            if len(sentence) > self._max_chars_per_chunk:
                if current:
                    chunks.append(current.strip())
                    current = ""
                for start in range(0, len(sentence), self._max_chars_per_chunk):
                    chunks.append(sentence[start:start + self._max_chars_per_chunk].strip())
                continue

            candidate = f"{current} {sentence}".strip() if current else sentence
            if len(candidate) > self._max_chars_per_chunk:
                if current:
                    chunks.append(current.strip())
                current = sentence
            else:
                current = candidate

        if current.strip():
            chunks.append(current.strip())

        return [chunk for chunk in chunks if chunk]

    @staticmethod
    def _concatenate_results(
        results: List[SynthesisResult], silence_gap_seconds: float = 0.15
    ) -> SynthesisResult:
        """
        Concatenate multiple synthesis results into a single waveform,
        inserting a short silence between chunks for natural pacing.

        Args:
            results: Ordered list of per-chunk synthesis results.
            silence_gap_seconds: Duration of silence inserted between
                consecutive chunks.

        Returns:
            SynthesisResult: The merged waveform and sample rate.

        Raises:
            TTSInferenceError: If `results` is empty or sample rates differ.
        """
        if not results:
            raise TTSInferenceError("No synthesis results to concatenate.")

        sample_rate = results[0].sample_rate
        if any(result.sample_rate != sample_rate for result in results):
            raise TTSInferenceError(
                "Cannot concatenate audio chunks with mismatched sample rates."
            )

        if len(results) == 1:
            return results[0]

        silence_samples = int(sample_rate * silence_gap_seconds)
        silence = np.zeros(silence_samples, dtype=np.float32)

        pieces: List[np.ndarray] = []
        for idx, result in enumerate(results):
            pieces.append(result.audio)
            if idx < len(results) - 1:
                pieces.append(silence)

        merged_audio = np.concatenate(pieces).astype(np.float32)
        return SynthesisResult(audio=merged_audio, sample_rate=sample_rate)

    def synthesize(self, request: SynthesisRequest) -> SynthesisResponse:
        """
        Synchronously synthesize speech for a given request, handling
        text chunking and audio concatenation transparently.

        Args:
            request: The synthesis request to process.

        Returns:
            SynthesisResponse: WAV audio bytes and synthesis metadata.

        Raises:
            TTSInferenceError: If validation, chunking, or synthesis fails
                (including device-mismatch failures, which are now
                reported with a distinguishable message -- see FIX below).
        """
        start_time = time.monotonic()
        cleaned_text = self._validate_request(request)
        chunks = self._chunk_text(cleaned_text)
        voice_preset = request.voice_preset or self._default_voice_preset
        max_new_tokens = request.max_new_tokens or self._max_new_tokens

        logger.info(
            "Starting synthesis: %d chunk(s), voice_preset=%s, language=%s, device=%s.",
            len(chunks),
            voice_preset,
            request.language,
            self._model.device,
        )

        results: List[SynthesisResult] = []
        try:
            with self._synthesis_lock:
                for idx, chunk in enumerate(chunks):
                    logger.debug(
                        "Synthesizing chunk %d/%d (%d chars).",
                        idx + 1,
                        len(chunks),
                        len(chunk),
                    )
                    result = self._model.synthesize(
                        text=chunk,
                        voice_preset=voice_preset,
                        max_new_tokens=max_new_tokens,
                        age=request.age,
                        pitch_semitones=request.pitch_semitones,
                    )
                    results.append(result)

            merged = self._concatenate_results(results)
            audio_bytes = merged.to_wav_bytes()
            duration_seconds = len(merged.audio) / float(merged.sample_rate)
            latency = time.monotonic() - start_time

            logger.info(
                "Synthesis complete: duration=%.2fs, chunks=%d, latency=%.2fs.",
                duration_seconds,
                len(chunks),
                latency,
            )

            return SynthesisResponse(
                audio_bytes=audio_bytes,
                sample_rate=merged.sample_rate,
                duration_seconds=duration_seconds,
                chunk_count=len(chunks),
                latency_seconds=latency,
                voice_preset=voice_preset,
            )

        except TTSInferenceError:
            raise
        except TTSDeviceMismatchError as exc:
            # FIX: Handle the new, more specific exception type raised by
            # model.py separately from generic TTSSynthesisError, so logs
            # and any upstream error handling (services/api/exceptions.py)
            # can clearly identify this as a device-placement bug rather
            # than an ambiguous model failure.
            logger.exception(
                "CUDA/CPU device mismatch during synthesis (voice_preset=%s).", voice_preset
            )
            raise TTSInferenceError(f"Device mismatch during synthesis: {exc}") from exc
        except (TTSModelLoadError, TTSSynthesisError, TTSModelError) as exc:
            logger.exception("Model-level failure during synthesis.")
            raise TTSInferenceError(f"Synthesis failed: {exc}") from exc
        except Exception as exc:
            logger.exception("Unexpected failure during synthesis.")
            raise TTSInferenceError(f"Unexpected synthesis failure: {exc}") from exc

    async def synthesize_async(self, request: SynthesisRequest) -> SynthesisResponse:
        """
        Asynchronously synthesize speech by delegating the blocking
        model call to a background thread pool, keeping the event loop
        free for other FastAPI request handling.

        Args:
            request: The synthesis request to process.

        Returns:
            SynthesisResponse: WAV audio bytes and synthesis metadata.

        Raises:
            TTSInferenceError: If synthesis fails.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self.synthesize, request)

    def shutdown(self) -> None:
        """
        Gracefully release resources held by the inference service,
        including the thread pool and the underlying model.
        """
        logger.info("Shutting down TTS inference service...")
        self._executor.shutdown(wait=True)
        self._model.unload()
        logger.info("TTS inference service shut down.")


_service_instance: Optional[TTSInferenceService] = None
_service_lock = threading.Lock()


def get_tts_service() -> TTSInferenceService:
    """
    Retrieve the process-wide `TTSInferenceService` singleton, creating
    it on first access. Intended for use as a FastAPI dependency.

    Returns:
        TTSInferenceService: The shared inference service instance.
    """
    global _service_instance

    if _service_instance is None:
        with _service_lock:
            if _service_instance is None:
                _service_instance = TTSInferenceService()

    return _service_instance