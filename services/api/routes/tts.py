"""
services/api/routes/tts.py

TTS API routes for Valora AI.

Exposes speech synthesis, voice discovery, health, model, and metrics
endpoints under the `/tts` prefix, built on top of
`services/tts/inference.py`, `services/tts/preprocess.py`, and
`services/tts/config.py`.
"""

from __future__ import annotations

import io
import threading
import time
from typing import Dict, List

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from services.api.dependencies import get_inference_service, get_text_preprocessor
from services.api.schemas import (
    GenerateJSONResponse,
    GenerateRequest,
    HealthResponse,
    MetricsResponse,
    ModelInfoResponse,
    VoicesResponse,
)
from services.tts.config import settings
from services.tts.inference import (
    SynthesisRequest,
    TTSInferenceError,
    TTSInferenceService,
)
from services.tts.preprocess import TextPreprocessor, TextValidationError
from services.tts.utils import generate_unique_filename, get_gpu_info, get_logger

logger = get_logger("valora.api.routes.tts")

router = APIRouter(prefix="/tts", tags=["TTS"])

BARK_VOICE_PRESETS: List[str] = [
    "v2/en_speaker_0", "v2/en_speaker_1", "v2/en_speaker_2", "v2/en_speaker_3",
    "v2/en_speaker_4", "v2/en_speaker_5", "v2/en_speaker_6", "v2/en_speaker_7",
    "v2/en_speaker_8", "v2/en_speaker_9",
    "v2/hi_speaker_0", "v2/hi_speaker_1", "v2/hi_speaker_2", "v2/hi_speaker_3",
    "v2/hi_speaker_4", "v2/hi_speaker_5", "v2/hi_speaker_6", "v2/hi_speaker_7",
    "v2/hi_speaker_8", "v2/hi_speaker_9",
    "v2/fr_speaker_0", "v2/fr_speaker_1", "v2/fr_speaker_2", "v2/fr_speaker_3",
    "v2/fr_speaker_4", "v2/fr_speaker_5", "v2/fr_speaker_6", "v2/fr_speaker_7",
    "v2/fr_speaker_8", "v2/fr_speaker_9",
    "v2/ja_speaker_0", "v2/ja_speaker_1", "v2/ja_speaker_2", "v2/ja_speaker_3",
    "v2/ja_speaker_4", "v2/ja_speaker_5", "v2/ja_speaker_6", "v2/ja_speaker_7",
    "v2/ja_speaker_8", "v2/ja_speaker_9",
    "v2/es_speaker_0", "v2/es_speaker_1", "v2/es_speaker_2", "v2/es_speaker_3",
    "v2/es_speaker_4", "v2/es_speaker_5", "v2/es_speaker_6", "v2/es_speaker_7",
    "v2/es_speaker_8", "v2/es_speaker_9",
]

VOICE_LANGUAGE_MAP: Dict[str, str] = {
    "en": "English",
    "hi": "Hindi",
    "fr": "French",
    "ja": "Japanese",
    "es": "Spanish",
}


class _MetricsTracker:
    """
    Thread-safe in-memory tracker for basic request-serving metrics.

    Attributes:
        requests_served: Total number of successfully completed requests.
        total_latency_seconds: Cumulative latency across all requests.
        start_time: Monotonic timestamp when the tracker was created.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.requests_served: int = 0
        self.total_latency_seconds: float = 0.0
        self.start_time: float = time.monotonic()

    def record(self, latency_seconds: float) -> None:
        """
        Record a completed request's latency.

        Args:
            latency_seconds: Time taken to serve the request, in seconds.
        """
        with self._lock:
            self.requests_served += 1
            self.total_latency_seconds += latency_seconds

    def snapshot(self) -> Dict[str, float]:
        """
        Capture a consistent snapshot of current metrics.

        Returns:
            Dict[str, float]: Requests served, average latency, and
                uptime in seconds.
        """
        with self._lock:
            requests_served = self.requests_served
            total_latency = self.total_latency_seconds

        average_latency = (
            total_latency / requests_served if requests_served > 0 else 0.0
        )
        uptime = time.monotonic() - self.start_time

        return {
            "requests_served": requests_served,
            "average_latency_seconds": round(average_latency, 4),
            "uptime_seconds": round(uptime, 2),
        }


_metrics = _MetricsTracker()


def _preprocess_and_join(
    text: str, preprocessor: TextPreprocessor
) -> str:
    """
    Run text through the preprocessing pipeline and rejoin chunks into
    a single normalized string, since chunking is re-applied downstream
    by the inference service.

    Args:
        text: Raw input text.
        preprocessor: The text preprocessor instance.

    Returns:
        str: Normalized text ready for synthesis.

    Raises:
        HTTPException: If the text fails validation.
    """
    try:
        result = preprocessor.preprocess(text)
        return result.normalized_text
    except TextValidationError as exc:
        logger.warning("Text validation failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid input text: {exc}",
        ) from exc


@router.post(
    "/generate",
    summary="Generate speech and stream WAV audio",
    response_class=StreamingResponse,
)
async def generate_speech(
    request: GenerateRequest,
    service: TTSInferenceService = Depends(get_inference_service),
    preprocessor: TextPreprocessor = Depends(get_text_preprocessor),
) -> StreamingResponse:
    """
    Synthesize speech from input text and stream the result as a WAV
    audio response.

    Args:
        request: The synthesis request payload (text, voice preset,
            and optional generation parameters).
        service: Injected TTS inference service.
        preprocessor: Injected text preprocessor.

    Returns:
        StreamingResponse: WAV audio content with `audio/wav` media type.

    Raises:
        HTTPException: If validation fails (400) or synthesis fails (500).
    """
    normalized_text = _preprocess_and_join(request.text, preprocessor)

    synthesis_request = SynthesisRequest(
        text=normalized_text,
        voice_preset=request.voice_preset,
        max_new_tokens=request.max_new_tokens,
        language=request.language,
        age=request.age,
        pitch_semitones=request.pitch_semitones,
    )

    try:
        response = await service.synthesize_async(synthesis_request)
    except TTSInferenceError as exc:
        logger.error("Speech generation failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Speech generation failed: {exc}",
        ) from exc

    _metrics.record(response.latency_seconds)

    filename = generate_unique_filename(prefix="valora_tts", extension="wav")
    audio_stream = io.BytesIO(response.audio_bytes)

    headers = {
        "Content-Disposition": f'attachment; filename="{filename}"',
        "X-Audio-Duration-Seconds": f"{response.duration_seconds:.3f}",
        "X-Sample-Rate": str(response.sample_rate),
        "X-Latency-Seconds": f"{response.latency_seconds:.3f}",
        "X-Voice-Preset": response.voice_preset,
    }

    return StreamingResponse(audio_stream, media_type="audio/wav", headers=headers)


@router.post(
    "/generate-json",
    response_model=GenerateJSONResponse,
    summary="Generate speech and return metadata as JSON",
)
async def generate_speech_json(
    request: GenerateRequest,
    service: TTSInferenceService = Depends(get_inference_service),
    preprocessor: TextPreprocessor = Depends(get_text_preprocessor),
) -> GenerateJSONResponse:
    """
    Synthesize speech from input text and return structured metadata
    without embedding the raw audio bytes in the response body.

    Args:
        request: The synthesis request payload.
        service: Injected TTS inference service.
        preprocessor: Injected text preprocessor.

    Returns:
        GenerateJSONResponse: Synthesis metadata (duration, sample rate,
            latency, voice preset, and a generated audio filename).

    Raises:
        HTTPException: If validation fails (400) or synthesis fails (500).
    """
    normalized_text = _preprocess_and_join(request.text, preprocessor)

    synthesis_request = SynthesisRequest(
        text=normalized_text,
        voice_preset=request.voice_preset,
        max_new_tokens=request.max_new_tokens,
        language=request.language,
    )

    try:
        response = await service.synthesize_async(synthesis_request)
    except TTSInferenceError as exc:
        logger.error("Speech generation failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Speech generation failed: {exc}",
        ) from exc

    _metrics.record(response.latency_seconds)

    filename = generate_unique_filename(prefix="valora_tts", extension="wav")
    output_path = settings.OUTPUT_DIR / filename

    try:
        output_path.write_bytes(response.audio_bytes)
    except OSError as exc:
        logger.error("Failed to persist audio file '%s': %s", output_path, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to persist generated audio file.",
        ) from exc

    return GenerateJSONResponse(
        duration_seconds=response.duration_seconds,
        sample_rate=response.sample_rate,
        latency_seconds=response.latency_seconds,
        voice_preset=response.voice_preset,
        audio_filename=filename,
        chunk_count=response.chunk_count,
    )


@router.get(
    "/voices",
    response_model=VoicesResponse,
    summary="List available voice presets",
)
async def list_voices() -> VoicesResponse:
    """
    List all available Bark voice presets grouped by language.

    Returns:
        VoicesResponse: All supported voice presets and their languages.
    """
    return VoicesResponse(
        voices=BARK_VOICE_PRESETS,
        languages=VOICE_LANGUAGE_MAP,
        default_voice_preset=settings.TTS_DEFAULT_VOICE_PRESET,
    )


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Report TTS inference health",
)
async def tts_health(
    service: TTSInferenceService = Depends(get_inference_service),
) -> HealthResponse:
    """
    Report the health status of the TTS inference service and its
    underlying model.

    Args:
        service: Injected TTS inference service.

    Returns:
        HealthResponse: Structured health information.

    Raises:
        HTTPException: If the health check itself fails unexpectedly.
    """
    try:
        model_health = service.health_check()
    except Exception as exc:
        logger.exception("TTS health check failed.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Health check failed: {exc}",
        ) from exc

    is_healthy = bool(model_health.get("is_loaded", False))

    return HealthResponse(
        status="healthy" if is_healthy else "degraded",
        is_loaded=is_healthy,
        device=model_health.get("device", "unknown"),
        sample_rate=model_health.get("sample_rate", 0),
    )


@router.get(
    "/model",
    response_model=ModelInfoResponse,
    summary="Report model and device information",
)
async def model_info(
    service: TTSInferenceService = Depends(get_inference_service),
) -> ModelInfoResponse:
    """
    Report details about the loaded TTS model and compute device.

    Args:
        service: Injected TTS inference service.

    Returns:
        ModelInfoResponse: Model name, CUDA availability, GPU name,
            sample rate, and load status.
    """
    model_health = service.health_check()
    gpu_info = get_gpu_info()

    cuda_available = bool(gpu_info.get("cuda_available", False))
    gpu_name = "N/A"
    if cuda_available:
        devices = gpu_info.get("devices") or []
        if devices:
            gpu_name = devices[0].get("name", "Unknown GPU")

    return ModelInfoResponse(
        model_name=model_health.get("model_name", settings.TTS_MODEL_NAME),
        cuda_available=cuda_available,
        gpu_name=gpu_name,
        sample_rate=model_health.get("sample_rate", 0),
        is_loaded=bool(model_health.get("is_loaded", False)),
    )


@router.get(
    "/metrics",
    response_model=MetricsResponse,
    summary="Report service metrics",
)
async def metrics(
    service: TTSInferenceService = Depends(get_inference_service),
) -> MetricsResponse:
    """
    Report operational metrics for the TTS service, including request
    volume, average latency, uptime, and model status.

    Args:
        service: Injected TTS inference service.

    Returns:
        MetricsResponse: Aggregated service metrics.
    """
    snapshot = _metrics.snapshot()
    model_health = service.health_check()

    return MetricsResponse(
        requests_served=snapshot["requests_served"],
        average_latency_seconds=snapshot["average_latency_seconds"],
        uptime_seconds=snapshot["uptime_seconds"],
        model_status="loaded" if model_health.get("is_loaded", False) else "not_loaded",
    )