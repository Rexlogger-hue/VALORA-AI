"""
services/api/schemas.py

Pydantic v2 request/response schemas for the Valora AI API.

Defines the full set of data contracts used by `services/api/routes/tts.py`
and `services/api/main.py`, including request validation, response
serialization, and structured error payloads.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class TTSRequest(BaseModel):
    """
    Request payload for synthesizing speech from text.

    Attributes:
        text: The input text to convert to speech.
        voice_preset: Optional voice/speaker preset identifier.
        max_new_tokens: Optional cap on generated audio tokens.
        language: Optional ISO language hint for logging/metrics.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "text": "Welcome to Valora AI, the future of voice synthesis.",
                "voice_preset": "v2/en_speaker_6",
                "max_new_tokens": None,
                "language": "en",
            }
        }
    )

    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Input text to synthesize into speech.",
        examples=["Welcome to Valora AI, the future of voice synthesis."],
    )
    voice_preset: Optional[str] = Field(
        default=None,
        description="Voice/speaker preset identifier (e.g. 'v2/en_speaker_6'). "
        "Falls back to the server-configured default if omitted.",
        examples=["v2/en_speaker_6"],
    )
    max_new_tokens: Optional[int] = Field(
        default=None,
        gt=0,
        description="Optional cap on generated audio tokens per synthesis chunk.",
        examples=[None],
    )
    language: Optional[str] = Field(
        default=None,
        min_length=2,
        max_length=10,
        description="Optional ISO language hint (e.g. 'en', 'hi', 'fr', 'ja', 'es').",
        examples=["en"],
    )
    age: Optional[int] = Field(
        default=None,
        ge=1,
        le=100,
        description="Optional target apparent age (1-100) for pitch-shifting the voice younger or older.",
        examples=[25],
    )
    pitch_semitones: Optional[float] = Field(
        default=None,
        ge=-12.0,
        le=12.0,
        description="Optional direct pitch shift in semitones for manual control. Overrides `age` if both are set.",
        examples=[None],
    )

    @field_validator("text")
    @classmethod
    def _validate_text_not_blank(cls, value: str) -> str:
        """
        Ensure the text field is not blank after stripping whitespace.

        Args:
            value: The raw text field value.

        Returns:
            str: The validated text value.

        Raises:
            ValueError: If the text is empty or whitespace-only.
        """
        if not value.strip():
            raise ValueError("text must not be empty or whitespace-only.")
        return value

    @field_validator("voice_preset")
    @classmethod
    def _validate_voice_preset_format(cls, value: Optional[str]) -> Optional[str]:
        """
        Ensure the voice preset, if provided, follows the expected
        'v2/{lang}_speaker_{n}' naming convention.

        Args:
            value: The raw voice preset value.

        Returns:
            Optional[str]: The validated voice preset value.

        Raises:
            ValueError: If the preset does not match the expected format.
        """
        if value is None:
            return value
        stripped = value.strip()
        if not stripped:
            raise ValueError("voice_preset must not be blank if provided.")
        return stripped


class TTSResponse(BaseModel):
    """
    Response payload describing the result of a speech synthesis
    request, returned by JSON-based synthesis endpoints.

    Attributes:
        duration_seconds: Total duration of the generated audio.
        sample_rate: Sample rate (Hz) of the generated audio.
        latency_seconds: Wall-clock time taken to synthesize the audio.
        voice_preset: The voice preset actually used for synthesis.
        audio_filename: Filename under which the audio was persisted.
        chunk_count: Number of text chunks synthesized and concatenated.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "duration_seconds": 4.32,
                "sample_rate": 24000,
                "latency_seconds": 1.87,
                "voice_preset": "v2/en_speaker_6",
                "audio_filename": "valora_tts_1732550400000_a1b2c3d4e5f6.wav",
                "chunk_count": 1,
            }
        }
    )

    duration_seconds: float = Field(
        ..., ge=0.0, description="Total duration of the generated audio, in seconds."
    )
    sample_rate: int = Field(
        ..., gt=0, description="Sample rate of the generated audio, in Hz."
    )
    latency_seconds: float = Field(
        ..., ge=0.0, description="Wall-clock time taken to synthesize the audio, in seconds."
    )
    voice_preset: str = Field(
        ..., description="Voice/speaker preset actually used for synthesis."
    )
    audio_filename: str = Field(
        ..., description="Filename under which the generated audio was persisted."
    )
    chunk_count: int = Field(
        default=1, ge=1, description="Number of text chunks synthesized and concatenated."
    )


class HealthResponse(BaseModel):
    """
    Health status of the TTS inference service and underlying model.

    Attributes:
        status: Overall health status ('healthy' or 'degraded').
        is_loaded: Whether the TTS model is currently loaded in memory.
        device: Compute device the model is running on ('cuda' or 'cpu').
        sample_rate: Sample rate (Hz) produced by the loaded model.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "healthy",
                "is_loaded": True,
                "device": "cuda",
                "sample_rate": 24000,
            }
        }
    )

    status: str = Field(..., description="Overall health status: 'healthy' or 'degraded'.")
    is_loaded: bool = Field(..., description="Whether the TTS model is currently loaded.")
    device: str = Field(..., description="Compute device in use ('cuda' or 'cpu').")
    sample_rate: int = Field(..., ge=0, description="Sample rate produced by the loaded model, in Hz.")


class ModelResponse(BaseModel):
    """
    Details about the loaded TTS model and compute device.

    Attributes:
        model_name: HuggingFace model identifier in use.
        cuda_available: Whether a CUDA-capable GPU is available.
        gpu_name: Name of the active GPU device, or 'N/A' on CPU.
        sample_rate: Sample rate (Hz) produced by the loaded model.
        is_loaded: Whether the model is currently loaded in memory.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "model_name": "suno/bark",
                "cuda_available": True,
                "gpu_name": "NVIDIA A100-SXM4-40GB",
                "sample_rate": 24000,
                "is_loaded": True,
            }
        }
    )

    model_name: str = Field(..., description="HuggingFace model identifier in use.")
    cuda_available: bool = Field(..., description="Whether a CUDA-capable GPU is available.")
    gpu_name: str = Field(..., description="Name of the active GPU device, or 'N/A' on CPU.")
    sample_rate: int = Field(..., ge=0, description="Sample rate produced by the loaded model, in Hz.")
    is_loaded: bool = Field(..., description="Whether the model is currently loaded in memory.")


class VoiceResponse(BaseModel):
    """
    Available voice presets grouped by supported language.

    Attributes:
        voices: List of all available voice preset identifiers.
        languages: Mapping of language codes to display names.
        default_voice_preset: The server-configured default voice preset.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "voices": ["v2/en_speaker_0", "v2/en_speaker_6", "v2/hi_speaker_3"],
                "languages": {"en": "English", "hi": "Hindi", "fr": "French"},
                "default_voice_preset": "v2/en_speaker_6",
            }
        }
    )

    voices: List[str] = Field(..., description="List of all available voice preset identifiers.")
    languages: Dict[str, str] = Field(
        ..., description="Mapping of supported language codes to human-readable names."
    )
    default_voice_preset: str = Field(
        ..., description="The server-configured default voice preset."
    )


class MetricsResponse(BaseModel):
    """
    Operational metrics for the TTS service.

    Attributes:
        requests_served: Total number of successfully completed requests.
        average_latency_seconds: Mean synthesis latency across all requests.
        uptime_seconds: Time elapsed since the service started, in seconds.
        model_status: Current model status ('loaded' or 'not_loaded').
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "requests_served": 1024,
                "average_latency_seconds": 1.42,
                "uptime_seconds": 86400.0,
                "model_status": "loaded",
            }
        }
    )

    requests_served: int = Field(..., ge=0, description="Total number of successfully completed requests.")
    average_latency_seconds: float = Field(
        ..., ge=0.0, description="Mean synthesis latency across all served requests, in seconds."
    )
    uptime_seconds: float = Field(..., ge=0.0, description="Time elapsed since the service started, in seconds.")
    model_status: str = Field(..., description="Current model status: 'loaded' or 'not_loaded'.")


class ValidationErrorResponse(BaseModel):
    """
    Structured error payload returned for request validation failures
    and other handled API errors.

    Attributes:
        error: Machine-readable error category.
        detail: Human-readable (or structured) error details.
        path: The request path that produced the error.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "error": "validation_error",
                "detail": "text must not be empty or whitespace-only.",
                "path": "/tts/generate",
            }
        }
    )

    error: str = Field(..., description="Machine-readable error category.")
    detail: str = Field(..., description="Human-readable description of the error.")
    path: str = Field(..., description="The request path that produced the error.")


# --------------------------------------------------------------------- #
# Backward-compatible aliases for services/api/routes/tts.py
# --------------------------------------------------------------------- #

GenerateRequest = TTSRequest
GenerateJSONResponse = TTSResponse
ModelInfoResponse = ModelResponse
VoicesResponse = VoiceResponse