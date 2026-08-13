"""
services/tts/model.py

Production-grade Text-to-Speech (TTS) model wrapper for Valora AI.

This version uses Microsoft Edge's neural TTS voices (via the `edge-tts`
library) as the synthesis backend. This is a hosted-API backend, not a
local GPU model -- it requires no CUDA, no local model weights, and no
device placement, which eliminates the entire class of CUDA/CPU device
mismatch issues that local Bark inference was subject to.

This version also adds age-based (or manually specified) pitch shifting
via librosa, so a single synthesized voice can be made to sound younger
or older on demand -- a deterministic DSP transform, not a trained model,
so it works instantly with every voice/language edge-tts supports.

Public interface (class name, method signatures, exception types) is
kept backward compatible with the previous local-Bark implementation:
services/tts/inference.py and everything in services/api/ require NO
changes for existing calls; new `age`/`pitch_semitones` parameters are
additive and optional.
"""

from __future__ import annotations

import asyncio
import io
import logging
import threading
import wave
from dataclasses import dataclass
from typing import Dict, Optional

import librosa
import numpy as np
from pydub import AudioSegment

try:
    import edge_tts
except ImportError as _import_exc:  # pragma: no cover
    edge_tts = None
    _EDGE_TTS_IMPORT_ERROR = _import_exc
else:
    _EDGE_TTS_IMPORT_ERROR = None

logger = logging.getLogger("valora.tts.model")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


class TTSModelError(Exception):
    """Base exception for all TTS model related errors."""


class TTSModelLoadError(TTSModelError):
    """Raised when the TTS backend fails to initialize."""


class TTSSynthesisError(TTSModelError):
    """Raised when speech synthesis fails."""


class TTSDeviceMismatchError(TTSSynthesisError):
    """
    Kept for backward compatibility with callers that catch this
    exception type. Not raised by this hosted-API backend, since there
    is no local device placement involved.
    """


@dataclass(frozen=True)
class SynthesisResult:
    """
    Container for a synthesized speech result.

    Attributes:
        audio: 1-D numpy float32 array of waveform samples, range [-1, 1].
        sample_rate: Sample rate of the audio in Hz.
    """

    audio: np.ndarray
    sample_rate: int

    def to_wav_bytes(self) -> bytes:
        """
        Encode the audio waveform as WAV-formatted bytes.

        Uses Python's built-in `wave` module rather than torchaudio.save(),
        since newer torchaudio releases require the optional `torchcodec`
        backend for WAV encoding, which is not installed. The stdlib
        `wave` module needs no extra dependencies and is a natural fit
        here since we already have PCM sample data.

        Returns:
            bytes: In-memory WAV file content.

        Raises:
            TTSSynthesisError: If encoding fails.
        """
        try:
            # Convert float32 samples in [-1, 1] to 16-bit PCM integers.
            clipped = np.clip(self.audio, -1.0, 1.0)
            pcm_samples = (clipped * 32767.0).astype(np.int16)

            buffer = io.BytesIO()
            with wave.open(buffer, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)  # 16-bit = 2 bytes per sample
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(pcm_samples.tobytes())

            buffer.seek(0)
            return buffer.read()
        except Exception as exc:
            raise TTSSynthesisError(f"Failed to encode audio to WAV: {exc}") from exc


def map_age_to_semitones(age: int) -> float:
    """
    Map a human age (in years) to a pitch-shift amount in semitones,
    used to make a synthesized voice sound younger or older.

    Args:
        age: Target apparent age, roughly 1-100.

    Returns:
        float: Semitone shift to apply (positive = higher/younger,
            negative = lower/older). 0.0 at age 28 (neutral baseline).
    """
    age = max(1, min(100, age))
    baseline_age = 28

    if age <= baseline_age:
        # Younger than baseline: shift up, capped at +9 semitones for
        # very young children.
        fraction = (baseline_age - age) / (baseline_age - 1)
        return fraction * 9.0

    # Older than baseline: shift down, capped at -7 semitones for
    # elderly voices.
    fraction = (age - baseline_age) / (100 - baseline_age)
    return -fraction * 7.0


# Curated list of high-quality, natural-sounding Edge neural voices,
# grouped by language/accent. Full catalog can be listed with:
#   edge-tts --list-voices
VOICE_PRESETS: Dict[str, str] = {
    # English -- various accents
    "v2/en_speaker_6": "en-US-AriaNeural",
    "v2/en_speaker_0": "en-US-GuyNeural",
    "v2/en_speaker_1": "en-US-JennyNeural",
    "v2/en_speaker_2": "en-GB-SoniaNeural",
    "v2/en_speaker_3": "en-GB-RyanNeural",
    "v2/en_speaker_4": "en-AU-NatashaNeural",
    "v2/en_speaker_5": "en-IN-NeerjaNeural",
    "v2/en_in_speaker_0": "en-IN-NeerjaNeural",
    "v2/en_in_speaker_1": "en-IN-PrabhatNeural",

    # Hindi
    "v2/hi_speaker_0": "hi-IN-SwaraNeural",
    "v2/hi_speaker_1": "hi-IN-MadhurNeural",
    "v2/hi_speaker_2": "hi-IN-SwaraNeural",
    "v2/hi_speaker_3": "hi-IN-MadhurNeural",

    # Other Indian languages
    "v2/ta_speaker_0": "ta-IN-PallaviNeural",
    "v2/ta_speaker_1": "ta-IN-ValluvarNeural",
    "v2/te_speaker_0": "te-IN-ShrutiNeural",
    "v2/te_speaker_1": "te-IN-MohanNeural",
    "v2/mr_speaker_0": "mr-IN-AarohiNeural",
    "v2/mr_speaker_1": "mr-IN-ManoharNeural",
    "v2/bn_speaker_0": "bn-IN-TanishaaNeural",
    "v2/bn_speaker_1": "bn-IN-BashkarNeural",
    "v2/gu_speaker_0": "gu-IN-DhwaniNeural",
    "v2/gu_speaker_1": "gu-IN-NiranjanNeural",
    "v2/kn_speaker_0": "kn-IN-SapnaNeural",
    "v2/kn_speaker_1": "kn-IN-GaganNeural",
    "v2/ml_speaker_0": "ml-IN-SobhanaNeural",
    "v2/ml_speaker_1": "ml-IN-MidhunNeural",
    "v2/pa_speaker_0": "pa-IN-OjasNeural",
    "v2/pa_speaker_1": "pa-IN-VaaniNeural",

    # French
    "v2/fr_speaker_0": "fr-FR-DeniseNeural",
    "v2/fr_speaker_1": "fr-FR-HenriNeural",

    # Japanese
    "v2/ja_speaker_0": "ja-JP-NanamiNeural",
    "v2/ja_speaker_1": "ja-JP-KeitaNeural",

    # Spanish
    "v2/es_speaker_0": "es-ES-ElviraNeural",
    "v2/es_speaker_1": "es-ES-AlvaroNeural",
}


class TextToSpeechModel:
    """
    Reusable, production-ready wrapper around Microsoft Edge's neural
    TTS voices for the Valora AI voice platform.

    This class handles:
        - Async-to-sync bridging (edge_tts is asyncio-native; our
          synthesize() interface is synchronous, matching the previous
          local-model contract used throughout the codebase).
        - Voice preset resolution (maps Bark-style preset names used
          elsewhere in the codebase, e.g. "v2/en_speaker_6", to Edge
          neural voice names, e.g. "en-US-AriaNeural").
        - MP3 -> WAV/PCM conversion via pydub+ffmpeg, since edge_tts
          returns MP3-encoded audio and the rest of the pipeline
          (SynthesisResult, WAV streaming) expects a numpy waveform.
        - Age-based or manual pitch shifting via librosa, so any voice
          can be made to sound younger or older on demand.
        - Robust error handling and structured logging.

    Example:
        tts = TextToSpeechModel()
        tts.load()
        result = tts.synthesize("Hello, welcome to Valora AI.", age=10)
        wav_bytes = result.to_wav_bytes()
    """

    DEFAULT_MODEL_NAME: str = "edge-tts"
    DEFAULT_VOICE_PRESET: str = "v2/en_speaker_6"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: Optional[str] = None,
        use_fp16: bool = True,
    ) -> None:
        """
        Initialize the TTS model wrapper.

        Args:
            model_name: Kept for interface compatibility; unused (no
                local model weights are loaded for this backend).
            device: Kept for interface compatibility; unused (synthesis
                happens via a remote API call, not local compute).
            use_fp16: Kept for interface compatibility; unused.
        """
        self.model_name: str = model_name
        # Kept as attributes so health_check()/callers that read
        # `.device` don't break; always reports "api" since there is no
        # local compute device involved in this backend.
        self.device: str = "api"
        self.use_fp16: bool = False

        self._sample_rate: int = 24000
        self._is_loaded: bool = False
        self._load_lock = threading.Lock()

        logger.info("TextToSpeechModel initialized (backend=edge-tts, model=%s)", self.model_name)

    @property
    def is_loaded(self) -> bool:
        """Whether the TTS backend has been verified available."""
        return self._is_loaded

    @property
    def sample_rate(self) -> int:
        """Sample rate (Hz) of audio produced by this backend."""
        return self._sample_rate

    def load(self) -> None:
        """
        Verify the edge-tts backend is available. No model weights are
        downloaded or loaded -- this simply confirms the `edge_tts`
        package is importable and the async bridge functions correctly.

        Raises:
            TTSModelLoadError: If the edge_tts package is unavailable or
                a connectivity check fails.
        """
        if self._is_loaded:
            logger.debug("Backend already verified; skipping re-check.")
            return

        with self._load_lock:
            if self._is_loaded:
                return

            logger.info("Initializing TTS backend 'edge-tts'...")

            if edge_tts is None:
                raise TTSModelLoadError(
                    f"edge_tts package is not installed: {_EDGE_TTS_IMPORT_ERROR}. "
                    f"Install it with: pip install edge-tts"
                )

            try:
                # Lightweight connectivity check: synthesize one short
                # word and confirm we get audio bytes back.
                self._run_async(self._synthesize_mp3("Ready.", VOICE_PRESETS[self.DEFAULT_VOICE_PRESET]))
                self._is_loaded = True
                logger.info(
                    "TTS backend 'edge-tts' verified successfully (sample_rate=%d Hz).",
                    self._sample_rate,
                )
            except Exception as exc:
                self._is_loaded = False
                logger.exception("Failed to verify TTS backend 'edge-tts'.")
                raise TTSModelLoadError(f"Failed to initialize edge-tts backend: {exc}") from exc

    def unload(self) -> None:
        """
        Reset the backend's loaded state. No local resources to release
        for this API-based backend, but kept for interface compatibility.
        """
        with self._load_lock:
            self._is_loaded = False
            logger.info("TTS backend 'edge-tts' marked as unloaded.")

    @staticmethod
    def _run_async(coro):
        """
        Run an async coroutine from synchronous code, safely, even when
        called from a thread that already has a running event loop (as
        is the case when this is invoked from within FastAPI's async
        lifespan/startup handler, which runs on uvicorn's event-loop
        thread).

        WHY A SEPARATE THREAD: asyncio tracks "is a loop currently
        running" per OS thread, not per loop object. If the calling
        thread already has a running loop (e.g. uvicorn's loop during
        startup), neither asyncio.run() nor loop.run_until_complete()
        can start a second loop on that same thread -- both raise
        RuntimeError regardless of which loop instance is used. Running
        the coroutine on a brand-new thread (which has no loop of its
        own) avoids this conflict entirely, at the cost of a small
        thread-creation overhead per call.

        Args:
            coro: The coroutine to run to completion.

        Returns:
            Any: The coroutine's return value.

        Raises:
            Exception: Re-raises whatever exception the coroutine itself
                raised, preserving the original traceback context.
        """
        result_box: dict = {}

        def _runner() -> None:
            try:
                result_box["result"] = asyncio.run(coro)
            except BaseException as exc:  # noqa: BLE001
                result_box["error"] = exc

        thread = threading.Thread(target=_runner, daemon=True)
        thread.start()
        thread.join()

        if "error" in result_box:
            raise result_box["error"]
        return result_box.get("result")

    @staticmethod
    async def _synthesize_mp3(text: str, edge_voice: str) -> bytes:
        """
        Call the edge-tts API and collect the resulting MP3 audio bytes.

        Args:
            text: Text to synthesize.
            edge_voice: Edge neural voice identifier (e.g. 'en-US-AriaNeural').

        Returns:
            bytes: Raw MP3-encoded audio.

        Raises:
            TTSSynthesisError: If no audio data is returned.
        """
        communicator = edge_tts.Communicate(text=text, voice=edge_voice)
        audio_chunks = bytearray()

        async for chunk in communicator.stream():
            if chunk["type"] == "audio":
                audio_chunks.extend(chunk["data"])

        if not audio_chunks:
            raise TTSSynthesisError("edge-tts returned no audio data.")

        return bytes(audio_chunks)

    def _resolve_voice(self, preset: str) -> str:
        """
        Resolve a Bark-style voice preset name to an Edge neural voice
        identifier, falling back to the default voice with a warning if
        the preset is unrecognized.

        Args:
            preset: Voice preset identifier (e.g. 'v2/en_speaker_6').

        Returns:
            str: The corresponding Edge neural voice name.
        """
        if preset in VOICE_PRESETS:
            return VOICE_PRESETS[preset]

        logger.warning(
            "Unrecognized voice preset '%s'; falling back to default preset '%s'.",
            preset,
            self.DEFAULT_VOICE_PRESET,
        )
        return VOICE_PRESETS[self.DEFAULT_VOICE_PRESET]

    def synthesize(
        self,
        text: str,
        voice_preset: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        age: Optional[int] = None,
        pitch_semitones: Optional[float] = None,
    ) -> SynthesisResult:
        """
        Synthesize speech audio from input text using edge-tts.

        Args:
            text: The input text to convert to speech. Must be non-empty.
            voice_preset: Optional voice preset identifier (Bark-style
                naming, e.g. 'v2/en_speaker_6'). Falls back to the
                default preset if not provided or unrecognized.
            max_new_tokens: Unused for this backend (kept for interface
                compatibility with local-model callers).
            age: Optional target apparent age (1-100). If provided, the
                synthesized voice's pitch is shifted to sound younger or
                older, via map_age_to_semitones(). Ignored if
                pitch_semitones is also provided.
            pitch_semitones: Optional direct pitch shift in semitones,
                for advanced/manual control. Takes precedence over `age`
                if both are provided.

        Returns:
            SynthesisResult: Waveform and sample rate of the generated speech.

        Raises:
            ValueError: If the input text is empty or invalid.
            TTSModelLoadError: If the backend is not loaded and auto-load fails.
            TTSSynthesisError: If synthesis fails for any reason.
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Input 'text' must be a non-empty string.")

        if not self._is_loaded:
            logger.warning("Backend not loaded prior to synthesize(); loading now.")
            self.load()

        preset = voice_preset or self.DEFAULT_VOICE_PRESET
        edge_voice = self._resolve_voice(preset)

        try:
            mp3_bytes = self._run_async(self._synthesize_mp3(text, edge_voice))

            # Convert MP3 -> PCM waveform via pydub (requires ffmpeg on PATH).
            segment = AudioSegment.from_file(io.BytesIO(mp3_bytes), format="mp3")
            segment = segment.set_channels(1).set_frame_rate(self._sample_rate)

            samples = np.array(segment.get_array_of_samples(), dtype=np.float32)
            # 16-bit PCM range normalization to [-1, 1]
            samples = samples / 32768.0

            # Apply age-based (or manual) pitch shifting, if requested.
            # This is a deterministic DSP transform (librosa's phase-vocoder
            # based pitch shift), not a neural model -- it works instantly
            # on any voice/language edge-tts produces, with no training
            # and no GPU required.
            shift_semitones = pitch_semitones if pitch_semitones is not None else (
                map_age_to_semitones(age) if age is not None else None
            )
            if shift_semitones is not None and abs(shift_semitones) > 0.01:
                samples = librosa.effects.pitch_shift(
                    y=samples,
                    sr=self._sample_rate,
                    n_steps=shift_semitones,
                )
                logger.info("Applied pitch shift of %.2f semitones (age=%s).", shift_semitones, age)

            logger.info(
                "Synthesized %.2f seconds of audio for text of length %d chars (voice=%s).",
                len(samples) / float(self._sample_rate),
                len(text),
                edge_voice,
            )

            return SynthesisResult(audio=samples.astype(np.float32), sample_rate=self._sample_rate)

        except (ValueError, TTSModelLoadError):
            raise
        except Exception as exc:
            logger.exception("TTS synthesis failed for input text of length %d.", len(text))
            raise TTSSynthesisError(f"TTS synthesis failed: {exc}") from exc

    def synthesize_to_wav_bytes(
        self,
        text: str,
        voice_preset: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        age: Optional[int] = None,
        pitch_semitones: Optional[float] = None,
    ) -> bytes:
        """
        Convenience method: synthesize speech and return WAV-encoded bytes.

        Args:
            text: The input text to convert to speech.
            voice_preset: Optional voice preset identifier.
            max_new_tokens: Unused for this backend.
            age: Optional target apparent age (1-100) for pitch shifting.
            pitch_semitones: Optional direct pitch shift in semitones.

        Returns:
            bytes: WAV-encoded audio content.
        """
        result = self.synthesize(
            text=text,
            voice_preset=voice_preset,
            max_new_tokens=max_new_tokens,
            age=age,
            pitch_semitones=pitch_semitones,
        )
        return result.to_wav_bytes()

    def health_check(self) -> dict:
        """
        Report the current health/status of the TTS backend.

        Returns:
            dict: Status information suitable for a FastAPI health endpoint.
        """
        return {
            "model_name": self.model_name,
            "is_loaded": self._is_loaded,
            "device": self.device,
            "cuda_available": False,
            "sample_rate": self._sample_rate,
            "backend": "edge-tts (hosted)",
        }

    def __repr__(self) -> str:
        return f"TextToSpeechModel(backend='edge-tts', is_loaded={self._is_loaded})"