"""
services/stt/inference.py

Singleton accessor for the STT service, following the same pattern as
services/tts/inference.py.
"""

from __future__ import annotations

import threading
from typing import Optional

from services.stt.model import SpeechToTextModel

_stt_instance: Optional[SpeechToTextModel] = None
_stt_lock = threading.Lock()


def get_stt_service() -> SpeechToTextModel:
    """Retrieve the process-wide SpeechToTextModel singleton."""
    global _stt_instance
    if _stt_instance is None:
        with _stt_lock:
            if _stt_instance is None:
                _stt_instance = SpeechToTextModel()
    return _stt_instance