"""Voice input/output module for Particle.

Text-to-Speech:
  Uses ``kokoro-onnx`` (Kokoro TTS, runs fully locally via ONNX runtime).
  Falls back to printing text to stdout if the package is unavailable.

Speech-to-Text:
  Uses Voxtral via the Hugging Face Inference API (requires HF token).
  Input audio can be recorded from the default microphone via ``sounddevice``
  or provided as a file path.

All heavy imports are guarded so the module loads cleanly even when the
optional audio packages are not installed.
"""

from __future__ import annotations

import logging
import os
import tempfile
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import requests

from modules.config_loader import get_config

logger = logging.getLogger("particle.voice")

if TYPE_CHECKING:
    import numpy as np

# ---------------------------------------------------------------------------
# Optional imports
# ---------------------------------------------------------------------------

try:
    import numpy as _np  # type: ignore
    _NUMPY_AVAILABLE = True
except ImportError:
    _np = None  # type: ignore
    _NUMPY_AVAILABLE = False

try:
    import sounddevice as sd  # type: ignore
    _SOUNDDEVICE_AVAILABLE = True
except ImportError:
    _SOUNDDEVICE_AVAILABLE = False
    logger.warning("sounddevice not installed — microphone recording unavailable")

try:
    from kokoro_onnx import Kokoro  # type: ignore
    _KOKORO_AVAILABLE = True
except ImportError:
    _KOKORO_AVAILABLE = False
    logger.warning("kokoro-onnx not installed — TTS will fall back to text output")

_SAMPLE_RATE = 24000   # kokoro default output sample rate
_MIC_SAMPLE_RATE = 16000  # STT expects 16 kHz
_DEFAULT_VOICE = "af_heart"


class VoiceEngine:
    """Local TTS (kokoro-onnx) and STT (Voxtral) engine."""

    def __init__(self) -> None:
        cfg = get_config()
        voice_cfg = cfg.voice
        self._enabled: bool = bool(getattr(voice_cfg, "enabled", True))
        self._voice_engine: str = str(getattr(voice_cfg, "voice_engine", "kokoro")).lower()
        self._stt_ready = False
        self._lock = threading.Lock()
        self._kokoro: Optional[object] = None

        if self._enabled:
            self._init_kokoro()
            self._init_voxtral()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_kokoro(self) -> None:
        if not _KOKORO_AVAILABLE:
            return
        try:
            self._kokoro = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")
            logger.info("Kokoro TTS engine initialised")
        except Exception as exc:
            logger.error("Failed to initialise Kokoro TTS: %s", exc)
            self._kokoro = None

    def _init_voxtral(self) -> None:
        cfg = get_config()
        self._hf_token = str(getattr(cfg.llm, "hf_token", ""))
        self._hf_url = "https://api-inference.huggingface.co/models/mistralai/Voxtral-Mini-3B-2507"
        self._hf_headers = {"Authorization": f"Bearer {self._hf_token}"}
        if not self._hf_token:
            logger.warning("HF_TOKEN missing — Voxtral STT unavailable")
            self._stt_ready = False
            return
        self._stt_ready = True
        logger.info("HF Voxtral STT initialised")

    # ------------------------------------------------------------------
    # Text-to-Speech
    # ------------------------------------------------------------------

    def speak(self, text: str, voice: str = _DEFAULT_VOICE, speed: float = 1.0) -> bool:
        """Synthesise and play *text* via the configured TTS engine.

        Returns True on success, False if TTS is unavailable.
        """
        if self._voice_engine == "voxtral":
            if not self._enabled:
                logger.debug("Voice disabled — skipping TTS for: %s", text[:80])
                return False
            if self.speak_voxtral(text):
                return True
        return self._speak_kokoro(text, voice, speed)

    def _speak_kokoro(self, text: str, voice: str, speed: float) -> bool:
        if not self._enabled:
            logger.debug("Voice disabled — skipping TTS for: %s", text[:80])
            return False

        if self._kokoro is None:
            print(f"[Particle TTS] {text}")
            return False

        if not _SOUNDDEVICE_AVAILABLE:
            logger.warning("sounddevice unavailable — cannot play audio")
            return False

        try:
            with self._lock:
                samples, sample_rate = self._kokoro.create(  # type: ignore[union-attr]
                    text, voice=voice, speed=speed, lang="en-us"
                )
            sd.play(samples, samplerate=sample_rate)
            sd.wait()
            logger.debug("TTS playback complete for %d chars", len(text))
            return True
        except Exception as exc:
            logger.error("TTS speak error: %s", exc)
            return False

    def speak_voxtral(self, text: str) -> bool:
        """Synthesise and play *text* via Voxtral."""
        logger.warning(
            "Voxtral TTS not available on HF free tier, falling back to Kokoro"
        )
        return self._speak_kokoro(text, _DEFAULT_VOICE, 1.0)

    def synthesize_to_file(
        self,
        text: str,
        output_path: "str | Path",
        voice: str = _DEFAULT_VOICE,
        speed: float = 1.0,
    ) -> bool:
        """Synthesise *text* and write audio to *output_path* (.wav).

        Returns True on success.
        """
        if self._kokoro is None:
            logger.warning("Kokoro TTS unavailable — cannot synthesize to file")
            return False

        try:
            import soundfile as sf  # type: ignore
        except ImportError:
            logger.error("soundfile not installed — cannot write audio file")
            return False

        try:
            with self._lock:
                samples, sample_rate = self._kokoro.create(  # type: ignore[union-attr]
                    text, voice=voice, speed=speed, lang="en-us"
                )
            sf.write(str(output_path), samples, sample_rate)
            logger.info("TTS audio written to %s", output_path)
            return True
        except Exception as exc:
            logger.error("synthesize_to_file error: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Speech-to-Text
    # ------------------------------------------------------------------

    def transcribe_file(self, audio_path: "str | Path") -> str:
        """Transcribe an audio file using Voxtral.

        Returns the transcript string, or empty string on failure.
        """
        if not self._stt_ready:
            return ""

        try:
            audio_path = Path(audio_path)
            with audio_path.open("rb") as handle:
                response = requests.post(
                    self._hf_url,
                    headers=self._hf_headers,
                    data=handle.read(),
                    timeout=60,
                )
            response.raise_for_status()
            payload = response.json()
            if isinstance(payload, dict):
                return str(payload.get("text", "")).strip()
            if isinstance(payload, list):
                if not payload:
                    return ""
                entry = payload[0]
                if isinstance(entry, dict):
                    return str(entry.get("text", "")).strip()
            return ""
        except Exception as exc:
            logger.error("transcribe_file error: %s", exc)
            return ""

    def record_and_transcribe(self, duration_seconds: float = 5.0) -> str:
        """Record from the default microphone and transcribe with Voxtral.

        Returns the transcript string, or empty string on failure.
        """
        if not _SOUNDDEVICE_AVAILABLE:
            logger.warning("sounddevice unavailable — cannot record")
            return ""
        if not self._stt_ready:
            logger.warning("Voxtral STT unavailable — cannot transcribe recording")
            return ""

        logger.info("Recording %.1fs of audio at %dHz…", duration_seconds, _MIC_SAMPLE_RATE)
        try:
            audio = sd.rec(
                int(duration_seconds * _MIC_SAMPLE_RATE),
                samplerate=_MIC_SAMPLE_RATE,
                channels=1,
                dtype="float32",
            )
            sd.wait()
        except Exception as exc:
            logger.error("Audio recording error: %s", exc)
            return ""

        # Write to temp WAV and transcribe
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            import soundfile as sf  # type: ignore
            sf.write(tmp_path, audio.squeeze(), _MIC_SAMPLE_RATE)
            return self.transcribe_file(tmp_path)
        except ImportError:
            logger.error("soundfile not installed — cannot write temp audio")
            return ""
        except Exception as exc:
            logger.error("record_and_transcribe write error: %s", exc)
            return ""
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def transcribe_audio_array(self, audio: object, sample_rate: int = _MIC_SAMPLE_RATE) -> str:
        """Transcribe a NumPy float32 audio array using Voxtral.

        Useful when audio is already in memory (e.g. from meeting recorder).
        """
        if not self._stt_ready:
            return ""
        if not _NUMPY_AVAILABLE or _np is None:
            logger.warning("numpy unavailable — cannot transcribe audio array")
            return ""

        try:
            import soundfile as sf  # type: ignore
        except ImportError:
            logger.error("soundfile not installed — cannot write temp audio")
            return ""

        arr = _np.asarray(audio, dtype="float32")
        if arr.ndim > 1:
            arr = arr.mean(axis=1)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            sf.write(tmp_path, arr, sample_rate)
            return self.transcribe_file(tmp_path)
        except Exception as exc:
            logger.error("transcribe_audio_array error: %s", exc)
            return ""
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_instance: Optional[VoiceEngine] = None
_singleton_lock = threading.Lock()


def get_voice_engine() -> VoiceEngine:
    """Return the module-level :class:`VoiceEngine` singleton."""
    global _instance
    with _singleton_lock:
        if _instance is None:
            _instance = VoiceEngine()
    return _instance


def speak(text: str) -> bool:
    """Convenience function: speak *text* via the global engine."""
    return get_voice_engine().speak(text)


def transcribe(audio_path: "str | Path") -> str:
    """Convenience function: transcribe *audio_path* via the global engine."""
    return get_voice_engine().transcribe_file(audio_path)
