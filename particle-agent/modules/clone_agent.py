"""Persona cloning module for Particle meetings.

Full pipeline:
  HF Voxtral STT → Gemini/OpenRouter LLM → ElevenLabs TTS (your cloned voice)
                                        → Deep-Live-Cam (your face, local)

Setup requirements:
- Get HF_TOKEN from huggingface.co (free)
- Clone your voice on elevenlabs.io (free tier), get voice ID
- Save front-facing photo as assets/my_face.jpg
- git clone https://github.com/hacksider/Deep-Live-Cam
- Set clone.enabled: true in config.yaml
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional

from modules.config_loader import get_config

logger = logging.getLogger("particle.clone_agent")

# ---------------------------------------------------------------------------
# Optional imports
# ---------------------------------------------------------------------------

try:
    import requests  # type: ignore
    _REQUESTS_AVAILABLE = True
except ImportError:
    requests = None  # type: ignore
    _REQUESTS_AVAILABLE = False
    logger.warning("requests not installed — clone agent HTTP calls unavailable")

try:
    import numpy as _np  # type: ignore
    _NUMPY_AVAILABLE = True
except ImportError:
    _np = None  # type: ignore
    _NUMPY_AVAILABLE = False
    logger.warning("numpy not installed — audio array transcription unavailable")

try:
    import sounddevice as _sd  # type: ignore
    _SOUNDDEVICE_AVAILABLE = True
except ImportError:
    _sd = None  # type: ignore
    _SOUNDDEVICE_AVAILABLE = False
    logger.warning("sounddevice not installed — audio playback unavailable")

try:
    import soundfile as _sf  # type: ignore
    _SOUNDFILE_AVAILABLE = True
except ImportError:
    _sf = None  # type: ignore
    _SOUNDFILE_AVAILABLE = False
    logger.warning("soundfile not installed — audio file decoding unavailable")


def _cfg_get(section: object, key: str, default: object) -> object:
    if section is None:
        return default
    if isinstance(section, dict):
        return section.get(key, default)
    return getattr(section, key, default)


class CloneAgent:
    """Real-time persona cloning for meetings."""

    def __init__(self) -> None:
        cfg = get_config()
        clone_cfg = getattr(cfg, "clone", None)
        voice_cfg = getattr(cfg, "voice", None)
        llm_cfg = getattr(cfg, "llm", None)

        self._enabled: bool = bool(_cfg_get(clone_cfg, "enabled", False))
        self._voice_sample = self._resolve_path(
            str(_cfg_get(clone_cfg, "voice_sample", "assets/my_voice.wav"))
        )
        self._face_photo = self._resolve_path(
            str(_cfg_get(clone_cfg, "face_photo", "assets/my_face.jpg"))
        )
        self._deep_live_cam_dir = self._resolve_path(
            str(_cfg_get(clone_cfg, "deep_live_cam_dir", "Deep-Live-Cam"))
        )
        self._execution_provider: str = str(_cfg_get(clone_cfg, "execution_provider", "cpu"))
        self._language: str = str(_cfg_get(clone_cfg, "language", "en"))
        self._voice_engine: str = str(_cfg_get(clone_cfg, "voice_engine", "elevenlabs"))
        self._hf_token: str = str(_cfg_get(llm_cfg, "hf_token", ""))
        self._eleven_key: str = str(_cfg_get(voice_cfg, "elevenlabs_api_key", ""))
        self._eleven_voice_id: str = str(_cfg_get(voice_cfg, "elevenlabs_voice_id", ""))

        self._hf_api_url: str = ""
        self._hf_headers: dict[str, str] = {}
        self._stt_ready = False
        self._tts_ready = False

        self._lock = threading.Lock()
        self._dlc_proc: Optional[subprocess.Popen] = None

        if self._enabled:
            self._init_stt()
            self._init_tts()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _resolve_path(self, path_value: str) -> Path:
        path = Path(path_value).expanduser()
        if not path.is_absolute():
            base_dir = Path(__file__).resolve().parent.parent
            path = (base_dir / path).resolve()
        return path

    def _init_stt(self) -> None:
        self._hf_api_url = (
            "https://api-inference.huggingface.co/models/mistralai/Voxtral-Mini-3B-2507"
        )
        self._hf_headers = {"Authorization": f"Bearer {self._hf_token}"}
        if not self._hf_token:
            logger.warning("HF token missing — Voxtral STT unavailable")
            self._stt_ready = False
            return
        self._stt_ready = True

    def _init_tts(self) -> None:
        if not self._eleven_key:
            logger.warning("ElevenLabs API key missing — TTS unavailable")
            self._tts_ready = False
            return
        if not self._eleven_voice_id:
            logger.warning(
                "ElevenLabs voice ID missing — get your voice ID from elevenlabs.io after cloning your voice"
            )
            self._tts_ready = False
            return
        self._tts_ready = True

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def available(self) -> bool:
        return self._enabled and self._stt_ready and self._tts_ready

    # ------------------------------------------------------------------
    # Face cloning
    # ------------------------------------------------------------------

    def start_face_clone(self) -> bool:
        if not self._face_photo.exists():
            logger.warning("Face photo not found at %s", self._face_photo)
            return False

        run_path = self._deep_live_cam_dir / "run.py"
        if not run_path.exists():
            logger.warning(
                "Deep-Live-Cam not found at %s. Clone it with: git clone https://github.com/hacksider/Deep-Live-Cam",
                self._deep_live_cam_dir,
            )
            return False

        try:
            proc = subprocess.Popen(
                [
                    "python",
                    str(run_path),
                    "--source",
                    str(self._face_photo),
                    "--execution-provider",
                    self._execution_provider,
                    "--live",
                ],
                cwd=str(self._deep_live_cam_dir),
            )
            time.sleep(3)
            if proc.poll() is not None:
                logger.warning("Deep-Live-Cam exited early with code %s", proc.returncode)
                return False
            self._dlc_proc = proc
            logger.info("Deep-Live-Cam running (pid=%s)", proc.pid)
            return True
        except Exception as exc:
            logger.error("Failed to start Deep-Live-Cam: %s", exc)
            return False

    def stop_face_clone(self) -> None:
        proc = self._dlc_proc
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        self._dlc_proc = None
        logger.info("Deep-Live-Cam stopped")

    # ------------------------------------------------------------------
    # Speech-to-Text
    # ------------------------------------------------------------------

    def transcribe(self, audio_path: str | Path) -> str:
        if not self._stt_ready:
            return ""
        if not _REQUESTS_AVAILABLE or requests is None:
            logger.warning("requests unavailable — cannot call Voxtral STT")
            return ""

        try:
            audio_path = Path(audio_path)
            with audio_path.open("rb") as handle:
                resp = requests.post(
                    self._hf_api_url,
                    headers=self._hf_headers,
                    data=handle,
                    timeout=60,
                )
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, dict):
                return str(data.get("text", "")).strip()
            if isinstance(data, list) and data and isinstance(data[0], dict):
                return str(data[0].get("text", "")).strip()
            return ""
        except Exception as exc:
            logger.error("STT transcription error: %s", exc)
            return ""

    def transcribe_array(self, audio: object, sample_rate: int = 16000) -> str:
        if not _NUMPY_AVAILABLE or _np is None:
            logger.warning("numpy unavailable — cannot transcribe audio array")
            return ""
        if not _SOUNDFILE_AVAILABLE or _sf is None:
            logger.warning("soundfile unavailable — cannot write temp WAV")
            return ""

        arr = _np.asarray(audio, dtype="float32")
        if arr.ndim > 1:
            arr = arr.mean(axis=1)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            _sf.write(tmp_path, arr, sample_rate)
            return self.transcribe(tmp_path)
        except Exception as exc:
            logger.error("Audio array transcription error: %s", exc)
            return ""
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    # ------------------------------------------------------------------
    # Text-to-Speech
    # ------------------------------------------------------------------

    def speak_as_me(self, text: str, play: bool = True) -> bool:
        if not self._tts_ready:
            logger.warning("TTS unavailable — cannot synthesize speech")
            return False
        if not _REQUESTS_AVAILABLE or requests is None:
            logger.warning("requests unavailable — cannot call ElevenLabs")
            return False

        url = f"https://api.elevenlabs.io/v1/text-to-speech/{self._eleven_voice_id}"
        headers = {
            "xi-api-key": self._eleven_key,
            "Content-Type": "application/json",
            "Accept": "audio/wav",
        }
        payload = {"text": text, "model_id": "eleven_monolingual_v1"}

        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=60)
            resp.raise_for_status()
            audio_bytes = resp.content
        except Exception as exc:
            logger.error("ElevenLabs TTS error: %s", exc)
            return False

        if not play:
            return True

        if not _SOUNDFILE_AVAILABLE or _sf is None:
            logger.warning("soundfile unavailable — cannot decode ElevenLabs audio")
            return False
        if not _SOUNDDEVICE_AVAILABLE or _sd is None:
            logger.warning("sounddevice unavailable — cannot play ElevenLabs audio")
            return False

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            with open(tmp_path, "wb") as handle:
                handle.write(audio_bytes)
            samples, sample_rate = _sf.read(tmp_path, dtype="float32")
            with self._lock:
                _sd.play(samples, samplerate=sample_rate)
                _sd.wait()
            return True
        except Exception as exc:
            logger.error("ElevenLabs playback error: %s", exc)
            return False
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    # ------------------------------------------------------------------
    # Response pipeline
    # ------------------------------------------------------------------

    def respond_to(self, what_they_said: str, meeting_context: str = "") -> str:
        from modules.llm_router import complete

        prompt = (
            "You are attending a meeting on behalf of the user.\n"
            "Respond concisely and professionally as if you were the user.\n"
            f"Meeting: {meeting_context}\n"
            f"They said: {what_they_said}\n"
            "Your response:"
        )
        try:
            response_text = complete(prompt)
        except Exception as exc:
            logger.error("LLM response error: %s", exc)
            response_text = "Sorry, could you repeat that?"

        self.speak_as_me(response_text)
        return response_text


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_instance: Optional[CloneAgent] = None
_singleton_lock = threading.Lock()


def get_clone_agent() -> CloneAgent:
    """Return the module-level :class:`CloneAgent` singleton."""
    global _instance
    with _singleton_lock:
        if _instance is None:
            _instance = CloneAgent()
    return _instance
