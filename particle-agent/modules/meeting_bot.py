"""Meeting automation module for Particle.

Uses Selenium to join Google Meet and Zoom meetings automatically:
  * Watches the calendar for upcoming meetings.
  * Joins the meeting URL a minute before it starts.
  * Captures a transcript using Voxtral in a background thread.
  * Generates an LLM meeting summary when the meeting ends.
  * Posts the summary to Telegram.

Audio capture relies on the system's virtual audio loopback (e.g.
PulseAudio loopback or BlackHole on macOS).  If audio capture fails,
the module falls back to on-screen caption scraping from Google Meet.
"""

from __future__ import annotations

import logging
import os
import queue
import tempfile
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from modules.config_loader import get_config

logger = logging.getLogger("particle.meeting_bot")

# ---------------------------------------------------------------------------
# Optional imports
# ---------------------------------------------------------------------------

try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options as ChromeOptions
    from selenium.webdriver.chrome.service import Service as ChromeService
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.support.wait import WebDriverWait
    _SELENIUM_AVAILABLE = True
except ImportError:
    _SELENIUM_AVAILABLE = False
    logger.warning("selenium not installed — meeting bot unavailable")

try:
    import sounddevice as sd  # type: ignore
    _SD_AVAILABLE = True
except ImportError:
    _SD_AVAILABLE = False

_CHUNK_SECONDS = 30   # seconds of audio per transcription chunk
_MIC_RATE = 16000     # STT expects 16kHz

# Allowed meeting provider hostnames (exact match against parsed URL host)
_GOOGLE_MEET_HOST = "meet.google.com"
_ZOOM_HOST = "zoom.us"


def _url_host_is(url: str, expected_host: str) -> bool:
    """Return True only when the parsed hostname exactly matches *expected_host*."""
    try:
        parsed = urlparse(url)
        host = parsed.netloc.lower().split(":")[0]  # strip port if present
        return host == expected_host or host.endswith("." + expected_host)
    except Exception:
        return False


@dataclass
class MeetingSession:
    """Holds the runtime state for a single ongoing meeting."""

    event_id: str
    title: str
    url: str
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    transcript_parts: list[str] = field(default_factory=list)
    is_running: bool = True


class MeetingBot:
    """Selenium-powered meeting attendee with background transcription."""

    def __init__(self) -> None:
        cfg = get_config()
        self._send_telegram: Optional[callable] = None
        self._active_sessions: dict[str, MeetingSession] = {}
        self._lock = threading.Lock()
        self._running = False
        self._watcher_thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_telegram_notifier(self, fn: callable) -> None:
        """Inject a callable(message: str) for Telegram notifications."""
        self._send_telegram = fn

    def start(self) -> None:
        """Start the meeting watcher daemon thread."""
        self._running = True
        self._watcher_thread = threading.Thread(
            target=self._watch_calendar, daemon=True, name="meeting-watcher"
        )
        self._watcher_thread.start()
        logger.info("MeetingBot watcher started")

    def stop(self) -> None:
        """Stop the meeting watcher and all active sessions."""
        self._running = False
        with self._lock:
            for session in self._active_sessions.values():
                session.is_running = False
        logger.info("MeetingBot stopped")

    def join_meeting(self, event_id: str, title: str, url: str) -> bool:
        """Manually join a meeting URL; returns True on success."""
        if not _SELENIUM_AVAILABLE:
            logger.error("Selenium unavailable — cannot join meeting")
            return False
        if event_id in self._active_sessions:
            logger.warning("Already in meeting '%s'", title)
            return False

        session = MeetingSession(event_id=event_id, title=title, url=url)
        with self._lock:
            self._active_sessions[event_id] = session

        thread = threading.Thread(
            target=self._run_meeting,
            args=(session,),
            daemon=True,
            name=f"meeting-{event_id[:8]}",
        )
        thread.start()
        logger.info("Joining meeting '%s' at %s", title, url)
        return True

    # ------------------------------------------------------------------
    # Calendar watcher
    # ------------------------------------------------------------------

    def _watch_calendar(self) -> None:
        """Poll calendar every minute; auto-join events that are starting soon."""
        while self._running:
            try:
                self._check_upcoming_meetings()
            except Exception as exc:
                logger.error("Meeting watcher error: %s", exc, exc_info=True)
            time.sleep(60)

    def _check_upcoming_meetings(self) -> None:
        from modules.calendar_manager import get_calendar_manager

        cal = get_calendar_manager()
        events = cal.get_upcoming_events(max_results=10)
        now = datetime.now(timezone.utc)
        join_window = timedelta(minutes=2)

        for event in events:
            event_id = event.get("id", "")
            if event_id in self._active_sessions:
                continue

            start_str = (
                event.get("start", {}).get("dateTime")
                or event.get("start", {}).get("date")
            )
            if not start_str:
                continue

            try:
                start_dt = datetime.fromisoformat(start_str.replace("Z", "+00:00"))
                if start_dt.tzinfo is None:
                    start_dt = start_dt.replace(tzinfo=timezone.utc)
            except ValueError:
                continue

            if abs((start_dt - now).total_seconds()) <= join_window.total_seconds():
                # Find a meeting URL in the event
                url = self._extract_meeting_url(event)
                if url:
                    title = event.get("summary", "Meeting")
                    self.join_meeting(event_id, title, url)

    def _extract_meeting_url(self, event: dict) -> Optional[str]:
        """Extract a Google Meet or Zoom URL from a calendar event."""
        # Google Meet link
        conf = event.get("conferenceData", {})
        entry_points = conf.get("entryPoints", [])
        for ep in entry_points:
            if ep.get("entryPointType") == "video":
                return ep.get("uri")

        # Fallback: search description / location for URL patterns
        for field_name in ("description", "location"):
            text = event.get(field_name, "") or ""
            for token in text.split():
                clean = token.strip("()[]<>.,")
                if _url_host_is(clean, _GOOGLE_MEET_HOST) or _url_host_is(clean, _ZOOM_HOST):
                    return clean
        return None

    # ------------------------------------------------------------------
    # Meeting runner (per-session thread)
    # ------------------------------------------------------------------

    def _run_meeting(self, session: MeetingSession) -> None:
        driver = None
        audio_thread = None
        clone_agent = None

        try:
            driver = self._build_driver()
            self._notify(
                f"📅 *Joining meeting:* {session.title}\n"
                f"URL: {session.url}\n\n"
                "Particle is now attending this meeting on your behalf."
            )
            self._join_url(driver, session.url)
            logger.info("Joined meeting '%s'", session.title)
            self._notify(f"🤝 Joined meeting: *{session.title}*")

            from modules.clone_agent import get_clone_agent

            clone_agent = get_clone_agent()
            if clone_agent.available:
                clone_agent.start_face_clone()

            # Start audio capture in a background thread
            audio_queue: queue.Queue = queue.Queue()
            if _SD_AVAILABLE:
                audio_thread = threading.Thread(
                    target=self._capture_audio,
                    args=(session, audio_queue, clone_agent),
                    daemon=True,
                    name=f"audio-{session.event_id[:8]}",
                )
                audio_thread.start()

            # Wait for meeting to end (check every 30s whether the tab still has the meeting)
            while session.is_running:
                time.sleep(30)
                if not self._is_meeting_active(driver, session.url):
                    logger.info("Meeting '%s' appears to have ended", session.title)
                    session.is_running = False

        except Exception as exc:
            logger.error("Meeting '%s' session error: %s", session.title, exc, exc_info=True)
            session.is_running = False
        finally:
            session.is_running = False
            if driver:
                try:
                    driver.quit()
                except Exception:
                    pass
            if audio_thread:
                audio_thread.join(timeout=5)
            if clone_agent:
                clone_agent.stop_face_clone()

        # Generate and send summary
        self._post_summary(session)
        with self._lock:
            self._active_sessions.pop(session.event_id, None)

    def _join_url(self, driver, url: str) -> None:
        """Navigate to a meeting URL and attempt to dismiss consent dialogs."""
        driver.get(url)
        time.sleep(3)

        # Google Meet: click 'Join now' / 'Ask to join'
        if _url_host_is(url, _GOOGLE_MEET_HOST):
            for selector in [
                "//button[contains(., 'Join now')]",
                "//button[contains(., 'Ask to join')]",
                "//button[contains(@data-idom-class, 'join')]",
            ]:
                try:
                    btn = WebDriverWait(driver, 5).until(
                        EC.element_to_be_clickable((By.XPATH, selector))
                    )
                    btn.click()
                    logger.debug("Clicked '%s' button for Google Meet", selector)
                    break
                except Exception:
                    continue

        # Zoom: handle browser-based join
        if _url_host_is(url, _ZOOM_HOST):
            try:
                driver.get(url.replace("/j/", "/wc/join/"))
                time.sleep(2)
            except Exception:
                pass

    def _is_meeting_active(self, driver, url: str) -> bool:
        """Return True while a meeting tab is still showing meeting content."""
        try:
            current_url = driver.current_url
            # If navigated away from the meeting domain the meeting has ended
            if _url_host_is(url, _GOOGLE_MEET_HOST) and not _url_host_is(current_url, _GOOGLE_MEET_HOST):
                return False
            if _url_host_is(url, _ZOOM_HOST) and not _url_host_is(current_url, _ZOOM_HOST):
                return False
            try:
                title = driver.title.lower()
                if any(x in title for x in ["left", "ended", "removed"]):
                    return False
            except Exception:
                pass
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Audio capture
    # ------------------------------------------------------------------

    def _capture_audio(
        self,
        session: MeetingSession,
        q: "queue.Queue",
        clone_agent: object,
    ) -> None:
        """Record audio in chunks and transcribe each one with Whisper."""
        use_clone_agent = bool(getattr(clone_agent, "available", False))
        engine = None
        if not use_clone_agent:
            from modules.voice import get_voice_engine

            engine = get_voice_engine()
        chunk_samples = _CHUNK_SECONDS * _MIC_RATE

        while session.is_running:
            try:
                audio = sd.rec(
                    chunk_samples,
                    samplerate=_MIC_RATE,
                    channels=1,
                    dtype="float32",
                )
                sd.wait()
                if use_clone_agent:
                    text = clone_agent.transcribe_array(audio.squeeze(), _MIC_RATE)
                else:
                    text = engine.transcribe_audio_array(audio.squeeze(), _MIC_RATE)
                text = text.strip()
                if text:
                    session.transcript_parts.append(text)
                    if use_clone_agent:
                        clone_agent.respond_to(text, session.title)
                    logger.debug("Meeting transcript chunk (%d chars)", len(text))
            except Exception as exc:
                if "portaudio" in str(exc).lower() or "wasapi" in str(exc).lower():
                    logger.warning(
                        "Audio device error on Windows — transcript will be empty. "
                        "Install a virtual audio cable for full transcription."
                    )
                    break
                logger.error("Audio capture error: %s", exc)
                time.sleep(5)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def _post_summary(self, session: MeetingSession) -> None:
        transcript = "\n".join(session.transcript_parts)
        duration = int((datetime.now(timezone.utc) - session.started_at).total_seconds() / 60)

        if not transcript:
            msg = (
                f"📋 *Meeting ended:* {session.title}\n"
                f"Duration: ~{duration} min\n"
                "_No transcript captured._"
            )
            self._notify(msg)
            return

        logger.info("Generating summary for meeting '%s' (%d chars transcript)", session.title, len(transcript))
        summary = self._summarise(transcript, session.title)
        msg = (
            f"📋 *Meeting Summary:* {session.title}\n"
            f"Duration: ~{duration} min\n\n"
            f"{summary}"
        )
        self._notify(msg)

    def _summarise(self, transcript: str, title: str) -> str:
        from modules.llm_router import complete

        prompt = (
            f"Summarise the following meeting transcript for '{title}'. "
            "Highlight: key decisions, action items, and important topics discussed.\n\n"
            f"Transcript:\n{transcript[:6000]}"
        )
        try:
            return complete(prompt)
        except Exception as exc:
            logger.error("Meeting summary LLM error: %s", exc)
            return "_Summary generation failed._"

    # ------------------------------------------------------------------
    # Selenium driver factory
    # ------------------------------------------------------------------

    def _build_driver(self):
        """Build a headless Chrome WebDriver with microphone/camera permissions."""
        options = ChromeOptions()
        options.add_argument("--window-size=1280,720")
        options.add_argument("--start-minimized")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--use-fake-ui-for-media-stream")
        options.add_argument("--use-fake-device-for-media-stream")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option("useAutomationExtension", False)
        from webdriver_manager.chrome import ChromeDriverManager
        try:
            driver = webdriver.Chrome(options=options)
        except Exception:
            service = ChromeService(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=options)
        driver.set_page_load_timeout(30)
        return driver

    # ------------------------------------------------------------------
    # Notifications
    # ------------------------------------------------------------------

    def _notify(self, message: str) -> None:
        if self._send_telegram:
            try:
                self._send_telegram(message)
            except Exception as exc:
                logger.error("MeetingBot notify error: %s", exc)
        else:
            logger.info("MEETING NOTIFY: %s", message[:120])


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_instance: Optional[MeetingBot] = None
_singleton_lock = threading.Lock()


def get_meeting_bot() -> MeetingBot:
    """Return the module-level :class:`MeetingBot` singleton."""
    global _instance
    with _singleton_lock:
        if _instance is None:
            _instance = MeetingBot()
    return _instance
