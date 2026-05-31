"""Meeting automation module for Particle.

Uses Selenium to join Google Meet and Zoom meetings automatically:
  * Watches the calendar for upcoming meetings.
  * Joins the meeting URL a minute before it starts.
  * Loads saved Google cookies so no CAPTCHA / robot check ever fires.
  * Falls back to password login if cookies are missing or expired.
  * Captures a transcript using Voxtral in a background thread.
  * Generates an LLM meeting summary when the meeting ends.
  * Posts the summary to Telegram.

Cookie setup (one-time):
  Run:  python setup_cookies.py
  This opens Chrome, lets you sign in manually, then saves cookies to
  data/google_cookies.json.  After that, meeting_bot.py loads them
  automatically on every join — no password prompts, no CAPTCHA.
"""

from __future__ import annotations

import json
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
    import sounddevice as sd
    _SD_AVAILABLE = True
except ImportError:
    _SD_AVAILABLE = False

_CHUNK_SECONDS = 30
_MIC_RATE = 16000

_GOOGLE_MEET_HOST = "meet.google.com"
_ZOOM_HOST = "zoom.us"

# Where cookies are saved after setup_cookies.py is run
_COOKIE_FILE = Path("data/google_cookies.json")


def _url_host_is(url: str, expected_host: str) -> bool:
    try:
        parsed = urlparse(url)
        host = parsed.netloc.lower().split(":")[0]
        return host == expected_host or host.endswith("." + expected_host)
    except Exception:
        return False


@dataclass
class MeetingSession:
    event_id: str
    title: str
    url: str
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    transcript_parts: list[str] = field(default_factory=list)
    is_running: bool = True


class MeetingBot:
    """Selenium-powered meeting attendee with cookie-based Google auth."""

    def __init__(self) -> None:
        cfg = get_config()
        self._send_telegram: Optional[callable] = None
        self._active_sessions: dict[str, MeetingSession] = {}
        self._lock = threading.Lock()
        self._running = False
        self._watcher_thread: Optional[threading.Thread] = None

        # Fallback password credentials (only used if cookies missing/expired)
        self._google_email = str(getattr(cfg.meeting_bot, "google_email", ""))
        self._google_password = str(getattr(cfg.meeting_bot, "google_password", ""))

        # Cookie file path (can be overridden in config)
        cookie_path = str(getattr(cfg.meeting_bot, "cookie_file", str(_COOKIE_FILE)))
        self._cookie_file = Path(cookie_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_telegram_notifier(self, fn: callable) -> None:
        self._send_telegram = fn

    def start(self) -> None:
        self._running = True
        self._watcher_thread = threading.Thread(
            target=self._watch_calendar, daemon=True, name="meeting-watcher"
        )
        self._watcher_thread.start()
        logger.info("MeetingBot watcher started")

    def stop(self) -> None:
        self._running = False
        with self._lock:
            for session in self._active_sessions.values():
                session.is_running = False
        logger.info("MeetingBot stopped")

    def join_meeting(self, event_id: str, title: str, url: str) -> bool:
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
                url = self._extract_meeting_url(event)
                if url:
                    title = event.get("summary", "Meeting")
                    self.join_meeting(event_id, title, url)

    def _extract_meeting_url(self, event: dict) -> Optional[str]:
        conf = event.get("conferenceData", {})
        entry_points = conf.get("entryPoints", [])
        for ep in entry_points:
            if ep.get("entryPointType") == "video":
                return ep.get("uri")

        for field_name in ("description", "location"):
            text = event.get(field_name, "") or ""
            for token in text.split():
                clean = token.strip("()[]<>.,")
                if _url_host_is(clean, _GOOGLE_MEET_HOST) or _url_host_is(clean, _ZOOM_HOST):
                    return clean
        return None

    # ------------------------------------------------------------------
    # Meeting runner
    # ------------------------------------------------------------------

    def _run_meeting(self, session: MeetingSession) -> None:
        driver = None
        audio_thread = None
        clone_agent = None

        try:
            driver = self._build_driver()

            # Sign in with credentials directly
            self._sign_in_google(driver)

            self._notify(
                f"Joining meeting: {session.title}\n"
                f"URL: {session.url}\n\n"
                "Particle is now attending this meeting on your behalf."
            )

            self._join_url(driver, session.url)
            logger.info("Joined meeting '%s'", session.title)
            self._notify(f"Joined meeting: {session.title}")

            from modules.clone_agent import get_clone_agent
            clone_agent = get_clone_agent()
            if clone_agent.available:
                clone_agent.start_face_clone()

            audio_queue: queue.Queue = queue.Queue()
            if _SD_AVAILABLE:
                audio_thread = threading.Thread(
                    target=self._capture_audio,
                    args=(session, audio_queue, clone_agent),
                    daemon=True,
                    name=f"audio-{session.event_id[:8]}",
                )
                audio_thread.start()

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

        self._post_summary(session)
        with self._lock:
            self._active_sessions.pop(session.event_id, None)

    def _join_url(self, driver, url: str) -> None:
        if not url.startswith("http"):
            url = "https://" + url
        driver.get(url)
        time.sleep(5)

        if _url_host_is(url, _GOOGLE_MEET_HOST):
            # Handle "Continue as [name]" account picker if it appears
            self._handle_account_picker(driver)

            # Mute mic and turn off camera before joining
            self._mute_before_join(driver)

            for selector in [
                "//button[contains(., 'Join now')]",
                "//button[contains(., 'Ask to join')]",
                "//button[contains(., 'Join')]",
                "//button[contains(@data-idom-class, 'join')]",
            ]:
                try:
                    btn = WebDriverWait(driver, 8).until(
                        EC.element_to_be_clickable((By.XPATH, selector))
                    )
                    btn.click()
                    logger.info("Clicked join button for Google Meet")
                    time.sleep(3)
                    break
                except Exception:
                    continue

            # Mute again after joining in case it got re-enabled
            time.sleep(2)
            self._mute_after_join(driver)

        if _url_host_is(url, _ZOOM_HOST):
            try:
                driver.get(url.replace("/j/", "/wc/join/"))
                time.sleep(2)
            except Exception:
                pass

    def _is_meeting_active(self, driver, url: str) -> bool:
        try:
            current_url = driver.current_url
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
    # Cookie-based auth (primary method — no CAPTCHA)
    # ------------------------------------------------------------------

    def _load_cookies(self, driver) -> bool:
        """Load saved Google cookies into Chrome."""
        if not self._cookie_file.exists():
            logger.info(
                "No cookie file at %s — run: python setup_cookies.py",
                self._cookie_file,
            )
            return False

        try:
            cookies = json.loads(self._cookie_file.read_text(encoding="utf-8"))
            if not cookies:
                return False

            driver.get("https://google.com")
            time.sleep(2)

            for cookie in cookies:
                cookie.pop("sameSite", None)
                cookie.pop("expiry", None)
                try:
                    driver.add_cookie(cookie)
                except Exception:
                    pass

            logger.info("Loaded %d cookies — going straight to Meet", len(cookies))
            return True

        except Exception as exc:
            logger.warning("Failed to load cookies: %s", exc)
            return False

    def _save_cookies(self, driver) -> None:
        """Save current Chrome cookies to file for reuse next time."""
        try:
            driver.get("https://accounts.google.com")
            time.sleep(2)
            cookies = driver.get_cookies()
            self._cookie_file.parent.mkdir(parents=True, exist_ok=True)
            self._cookie_file.write_text(
                json.dumps(cookies, indent=2), encoding="utf-8"
            )
            logger.info("Saved %d cookies to %s", len(cookies), self._cookie_file)
        except Exception as exc:
            logger.warning("Could not save cookies: %s", exc)

    # ------------------------------------------------------------------
    # Password login (fallback only — used when cookies missing/expired)
    # ------------------------------------------------------------------

    def _sign_in_google(self, driver) -> bool:
        """Sign in to Google using keyboard input — more reliable than button clicks."""
        if not self._google_email or not self._google_password:
            logger.warning("No Google credentials — set GOOGLE_MEET_EMAIL/PASSWORD in .env")
            return False

        logger.info("Signing in as %s", self._google_email)
        try:
            from selenium.webdriver.common.keys import Keys
            driver.get("https://accounts.google.com/signin")
            wait = WebDriverWait(driver, 30)

            # Enter email and press Enter
            email_field = wait.until(
                EC.presence_of_element_located((By.ID, "identifierId"))
            )
            time.sleep(1)
            email_field.click()
            email_field.clear()
            email_field.send_keys(self._google_email)
            email_field.send_keys(Keys.RETURN)
            time.sleep(4)

            # Enter password and press Enter
            password_field = wait.until(
                EC.presence_of_element_located((By.NAME, "Passwd"))
            )
            time.sleep(1)
            password_field.click()
            password_field.clear()
            password_field.send_keys(self._google_password)
            password_field.send_keys(Keys.RETURN)
            time.sleep(6)

            current = driver.current_url
            logger.info("After sign-in URL: %s", current[:80])

            if ("accounts.google.com/signin" not in current and
                    "accounts.google.com/v3/signin" not in current):
                logger.info("Signed in successfully as %s", self._google_email)
                return True

            logger.warning("Sign-in may have failed — URL: %s", current[:80])
            return False

        except Exception as exc:
            logger.warning("Sign-in failed: %s", exc)
            return False

    def _handle_account_picker(self, driver) -> None:
        """Click through any 'Continue as [name]' account picker screens."""
        try:
            # Wait a moment for the popup to appear
            time.sleep(2)
            wait = WebDriverWait(driver, 8)

            for selector in [
                "//button[contains(., 'Continue as')]",
                "//a[contains(., 'Continue as')]",
                "//button[contains(@aria-label, 'Continue as')]",
                "//div[contains(., 'Continue as')]//button",
                "//button[contains(., 'Continue')]",
                # Chrome's signin popup dismiss buttons
                "//button[contains(., 'No thanks')]",
                "//button[contains(., 'No, thanks')]",
                "//button[contains(., 'Cancel')]",
            ]:
                try:
                    btn = wait.until(EC.element_to_be_clickable((By.XPATH, selector)))
                    btn.click()
                    logger.info("Clicked account picker: %s", selector)
                    time.sleep(2)
                    return
                except Exception:
                    continue

            # Also try dismissing via JavaScript if buttons not clickable
            try:
                driver.execute_script("""
                    var buttons = document.querySelectorAll('button');
                    for (var b of buttons) {
                        if (b.innerText && (
                            b.innerText.includes('Continue as') ||
                            b.innerText.includes('No thanks')
                        )) {
                            b.click();
                            break;
                        }
                    }
                """)
                logger.info("Dismissed account picker via JavaScript")
                time.sleep(2)
            except Exception:
                pass

        except Exception as exc:
            logger.debug("Account picker handler: %s", exc)

    def _capture_audio(
        self,
        session: MeetingSession,
        q: "queue.Queue",
        clone_agent: object,
    ) -> None:
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
            self._notify(
                f"Meeting ended: {session.title}\n"
                f"Duration: ~{duration} min\n"
                "No transcript captured."
            )
            return

        logger.info("Generating summary for '%s'", session.title)
        summary = self._summarise(transcript, session.title)
        self._notify(
            f"Meeting Summary: {session.title}\n"
            f"Duration: ~{duration} min\n\n"
            f"{summary}"
        )

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
            return "Summary generation failed."

    # ------------------------------------------------------------------
    # Chrome driver
    # ------------------------------------------------------------------

    def _build_driver(self):
        """Build Chrome using undetected-chromedriver so Google doesn't block it."""
        tmp_profile = tempfile.mkdtemp(prefix="particle_chrome_")

        try:
            import undetected_chromedriver as uc
            options = uc.ChromeOptions()
            options.add_argument(f"--user-data-dir={tmp_profile}")
            options.add_argument("--window-size=1280,720")
            options.add_argument("--use-fake-ui-for-media-stream")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            options.add_experimental_option("prefs", {
                "profile.default_content_setting_values.media_stream_mic": 1,
                "profile.default_content_setting_values.media_stream_camera": 1,
                "profile.default_content_setting_values.notifications": 1,
            })
            driver = uc.Chrome(options=options, headless=False, version_main=148)
            logger.info("Chrome started via undetected-chromedriver")
        except ImportError:
            # Fallback to regular selenium if undetected-chromedriver not installed
            logger.warning(
                "undetected-chromedriver not installed — falling back to selenium. "
                "Run: pip install undetected-chromedriver"
            )
            from webdriver_manager.chrome import ChromeDriverManager
            options = ChromeOptions()
            options.add_argument(f"--user-data-dir={tmp_profile}")
            options.add_argument("--window-size=1280,720")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("--disable-gpu")
            options.add_argument("--disable-blink-features=AutomationControlled")
            options.add_argument("--use-fake-ui-for-media-stream")
            options.add_experimental_option("prefs", {
                "profile.default_content_setting_values.media_stream_mic": 1,
                "profile.default_content_setting_values.media_stream_camera": 1,
                "profile.default_content_setting_values.notifications": 1,
            })
            options.add_experimental_option("excludeSwitches", ["enable-automation"])
            options.add_experimental_option("useAutomationExtension", False)
            service = ChromeService(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=options)

        driver.set_page_load_timeout(30)
        driver.implicitly_wait(10)
        return driver

    # ------------------------------------------------------------------
    # Mute mic and camera
    # ------------------------------------------------------------------

    def _mute_before_join(self, driver) -> None:
        """Mute mic and turn off camera on the pre-join lobby screen."""
        from selenium.webdriver.common.keys import Keys
        try:
            # Click mic button if it shows as enabled (not already muted)
            mic_selectors = [
                "//div[@data-is-muted='false' and @data-tooltip and contains(@data-tooltip, 'mic')]",
                "//button[@data-is-muted='false' and contains(@aria-label, 'microphone')]",
                "//button[contains(@aria-label, 'Turn off microphone')]",
                "//button[contains(@aria-label, 'Mute microphone')]",
            ]
            for sel in mic_selectors:
                try:
                    btn = driver.find_element(By.XPATH, sel)
                    btn.click()
                    logger.info("Mic muted on lobby")
                    time.sleep(0.5)
                    break
                except Exception:
                    continue

            # Turn off camera
            cam_selectors = [
                "//button[contains(@aria-label, 'Turn off camera')]",
                "//button[contains(@aria-label, 'Stop camera')]",
                "//div[@data-is-muted='false' and contains(@data-tooltip, 'camera')]",
            ]
            for sel in cam_selectors:
                try:
                    btn = driver.find_element(By.XPATH, sel)
                    btn.click()
                    logger.info("Camera off on lobby")
                    time.sleep(0.5)
                    break
                except Exception:
                    continue

        except Exception as exc:
            logger.debug("Pre-join mute error (non-fatal): %s", exc)

    def _mute_after_join(self, driver) -> None:
        """Ensure mic and camera are off after joining the meeting."""
        from selenium.webdriver.common.keys import Keys
        try:
            time.sleep(2)

            # Use keyboard shortcuts — most reliable method
            # Ctrl+D = toggle mic, Ctrl+E = toggle camera in Google Meet
            from selenium.webdriver.common.action_chains import ActionChains
            actions = ActionChains(driver)

            # Focus the page first
            driver.find_element(By.TAG_NAME, "body").click()
            time.sleep(0.5)

            # Mute mic with Ctrl+D
            actions.key_down(Keys.CONTROL).send_keys("d").key_up(Keys.CONTROL).perform()
            time.sleep(0.5)
            logger.info("Sent Ctrl+D to mute mic")

            # Turn off camera with Ctrl+E
            actions = ActionChains(driver)
            actions.key_down(Keys.CONTROL).send_keys("e").key_up(Keys.CONTROL).perform()
            time.sleep(0.5)
            logger.info("Sent Ctrl+E to turn off camera")

            # Also try clicking mute buttons as backup
            for aria in ["Turn off microphone", "Mute microphone", "Mute mic"]:
                try:
                    btn = driver.find_element(
                        By.XPATH, f"//button[contains(@aria-label, '{aria}')]"
                    )
                    btn.click()
                    logger.info("Clicked mute button: %s", aria)
                    break
                except Exception:
                    continue

            for aria in ["Turn off camera", "Stop camera"]:
                try:
                    btn = driver.find_element(
                        By.XPATH, f"//button[contains(@aria-label, '{aria}')]"
                    )
                    btn.click()
                    logger.info("Clicked camera off button: %s", aria)
                    break
                except Exception:
                    continue

            logger.info("Mic and camera muted after joining")

        except Exception as exc:
            logger.debug("Post-join mute error (non-fatal): %s", exc)

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
    global _instance
    with _singleton_lock:
        if _instance is None:
            _instance = MeetingBot()
    return _instance