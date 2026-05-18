"""Daily briefing module for Particle.

Schedules and delivers a morning briefing every day at 09:00 UTC via Telegram.
The briefing includes:
  * Date / time greeting.
  * Weather forecast for the user's configured location (Open-Meteo, free, no key).
  * Calendar events for the day.
  * Pending task summary with any deadlines due today / tomorrow.
  * Email digest (unread count, urgent senders).
  * An LLM-generated motivational or actionable focus note.

The briefing can also be triggered manually via the Telegram /briefing command
or programmatically by calling ``trigger_now()``.
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone
from typing import Optional

import requests

from modules.config_loader import get_config

logger = logging.getLogger("particle.briefing")

# ---------------------------------------------------------------------------
# Optional APScheduler import
# ---------------------------------------------------------------------------

try:
    from apscheduler.schedulers.background import BackgroundScheduler  # type: ignore
    from apscheduler.triggers.cron import CronTrigger  # type: ignore
    _APS_AVAILABLE = True
except ImportError:
    _APS_AVAILABLE = False
    logger.warning("APScheduler not installed — briefing scheduler unavailable")

_WEATHER_URL = (
    "https://api.open-meteo.com/v1/forecast"
    "?latitude={lat}&longitude={lon}"
    "&current_weather=true"
    "&daily=weathercode,temperature_2m_max,temperature_2m_min"
    "&forecast_days=1&timezone=UTC"
)

# Default: London, UK
_DEFAULT_LAT = 51.5074
_DEFAULT_LON = -0.1278

_WMO_CODES = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Foggy", 48: "Icy fog",
    51: "Light drizzle", 61: "Light rain", 63: "Moderate rain", 65: "Heavy rain",
    71: "Light snow", 73: "Moderate snow", 75: "Heavy snow",
    80: "Light showers", 81: "Moderate showers", 82: "Violent showers",
    95: "Thunderstorm", 96: "Thunderstorm + hail", 99: "Thunderstorm + heavy hail",
}


def _weather_description(code: int) -> str:
    return _WMO_CODES.get(code, f"Weather code {code}")


class BriefingManager:
    """Scheduler and composer for the daily Particle briefing."""

    def __init__(self) -> None:
        cfg = get_config()
        self._cfg = cfg
        self._digest_hour: int = int(getattr(cfg.email, "digest_hour", 9))
        self._send_telegram: Optional[callable] = None
        self._scheduler: Optional[object] = None
        self._lat: float = float(getattr(cfg, "weather_lat", _DEFAULT_LAT) if hasattr(cfg, "weather_lat") else _DEFAULT_LAT)
        self._lon: float = float(getattr(cfg, "weather_lon", _DEFAULT_LON) if hasattr(cfg, "weather_lon") else _DEFAULT_LON)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_telegram_notifier(self, fn: callable) -> None:
        """Inject a callable(message: str) used to send the briefing."""
        self._send_telegram = fn

    def start(self) -> None:
        """Schedule the daily briefing using APScheduler."""
        if not _APS_AVAILABLE:
            logger.warning("APScheduler not available — starting simple fallback loop")
            t = threading.Thread(target=self._fallback_loop, daemon=True, name="briefing-loop")
            t.start()
            return

        self._scheduler = BackgroundScheduler(timezone="UTC")
        self._scheduler.add_job(  # type: ignore[union-attr]
            self.trigger_now,
            trigger=CronTrigger(hour=self._digest_hour, minute=0, timezone="UTC"),
            id="daily_briefing",
            name="Daily morning briefing",
            replace_existing=True,
        )
        self._scheduler.start()  # type: ignore[union-attr]
        logger.info(
            "Briefing scheduler started — daily at %02d:00 UTC", self._digest_hour
        )

    def stop(self) -> None:
        """Shut down the scheduler."""
        if self._scheduler is not None:
            self._scheduler.shutdown(wait=False)  # type: ignore[union-attr]
            logger.info("Briefing scheduler stopped")

    def trigger_now(self) -> None:
        """Compose and send a briefing immediately."""
        logger.info("Generating daily briefing…")
        try:
            text = self._compose()
            self._notify(text)
            logger.info("Briefing sent (%d chars)", len(text))
        except Exception as exc:
            logger.error("Briefing generation failed: %s", exc, exc_info=True)
            self._notify("⚠️ Briefing generation failed — check logs.")

    # ------------------------------------------------------------------
    # Composition
    # ------------------------------------------------------------------

    def _compose(self) -> str:
        now = datetime.now(timezone.utc)
        sections: list[str] = [
            f"🌅 *Good morning!*  {now.strftime('%A, %d %B %Y')}",
            "",
        ]

        # Weather
        weather = self._get_weather()
        if weather:
            sections.append(f"🌤 *Weather:* {weather}")
            sections.append("")

        # Calendar
        cal_section = self._get_calendar_section()
        if cal_section:
            sections.append(cal_section)
            sections.append("")

        # Tasks
        tasks_section = self._get_tasks_section()
        if tasks_section:
            sections.append(tasks_section)
            sections.append("")

        # Email
        email_section = self._get_email_section()
        if email_section:
            sections.append(email_section)
            sections.append("")

        # LLM focus note
        focus = self._get_focus_note(sections)
        if focus:
            sections.append(f"💡 *Focus note:*\n{focus}")

        return "\n".join(sections)

    # ------------------------------------------------------------------
    # Section builders
    # ------------------------------------------------------------------

    def _get_weather(self) -> str:
        try:
            url = _WEATHER_URL.format(lat=self._lat, lon=self._lon)
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            cw = data.get("current_weather", {})
            temp = cw.get("temperature", "?")
            wind = cw.get("windspeed", "?")
            code = cw.get("weathercode", -1)
            description = _weather_description(int(code)) if code != -1 else "Unknown"
            daily = data.get("daily", {})
            t_max = daily.get("temperature_2m_max", ["?"])[0]
            t_min = daily.get("temperature_2m_min", ["?"])[0]
            return (
                f"{description}  🌡 {temp}°C  "
                f"↑{t_max}°  ↓{t_min}°  💨 {wind} km/h"
            )
        except Exception as exc:
            logger.warning("Weather fetch failed: %s", exc)
            return ""

    def _get_calendar_section(self) -> str:
        try:
            from modules.calendar_manager import get_calendar_manager

            events = get_calendar_manager().get_upcoming_events(max_results=5)
            if not events:
                return "📅 *Calendar:* No upcoming events today."

            today = datetime.now(timezone.utc).date()
            today_events = []
            for ev in events:
                start_str = ev.get("start", {}).get("dateTime") or ev.get("start", {}).get("date")
                if not start_str:
                    continue
                try:
                    dt = datetime.fromisoformat(start_str.replace("Z", "+00:00"))
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    if dt.date() == today:
                        today_events.append((dt, ev.get("summary", "Meeting")))
                except ValueError:
                    continue

            if not today_events:
                return "📅 *Calendar:* No meetings today."

            lines = ["📅 *Today's meetings:*"]
            for dt, title in today_events:
                lines.append(f"  • {dt.strftime('%H:%M')} UTC — {title}")
            return "\n".join(lines)
        except Exception as exc:
            logger.warning("Calendar section error: %s", exc)
            return ""

    def _get_tasks_section(self) -> str:
        try:
            from modules.task_manager import get_task_manager

            tm = get_task_manager()
            due_today = tm.due_soon(days_ahead=0)
            due_tomorrow = tm.due_soon(days_ahead=1)
            all_pending = tm.pending()

            lines = [f"✅ *Tasks:* {len(all_pending)} pending"]
            if due_today:
                lines.append("  ⚠️ Due today:")
                for t in due_today[:3]:
                    lines.append(f"    [{t['priority'].upper()}] {t['title']}")
            if due_tomorrow:
                lines.append("  📌 Due tomorrow:")
                for t in due_tomorrow[:3]:
                    lines.append(f"    [{t['priority'].upper()}] {t['title']}")
            return "\n".join(lines)
        except Exception as exc:
            logger.warning("Tasks section error: %s", exc)
            return ""

    def _get_email_section(self) -> str:
        try:
            from modules.email_manager import get_email_manager

            em = get_email_manager()
            messages = em._fetch_unread()  # noqa: SLF001 — intentional internal access
            if not messages:
                return "📧 *Email:* Inbox is clear."

            total = len(messages)
            urgent = [m for m in messages if m["category"] == "urgent"]
            lines = [f"📧 *Email:* {total} unread"]
            if urgent:
                lines.append("  🚨 Urgent:")
                for m in urgent[:3]:
                    lines.append(f"    • {m['from'][:30]} — {m['subject'][:40]}")
            return "\n".join(lines)
        except Exception as exc:
            logger.warning("Email section error: %s", exc)
            return ""

    def _get_focus_note(self, sections: list[str]) -> str:
        try:
            from modules.llm_router import complete

            context = "\n".join(sections)
            prompt = (
                "Based on this morning briefing, write one short (2-3 sentence) "
                "motivational or productivity tip for the day.\n\n"
                f"Briefing:\n{context[:1500]}"
            )
            return complete(prompt)
        except Exception as exc:
            logger.warning("Focus note LLM error: %s", exc)
            return ""

    # ------------------------------------------------------------------
    # Fallback scheduler (no APScheduler)
    # ------------------------------------------------------------------

    def _fallback_loop(self) -> None:
        """Simple loop that triggers the briefing once per day at the configured hour."""
        triggered_today: Optional[int] = None
        while True:
            now = datetime.now(timezone.utc)
            today_key = now.toordinal()
            if now.hour == self._digest_hour and triggered_today != today_key:
                triggered_today = today_key
                try:
                    self.trigger_now()
                except Exception as exc:
                    logger.error("Fallback briefing error: %s", exc)
            time.sleep(60)

    # ------------------------------------------------------------------
    # Notifications
    # ------------------------------------------------------------------

    def _notify(self, message: str) -> None:
        if self._send_telegram:
            try:
                self._send_telegram(message)
            except Exception as exc:
                logger.error("Briefing notify error: %s", exc)
        else:
            logger.info("BRIEFING (no Telegram):\n%s", message)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_instance: Optional[BriefingManager] = None
_singleton_lock = threading.Lock()


def get_briefing_manager() -> BriefingManager:
    """Return the module-level :class:`BriefingManager` singleton."""
    global _instance
    with _singleton_lock:
        if _instance is None:
            _instance = BriefingManager()
    return _instance
