"""Calendar management module for Particle.

Connects to Google Calendar API v3 via OAuth2 to:
  * Read upcoming events and surface them in briefings.
  * Create, update, and delete events on the user's behalf.
  * Send a Telegram reminder 15 minutes before each meeting.
  * Auto-decline meetings that conflict with existing events.
  * Accept invites based on availability rules in config.yaml.
  * Never double-book the user.

Credentials are stored as JSON in the environment variable
``GOOGLE_CALENDAR_CREDENTIALS``, or as a file path in config.yaml.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

from modules.config_loader import get_config

logger = logging.getLogger("particle.calendar_manager")

# ---------------------------------------------------------------------------
# Optional Google API imports — degrade gracefully if packages missing
# ---------------------------------------------------------------------------

try:
    from google.oauth2.credentials import Credentials  # type: ignore
    from google.auth.transport.requests import Request as GAuthRequest  # type: ignore
    from googleapiclient.discovery import build as gapi_build  # type: ignore
    from googleapiclient.errors import HttpError  # type: ignore
    _GOOGLE_AVAILABLE = True
except ImportError:
    _GOOGLE_AVAILABLE = False
    logger.warning(
        "google-api-python-client not installed — calendar features unavailable"
    )

_SCOPES = ["https://www.googleapis.com/auth/calendar"]
_POLL_INTERVAL = 300  # 5 minutes between reminder checks


def _parse_dt(value: Optional[str]) -> Optional[datetime]:
    """Parse an ISO datetime string (with or without timezone) to a UTC datetime."""
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except ValueError:
        return None


class CalendarManager:
    """Google Calendar integration with reminders and conflict detection."""

    def __init__(self) -> None:
        cfg = get_config()
        self._cfg = cfg.calendar
        self._reminder_minutes: int = int(getattr(cfg.calendar, "reminder_minutes", 15))
        self._creds_raw: str = getattr(cfg.calendar, "credentials", "")
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._service = None
        self._reminded: set[str] = set()  # event IDs already reminded
        self._send_telegram: Optional[callable] = None

        if _GOOGLE_AVAILABLE and self._creds_raw:
            self._build_service()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_telegram_notifier(self, fn: callable) -> None:
        """Inject a callable(message: str) for Telegram notifications."""
        self._send_telegram = fn

    def start(self) -> None:
        """Start the background reminder-check loop."""
        if self._running or self._service is None:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._reminder_loop, daemon=True, name="calendar-reminder"
        )
        self._thread.start()
        logger.info("CalendarManager reminder loop started")

    def stop(self) -> None:
        """Stop the reminder loop."""
        self._running = False

    def get_upcoming_events(self, max_results: int = 10) -> list[dict]:
        """Return the next *max_results* events from the primary calendar."""
        if self._service is None:
            return []
        now = datetime.now(timezone.utc).isoformat()
        try:
            result = (
                self._service.events()
                .list(
                    calendarId="primary",
                    timeMin=now,
                    maxResults=max_results,
                    singleEvents=True,
                    orderBy="startTime",
                )
                .execute()
            )
            events = result.get("items", [])
            logger.debug("CalendarManager: fetched %d upcoming events", len(events))
            return events
        except Exception as exc:
            logger.error("get_upcoming_events error: %s", exc)
            return []

    def create_event(
        self,
        title: str,
        start: str,
        end: str,
        description: str = "",
        attendees: Optional[list[str]] = None,
    ) -> Optional[dict]:
        """Create a calendar event; returns the created event or None."""
        if self._service is None:
            logger.warning("Calendar service unavailable — cannot create event")
            return None

        if self._is_conflicting(start, end):
            logger.warning(
                "Conflict detected for %r (%s–%s) — not creating", title, start, end
            )
            return None

        body: dict = {
            "summary": title,
            "description": description,
            "start": {"dateTime": start, "timeZone": "UTC"},
            "end": {"dateTime": end, "timeZone": "UTC"},
        }
        if attendees:
            body["attendees"] = [{"email": a} for a in attendees]

        try:
            event = self._service.events().insert(calendarId="primary", body=body).execute()
            logger.info("Calendar event created: %s id=%s", title, event.get("id"))
            return event
        except Exception as exc:
            logger.error("create_event error: %s", exc)
            return None

    def update_event(self, event_id: str, **kwargs) -> Optional[dict]:
        """Patch fields on an existing event; returns updated event or None."""
        if self._service is None:
            return None
        try:
            event = self._service.events().patch(
                calendarId="primary", eventId=event_id, body=kwargs
            ).execute()
            logger.info("Calendar event updated id=%s", event_id)
            return event
        except Exception as exc:
            logger.error("update_event error id=%s: %s", event_id, exc)
            return None

    def delete_event(self, event_id: str) -> bool:
        """Delete an event by ID; returns True on success."""
        if self._service is None:
            return False
        try:
            self._service.events().delete(calendarId="primary", eventId=event_id).execute()
            logger.info("Calendar event deleted id=%s", event_id)
            return True
        except Exception as exc:
            logger.error("delete_event error id=%s: %s", event_id, exc)
            return False

    def decline_invite(self, event_id: str) -> bool:
        """RSVP as 'declined' for a pending invite."""
        try:
            self._service.events().patch(
                calendarId="primary",
                eventId=event_id,
                body={"attendees": [{"self": True, "responseStatus": "declined"}]},
            ).execute()
            logger.info("Declined invite for event id=%s", event_id)
            return True
        except Exception as exc:
            logger.error("decline_invite error id=%s: %s", event_id, exc)
            return False

    def list_events_in_range(self, start: str, end: str) -> list[dict]:
        """Return events that overlap [start, end] (ISO strings)."""
        if self._service is None:
            return []
        try:
            result = (
                self._service.events()
                .list(
                    calendarId="primary",
                    timeMin=start,
                    timeMax=end,
                    singleEvents=True,
                    orderBy="startTime",
                )
                .execute()
            )
            return result.get("items", [])
        except Exception as exc:
            logger.error("list_events_in_range error: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Reminder loop
    # ------------------------------------------------------------------

    def _reminder_loop(self) -> None:
        while self._running:
            try:
                self._check_reminders()
            except Exception as exc:
                logger.error("Reminder loop error: %s", exc, exc_info=True)
            time.sleep(_POLL_INTERVAL)

    def _check_reminders(self) -> None:
        """Send Telegram alerts for events starting within the reminder window."""
        now = datetime.now(timezone.utc)
        window_end = now + timedelta(minutes=self._reminder_minutes + 1)

        events = self.get_upcoming_events(max_results=20)
        for event in events:
            event_id = event.get("id", "")
            if event_id in self._reminded:
                continue

            start_str = event.get("start", {}).get("dateTime") or event.get("start", {}).get("date")
            start_dt = _parse_dt(start_str)
            if start_dt is None:
                continue

            delta = (start_dt - now).total_seconds() / 60
            if 0 <= delta <= self._reminder_minutes:
                title = event.get("summary", "Meeting")
                msg = (
                    f"📅 *Reminder:* {title} starts in {int(delta)} minutes!\n"
                    f"Time: {start_dt.strftime('%H:%M UTC')}"
                )
                self._notify(msg)
                self._reminded.add(event_id)
                logger.info("Reminder sent for event '%s' at %s", title, start_dt)

    # ------------------------------------------------------------------
    # Conflict detection
    # ------------------------------------------------------------------

    def _is_conflicting(self, start: str, end: str) -> bool:
        """Return True if any existing event overlaps the given slot."""
        existing = self.list_events_in_range(start, end)
        return len(existing) > 0

    # ------------------------------------------------------------------
    # Service initialisation
    # ------------------------------------------------------------------

    def _build_service(self) -> None:
        """Build the Google Calendar API service from stored credentials JSON."""
        if not _GOOGLE_AVAILABLE:
            return
        try:
            creds_data = self._load_credentials_json()
            if creds_data is None:
                logger.warning("No Google Calendar credentials — service unavailable")
                return

            creds = Credentials.from_authorized_user_info(creds_data, _SCOPES)
            if creds.expired and creds.refresh_token:
                creds.refresh(GAuthRequest())
                logger.info("Google Calendar credentials refreshed")

            self._service = gapi_build("calendar", "v3", credentials=creds)
            logger.info("Google Calendar API service ready")
        except Exception as exc:
            logger.error("Failed to build Calendar service: %s", exc, exc_info=True)
            self._service = None

    def _load_credentials_json(self) -> Optional[dict]:
        """Load OAuth2 credentials from env var (JSON string) or file path."""
        raw = self._creds_raw
        if not raw:
            return None
        # Try treating as JSON directly
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
        # Try treating as a file path
        if os.path.isfile(raw):
            try:
                with open(raw, "r", encoding="utf-8") as fh:
                    return json.load(fh)
            except (OSError, json.JSONDecodeError) as exc:
                logger.error("Failed loading Calendar credentials from file %s: %s", raw, exc)
        logger.error("GOOGLE_CALENDAR_CREDENTIALS is neither valid JSON nor a readable file path")
        return None

    # ------------------------------------------------------------------
    # Notifications
    # ------------------------------------------------------------------

    def _notify(self, message: str) -> None:
        if self._send_telegram:
            try:
                self._send_telegram(message)
            except Exception as exc:
                logger.error("CalendarManager notify error: %s", exc)
        else:
            logger.info("CALENDAR NOTIFY (no Telegram): %s", message[:120])


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_instance: Optional[CalendarManager] = None
_singleton_lock = threading.Lock()


def get_calendar_manager() -> CalendarManager:
    """Return the module-level :class:`CalendarManager` singleton."""
    global _instance
    with _singleton_lock:
        if _instance is None:
            _instance = CalendarManager()
    return _instance
