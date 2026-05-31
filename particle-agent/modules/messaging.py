"""Telegram messaging integration for Particle.

Runs an async Telegram bot that:
  * Monitors all incoming messages 24/7.
  * When status is 'available': delivers the message to the user directly.
  * When status is 'busy' or 'away': auto-replies using the LLM + context.
  * Always escalates urgent messages regardless of status.
  * Handles bot commands:
      /start               — welcome message
      /status              — show current Particle status
      /setstatus <x>       — change status to available | busy | away
      /tasks               — list all pending tasks
      /addtask <text>      — quickly add a task
      /briefing            — trigger a manual briefing now
      /logs                — tail the last 30 lines of particle.log
      /schedule <text>     — schedule a meeting in natural language
      /join <url>          — join a meeting URL immediately
      /leave               — leave the current meeting
      /meetings            — list active meetings
"""

from __future__ import annotations

import asyncio
import logging
import re
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from modules.config_loader import get_config

logger = logging.getLogger("particle.messaging")

# ---------------------------------------------------------------------------
# Optional imports
# ---------------------------------------------------------------------------

try:
    from telegram import Update
    from telegram.ext import (
        Application,
        CommandHandler,
        ContextTypes,
        MessageHandler,
        filters,
    )
    from telegram.constants import ParseMode
    _TELEGRAM_AVAILABLE = True
except ImportError:
    _TELEGRAM_AVAILABLE = False
    logger.warning("python-telegram-bot not installed — Telegram unavailable")

try:
    import dateparser
    _DATEPARSER_AVAILABLE = True
except ImportError:
    _DATEPARSER_AVAILABLE = False
    logger.warning("dateparser not installed — run: pip install dateparser")

_STATUS_OPTIONS = ("available", "busy", "away")
_URGENT_KEYWORDS = (
    "urgent", "asap", "emergency", "critical", "help", "important",
)

_IST_OFFSET = timezone(timedelta(hours=5, minutes=30))


def _is_urgent(text: str) -> bool:
    t = text.lower()
    return any(k in t for k in _URGENT_KEYWORDS)


class MessagingManager:
    """Async Telegram bot with status management and LLM-powered auto-replies."""

    def __init__(self) -> None:
        cfg = get_config()
        self._token: str = getattr(cfg.telegram, "bot_token", "")
        self._home_id: str = str(getattr(cfg.telegram, "home_id", ""))
        self._status: str = getattr(cfg.telegram, "default_status", "available")
        self._enabled: bool = bool(getattr(cfg.telegram, "enabled", True))
        self._log_path: str = str(getattr(cfg.paths, "log_file", "logs/particle.log"))
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._app = None
        self._thread: Optional[threading.Thread] = None
        self._ready = threading.Event()
        self._briefing_callback: Optional[callable] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_briefing_callback(self, fn: callable) -> None:
        self._briefing_callback = fn

    def start(self) -> None:
        if not self._enabled or not _TELEGRAM_AVAILABLE:
            logger.warning("Telegram disabled or python-telegram-bot missing — skipping")
            return
        if not self._token:
            logger.warning("TELEGRAM_BOT_TOKEN not configured — Telegram unavailable")
            return

        self._thread = threading.Thread(
            target=self._run_bot, daemon=True, name="telegram-bot"
        )
        self._thread.start()
        self._ready.wait(timeout=30)
        logger.info("MessagingManager started (status=%s)", self._status)

    def stop(self) -> None:
        if self._app is not None and self._loop is not None:
            asyncio.run_coroutine_threadsafe(self._app.stop(), self._loop)
        logger.info("MessagingManager stop requested")

    def send_message(self, text: str, chat_id: Optional[str] = None) -> None:
        if self._app is None or self._loop is None:
            logger.debug("Telegram not ready — message dropped: %s", text[:80])
            return
        target = chat_id or self._home_id
        if not target:
            logger.warning("No chat_id configured — cannot send Telegram message")
            return
        coro = self._app.bot.send_message(
            chat_id=target,
            text=text,
            parse_mode=ParseMode.MARKDOWN,
        )
        asyncio.run_coroutine_threadsafe(coro, self._loop)

    def get_status(self) -> str:
        return self._status

    def set_status(self, status: str) -> bool:
        if status not in _STATUS_OPTIONS:
            return False
        self._status = status
        logger.info("Particle status changed to '%s'", status)
        return True

    # ------------------------------------------------------------------
    # Bot runner
    # ------------------------------------------------------------------

    def _run_bot(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._async_start())
        except Exception as exc:
            logger.error("Telegram bot error: %s", exc, exc_info=True)
        finally:
            self._loop.close()

    async def _async_start(self) -> None:
        self._app = Application.builder().token(self._token).build()
        app = self._app

        app.add_handler(CommandHandler("start",     self._cmd_start))
        app.add_handler(CommandHandler("status",    self._cmd_status))
        app.add_handler(CommandHandler("setstatus", self._cmd_setstatus))
        app.add_handler(CommandHandler("tasks",     self._cmd_tasks))
        app.add_handler(CommandHandler("addtask",   self._cmd_addtask))
        app.add_handler(CommandHandler("briefing",  self._cmd_briefing))
        app.add_handler(CommandHandler("logs",      self._cmd_logs))
        app.add_handler(CommandHandler("schedule",  self._cmd_schedule))
        app.add_handler(CommandHandler("join",      self._cmd_join))
        app.add_handler(CommandHandler("leave",     self._cmd_leave))
        app.add_handler(CommandHandler("meetings",  self._cmd_meetings))
        app.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self._on_message)
        )

        await app.initialize()
        await app.start()
        self._ready.set()
        logger.info("Telegram bot polling started")
        await app.updater.start_polling(drop_pending_updates=True)
        stop_event = asyncio.Event()
        await stop_event.wait()

    # ------------------------------------------------------------------
    # Command handlers
    # ------------------------------------------------------------------

    async def _cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await update.message.reply_text(
            "👋 *Particle* is online.\n"
            "I'm your personal AI Chief of Staff.\n\n"
            "*Commands:*\n"
            "/status — current status\n"
            "/setstatus available|busy|away — change status\n"
            "/tasks — pending tasks\n"
            "/addtask <text> — add a task\n"
            "/briefing — get a briefing now\n"
            "/schedule <text> — schedule a meeting\n"
            "  e.g. `/schedule Team Standup 3pm 30mins`\n"
            "/join <url> — join a meeting immediately\n"
            "  e.g. `/join https://meet.google.com/abc-def-ghi`\n"
            "/leave — leave the current meeting\n"
            "/meetings — list active meetings\n"
            "/logs — recent logs",
            parse_mode=ParseMode.MARKDOWN,
        )

    async def _cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        status_emoji = {"available": "🟢", "busy": "🟡", "away": "🔴"}.get(self._status, "⚪")
        await update.message.reply_text(
            f"Particle status: {status_emoji} *{self._status}*",
            parse_mode=ParseMode.MARKDOWN,
        )

    async def _cmd_setstatus(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        args = context.args or []
        if not args:
            await update.message.reply_text("Usage: /setstatus available|busy|away")
            return
        new_status = args[0].lower()
        if self.set_status(new_status):
            await update.message.reply_text(
                f"Status updated to *{new_status}*.",
                parse_mode=ParseMode.MARKDOWN,
            )
        else:
            await update.message.reply_text(
                f"Unknown status '{new_status}'. Use: available, busy, away"
            )

    async def _cmd_tasks(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        from modules.task_manager import get_task_manager

        tasks = get_task_manager().pending()
        if not tasks:
            await update.message.reply_text("✅ No pending tasks!")
            return
        lines = ["📋 *Pending Tasks:*"]
        for t in tasks[:20]:
            due = f" (due {t['due_date']})" if t.get("due_date") else ""
            lines.append(f"  [{t['id']}] {t['priority'].upper()} — {t['title']}{due}")
        await update.message.reply_text(
            "\n".join(lines), parse_mode=ParseMode.MARKDOWN
        )

    async def _cmd_addtask(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        from modules.task_manager import get_task_manager

        text = " ".join(context.args or []).strip()
        if not text:
            await update.message.reply_text("Usage: /addtask <task title>")
            return
        task_id = get_task_manager().create(title=text)
        await update.message.reply_text(f"✅ Task #{task_id} added: {text}")

    async def _cmd_briefing(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await update.message.reply_text("Generating briefing…")
        if self._briefing_callback:
            try:
                self._briefing_callback()
            except Exception as exc:
                logger.error("Manual briefing error: %s", exc)
                await update.message.reply_text("⚠️ Briefing failed — check logs.")
        else:
            await update.message.reply_text("ℹ️ No briefing module connected.")

    async def _cmd_logs(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        log_file = Path(self._log_path)
        if not log_file.exists():
            await update.message.reply_text("Log file not found.")
            return
        try:
            with log_file.open("r", encoding="utf-8", errors="replace") as fh:
                lines = fh.readlines()
            last_lines = "".join(lines[-30:]).strip()
            await update.message.reply_text(
                f"```\n{last_lines[-3800:]}\n```",
                parse_mode=ParseMode.MARKDOWN,
            )
        except OSError as exc:
            await update.message.reply_text(f"Error reading logs: {exc}")

    async def _cmd_join(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Join a meeting URL immediately via /join <url>."""
        args = context.args or []
        if not args:
            await update.message.reply_text(
                "📅 *Join a Meeting*\n\n"
                "Usage: `/join <meeting_url>`\n\n"
                "Examples:\n"
                "`/join https://meet.google.com/abc-defg-hij`\n"
                "`/join https://zoom.us/j/123456789`",
                parse_mode=ParseMode.MARKDOWN,
            )
            return

        url = args[0].strip()

        # Validate URL
        if "meet.google.com" not in url and "zoom.us" not in url:
            await update.message.reply_text(
                "❌ Only Google Meet and Zoom links are supported.\n\n"
                "Example: `/join https://meet.google.com/abc-defg-hij`",
                parse_mode=ParseMode.MARKDOWN,
            )
            return

        await update.message.reply_text(
            f"📅 *Joining meeting...*\n{url}\n\n"
            "Particle will sign in and join shortly.",
            parse_mode=ParseMode.MARKDOWN,
        )

        # Join via meeting bot
        try:
            from modules.meeting_bot import get_meeting_bot
            import time as _time

            bot = get_meeting_bot()
            event_id = f"manual-{int(_time.time())}"
            title = "Manual Meeting"

            # Try to extract a nicer title from URL
            if "meet.google.com" in url:
                title = "Google Meet"
            elif "zoom.us" in url:
                title = "Zoom Meeting"

            success = bot.join_meeting(event_id, title, url)

            if not success:
                await update.message.reply_text(
                    "❌ Failed to join meeting.\n"
                    "Make sure Selenium and Chrome are installed."
                )
        except Exception as exc:
            logger.error("Join meeting error: %s", exc)
            await update.message.reply_text(f"❌ Error joining meeting: {exc}")

    async def _cmd_leave(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Leave all active meetings."""
        try:
            from modules.meeting_bot import get_meeting_bot
            bot = get_meeting_bot()

            if not bot._active_sessions:
                await update.message.reply_text("ℹ️ No active meetings to leave.")
                return

            count = len(bot._active_sessions)
            for session in bot._active_sessions.values():
                session.is_running = False

            await update.message.reply_text(
                f"👋 Left {count} active meeting(s)."
            )
        except Exception as exc:
            logger.error("Leave meeting error: %s", exc)
            await update.message.reply_text(f"❌ Error leaving meeting: {exc}")

    async def _cmd_meetings(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """List all active meetings."""
        try:
            from modules.meeting_bot import get_meeting_bot
            bot = get_meeting_bot()

            if not bot._active_sessions:
                await update.message.reply_text("ℹ️ No active meetings.")
                return

            lines = ["📅 *Active Meetings:*"]
            for session in bot._active_sessions.values():
                duration = int(
                    (datetime.now(timezone.utc) - session.started_at).total_seconds() / 60
                )
                lines.append(
                    f"  • *{session.title}*\n"
                    f"    Duration: {duration} min\n"
                    f"    URL: {session.url}"
                )
            await update.message.reply_text(
                "\n".join(lines), parse_mode=ParseMode.MARKDOWN
            )
        except Exception as exc:
            logger.error("List meetings error: %s", exc)
            await update.message.reply_text(f"❌ Error listing meetings: {exc}")

    async def _cmd_schedule(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Schedule a meeting using natural language.

        Examples:
          /schedule Team Standup 3pm
          /schedule Client Call tomorrow 2:30pm 45mins
          /schedule Daily Standup today 9am 30mins
          /schedule Hackathon Review 6pm 1hr
        """
        raw = " ".join(context.args or []).strip()

        if not raw:
            await update.message.reply_text(
                "📅 *Schedule a Meeting*\n\n"
                "Just tell me the title and time:\n\n"
                "`/schedule Team Standup 3pm`\n"
                "`/schedule Client Call tomorrow 2:30pm 45mins`\n"
                "`/schedule Daily Standup today 9am 30mins`\n"
                "`/schedule Review 6pm 1hr`\n\n"
                "Time is in IST. Duration defaults to 60 minutes.",
                parse_mode=ParseMode.MARKDOWN,
            )
            return

        if not _DATEPARSER_AVAILABLE:
            await update.message.reply_text(
                "⚠️ dateparser not installed. Run:\n`pip install dateparser`",
                parse_mode=ParseMode.MARKDOWN,
            )
            return

        # Extract duration (e.g. 45mins, 30min, 1hr, 2hours)
        duration_mins = 60
        duration_match = re.search(
            r"(\d+)\s*(hours?|hr|mins?|minutes?)", raw, re.IGNORECASE
        )
        if duration_match:
            val = int(duration_match.group(1))
            unit = duration_match.group(2).lower()
            duration_mins = val * 60 if unit.startswith("h") else val
            raw = (raw[: duration_match.start()] + " " + raw[duration_match.end():]).strip()

        # Split title and time
        time_match = re.search(
            r"(\b\d{1,2}:\d{2}\s*(am|pm)?\b|\b\d{1,2}\s*(am|pm)\b"
            r"|\btoday\b|\btomorrow\b|\btonight\b"
            r"|\b(next\s+)?(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b)",
            raw,
            re.IGNORECASE,
        )

        if time_match:
            title = raw[: time_match.start()].strip(" ,-") or "Meeting"
            time_str = raw[time_match.start():].strip()
        else:
            await update.message.reply_text(
                "⚠️ Couldn't find a time. Try:\n"
                "`/schedule Team Standup 3pm`\n"
                "`/schedule Client Call tomorrow 2:30pm`",
                parse_mode=ParseMode.MARKDOWN,
            )
            return

        parsed_dt = dateparser.parse(
            time_str,
            settings={
                "PREFER_DATES_FROM": "future",
                "TIMEZONE": "Asia/Kolkata",
                "RETURN_AS_TIMEZONE_AWARE": True,
            },
        )

        if not parsed_dt:
            await update.message.reply_text(
                "⚠️ Couldn't understand the time. Try:\n"
                "`/schedule Team Standup 3pm`\n"
                "`/schedule Client Call tomorrow 2:30pm`",
                parse_mode=ParseMode.MARKDOWN,
            )
            return

        start_utc = parsed_dt.astimezone(timezone.utc)
        end_utc = start_utc + timedelta(minutes=duration_mins)
        display_time = parsed_dt.strftime("%d %b %Y, %I:%M %p")

        await update.message.reply_text(
            f"⏳ Creating *{title}* at {display_time} IST…",
            parse_mode=ParseMode.MARKDOWN,
        )

        try:
            from modules.calendar_manager import get_calendar_manager

            cal = get_calendar_manager()
            if cal._service is None:
                await update.message.reply_text(
                    "⚠️ Google Calendar is not connected. Restart Particle to authorize."
                )
                return

            event_body = {
                "summary": title,
                "start": {"dateTime": start_utc.isoformat(), "timeZone": "UTC"},
                "end": {"dateTime": end_utc.isoformat(), "timeZone": "UTC"},
                "conferenceData": {
                    "createRequest": {
                        "requestId": f"particle-{int(datetime.now().timestamp())}",
                        "conferenceSolutionKey": {"type": "hangoutsMeet"},
                    }
                },
            }

            event = (
                cal._service.events()
                .insert(
                    calendarId="primary",
                    body=event_body,
                    conferenceDataVersion=1,
                )
                .execute()
            )

            meet_url = None
            for ep in event.get("conferenceData", {}).get("entryPoints", []):
                if ep.get("entryPointType") == "video":
                    meet_url = ep.get("uri")
                    break

            await update.message.reply_text(
                f"✅ *Meeting scheduled!*\n\n"
                f"📅 *{title}*\n"
                f"🕐 {display_time} IST ({duration_mins} mins)\n"
                f"🔗 {meet_url or 'No Meet link generated'}\n\n"
                "Particle will join automatically at meeting time.",
                parse_mode=ParseMode.MARKDOWN,
            )

            logger.info(
                "Meeting scheduled via Telegram: '%s' at %s meet=%s",
                title, start_utc.isoformat(), meet_url,
            )

        except Exception as exc:
            logger.error("Failed to create meeting: %s", exc, exc_info=True)
            await update.message.reply_text(f"⚠️ Failed to create meeting: {exc}")

    # ------------------------------------------------------------------
    # Message handler (non-command)
    # ------------------------------------------------------------------

    async def _on_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if update.message is None:
            return
        text = update.message.text or ""
        sender_id = str(update.effective_user.id) if update.effective_user else "unknown"
        sender_name = update.effective_user.first_name if update.effective_user else "Someone"

        logger.info(
            "Incoming Telegram message from %s (%s): %s",
            sender_name, sender_id, text[:80],
        )

        if _is_urgent(text):
            alert = f"🚨 *URGENT message from {sender_name}:*\n{text}"
            self.send_message(alert)
            await update.message.reply_text(
                "⚡ Your message has been escalated as urgent."
            )
            return

        if self._status == "available":
            if sender_id != self._home_id and self._home_id:
                self.send_message(f"💬 Message from {sender_name}: {text}")
            return

        reply = self._generate_auto_reply(text, sender_name)
        await update.message.reply_text(reply)

    # ------------------------------------------------------------------
    # Auto-reply
    # ------------------------------------------------------------------

    def _generate_auto_reply(self, message: str, sender_name: str) -> str:
        from modules.llm_router import complete
        from modules.context_loader import get_context_loader

        ctx = get_context_loader().build_context_string(message, n_results=3)
        system = (
            f"You are Particle, a personal AI assistant replying on behalf of your user. "
            f"The user is currently {self._status}. "
            "Write a brief, professional auto-reply. "
            "Do not pretend to be human — make it clear you are an AI assistant."
        )
        if ctx:
            system += f"\n\n{ctx}"
        prompt = f"Message from {sender_name}:\n{message}\n\nWrite an auto-reply:"
        try:
            return complete(prompt, system)
        except Exception as exc:
            logger.error("Auto-reply LLM error: %s", exc)
            return (
                f"Thank you for your message. "
                f"The user is currently {self._status} and will respond shortly."
            )


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_instance: Optional[MessagingManager] = None
_singleton_lock = threading.Lock()


def get_messaging_manager() -> MessagingManager:
    global _instance
    with _singleton_lock:
        if _instance is None:
            _instance = MessagingManager()
    return _instance


def send_telegram(message: str) -> None:
    get_messaging_manager().send_message(message)