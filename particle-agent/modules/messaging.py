"""Telegram messaging integration for Particle.

Runs an async Telegram bot that:
  * Monitors all incoming messages 24/7.
  * When status is 'available': delivers the message to the user directly
    (no auto-reply — just log and forward to the home chat if different).
  * When status is 'busy' or 'away': auto-replies using the LLM + context.
  * Always escalates urgent messages regardless of status.
  * Handles bot commands:
      /start          — welcome message
      /status         — show current Particle status
      /setstatus <x>  — change status to available | busy | away
      /tasks          — list all pending tasks
      /addtask <text> — quickly add a task
      /briefing       — trigger a manual briefing now
      /logs           — tail the last 30 lines of particle.log

The bot token and home-chat ID come from config/env.  An outbound helper
``send_message`` is exposed for other modules to push Telegram notifications.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from pathlib import Path
from typing import Optional

from modules.config_loader import get_config

logger = logging.getLogger("particle.messaging")

# ---------------------------------------------------------------------------
# Optional telegram import
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

_STATUS_OPTIONS = ("available", "busy", "away")
_URGENT_KEYWORDS = (
    "urgent", "asap", "emergency", "critical", "help", "important",
)


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
        # Optional briefing callback injected by orchestrator
        self._briefing_callback: Optional[callable] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_briefing_callback(self, fn: callable) -> None:
        """Inject a callable that triggers an immediate briefing."""
        self._briefing_callback = fn

    def start(self) -> None:
        """Start the Telegram bot in a background daemon thread."""
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
        """Request a graceful shutdown of the Telegram bot."""
        if self._app is not None and self._loop is not None:
            asyncio.run_coroutine_threadsafe(self._app.stop(), self._loop)
        logger.info("MessagingManager stop requested")

    def send_message(self, text: str, chat_id: Optional[str] = None) -> None:
        """Send a Telegram message from any thread."""
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
        """Return the current agent status string."""
        return self._status

    def set_status(self, status: str) -> bool:
        """Update the agent status; returns False for unknown values."""
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

        app.add_handler(CommandHandler("start", self._cmd_start))
        app.add_handler(CommandHandler("status", self._cmd_status))
        app.add_handler(CommandHandler("setstatus", self._cmd_setstatus))
        app.add_handler(CommandHandler("tasks", self._cmd_tasks))
        app.add_handler(CommandHandler("addtask", self._cmd_addtask))
        app.add_handler(CommandHandler("briefing", self._cmd_briefing))
        app.add_handler(CommandHandler("logs", self._cmd_logs))
        app.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self._on_message)
        )

        await app.initialize()
        await app.start()
        self._ready.set()
        logger.info("Telegram bot polling started")
        await app.updater.start_polling(drop_pending_updates=True)
        # Keep running until stopped
        stop_event = asyncio.Event()
        await stop_event.wait()

    # ------------------------------------------------------------------
    # Command handlers
    # ------------------------------------------------------------------

    async def _cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await update.message.reply_text(
            "👋 *Particle* is online.\n"
            "I'm your personal AI Chief of Staff.\n"
            "Use /status to check my current mode.",
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
            await update.message.reply_text(
                "Usage: /setstatus available|busy|away"
            )
            return
        new_status = args[0].lower()
        if self.set_status(new_status):
            await update.message.reply_text(f"Status updated to *{new_status}*.", parse_mode=ParseMode.MARKDOWN)
        else:
            await update.message.reply_text(f"Unknown status '{new_status}'. Use: available, busy, away")

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
        await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN)

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
            await update.message.reply_text(f"```\n{last_lines[-3800:]}\n```", parse_mode=ParseMode.MARKDOWN)
        except OSError as exc:
            await update.message.reply_text(f"Error reading logs: {exc}")

    # ------------------------------------------------------------------
    # Message handler (non-command)
    # ------------------------------------------------------------------

    async def _on_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if update.message is None:
            return
        text = update.message.text or ""
        sender_id = str(update.effective_user.id) if update.effective_user else "unknown"
        sender_name = update.effective_user.first_name if update.effective_user else "Someone"

        logger.info("Incoming Telegram message from %s (%s): %s", sender_name, sender_id, text[:80])

        is_urgent = _is_urgent(text)

        if is_urgent:
            # Always escalate urgent messages
            alert = f"🚨 *URGENT message from {sender_name}:*\n{text}"
            self.send_message(alert)
            await update.message.reply_text("⚡ Your message has been escalated as urgent.")
            return

        if self._status == "available":
            # Forward to home chat if the message isn't already from the home user
            if sender_id != self._home_id and self._home_id:
                self.send_message(f"💬 Message from {sender_name}: {text}")
            return

        # busy or away — generate auto-reply
        reply = self._generate_auto_reply(text, sender_name)
        await update.message.reply_text(reply)

    # ------------------------------------------------------------------
    # Auto-reply
    # ------------------------------------------------------------------

    def _generate_auto_reply(self, message: str, sender_name: str) -> str:
        """Produce an LLM-powered auto-reply with user context."""
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
    """Return the module-level :class:`MessagingManager` singleton."""
    global _instance
    with _singleton_lock:
        if _instance is None:
            _instance = MessagingManager()
    return _instance


def send_telegram(message: str) -> None:
    """Convenience function: send a Telegram message via the global manager."""
    get_messaging_manager().send_message(message)
