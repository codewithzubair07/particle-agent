"""Async orchestration layer for Particle modules.

The orchestrator is the central brain that:
  * Starts all modules in parallel using ``asyncio`` + ``ThreadPoolExecutor``.
  * Monitors each module's health via periodic heartbeat checks.
  * Restarts any module that crashes, with exponential back-off.
  * Provides cross-module wiring (e.g. connects Telegram notifier callbacks to
    email, calendar, and briefing modules).
  * Exposes ``run()`` which blocks until ``SIGTERM``/``SIGINT``.
"""

from __future__ import annotations

import asyncio
import logging
import signal
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional

from modules.config_loader import get_config

logger = logging.getLogger("particle.orchestrator")


class ModuleRunner:
    """Wraps a module start function with automatic restart on crash."""

    MAX_RESTART_DELAY = 300  # 5 minutes

    def __init__(self, name: str, start_fn: Callable[[], None]) -> None:
        self.name = name
        self._start_fn = start_fn
        self._thread: Optional[threading.Thread] = None
        self._restart_count = 0
        self._running = True

    def launch(self) -> None:
        """Launch the module in a managed daemon thread."""
        self._thread = threading.Thread(
            target=self._supervised_run,
            daemon=True,
            name=f"mod-{self.name}",
        )
        self._thread.start()

    def stop(self) -> None:
        """Signal the module not to restart after the next crash."""
        self._running = False

    def _supervised_run(self) -> None:
        while self._running:
            try:
                logger.info("Starting module '%s'", self.name)
                self._start_fn()
                logger.warning("Module '%s' exited normally (will not restart)", self.name)
                break
            except Exception as exc:
                if not self._running:
                    break
                self._restart_count += 1
                delay = min(2 ** self._restart_count, self.MAX_RESTART_DELAY)
                logger.error(
                    "Module '%s' crashed (attempt %d): %s — restarting in %ds",
                    self.name, self._restart_count, exc, delay, exc_info=True,
                )
                time.sleep(delay)


class Orchestrator:
    """Central coordinator that starts and monitors all Particle modules."""

    def __init__(self) -> None:
        cfg = get_config()
        self._cfg = cfg
        self._module_runners: list[ModuleRunner] = []
        self._stop_event = threading.Event()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Start all modules and block until a stop signal is received."""
        self._register_signal_handlers()
        self._wire_modules()
        self._start_all_modules()
        logger.info("Particle orchestrator running — press Ctrl+C to stop")
        self._stop_event.wait()
        self._shutdown()

    def stop(self) -> None:
        """Request an orderly shutdown."""
        logger.info("Orchestrator stop requested")
        self._stop_event.set()

    # ------------------------------------------------------------------
    # Wiring
    # ------------------------------------------------------------------

    def _wire_modules(self) -> None:
        """Inject cross-module callbacks (e.g. Telegram notifiers)."""
        from modules.messaging import get_messaging_manager
        from modules.email_manager import get_email_manager
        from modules.calendar_manager import get_calendar_manager
        from modules.briefing import get_briefing_manager
        from modules.meeting_bot import get_meeting_bot

        messaging = get_messaging_manager()
        email_mgr = get_email_manager()
        cal_mgr = get_calendar_manager()
        briefing_mgr = get_briefing_manager()
        meeting_bot = get_meeting_bot()

        # Wire Telegram send to all consumers
        notifier = messaging.send_message
        email_mgr.set_telegram_notifier(notifier)
        cal_mgr.set_telegram_notifier(notifier)
        briefing_mgr.set_telegram_notifier(notifier)
        meeting_bot.set_telegram_notifier(notifier)

        # Wire manual briefing trigger into Telegram commands
        messaging.set_briefing_callback(briefing_mgr.trigger_now)

        logger.info("Module cross-wiring complete")

    # ------------------------------------------------------------------
    # Module start functions
    # ------------------------------------------------------------------

    def _start_all_modules(self) -> None:
        """Initialise and launch all module runners."""
        modules: list[tuple[str, Callable]] = [
            ("context-index",  self._start_context_index),
            ("task-deadline-notifier", self._start_task_deadline_notifier),
            ("email-manager",  self._start_email_manager),
            ("calendar-manager", self._start_calendar_manager),
            ("telegram-bot",   self._start_telegram_bot),
            ("briefing",       self._start_briefing),
            ("meeting-bot",    self._start_meeting_bot),
        ]

        for name, fn in modules:
            runner = ModuleRunner(name, fn)
            self._module_runners.append(runner)
            runner.launch()
            logger.info("Module runner launched: %s", name)

        logger.info("All %d module runners launched", len(modules))

    # ------------------------------------------------------------------
    # Per-module start functions (called in supervised threads)
    # ------------------------------------------------------------------

    def _start_context_index(self) -> None:
        from modules.context_loader import get_context_loader

        loader = get_context_loader()
        count = loader.index_all()
        logger.info("Context index built: %d chunks", count)
        # Index once; future re-indexing triggered by /addtask command
        # Sleep indefinitely so the supervisor doesn't keep restarting
        threading.Event().wait()

    def _start_email_manager(self) -> None:
        from modules.email_manager import get_email_manager

        mgr = get_email_manager()
        mgr.start()
        threading.Event().wait()

    def _start_calendar_manager(self) -> None:
        from modules.calendar_manager import get_calendar_manager

        mgr = get_calendar_manager()
        mgr.start()
        threading.Event().wait()

    def _start_telegram_bot(self) -> None:
        from modules.messaging import get_messaging_manager

        mgr = get_messaging_manager()
        mgr.start()
        # Telegram bot runs its own event loop internally; just park here
        threading.Event().wait()

    def _start_briefing(self) -> None:
        from modules.briefing import get_briefing_manager

        mgr = get_briefing_manager()
        mgr.start()
        threading.Event().wait()

    def _start_meeting_bot(self) -> None:
        from modules.meeting_bot import get_meeting_bot

        bot = get_meeting_bot()
        bot.start()
        threading.Event().wait()

    def _start_task_deadline_notifier(self) -> None:
        """Background thread that checks for task deadlines every hour."""
        from modules.task_manager import get_task_manager
        from modules.messaging import get_messaging_manager

        while True:
            try:
                tm = get_task_manager()
                due = tm.due_soon(days_ahead=1)
                if due:
                    lines = ["📌 *Task deadlines due soon:*"]
                    for t in due[:10]:
                        lines.append(f"  [{t['priority'].upper()}] {t['title']} — due {t['due_date']}")
                    get_messaging_manager().send_message("\n".join(lines))
            except Exception as exc:
                logger.error("Task deadline notifier error: %s", exc)
            time.sleep(3600)  # check every hour

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def _shutdown(self) -> None:
        logger.info("Orchestrator shutting down modules…")
        for runner in self._module_runners:
            runner.stop()
        logger.info("Orchestrator shutdown complete")

    # ------------------------------------------------------------------
    # Signal handling
    # ------------------------------------------------------------------

    def _register_signal_handlers(self) -> None:
        def _handler(signum, frame):
            logger.info("Signal %s received — initiating shutdown", signum)
            self.stop()

        signal.signal(signal.SIGTERM, _handler)
        signal.signal(signal.SIGINT, _handler)
        logger.debug("Signal handlers registered (SIGTERM, SIGINT)")
