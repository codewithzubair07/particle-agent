"""CLI interface for Particle.

Provides operator commands for managing the running agent process,
viewing logs, and directly interacting with core modules.

Usage examples
--------------
    python cli.py start                  # start the agent in the background
    python cli.py stop                   # send SIGTERM to the running agent
    python cli.py status                 # check if the agent process is running
    python cli.py logs [-n N]            # tail the last N lines of the log file
    python cli.py tasks list             # list pending tasks
    python cli.py tasks add "Buy milk"   # add a quick task
    python cli.py tasks done <id>        # mark task <id> as completed
    python cli.py briefing               # trigger a briefing immediately
    python cli.py ask "What's on today?" # one-shot LLM query
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# PID-file helpers
# ---------------------------------------------------------------------------

_PID_FILE = Path("particle.pid")


def _write_pid(pid: int) -> None:
    _PID_FILE.write_text(str(pid))


def _read_pid() -> int | None:
    if not _PID_FILE.exists():
        return None
    try:
        return int(_PID_FILE.read_text().strip())
    except ValueError:
        return None


def _clear_pid() -> None:
    try:
        _PID_FILE.unlink(missing_ok=True)
    except OSError:
        pass


def _is_running(pid: int) -> bool:
    """Return True if a process with *pid* exists."""
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


# ---------------------------------------------------------------------------
# Command implementations
# ---------------------------------------------------------------------------

def cmd_start(args: argparse.Namespace) -> int:
    pid = _read_pid()
    if pid and _is_running(pid):
        print(f"Particle is already running (PID {pid}).")
        return 1

    script = Path(__file__).parent / "main.py"
    env = {**os.environ}
    proc = subprocess.Popen(
        [sys.executable, str(script)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
        env=env,
    )
    _write_pid(proc.pid)
    print(f"Particle started (PID {proc.pid}).")
    return 0


def cmd_stop(args: argparse.Namespace) -> int:
    pid = _read_pid()
    if pid is None or not _is_running(pid):
        print("Particle is not running.")
        _clear_pid()
        return 1
    try:
        os.kill(pid, signal.SIGTERM)
        # Wait up to 10s for the process to exit
        for _ in range(20):
            time.sleep(0.5)
            if not _is_running(pid):
                break
        _clear_pid()
        print(f"Particle (PID {pid}) stopped.")
        return 0
    except OSError as exc:
        print(f"Error stopping Particle (PID {pid}): {exc}")
        return 1


def cmd_status(args: argparse.Namespace) -> int:
    pid = _read_pid()
    if pid and _is_running(pid):
        print(f"✅ Particle is running (PID {pid}).")
        return 0
    print("🔴 Particle is not running.")
    _clear_pid()
    return 1


def cmd_logs(args: argparse.Namespace) -> int:
    n: int = args.n

    # Resolve log path from config
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from modules.config_loader import get_config

        cfg = get_config()
        log_path = Path(cfg.paths.log_file)
    except Exception:
        log_path = Path("logs/particle.log")

    if not log_path.exists():
        print(f"Log file not found: {log_path}")
        return 1

    try:
        with log_path.open("r", encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()
        for line in lines[-n:]:
            print(line, end="")
        return 0
    except OSError as exc:
        print(f"Error reading log file: {exc}")
        return 1


def cmd_tasks(args: argparse.Namespace) -> int:
    sys.path.insert(0, str(Path(__file__).parent))
    from modules.task_manager import get_task_manager

    tm = get_task_manager()
    action = args.task_action

    if action == "list":
        tasks = tm.pending()
        if not tasks:
            print("No pending tasks.")
        else:
            print(f"{'ID':>4}  {'PRI':<6}  {'DUE':<12}  TITLE")
            print("-" * 60)
            for t in tasks:
                due = t.get("due_date") or "—"
                print(f"{t['id']:>4}  {t['priority']:<6}  {due:<12}  {t['title']}")
        return 0

    if action == "add":
        title = " ".join(args.title)
        task_id = tm.create(title=title, priority=args.priority or "medium")
        print(f"Task #{task_id} created: {title}")
        return 0

    if action == "done":
        ok = tm.complete(args.id)
        if ok:
            print(f"Task #{args.id} marked as completed.")
        else:
            print(f"Task #{args.id} not found.")
        return 0 if ok else 1

    if action == "delete":
        ok = tm.delete(args.id)
        if ok:
            print(f"Task #{args.id} deleted.")
        else:
            print(f"Task #{args.id} not found.")
        return 0 if ok else 1

    print(f"Unknown task action: {action}")
    return 1


def cmd_briefing(args: argparse.Namespace) -> int:
    sys.path.insert(0, str(Path(__file__).parent))
    from modules.briefing import get_briefing_manager
    from modules.config_loader import get_config

    # Bootstrap logging
    import logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

    mgr = get_briefing_manager()

    def _printer(msg: str) -> None:
        print(msg)

    mgr.set_telegram_notifier(_printer)
    mgr.trigger_now()
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    sys.path.insert(0, str(Path(__file__).parent))
    import logging
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(message)s")

    from modules.llm_router import complete
    from modules.context_loader import get_context_loader

    question = " ".join(args.question)
    ctx = get_context_loader().build_context_string(question)
    system = "You are Particle, a helpful personal AI assistant."
    if ctx:
        system += f"\n\n{ctx}"

    try:
        answer = complete(question, system)
        print(answer)
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="particle",
        description="Particle AI agent — command-line interface",
    )
    sub = root.add_subparsers(dest="command", required=True)

    # start
    sub.add_parser("start", help="Start the Particle agent in the background")

    # stop
    sub.add_parser("stop", help="Stop the running Particle agent")

    # status
    sub.add_parser("status", help="Check if the Particle agent is running")

    # logs
    logs_p = sub.add_parser("logs", help="Display the last N log lines")
    logs_p.add_argument("-n", type=int, default=50, metavar="N", help="Lines to show (default: 50)")

    # tasks
    tasks_p = sub.add_parser("tasks", help="Manage tasks")
    task_sub = tasks_p.add_subparsers(dest="task_action", required=True)
    task_sub.add_parser("list", help="List pending tasks")
    add_p = task_sub.add_parser("add", help="Add a new task")
    add_p.add_argument("title", nargs="+", help="Task title")
    add_p.add_argument("--priority", choices=["high", "medium", "low"], default="medium")
    done_p = task_sub.add_parser("done", help="Mark a task as completed")
    done_p.add_argument("id", type=int)
    del_p = task_sub.add_parser("delete", help="Delete a task")
    del_p.add_argument("id", type=int)

    # briefing
    sub.add_parser("briefing", help="Trigger a briefing immediately")

    # ask
    ask_p = sub.add_parser("ask", help="Ask the LLM a one-shot question")
    ask_p.add_argument("question", nargs="+", help="Your question")

    return root


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

_DISPATCH = {
    "start":    cmd_start,
    "stop":     cmd_stop,
    "status":   cmd_status,
    "logs":     cmd_logs,
    "tasks":    cmd_tasks,
    "briefing": cmd_briefing,
    "ask":      cmd_ask,
}


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    handler = _DISPATCH.get(args.command)
    if handler is None:
        parser.print_help()
        return 1
    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
