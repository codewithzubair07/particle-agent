from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock


def test_only_timed_events_in_the_reminder_window_notify(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "particle-agent"))
    from modules import calendar_manager

    config = SimpleNamespace(calendar=SimpleNamespace(reminder_minutes=7))
    monkeypatch.setattr(calendar_manager, "get_config", lambda: config)
    monkeypatch.setattr(calendar_manager.CalendarManager, "_build_service", lambda self: None)
    manager = calendar_manager.CalendarManager()

    clock = Mock(wraps=datetime)
    clock.now.return_value = datetime(2030, 1, 1, 23, 53, tzinfo=timezone.utc)
    monkeypatch.setattr(calendar_manager, "datetime", clock)
    events = [
        {"id": "holiday", "summary": "Holiday", "start": {"date": "2030-01-02"}},
        {
            "id": "standup",
            "summary": "Standup",
            "start": {"dateTime": "2030-01-02T05:30:00+05:30"},
        },
        {"id": "later", "summary": "Later", "start": {"dateTime": "2030-01-02T00:01:00Z"}},
    ]
    monkeypatch.setattr(manager, "get_upcoming_events", lambda max_results: events)
    messages = []
    manager.set_telegram_notifier(messages.append)

    manager._check_reminders()

    assert len(messages) == 1, messages
    assert "Standup" in messages[0]
    assert "00:00 UTC" in messages[0]
