"""Email management module for Particle.

Connects to Gmail via IMAP for reading and SMTP for sending.  Responsibilities:
  * Poll inbox every 15 minutes and categorise messages (urgent/normal/spam).
  * Draft LLM-powered replies using user context and writing style.
  * Notify the Telegram home chat when an urgent email arrives.
  * Auto-unsubscribe from promotional/spam by sending an unsubscribe reply.
  * Send a daily digest to Telegram at 09:00 UTC containing unread count,
    top-3 urgent summaries, and a list of emails awaiting a reply.

All credentials come from the config/env system — never hard-coded.
"""

from __future__ import annotations

import email as email_lib
import email.header
import imaplib
import logging
import smtplib
import threading
import time
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional

from modules.config_loader import get_config

logger = logging.getLogger("particle.email_manager")

_URGENT_KEYWORDS = (
    "urgent", "asap", "immediately", "critical", "emergency",
    "action required", "important", "deadline",
)
_SPAM_KEYWORDS = (
    "unsubscribe", "click here", "limited time", "free offer",
    "you won", "prize", "click to claim", "opt-out",
)


def _decode_header(value: str) -> str:
    """Decode an RFC-2047 encoded email header to a plain string."""
    parts = email.header.decode_header(value)
    decoded: list[str] = []
    for chunk, charset in parts:
        if isinstance(chunk, bytes):
            decoded.append(chunk.decode(charset or "utf-8", errors="replace"))
        else:
            decoded.append(str(chunk))
    return " ".join(decoded)


def _categorise(subject: str, body: str) -> str:
    """Heuristically categorise a message as 'urgent', 'spam', or 'normal'."""
    combined = (subject + " " + body).lower()
    if any(k in combined for k in _URGENT_KEYWORDS):
        return "urgent"
    if any(k in combined for k in _SPAM_KEYWORDS):
        return "spam"
    return "normal"


class EmailManager:
    """Gmail inbox poller, categoriser, and reply drafter."""

    def __init__(self) -> None:
        cfg = get_config()
        self._cfg = cfg.email
        self._address: str = getattr(cfg.email, "address", "")
        self._password: str = getattr(cfg.email, "password", "")
        self._imap_host: str = getattr(cfg.email, "imap_host", "imap.gmail.com")
        self._imap_port: int = int(getattr(cfg.email, "imap_port", 993))
        self._smtp_host: str = getattr(cfg.email, "smtp_host", "smtp.gmail.com")
        self._smtp_port: int = int(getattr(cfg.email, "smtp_port", 587))
        self._poll_interval: int = int(getattr(cfg.app, "polling_interval_minutes", 15)) * 60
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._send_telegram: Optional[callable] = None  # injected by orchestrator

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_telegram_notifier(self, fn: callable) -> None:
        """Inject a callable(message: str) used to push Telegram alerts."""
        self._send_telegram = fn

    def start(self) -> None:
        """Start background polling loop in a daemon thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True, name="email-poller")
        self._thread.start()
        logger.info("EmailManager polling started (interval=%ds)", self._poll_interval)

    def stop(self) -> None:
        """Signal the polling loop to stop."""
        self._running = False
        logger.info("EmailManager polling stopped")

    def send_digest(self) -> None:
        """Compose and send a daily digest to the Telegram home chat."""
        try:
            messages = self._fetch_unread()
        except Exception as exc:
            logger.error("Digest: failed to fetch unread messages: %s", exc)
            return

        total = len(messages)
        urgent = [m for m in messages if m["category"] == "urgent"]
        needs_reply = [m for m in messages if m["category"] in ("urgent", "normal")][:5]

        lines = [
            f"📧 *Email Digest — {datetime.now(timezone.utc).strftime('%Y-%m-%d')}*",
            f"Unread: {total}",
            "",
        ]
        if urgent:
            lines.append("⚠️ *Top urgent emails:*")
            for m in urgent[:3]:
                lines.append(f"  • From: {m['from']} — {m['subject'][:60]}")
        if needs_reply:
            lines.append("\n📝 *Awaiting reply:*")
            for m in needs_reply[:5]:
                lines.append(f"  • {m['subject'][:60]}")

        self._notify("\n".join(lines))

    # ------------------------------------------------------------------
    # Polling loop
    # ------------------------------------------------------------------

    def _poll_loop(self) -> None:
        while self._running:
            try:
                self._process_inbox()
            except Exception as exc:
                logger.error("EmailManager poll error: %s", exc, exc_info=True)
            time.sleep(self._poll_interval)

    def _process_inbox(self) -> None:
        """Fetch unread messages, categorise, and act on them."""
        messages = self._fetch_unread()
        if not messages:
            logger.debug("Email poll: no new messages")
            return

        logger.info("Email poll: %d unread messages", len(messages))
        for msg in messages:
            cat = msg["category"]
            logger.info("Email id=%s cat=%s from=%r sub=%r", msg["uid"], cat, msg["from"], msg["subject"])

            if cat == "urgent":
                alert = (
                    f"🚨 *Urgent email!*\n"
                    f"From: {msg['from']}\n"
                    f"Subject: {msg['subject']}\n\n"
                    f"{msg['body'][:300]}"
                )
                self._notify(alert)

            if cat == "spam":
                self._attempt_unsubscribe(msg)

    # ------------------------------------------------------------------
    # IMAP helpers
    # ------------------------------------------------------------------

    def _connect_imap(self) -> imaplib.IMAP4_SSL:
        conn = imaplib.IMAP4_SSL(self._imap_host, self._imap_port)
        conn.login(self._address, self._password)
        return conn

    def _fetch_unread(self) -> list[dict]:
        """Return a list of unread message dicts from the Gmail inbox."""
        if not self._address or not self._password:
            logger.warning("Email credentials not configured — skipping fetch")
            return []
        messages: list[dict] = []
        try:
            conn = self._connect_imap()
            conn.select("INBOX")
            _, data = conn.search(None, "UNSEEN")
            uid_list = data[0].split() if data[0] else []

            for uid in uid_list[:20]:  # cap at 20 per poll
                _, raw = conn.fetch(uid, "(RFC822)")
                if not raw or raw[0] is None:
                    continue
                raw_bytes = raw[0][1] if isinstance(raw[0], tuple) else b""
                msg = email_lib.message_from_bytes(raw_bytes)

                subject = _decode_header(msg.get("Subject", "(no subject)"))
                sender = _decode_header(msg.get("From", ""))
                body = self._extract_body(msg)
                category = _categorise(subject, body)

                messages.append(
                    {
                        "uid": uid.decode(),
                        "from": sender,
                        "subject": subject,
                        "body": body,
                        "category": category,
                    }
                )

            conn.logout()
        except imaplib.IMAP4.error as exc:
            logger.error("IMAP error: %s", exc)
        except Exception as exc:
            logger.error("Unexpected IMAP error: %s", exc, exc_info=True)
        return messages

    def _extract_body(self, msg: email_lib.message.Message) -> str:
        """Extract the plain-text body from an email message."""
        body_parts: list[str] = []
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    charset = part.get_content_charset() or "utf-8"
                    payload = part.get_payload(decode=True)
                    if payload:
                        body_parts.append(payload.decode(charset, errors="replace"))
        else:
            payload = msg.get_payload(decode=True)
            if payload:
                charset = msg.get_content_charset() or "utf-8"
                body_parts.append(payload.decode(charset, errors="replace"))
        return "\n".join(body_parts).strip()

    # ------------------------------------------------------------------
    # SMTP helpers
    # ------------------------------------------------------------------

    def send_email(self, to: str, subject: str, body: str) -> bool:
        """Send an email; returns True on success."""
        if not self._address or not self._password:
            logger.warning("Email credentials not configured — cannot send")
            return False
        try:
            msg = MIMEMultipart("alternative")
            msg["From"] = self._address
            msg["To"] = to
            msg["Subject"] = subject
            msg.attach(MIMEText(body, "plain", "utf-8"))

            with smtplib.SMTP(self._smtp_host, self._smtp_port) as server:
                server.ehlo()
                server.starttls()
                server.login(self._address, self._password)
                server.sendmail(self._address, [to], msg.as_string())
            logger.info("Email sent to=%s subject=%r", to, subject)
            return True
        except smtplib.SMTPException as exc:
            logger.error("SMTP error sending to %s: %s", to, exc)
            return False
        except Exception as exc:
            logger.error("Unexpected send error: %s", exc, exc_info=True)
            return False

    def draft_reply(self, original: dict) -> str:
        """Use the LLM to draft a reply to the given email dict."""
        from modules.llm_router import complete
        from modules.context_loader import get_context_loader

        ctx = get_context_loader().build_context_string(original["subject"])
        system = (
            "You are Particle, a personal AI assistant. "
            "Draft a professional, concise reply on behalf of the user. "
            "Match the formality of the original message.\n"
        )
        if ctx:
            system += f"\n{ctx}\n"
        prompt = (
            f"Original email from {original['from']}:\n"
            f"Subject: {original['subject']}\n\n"
            f"{original['body'][:1000]}\n\n"
            "Draft a reply:"
        )
        try:
            return complete(prompt, system)
        except Exception as exc:
            logger.error("LLM draft_reply failed: %s", exc)
            return ""

    def _attempt_unsubscribe(self, msg: dict) -> None:
        """Send a brief unsubscribe request reply for a spam message."""
        reply_body = "Please unsubscribe me from this mailing list. Thank you."
        to_addr = msg["from"]
        # Extract bare email address
        if "<" in to_addr:
            to_addr = to_addr.split("<")[1].rstrip(">")
        subject = f"Re: {msg['subject']}"
        sent = self.send_email(to_addr, subject, reply_body)
        if sent:
            logger.info("Unsubscribe sent to %s", to_addr)

    # ------------------------------------------------------------------
    # Notifications
    # ------------------------------------------------------------------

    def _notify(self, message: str) -> None:
        """Send a notification via the injected Telegram callable."""
        if self._send_telegram:
            try:
                self._send_telegram(message)
            except Exception as exc:
                logger.error("EmailManager notify error: %s", exc)
        else:
            logger.info("EMAIL NOTIFY (no Telegram): %s", message[:120])


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_instance: Optional[EmailManager] = None
_singleton_lock = threading.Lock()


def get_email_manager() -> EmailManager:
    """Return the module-level :class:`EmailManager` singleton."""
    global _instance
    with _singleton_lock:
        if _instance is None:
            _instance = EmailManager()
    return _instance
