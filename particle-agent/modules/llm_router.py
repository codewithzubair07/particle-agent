"""LLM provider routing module for Particle.

Provides multi-provider fallback with automatic retry logic.  Provider chain:
  1. Google Gemini 2.0 Flash (gemini-2.0-flash-exp)
  2. Google Gemini 1.5 Flash (gemini-1.5-flash)
  3. OpenRouter — rotating: llama-3.1-8b-instruct, nemotron-super, mistral-7b

On a 429 rate-limit the router immediately moves to the next provider.
Other errors are retried up to ``max_retries_per_provider`` times with
exponential back-off before the next provider is tried.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import requests

from modules.config_loader import get_config

logger = logging.getLogger("particle.llm_router")

_GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)
_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


class RateLimitError(Exception):
    """Raised when a provider returns HTTP 429."""


class LLMRouter:
    """Multi-provider LLM router with automatic fallback and retry logic."""

    def __init__(self) -> None:
        cfg = get_config()
        self._cfg = cfg.llm
        self._gemini_key: str = getattr(cfg.llm, "gemini_api_key", "")
        self._openrouter_key: str = getattr(cfg.llm, "openrouter_api_key", "")
        self._max_retries: int = int(getattr(cfg.llm, "max_retries_per_provider", 3))
        self._timeout: int = int(getattr(cfg.llm, "request_timeout_seconds", 45))
        self._or_model_idx: int = 0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def complete(self, prompt: str, system: str = "") -> str:
        """Return a completion string using the configured provider chain.

        Tries each provider in order, switching immediately on rate-limit and
        after all retries are exhausted for other errors.
        """
        providers: list[str] = list(self._cfg.provider_order)

        for provider in providers:
            result = self._try_provider(provider, prompt, system)
            if result is not None:
                return result

        raise RuntimeError(
            "All LLM providers exhausted without producing a successful response."
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _try_provider(self, provider: str, prompt: str, system: str) -> str | None:
        """Attempt completion on *provider*, retrying transient errors.

        Returns the response text on success, or ``None`` when the provider
        should be skipped (rate-limit or all retries exhausted).
        """
        for attempt in range(1, self._max_retries + 1):
            try:
                logger.info(
                    "LLM provider=%s attempt=%d/%d", provider, attempt, self._max_retries
                )
                if provider in ("gemini-2.0-flash-exp", "gemini-1.5-flash"):
                    return self._call_gemini(provider, prompt, system)
                if provider == "openrouter":
                    return self._call_openrouter(prompt, system)
                logger.warning("Unknown LLM provider '%s' — skipping", provider)
                return None
            except RateLimitError as exc:
                logger.warning(
                    "Rate limit on provider=%s attempt=%d: %s — switching provider",
                    provider,
                    attempt,
                    exc,
                )
                return None
            except Exception as exc:
                logger.warning(
                    "Provider=%s attempt=%d error: %s", provider, attempt, exc
                )
                if attempt < self._max_retries:
                    delay = 2**attempt
                    logger.info("Retrying in %ds …", delay)
                    time.sleep(delay)

        logger.error(
            "Provider=%s exhausted all %d retries — moving to next provider",
            provider,
            self._max_retries,
        )
        return None

    def _call_gemini(self, model: str, prompt: str, system: str) -> str:
        """Call the Google Gemini REST API."""
        if not self._gemini_key:
            raise RuntimeError("GEMINI_API_KEY is not configured")

        # Gemini does not have a dedicated system role in v1beta; prepend it.
        user_text = f"[System]\n{system}\n\n[User]\n{prompt}" if system else prompt
        payload: dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": user_text}]}],
            "generationConfig": {"temperature": 0.7, "maxOutputTokens": 2048},
        }

        url = _GEMINI_URL.format(model=model)
        resp = requests.post(
            url,
            params={"key": self._gemini_key},
            json=payload,
            timeout=self._timeout,
        )

        if resp.status_code == 429:
            raise RateLimitError(f"Gemini {model}: {resp.text[:200]}")
        resp.raise_for_status()

        data = resp.json()
        try:
            text: str = data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError) as exc:
            raise RuntimeError(
                f"Unexpected Gemini response format: {str(data)[:400]}"
            ) from exc

        logger.info("Gemini model=%s responded (%d chars)", model, len(text))
        return text.strip()

    def _call_openrouter(self, prompt: str, system: str) -> str:
        """Call the OpenRouter API, rotating through the free model list."""
        if not self._openrouter_key:
            raise RuntimeError("OPENROUTER_API_KEY is not configured")

        models: list[str] = list(self._cfg.openrouter_models)
        model = models[self._or_model_idx % len(models)]
        logger.info(
            "OpenRouter model=%s (rotation_idx=%d)", model, self._or_model_idx
        )

        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": 2048,
            "temperature": 0.7,
        }
        headers = {
            "Authorization": f"Bearer {self._openrouter_key}",
            "HTTP-Referer": "https://github.com/particle-agent",
            "X-Title": "Particle Agent",
        }

        resp = requests.post(
            _OPENROUTER_URL,
            headers=headers,
            json=payload,
            timeout=self._timeout,
        )

        if resp.status_code == 429:
            self._or_model_idx += 1
            raise RateLimitError(
                f"OpenRouter model={model} rate-limited: {resp.text[:200]}"
            )
        resp.raise_for_status()

        data = resp.json()
        try:
            text: str = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            raise RuntimeError(
                f"Unexpected OpenRouter response format: {str(data)[:400]}"
            ) from exc

        logger.info("OpenRouter model=%s responded (%d chars)", model, len(text))
        self._or_model_idx += 1
        return text.strip()


# ---------------------------------------------------------------------------
# Module-level singleton helpers
# ---------------------------------------------------------------------------

_instance: LLMRouter | None = None


def get_router() -> LLMRouter:
    """Return the module-level :class:`LLMRouter` singleton."""
    global _instance
    if _instance is None:
        _instance = LLMRouter()
    return _instance


def complete(prompt: str, system: str = "") -> str:
    """Convenience wrapper — generate a completion via the global router."""
    return get_router().complete(prompt, system)
