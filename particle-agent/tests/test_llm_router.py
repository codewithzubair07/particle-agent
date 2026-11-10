"""Tests for LLMRouter and OpenRouter authorization header handling."""

import os
import sys
import traceback
from unittest.mock import MagicMock, patch

import pytest
import requests

# Ensure particle-agent root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.config_loader import ConfigNamespace
from modules.llm_router import LLMRouter, RateLimitError


@pytest.fixture
def dummy_router():
    """Create an LLMRouter instance with controlled test configuration."""
    cfg = ConfigNamespace(
        {
            "llm": ConfigNamespace(
                {
                    "provider_order": ["openrouter"],
                    "gemini_api_key": "test-gemini-key",
                    "openrouter_api_key": "sk-or-v1-secret-test-token",
                    "max_retries_per_provider": 2,
                    "request_timeout_seconds": 10,
                    "openrouter_models": [
                        "meta-llama/llama-3.1-8b-instruct:free",
                        "mistralai/mistral-7b-instruct:free",
                    ],
                }
            )
        }
    )
    with patch("modules.llm_router.get_config", return_value=cfg):
        router = LLMRouter()
    return router


def test_openrouter_sends_bearer_authorization_header(dummy_router):
    """Verify OpenRouter request includes Authorization: Bearer <key>."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": "Hello from OpenRouter"}}]
    }

    with patch("modules.llm_router.requests.post", return_value=mock_resp) as mock_post:
        result = dummy_router._call_openrouter("Test prompt", system="System prompt")

        assert result == "Hello from OpenRouter"
        mock_post.assert_called_once()
        _, kwargs = mock_post.call_args
        headers = kwargs["headers"]
        assert headers["Authorization"] == "Bearer sk-or-v1-secret-test-token"
        assert headers["HTTP-Referer"] == "https://github.com/particle-agent"
        assert headers["X-Title"] == "Particle Agent"


def test_openrouter_strips_key_whitespace():
    """Verify extra whitespace on openrouter_api_key is cleanly stripped."""
    cfg = ConfigNamespace(
        {
            "llm": ConfigNamespace(
                {
                    "provider_order": ["openrouter"],
                    "openrouter_api_key": "  sk-or-v1-whitespace-key  \n",
                    "openrouter_models": ["model-a"],
                    "max_retries_per_provider": 1,
                    "request_timeout_seconds": 5,
                }
            )
        }
    )
    with patch("modules.llm_router.get_config", return_value=cfg):
        router = LLMRouter()

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": "Success"}}]
    }
    with patch("modules.llm_router.requests.post", return_value=mock_resp) as mock_post:
        router._call_openrouter("Prompt")
        _, kwargs = mock_post.call_args
        assert kwargs["headers"]["Authorization"] == "Bearer sk-or-v1-whitespace-key"


def test_openrouter_missing_key_raises(dummy_router):
    """Verify RuntimeError is raised when openrouter_api_key is empty."""
    dummy_router._openrouter_key = ""
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY is not configured"):
        dummy_router._call_openrouter("Prompt")


@pytest.mark.parametrize("prefix", ["Rate limit exceeded for key ", "x" * 190])
def test_openrouter_key_not_exposed_in_rate_limit(dummy_router, prefix):
    """Verify raw key is redacted if present in rate-limit response text."""
    secret = "sk-or-v1-secret-test-token"
    mock_resp = MagicMock()
    mock_resp.status_code = 429
    mock_resp.text = f"{prefix}{secret} on endpoint"

    with patch("modules.llm_router.requests.post", return_value=mock_resp):
        with pytest.raises(RateLimitError) as exc_info:
            dummy_router._call_openrouter("Prompt")

        error_message = str(exc_info.value)
        assert secret not in error_message
        assert "[REDACTED]" in error_message


def test_openrouter_key_not_exposed_in_http_error(dummy_router):
    """Verify raw key is redacted if present in HTTP error messages."""
    secret = "sk-or-v1-secret-test-token"
    mock_resp = MagicMock()
    mock_resp.status_code = 401
    mock_resp.raise_for_status.side_effect = requests.HTTPError(
        f"401 Client Error: Unauthorized for token {secret}"
    )

    with patch("modules.llm_router.requests.post", return_value=mock_resp):
        with pytest.raises(RuntimeError) as exc_info:
            dummy_router._call_openrouter("Prompt")

        error_message = str(exc_info.value)
        assert secret not in error_message
        assert "[REDACTED]" in error_message


def test_openrouter_key_not_exposed_in_network_error(dummy_router):
    """Verify raw key is redacted if present in network/request exceptions."""
    secret = "sk-or-v1-secret-test-token"
    with patch(
        "modules.llm_router.requests.post",
        side_effect=requests.ConnectionError(f"Connection refused with key {secret}"),
    ):
        with pytest.raises(RuntimeError) as exc_info:
            dummy_router._call_openrouter("Prompt")

        error_message = str(exc_info.value)
        assert secret not in error_message
        assert "[REDACTED]" in error_message


def test_openrouter_key_not_exposed_in_json_decode_error(dummy_router):
    secret = dummy_router._openrouter_key
    mock_resp = MagicMock(status_code=200)
    mock_resp.json.side_effect = requests.exceptions.JSONDecodeError(
        f"Invalid value {secret}", secret, 0
    )

    with (
        patch("modules.llm_router.requests.post", return_value=mock_resp),
        pytest.raises(RuntimeError, match="^OpenRouter returned invalid JSON$") as exc_info,
    ):
        dummy_router._call_openrouter("Prompt")

    assert secret not in "".join(traceback.format_exception(exc_info.value))


@pytest.mark.parametrize(
    "payload",
    [
        None,
        {"error": "sk-or-v1-secret-test-token"},
        {"choices": []},
        {"choices": [{"message": None}]},
        {"choices": [{"message": {"content": ["sk-or-v1-secret-test-token"]}}]},
    ],
)
def test_openrouter_key_not_exposed_in_malformed_response(dummy_router, payload):
    mock_resp = MagicMock(status_code=200)
    mock_resp.json.return_value = payload

    with (
        patch("modules.llm_router.requests.post", return_value=mock_resp),
        pytest.raises(
            RuntimeError, match="^Unexpected OpenRouter response format$"
        ) as exc_info,
    ):
        dummy_router._call_openrouter("Prompt")

    assert dummy_router._openrouter_key not in "".join(
        traceback.format_exception(exc_info.value)
    )


def test_openrouter_rotation_on_rate_limit(dummy_router):
    """Verify model index advances upon encountering 429."""
    assert dummy_router._or_model_idx == 0

    mock_resp = MagicMock()
    mock_resp.status_code = 429
    mock_resp.text = "Too many requests"

    with patch("modules.llm_router.requests.post", return_value=mock_resp):
        with pytest.raises(RateLimitError):
            dummy_router._call_openrouter("Prompt")

    assert dummy_router._or_model_idx == 1


def test_complete_fallback_from_gemini_to_openrouter():
    """Verify complete() falls back to OpenRouter when Gemini hits RateLimitError."""
    cfg = ConfigNamespace(
        {
            "llm": ConfigNamespace(
                {
                    "provider_order": ["gemini-2.0-flash-exp", "openrouter"],
                    "gemini_api_key": "gemini-key",
                    "openrouter_api_key": "sk-or-v1-key",
                    "openrouter_models": ["model-or"],
                    "max_retries_per_provider": 1,
                    "request_timeout_seconds": 5,
                }
            )
        }
    )
    with patch("modules.llm_router.get_config", return_value=cfg):
        router = LLMRouter()

    def mock_post(url, **kwargs):
        resp = MagicMock()
        if "generativelanguage" in url:
            resp.status_code = 429
            resp.text = "Rate limited"
        else:
            resp.status_code = 200
            resp.json.return_value = {
                "choices": [{"message": {"content": "OpenRouter Fallback Success"}}]
            }
        return resp

    with patch("modules.llm_router.requests.post", side_effect=mock_post):
        res = router.complete("Hello world")
        assert res == "OpenRouter Fallback Success"
