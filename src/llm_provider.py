"""
src/llm_provider.py

LLM provider abstraction for the Day 4 agent.

Design goals:
  - No API key is ever hardcoded.  Keys come from environment variables.
  - A deterministic fallback is used when no provider is configured, so the
    agent remains testable and runnable without secrets.
  - The provider interface is small and mockable for unit tests.
"""

import json
import os
import re
import time
from typing import Any

from .config import CONFIG


class LLMProvider:
    """
    Minimal LLM provider interface.

    Subclasses implement ``generate``.  The base class handles the fallback
    behaviour used during development/testing.
    """

    def generate(self, prompt: str, **kwargs: Any) -> dict:
        raise NotImplementedError

    def generate_json(self, prompt: str, **kwargs: Any) -> dict:
        """Generate and parse a JSON object.  Returns a dict on success."""
        raw = self.generate(prompt, **kwargs)
        return _parse_json(raw)


def _parse_json(raw: str) -> dict:
    """Best-effort JSON extraction from an LLM response."""
    if not raw:
        return {"_parse_error": True, "raw": ""}

    text = raw.strip()
    # Strip markdown code fences if present.
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find the first {...} block.
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass
    return {"_parse_error": True, "raw": raw}


class FallbackLLMProvider(LLMProvider):
    """
    Deterministic fallback used when no LLM provider is configured.

    It does NOT attempt to be clever.  It produces a safe, honest response
    that recommends escalation or clarification.  This keeps the agent
    testable and runnable without API access.
    """

    def generate(self, prompt: str, **kwargs: Any) -> str:
        # Extract the customer message and escalation decision from the prompt
        # so the fallback can behave sensibly.
        decision = _extract_field(prompt, "Decision:")
        reason = _extract_field(prompt, "Reason:")

        if decision and decision.strip().upper() == "ESCALATE":
            return (
                "Thank you for reaching out. Based on the information provided, "
                "this request needs to be reviewed by a support specialist who "
                "can access your account details. "
                + (f"Reason: {reason.strip()}" if reason else "")
            ).strip()

        return (
            "Thank you for contacting support. I would like to help, but I need "
            "a little more information to assist you properly. Could you please "
            "confirm the details of your request?"
        ).strip()


class RemoteLLMProvider(LLMProvider):
    """
    Thin wrapper around an OpenAI-compatible HTTP API.

    Configuration is read from environment variables ONLY.  No key is ever
    stored in source code, README, or configuration files.
    """

    def __init__(self, api_key: str | None = None,
                 model: str | None = None,
                 base_url: str | None = None,
                 timeout: float | None = None,
                 temperature: float | None = None,
                 min_interval: float | None = None):
        self.api_key = api_key or CONFIG.llm_api_key or os.environ.get("LLM_API_KEY", "")
        self.model = model or CONFIG.llm_model
        self.base_url = base_url or CONFIG.llm_base_url or os.environ.get("LLM_BASE_URL", "")
        self.timeout = timeout or CONFIG.llm_timeout_seconds
        self.temperature = (
            temperature if temperature is not None else CONFIG.llm_temperature
        )
        # Proactive rate limiter — reads LLM_MIN_INTERVAL_SECONDS from config.
        # Default: 5s (Google AI Studio 15 RPM). Set to 22s for OpenAI free tier.
        self._min_interval = min_interval if min_interval is not None else CONFIG.llm_min_interval_seconds
        self._last_call_time: float = 0.0

    def is_configured(self) -> bool:
        return bool(self.api_key)

    def generate(self, prompt: str, **kwargs: Any) -> str:
        if not self.is_configured():
            raise RuntimeError(
                "No LLM API key configured. Set LLM_API_KEY in your environment "
                "or .env file, or use the fallback provider."
            )

        try:
            import httpx
        except ImportError as exc:
            raise RuntimeError(
                "httpx is required for the remote LLM provider. "
                "Install it with: pip install httpx"
            ) from exc

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
        }
        if "max_tokens" in kwargs:
            payload["max_tokens"] = kwargs["max_tokens"]

        url = self.base_url.rstrip("/") + "/chat/completions" if self.base_url \
            else "https://api.openai.com/v1/chat/completions"

        # Proactive rate limiter: sleep BEFORE the request if needed so we
        # never hit 429 in the first place (free tier = 3 RPM = 1 call/20s).
        import logging as _logging
        _log = _logging.getLogger(__name__)
        elapsed = time.time() - self._last_call_time
        if elapsed < self._min_interval:
            wait_for = self._min_interval - elapsed
            _log.info(f"Rate limiter: waiting {wait_for:.1f}s before next API call...")
            time.sleep(wait_for)

        self._last_call_time = time.time()

        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(url, headers=headers, json=payload)

            # Handle transient server errors (503) with one retry.
            if resp.status_code == 503:
                _log.warning("503 Service Unavailable. Retrying in 10s...")
                time.sleep(10)
                self._last_call_time = time.time()
                resp = client.post(url, headers=headers, json=payload)

            if resp.status_code == 429:
                # Parse error body — Google wraps errors in a list, OpenAI uses a dict.
                try:
                    body = resp.json()
                    # Unwrap list format: [{"error": {...}}]
                    if isinstance(body, list) and body:
                        body = body[0]
                    err = body.get("error", {}) if isinstance(body, dict) else {}
                    code = err.get("code", "")
                    etype = err.get("type", "") or err.get("status", "")
                    if code in ("credit_balance_exhausted", "billing_hard_limit_reached") \
                            or etype in ("insufficient_quota", "BILLING_DISABLED"):
                        raise RuntimeError(
                            f"OpenAI/Google account has no credits or billing is disabled. "
                            f"(error: {code or etype})"
                        )
                except (ValueError, KeyError, AttributeError):
                    pass  # not JSON or unexpected shape — treat as a normal rate limit

                # True rate limit: wait for TPM window to reset (up to 3 retries).
                max_retries = 3
                wait = float(resp.headers.get("retry-after", 60))
                for attempt in range(max_retries):
                    _log.warning(
                        f"429 rate limit. Sleeping {wait:.0f}s "
                        f"(attempt {attempt + 1}/{max_retries})..."
                    )
                    time.sleep(wait)
                    self._last_call_time = time.time()
                    resp = client.post(url, headers=headers, json=payload)
                    if resp.status_code != 429:
                        break
                    wait = min(wait * 2, 120)  # cap at 2 minutes

            resp.raise_for_status()
            data = resp.json()
        return data["choices"][0]["message"]["content"]


def _extract_field(prompt: str, label: str) -> str:
    """Extract a 'Label: value' line from a prompt string."""
    for line in prompt.splitlines():
        stripped = line.strip()
        if stripped.startswith(label):
            return stripped[len(label):].strip()
    return ""


def get_provider() -> LLMProvider:
    """
    Return a configured provider if one is available, otherwise the fallback.

    This keeps the agent runnable in CI and on developer machines without
    secrets while still supporting a real provider when configured.
    """
    if CONFIG.llm_api_key or os.environ.get("LLM_API_KEY"):
        return RemoteLLMProvider()
    return FallbackLLMProvider()


def main():
    provider = get_provider()
    print(f"Provider: {type(provider).__name__}")
    print(f"Configured: {getattr(provider, 'is_configured', lambda: False)()}")
    reply = provider.generate("Hello")
    print(f"Sample reply: {reply}")


if __name__ == "__main__":
    main()