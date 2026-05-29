"""LLM provider adapters for Drifter's in-app chat.

A thin, unified interface over Claude (Anthropic), Gemini (Google) and OpenAI.
Calls go **straight from this machine to the provider** using a key the user
stores locally — there is no intermediary server. Provider SDKs are optional and
lazily imported, so the package installs and runs (drift engine, storage, UI
scaffolding) without any of them; a missing SDK only matters when you actually
chat with that provider.

The message format used throughout is a list of
``{"role": "user"|"assistant", "content": str}`` dicts. Per-provider conversion is
factored into pure functions so it can be unit-tested offline.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

from cdm import config

__all__ = [
    "PROVIDERS",
    "LLMError",
    "LLMClient",
    "load_keys",
    "save_key",
    "get_key",
    "to_openai_messages",
    "to_anthropic_messages",
    "to_gemini_contents",
]


class LLMError(RuntimeError):
    """Raised for any provider/SDK/credential problem (with a helpful message)."""


# Provider registry: label, default model, SDK package + pip install hint.
PROVIDERS: Dict[str, Dict[str, str]] = {
    "claude": {
        "label": "Claude (Anthropic)",
        "default_model": "claude-sonnet-4-6",
        "sdk": "anthropic",
        "pip": "anthropic",
    },
    "gemini": {
        "label": "Gemini (Google)",
        "default_model": "gemini-2.5-flash",
        "sdk": "google.genai",
        "pip": "google-genai",
    },
    "openai": {
        "label": "OpenAI",
        "default_model": "gpt-4o",
        "sdk": "openai",
        "pip": "openai",
    },
}


# --------------------------------------------------------------------------- #
# Local key storage (a 0600 JSON file in the data dir — nothing leaves the PC)
# --------------------------------------------------------------------------- #
def _creds_path() -> Path:
    config.ensure_data_dir()
    return config.DATA_DIR / "credentials.json"


def load_keys() -> Dict[str, str]:
    """Return the locally stored ``{provider: api_key}`` map (possibly empty)."""
    path = _creds_path()
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
        return {k: str(v) for k, v in data.items()} if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_key(provider: str, api_key: str) -> None:
    """Persist ``api_key`` for ``provider`` to the local credentials file (0600)."""
    keys = load_keys()
    if api_key:
        keys[provider] = api_key
    else:
        keys.pop(provider, None)
    path = _creds_path()
    path.write_text(json.dumps(keys, indent=2))
    try:
        os.chmod(path, 0o600)
    except Exception:
        pass


def get_key(provider: str) -> Optional[str]:
    """Return the stored key for ``provider``, falling back to its env var."""
    keys = load_keys()
    if keys.get(provider):
        return keys[provider]
    env = {
        "claude": "ANTHROPIC_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "openai": "OPENAI_API_KEY",
    }.get(provider)
    return os.environ.get(env) if env else None


# --------------------------------------------------------------------------- #
# Message-format conversion (pure, testable)
# --------------------------------------------------------------------------- #
def _normalise(messages: List[dict]) -> List[dict]:
    """Coerce to clean ``{role, content}`` dicts; assistant for non-user roles."""
    out: List[dict] = []
    for m in messages or []:
        role = "user" if str(m.get("role", "user")).lower() == "user" else "assistant"
        content = str(m.get("content", "") or "")
        if content.strip():
            out.append({"role": role, "content": content})
    return out


def to_openai_messages(messages: List[dict], system: Optional[str]) -> List[dict]:
    """OpenAI chat format: optional leading system message + role/content list."""
    out: List[dict] = []
    if system:
        out.append({"role": "system", "content": system})
    out.extend(_normalise(messages))
    return out


def to_anthropic_messages(messages: List[dict]) -> List[dict]:
    """Anthropic messages list (system is passed separately, not inline)."""
    return _normalise(messages)


def to_gemini_contents(messages: List[dict]) -> List[dict]:
    """Gemini ``contents``: role 'user'/'model', text under 'parts'."""
    contents: List[dict] = []
    for m in _normalise(messages):
        role = "user" if m["role"] == "user" else "model"
        contents.append({"role": role, "parts": [{"text": m["content"]}]})
    return contents


# --------------------------------------------------------------------------- #
# Unified client
# --------------------------------------------------------------------------- #
class LLMClient:
    """Send a conversation to a provider and get the assistant's reply text."""

    def __init__(
        self,
        provider: str,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: int = 1024,
    ) -> None:
        if provider not in PROVIDERS:
            raise LLMError(
                f"Unknown provider {provider!r}; choose from {list(PROVIDERS)}."
            )
        self.provider = provider
        self.api_key = api_key or get_key(provider)
        self.model = model or PROVIDERS[provider]["default_model"]
        self.max_tokens = int(max_tokens)
        if not self.api_key:
            raise LLMError(
                f"No API key for {PROVIDERS[provider]['label']}. Add one in Settings."
            )

    def chat(self, messages: List[dict], system: Optional[str] = None) -> str:
        """Return the assistant reply for ``messages`` (optional ``system`` prompt)."""
        if self.provider == "claude":
            return self._chat_claude(messages, system)
        if self.provider == "gemini":
            return self._chat_gemini(messages, system)
        return self._chat_openai(messages, system)

    # -- per-provider implementations (lazy SDK imports) --------------------- #
    def _require(self, import_name: str):
        """Import a provider SDK or raise a helpful LLMError."""
        try:
            return __import__(import_name, fromlist=["_"])
        except Exception as exc:  # pragma: no cover - exercised only when absent
            pip = PROVIDERS[self.provider]["pip"]
            raise LLMError(
                f"The {PROVIDERS[self.provider]['label']} SDK is not installed. "
                f"Run: pip install {pip}"
            ) from exc

    def _chat_claude(self, messages: List[dict], system: Optional[str]) -> str:
        anthropic = self._require("anthropic")
        try:
            client = anthropic.Anthropic(api_key=self.api_key)
            kwargs = dict(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=to_anthropic_messages(messages),
            )
            if system:
                kwargs["system"] = system
            resp = client.messages.create(**kwargs)
            return "".join(
                getattr(b, "text", "") for b in resp.content
                if getattr(b, "type", "") == "text"
            ).strip()
        except LLMError:
            raise
        except Exception as exc:
            raise LLMError(f"Claude request failed: {exc}") from exc

    def _chat_gemini(self, messages: List[dict], system: Optional[str]) -> str:
        genai_pkg = self._require("google.genai")
        try:
            from google import genai  # type: ignore
            from google.genai import types  # type: ignore

            client = genai.Client(api_key=self.api_key)
            cfg = types.GenerateContentConfig(system_instruction=system) if system else None
            resp = client.models.generate_content(
                model=self.model,
                contents=to_gemini_contents(messages),
                config=cfg,
            )
            return (getattr(resp, "text", "") or "").strip()
        except LLMError:
            raise
        except Exception as exc:
            raise LLMError(f"Gemini request failed: {exc}") from exc

    def _chat_openai(self, messages: List[dict], system: Optional[str]) -> str:
        openai_pkg = self._require("openai")
        try:
            client = openai_pkg.OpenAI(api_key=self.api_key)
            resp = client.chat.completions.create(
                model=self.model,
                messages=to_openai_messages(messages, system),
                max_tokens=self.max_tokens,
            )
            return (resp.choices[0].message.content or "").strip()
        except LLMError:
            raise
        except Exception as exc:
            raise LLMError(f"OpenAI request failed: {exc}") from exc
