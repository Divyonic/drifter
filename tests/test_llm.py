"""Tests for cdm.llm — provider registry, message conversion, local key storage.

Fully offline: no provider SDKs are imported, no network, no real keys. Key
storage is redirected to a tmp dir so it never touches the real data directory.
"""

from __future__ import annotations

import pytest

from cdm import config, llm
from cdm.llm import (
    PROVIDERS,
    LLMClient,
    LLMError,
    to_anthropic_messages,
    to_gemini_contents,
    to_openai_messages,
)


@pytest.fixture
def tmp_data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    return tmp_path


def test_provider_registry_shape():
    assert {"claude", "gemini", "openai", "claude-cli"} <= set(PROVIDERS)
    for meta in PROVIDERS.values():
        assert {"label", "default_model", "sdk", "pip"} <= set(meta)


def test_providers_have_models_and_key_help():
    from cdm.llm import curated_models, key_url

    for p, meta in PROVIDERS.items():
        assert curated_models(p), f"{p} has no curated models"
        assert meta.get("key_hint")
        assert meta["default_model"] in curated_models(p)
        if not meta.get("keyless"):  # keyless (subscription) providers have no key url
            assert key_url(p).startswith("https://")


def test_keyless_provider_uses_cli(tmp_data_dir, monkeypatch):
    import cdm.llm as llm
    from cdm.llm import LLMClient, claude_cli_available, provider_ready

    assert isinstance(claude_cli_available(), bool)
    # No CLI on PATH -> not ready, and constructing the client raises.
    monkeypatch.setattr(llm.shutil, "which", lambda _n: None)
    assert provider_ready("claude-cli") is False
    with pytest.raises(LLMError):
        LLMClient("claude-cli")
    # CLI present -> ready, and the client constructs with no API key.
    monkeypatch.setattr(llm.shutil, "which", lambda _n: "/usr/bin/claude")
    assert provider_ready("claude-cli") is True
    client = LLMClient("claude-cli")
    assert client.keyless and client.api_key is None


def test_list_models_without_key_raises(tmp_data_dir, monkeypatch):
    for env in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(env, raising=False)
    from cdm.llm import list_models

    with pytest.raises(LLMError):
        list_models("claude")


def test_to_openai_messages_prepends_system():
    msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]
    out = to_openai_messages(msgs, system="be brief")
    assert out[0] == {"role": "system", "content": "be brief"}
    assert out[1:] == msgs


def test_to_openai_messages_without_system():
    out = to_openai_messages([{"role": "user", "content": "hi"}], system=None)
    assert out == [{"role": "user", "content": "hi"}]


def test_to_anthropic_messages_no_inline_system():
    out = to_anthropic_messages([{"role": "user", "content": "hi"}])
    assert out == [{"role": "user", "content": "hi"}]
    assert all(m["role"] != "system" for m in out)


def test_to_gemini_contents_maps_assistant_to_model():
    out = to_gemini_contents(
        [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]
    )
    assert out[0] == {"role": "user", "parts": [{"text": "hi"}]}
    assert out[1] == {"role": "model", "parts": [{"text": "yo"}]}


def test_conversion_drops_blank_and_coerces_unknown_roles():
    msgs = [{"role": "system", "content": "x"}, {"role": "user", "content": "  "}]
    # "system"/unknown -> assistant; blank content dropped
    out = to_anthropic_messages(msgs)
    assert out == [{"role": "assistant", "content": "x"}]


def test_key_storage_roundtrip(tmp_data_dir):
    assert llm.load_keys() == {}
    llm.save_key("claude", "sk-test-123")
    assert llm.get_key("claude") == "sk-test-123"
    assert llm.load_keys()["claude"] == "sk-test-123"
    # saving empty removes it
    llm.save_key("claude", "")
    assert llm.get_key("claude") is None


def test_get_key_env_fallback(tmp_data_dir, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key")
    assert llm.get_key("claude") == "env-key"


def test_client_unknown_provider_raises(tmp_data_dir):
    with pytest.raises(LLMError):
        LLMClient("not-a-provider")


def test_client_missing_key_raises(tmp_data_dir, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(LLMError):
        LLMClient("claude")  # no stored key, no env key
