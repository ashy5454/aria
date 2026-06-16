"""
providers.py — Unified LLM provider for ARIA.

Supports:
  gemini      — google-genai SDK  (native web search grounding)
  openrouter  — OpenAI-compatible HTTP  (any model: GPT, Claude, Llama...)
  openai      — OpenAI SDK  (gpt-4o, gpt-4o-mini, o3-mini...)

Configure in research.yaml:
  loop:
    provider: gemini          # gemini | openrouter | openai
    model: gemini-3.5-flash   # or openai/gpt-4o / gpt-4o-mini / etc.
    use_web_search: true      # only available with gemini provider

Required env vars:
  GEMINI_API_KEY       if provider: gemini
  OPENROUTER_API_KEY   if provider: openrouter
  OPENAI_API_KEY       if provider: openai
"""

from __future__ import annotations
import os
import json
import urllib.request
import urllib.error


class LLMResponse:
    """Uniform response object — always has .text"""
    def __init__(self, text: str):
        self.text = text


class LLMClient:
    """Base interface — generate_content(prompt, system=None) -> LLMResponse"""
    def generate_content(self, prompt: str, system: str | None = None) -> LLMResponse:
        raise NotImplementedError

    def generate_with_search(self, prompt: str) -> LLMResponse:
        """Web-search-grounded generation. Falls back to plain if not supported."""
        return self.generate_content(prompt)


# ── Gemini ────────────────────────────────────────────────────────────────────

class GeminiClient(LLMClient):
    def __init__(self, model: str, use_search: bool = False):
        from google import genai
        from google.genai import types as gtypes
        self._model = model
        self._use_search = use_search
        self._types = gtypes
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set")
        self._client = genai.Client(api_key=api_key)

    def generate_content(self, prompt: str, system: str | None = None) -> LLMResponse:
        from google.genai import types as gtypes
        cfg = None
        if system:
            cfg = gtypes.GenerateContentConfig(system_instruction=system)
        resp = self._client.models.generate_content(
            model=self._model, contents=prompt, config=cfg
        )
        return LLMResponse(resp.text.strip())

    def generate_with_search(self, prompt: str) -> LLMResponse:
        from google.genai import types as gtypes
        if not self._use_search:
            return self.generate_content(prompt)
        cfg = gtypes.GenerateContentConfig(
            tools=[gtypes.Tool(google_search=gtypes.GoogleSearch())]
        )
        resp = self._client.models.generate_content(
            model=self._model, contents=prompt, config=cfg
        )
        return LLMResponse(resp.text.strip())


# ── OpenAI-compatible (OpenRouter + OpenAI) ───────────────────────────────────

class OpenAICompatClient(LLMClient):
    """
    Works for both OpenAI and OpenRouter — same API, different base_url + key.
    Web search falls back to plain generation (no native grounding support).
    """
    def __init__(self, model: str, api_key: str, base_url: str,
                 extra_headers: dict | None = None):
        self._model = model
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._extra_headers = extra_headers or {}

    def _call(self, messages: list[dict]) -> str:
        payload = json.dumps({
            "model": self._model,
            "messages": messages,
        }).encode()

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
            **self._extra_headers,
        }

        req = urllib.request.Request(
            f"{self._base_url}/chat/completions",
            data=payload,
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read())
                return data["choices"][0]["message"]["content"].strip()
        except urllib.error.HTTPError as e:
            body = e.read().decode()
            raise RuntimeError(f"LLM API error {e.code}: {body[:300]}")

    def generate_content(self, prompt: str, system: str | None = None) -> LLMResponse:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return LLMResponse(self._call(messages))

    def generate_with_search(self, prompt: str) -> LLMResponse:
        # No native web search — just generate normally
        return self.generate_content(prompt)


# ── Factory ───────────────────────────────────────────────────────────────────

def make_client(provider: str, model: str, use_search: bool = False) -> LLMClient:
    """
    Create the right client from research.yaml settings.

    provider: "gemini" | "openrouter" | "openai"
    model:    e.g. "gemini-3.5-flash" / "openai/gpt-4o" / "gpt-4o-mini"
    """
    provider = provider.lower()

    if provider == "gemini":
        return GeminiClient(model=model, use_search=use_search)

    if provider == "openrouter":
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY not set")
        return OpenAICompatClient(
            model=model,
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            extra_headers={
                "HTTP-Referer": "https://github.com/ashy5454/aria",
                "X-Title": "ARIA Research Agent",
            },
        )

    if provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        return OpenAICompatClient(
            model=model,
            api_key=api_key,
            base_url="https://api.openai.com/v1",
        )

    raise ValueError(f"Unknown provider: '{provider}'. Choose: gemini | openrouter | openai")
