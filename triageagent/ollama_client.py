"""Minimal client for Ollama's local /api/chat endpoint.

Uses only the standard library (urllib) instead of the `requests` package,
so this project keeps the portfolio's zero-external-runtime-dependency
convention even though it talks to a local LLM server. Ollama runs on the
same machine (http://localhost:11434 by default) - this never makes a
network call off the machine, and never costs money.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any


class OllamaError(RuntimeError):
    pass


class OllamaUnavailable(OllamaError):
    """Raised when the local Ollama server isn't reachable."""


@dataclass
class ChatResponse:
    role: str
    content: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


class OllamaClient:
    def __init__(self, model: str = "llama3.2", host: str = "http://localhost:11434", timeout: float = 120.0):
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> ChatResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }
        if tools:
            payload["tools"] = tools

        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.host}/api/chat",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise OllamaUnavailable(
                f"Could not reach Ollama at {self.host}. Is it running? "
                f"(start it, then `ollama pull {self.model}`)"
            ) from exc

        message = raw.get("message", {})
        return ChatResponse(
            role=message.get("role", "assistant"),
            content=message.get("content", "") or "",
            tool_calls=message.get("tool_calls", []) or [],
        )
