#!/usr/bin/env python3
"""LLM plumbing for the spendly agent: one /api/chat round-trip through the
`ollama` package. Everything Ollama-specific lives here, so swapping providers
later means touching only this module."""
from __future__ import annotations

import ollama

from constants import DEFAULT_MODEL, NUM_CTX, OLLAMA_HOST, TEMPERATURE


class LLM:
    """Thin wrapper around the ollama package's /api/chat endpoint."""

    def __init__(self, model: str = DEFAULT_MODEL, host: str = OLLAMA_HOST):
        self.model = model
        self.host = host

    def run(self, messages: list[dict], tools: list[dict]):
        return ollama.Client(host=self.host).chat(
            model=self.model,
            messages=messages,
            tools=tools,
            stream=False,
            options={"temperature": TEMPERATURE, "num_ctx": NUM_CTX},
        )
