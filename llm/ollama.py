"""
llm/ollama.py
-------------
Standalone Ollama API client used as a direct LLM backend or as an explicit
fallback when Gemini is unavailable.

Mirrors the interface expected by GeminiClient._ollama_call() so the two can
be swapped or tested independently.

Environment variables
---------------------
OLLAMA_BASE_URL   Base URL of the Ollama HTTP server (default: http://localhost:11434)
OLLAMA_MODEL      Model tag to use               (default: codellama:latest)

Usage
-----
    from llm.ollama import OllamaClient

    client = OllamaClient()
    text = client.generate("Explain Big-O notation in one paragraph.")
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults (overridden by env vars or constructor args)
# ---------------------------------------------------------------------------

_DEFAULT_BASE_URL = "http://localhost:11434"
_DEFAULT_MODEL    = "codellama:latest"
_DEFAULT_TIMEOUT  = 600   # seconds — 7 B models on CPU can take ~60-120 s for simple prompts, longer for complex code analysis


# ---------------------------------------------------------------------------
# OllamaClient
# ---------------------------------------------------------------------------

class OllamaClient:
    """
    Thin wrapper around Ollama's /api/generate endpoint.

    Parameters
    ----------
    base_url : str, optional
        Root URL of the Ollama server.  Falls back to the OLLAMA_BASE_URL
        environment variable, then to ``http://localhost:11434``.
    model : str, optional
        Model tag (e.g. ``"codellama:latest"``, ``"llama3:8b"``).
        Falls back to the OLLAMA_MODEL environment variable, then to
        ``"codellama:latest"``.
    timeout : int, optional
        HTTP request timeout in seconds.  Default: 300.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        model:    Optional[str] = None,
        timeout:  int = _DEFAULT_TIMEOUT,
    ) -> None:
        self.base_url = (
            base_url
            or os.environ.get("OLLAMA_BASE_URL", _DEFAULT_BASE_URL)
        ).rstrip("/")
        self.model   = model or os.environ.get("OLLAMA_MODEL", _DEFAULT_MODEL)
        self.timeout = timeout
        logger.info("OllamaClient ready (base_url=%s  model=%s)", self.base_url, self.model)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, prompt: str) -> str:
        """
        Send *prompt* to Ollama and return the model's text response.

        Uses the non-streaming ``/api/generate`` endpoint so the full
        response is returned in a single HTTP response body.

        Parameters
        ----------
        prompt : str
            The complete prompt string.  For multi-turn or system+user
            patterns, concatenate them before calling this method (or use
            :meth:`generate_with_system`).

        Returns
        -------
        str
            The model's text output.

        Raises
        ------
        RuntimeError
            If the HTTP request fails or the response lacks a ``"response"``
            field.
        """
        return self._post_generate(prompt=prompt)

    def generate_with_system(self, system: str, user: str) -> str:
        """
        Convenience wrapper that prepends a system message to the user prompt.

        Ollama's ``/api/generate`` endpoint does not have a dedicated system
        role parameter in non-chat mode, so the two strings are merged with a
        blank line separator — the same convention used by GeminiClient.

        Parameters
        ----------
        system : str
            System / instruction text.
        user : str
            User message / code to process.

        Returns
        -------
        str
            The model's text output.
        """
        combined = f"{system}\n\n{user}"
        return self._post_generate(prompt=combined)

    def is_available(self) -> bool:
        """
        Return True if the Ollama server is reachable, False otherwise.

        Performs a lightweight GET to ``/api/tags`` (lists local models)
        with a short timeout so health-checks don't block the pipeline.
        """
        try:
            resp = requests.get(
                f"{self.base_url}/api/tags",
                timeout=5,
            )
            return resp.status_code == 200
        except requests.RequestException:
            return False

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _post_generate(self, prompt: str) -> str:
        """
        POST to ``/api/generate`` and return the ``"response"`` field.

        Raises
        ------
        RuntimeError
            Wraps any ``requests`` exception or missing response field.
        """
        url     = f"{self.base_url}/api/generate"
        payload = {
            "model":  self.model,
            "prompt": prompt,
            "stream": False,
        }

        logger.info(
            "[ollama] POST %s  model=%s  prompt_chars=%d",
            url, self.model, len(prompt),
        )
        t0 = time.perf_counter()

        try:
            resp = requests.post(url, json=payload, timeout=self.timeout)
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(
                f"Ollama request failed (model={self.model}): {exc}"
            ) from exc

        elapsed = time.perf_counter() - t0

        try:
            data = resp.json()
        except ValueError as exc:
            raise RuntimeError(
                f"Ollama returned non-JSON response: {resp.text[:200]}"
            ) from exc

        if "response" not in data:
            raise RuntimeError(
                f"Ollama response missing 'response' field. Keys: {list(data.keys())}"
            )

        logger.info(
            "[ollama] done in %.2fs  eval_count=%s",
            elapsed,
            data.get("eval_count", "?"),
        )
        return data["response"]