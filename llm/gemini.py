"""
llm/gemini.py
-------------
Gemini API client with structured prompt templates for:
  1. Code analysis   — hotspot-aware optimisation opportunity scoring
  2. Code refactoring — full rewrite guided by suggestions + RAG patterns
  3. Report generation — Markdown summary of the full optimisation run

Ollama is used as an automatic fallback when Gemini is unavailable or
quota-exceeded. Quota/rate-limit errors skip retries and jump directly
to Ollama without waiting.

Dependencies
------------
    pip install google-genai tenacity
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from google import genai
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

# Load .env from project root
ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

from agent.state import OptimizationSuggestion, RetrievedPattern
from llm.ollama import OllamaClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

GEMINI_MODEL      = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
MAX_OUTPUT_TOKENS = 8192
TEMP_ANALYZE      = 0.2
TEMP_REFACTOR     = 0.3
TEMP_REPORT       = 0.5

# Keywords that indicate a non-retryable Gemini error (quota / billing / auth)
_QUOTA_KEYWORDS = ("quota", "rate limit", "429", "resource_exhausted", "billing",
                   "too many requests", "exceeded")


# ---------------------------------------------------------------------------
# Response dataclasses
# ---------------------------------------------------------------------------

@dataclass
class AnalysisResponse:
    suggestions: list[OptimizationSuggestion]
    score: float
    reasoning: str


@dataclass
class RefactorResponse:
    refactored_code: str
    change_summary: str


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_SYSTEM_PREAMBLE = """\
You are an expert software performance engineer specialising in Python \
optimisation. You have deep knowledge of algorithmic complexity, CPython \
internals, NumPy vectorisation, I/O patterns, and concurrency primitives. \
You are precise and evidence-based, always tying recommendations to the \
provided profiling data.\
"""

# ── Analysis ──────────────────────────────────────────────────────────────

_ANALYZE_SYSTEM = _SYSTEM_PREAMBLE + """

Your task: analyse a Python snippet with runtime profiling data, complexity
metrics, and retrieved optimisation patterns (RAG context).

Reason step-by-step, then return ONE JSON object — no fences, no preamble:

{
  "reasoning": "<step-by-step analysis>",
  "opt_score": <float 0.0–1.0>,
  "suggestions": [
    {
      "category": "<algorithm|data-structure|caching|io|concurrency|vectorisation|string|memory|other>",
      "severity": "<low|medium|high>",
      "target_function": "<function name or '__module__'>",
      "description": "<one sentence describing the problem>",
      "example_fix": "<short code snippet or null>"
    }
  ]
}

SCORING RUBRIC
  0.0–0.2  already well-optimised
  0.2–0.4  minor style/readability improvements
  0.4–0.6  meaningful algorithmic gains possible
  0.6–0.8  significant bottlenecks; targeted refactoring advised
  0.8–1.0  severe issues; full hot-path rewrite warranted

Return at most 8 suggestions, ordered by severity desc then cumtime desc.
"""

_ANALYZE_USER = """\
## Source Code
```python
{source_code}
```

## Profiling Hotspots (top {n_hotspots} by cumtime)
{hotspots_table}
## Profile Report

{profile_report}
## Complexity Metrics
{complexity_table}

## Retrieved Optimisation Patterns (RAG)
{rag_context}

Analyse the code and return the JSON object.
"""

# ── Refactor ──────────────────────────────────────────────────────────────

_REFACTOR_SYSTEM = _SYSTEM_PREAMBLE + """

Your task: rewrite the provided Python code to address the listed optimisation
suggestions. Rules:

  1. Apply ALL HIGH-severity suggestions.
  2. Apply MEDIUM suggestions unless they change the public API or hurt readability.
  3. Preserve all public function signatures, return types, and docstrings.
  4. Do NOT change observable behaviour — semantic equivalence is mandatory.
  5. Add a short inline comment above each changed block explaining why.
  6. Use the RAG before/after examples as concrete implementation guides.

Return ONE JSON object — no fences, no preamble:

{
  "refactored_code": "<full refactored source>",
  "change_summary": "<one sentence: what changed and why>"
}
"""

_REFACTOR_USER = """\
## Original Code
```python
{source_code}
```

## Suggestions to Apply
{suggestions_block}

## RAG Pattern Examples
{rag_examples}

Produce the refactored code and change summary.
"""

# ── Report ────────────────────────────────────────────────────────────────

_REPORT_SYSTEM = _SYSTEM_PREAMBLE + """

Write a concise professional Markdown optimisation report (≤ 600 words) for
a developer audience with these sections in order:

  1. **Executive Summary** — what was analysed, score, headline result
  2. **Profiling Findings** — bullet list of top hotspots + cumtimes
  3. **Optimisation Suggestions** — grouped by severity with brief rationale
  4. **Changes Made** — only if refactoring was performed
  5. **Benchmark Results** — original vs refactored runtime, speedup, test counts
  6. **Recommendations** — remaining issues or next steps

Do NOT include the full source code. Use code blocks sparingly.
"""

_REPORT_USER = """\
Language: {language}
Score: {opt_score:.2f} / 1.00
Reasoning: {reasoning}

Suggestions:
{suggestions_block}
## Profiling Report
{profile_report}
Hotspots:
{hotspots_table}

Refactoring performed: {refactored}
Change summary: {change_summary}

Tests passed: {tests_passed} / {tests_run}
Original runtime: {original_ms:.1f} ms
Refactored runtime: {refactored_ms:.1f} ms
Speedup: {speedup:.2f}×

Write the Markdown report.
"""


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt_hotspots(hotspots: list[dict]) -> str:
    if not hotspots:
        return "No profiling data available."
    header = f"{'Function':<45} {'ncalls':>7} {'tottime':>9} {'cumtime':>9}"
    sep    = "-" * 74
    rows   = [
        f"{h.get('name','?'):<45} {h.get('ncalls',0):>7} "
        f"{h.get('tottime',0.0):>9.4f} {h.get('cumtime',0.0):>9.4f}"
        for h in hotspots
    ]
    return "\n".join([header, sep, *rows])


def _fmt_complexity(metrics: list[dict]) -> str:
    if not metrics:
        return "No complexity data available."
    header = f"{'Function':<45} {'cyclomatic':>10} {'cognitive':>10} {'loc':>6}"
    sep    = "-" * 75
    rows   = [
        f"{m.get('name','?'):<45} {m.get('cyclomatic',0):>10} "
        f"{m.get('cognitive',0):>10} {m.get('loc',0):>6}"
        for m in metrics
    ]
    return "\n".join([header, sep, *rows])


def _fmt_rag_context(patterns: list[RetrievedPattern]) -> str:
    if not patterns:
        return "No relevant patterns retrieved."
    return "\n\n".join(
        f"### [{p['pattern_id']}] {p['title']}  (score: {p['score']:.3f})\n"
        f"{p['description']}\nTags: {', '.join(p.get('tags', []))}"
        for p in patterns
    )


def _fmt_rag_examples(patterns: list[RetrievedPattern]) -> str:
    blocks = [
        f"**{p['title']}**\n\nBefore:\n```python\n{p['before']}\n```\n\n"
        f"After:\n```python\n{p['after']}\n```"
        for p in patterns
        if p.get("before") or p.get("after")
    ]
    return "\n\n---\n\n".join(blocks) if blocks else "No code examples available."


def _fmt_suggestions(suggestions: list[OptimizationSuggestion]) -> str:
    if not suggestions:
        return "No suggestions."
    lines = []
    for i, s in enumerate(suggestions, 1):
        lines.append(
            f"{i}. [{s['severity'].upper()}] {s['category']} — "
            f"{s['target_function']}\n   {s['description']}"
        )
        if s.get("example_fix"):
            lines.append(f"   Example: {s['example_fix']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> dict:
    """Extract first JSON object from model output, handling markdown fences."""
    text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found in model response.")
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start: i + 1])
    raise ValueError("Unmatched braces in model response JSON.")


def _parse_analysis(raw: str) -> tuple[list[OptimizationSuggestion], float, str]:
    data      = _extract_json(raw)
    reasoning = data.get("reasoning", "")
    score     = max(0.0, min(1.0, float(data.get("opt_score", 0.0))))
    suggestions = [
        OptimizationSuggestion(
            category=s.get("category", "other"),
            severity=s.get("severity", "low"),
            target_function=s.get("target_function", "__module__"),
            description=s.get("description", ""),
            example_fix=s.get("example_fix"),
        )
        for s in data.get("suggestions", [])
    ]
    return suggestions, score, reasoning


def _parse_refactor(raw: str) -> tuple[str, str]:
    data = _extract_json(raw)
    return data.get("refactored_code", ""), data.get("change_summary", "")


# ---------------------------------------------------------------------------
# Retry helpers
# ---------------------------------------------------------------------------

def _is_retryable(exc: Exception) -> bool:
    """
    Return True for transient errors that warrant a retry (e.g. 5xx, network
    hiccup). Return False for quota / billing / auth errors so we skip the
    remaining retry attempts and fall through to the Ollama fallback immediately.
    """
    msg = str(exc).lower()
    return not any(kw in msg for kw in _QUOTA_KEYWORDS)


# ---------------------------------------------------------------------------
# GeminiClient
# ---------------------------------------------------------------------------

class GeminiClient:
    """
    Structured Gemini API client with Ollama fallback.

    Quota / rate-limit errors bypass the retry loop entirely and trigger
    the Ollama fallback on the first failure, avoiding wasted wait time.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = GEMINI_MODEL,
        ollama_fallback: bool = True,
    ) -> None:
        key = api_key or os.getenv("GEMINI_API_KEY")

        if not key:
            raise EnvironmentError(
                f"GEMINI_API_KEY not set. Expected in {ROOT_DIR / '.env'}"
            )

        print("\n" + "=" * 60)
        print("GEMINI DEBUG")
        print("API KEY PREFIX :", key[:15])
        print("MODEL          :", model)
        print("ROOT DIR       :", ROOT_DIR)
        print("=" * 60 + "\n")

        self._client = genai.Client(api_key=key)
        self._model = model
        self._ollama_fallback = ollama_fallback

        logger.info("GeminiClient ready (model=%s)", model)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze_code(
        self,
        source_code: str,
        hotspots: list[dict],
        profile_report: str,
        complexity_metrics: list[dict],
        retrieved_patterns: list[RetrievedPattern],
    ) -> AnalysisResponse:
        """Analyse code for optimisation opportunities."""
        prompt = _ANALYZE_USER.format(
            source_code=source_code,
            n_hotspots=len(hotspots),
            hotspots_table=_fmt_hotspots(hotspots),
            profile_report=profile_report,
            complexity_table=_fmt_complexity(complexity_metrics),
            rag_context=_fmt_rag_context(retrieved_patterns),
        )
        raw = self._call(
            system=_ANALYZE_SYSTEM,
            user=prompt,
            temperature=TEMP_ANALYZE,
            label="analyze",
        )
        try:
            suggestions, score, reasoning = _parse_analysis(raw)
        except Exception as e:
            logger.error("[gemini/analyze] parse error: %s", e)
            logger.debug("[gemini/analyze] raw output:\n%s", raw)
            suggestions = []
            score       = 0.0
            reasoning   = raw

        logger.info(
            "[gemini/analyze] score=%.2f  suggestions=%d", score, len(suggestions)
        )
        return AnalysisResponse(
            suggestions=suggestions, score=score, reasoning=reasoning
        )

    def refactor(
        self,
        source_code: str,
        suggestions: list[OptimizationSuggestion],
        retrieved_patterns: list[RetrievedPattern],
    ) -> RefactorResponse:
        """Produce an optimised rewrite of the source code."""
        prompt = _REFACTOR_USER.format(
            source_code=source_code,
            suggestions_block=_fmt_suggestions(suggestions),
            rag_examples=_fmt_rag_examples(retrieved_patterns),
        )
        raw = self._call(
            system=_REFACTOR_SYSTEM,
            user=prompt,
            temperature=TEMP_REFACTOR,
            label="refactor",
        )
        code, summary = _parse_refactor(raw)
        if not code.strip():
            logger.warning("[gemini/refactor] empty response; keeping original")
            code = source_code
        logger.info("[gemini/refactor] %s", summary)
        return RefactorResponse(refactored_code=code, change_summary=summary)

    def generate_report(
        self,
        *,
        language: str,
        opt_score: float,
        reasoning: str,
        suggestions: list[OptimizationSuggestion],
        hotspots: list[dict],
        refactored: bool,
        profile_report: str,
        change_summary: str,
        tests_passed: int,
        tests_run: int,
        original_ms: float,
        refactored_ms: float,
        speedup: float,
    ) -> str:
        """Generate the final Markdown optimisation report."""
        prompt = _REPORT_USER.format(
            language=language,
            opt_score=opt_score,
            reasoning=reasoning,
            suggestions_block=_fmt_suggestions(suggestions),
            hotspots_table=_fmt_hotspots(hotspots),
            profile_report=profile_report,
            refactored="Yes" if refactored else "No (score below threshold)",
            change_summary=change_summary or "N/A",
            tests_passed=tests_passed,
            tests_run=tests_run,
            original_ms=original_ms,
            refactored_ms=refactored_ms,
            speedup=speedup,
        )
        return self._call(
            system=_REPORT_SYSTEM,
            user=prompt,
            temperature=TEMP_REPORT,
            label="report",
        ).strip()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @retry(
        retry=retry_if_exception(_is_retryable),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    def _gemini_call(self, contents: list, config: dict) -> str:
        """Single Gemini API call, retried only on transient errors."""
        response = self._client.models.generate_content(
            model=self._model,
            contents=contents,
            config=config,
        )
        return response.text

    def _call(self, system: str, user: str, temperature: float, label: str) -> str:
        """
        Execute one Gemini call with retry + optional Ollama fallback.

        Quota / billing errors skip the retry loop and fall through to
        Ollama immediately on the first failure.
        """
        contents = [
            {"role": "user",  "parts": [{"text": system}]},
            {"role": "model", "parts": [{"text": "Understood. Ready."}]},
            {"role": "user",  "parts": [{"text": user}]},
        ]
        config = {
            "temperature": temperature,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
        }
        t0 = time.perf_counter()
        try:
            text = self._gemini_call(contents, config)
            logger.info("[gemini/%s] %.2fs", label, time.perf_counter() - t0)
            return text
        except Exception as exc:
            logger.error("[gemini/%s] failed: %s", label, exc)
            if self._ollama_fallback:
                logger.info("[gemini/%s] falling back to Ollama", label)
                return self._ollama_call(system=system, user=user, label=label)
            raise

    def _ollama_call(self, system: str, user: str, label: str) -> str:
        """
        Fallback to a local Ollama instance when Gemini is unavailable.

        Uses the OllamaClient class for consistency and to avoid code duplication.
        """
        try:
            client = OllamaClient()
            return client.generate_with_system(system=system, user=user)
        except Exception as exc:
            raise RuntimeError(
                f"Ollama fallback also failed [{label}]: {exc}"
            ) from exc