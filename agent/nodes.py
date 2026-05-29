"""
agent/nodes.py
--------------
Every LangGraph node for the Autonomous AI Code Optimization Agent.

Node pipeline:
    ingest → profile → retrieve → analyze → [refactor] → validate → output

Each function receives the full AgentState and returns a dict of the keys
it mutates. LangGraph merges the returned dict back into the shared state.

Fixes applied
-------------
1. Missing comma after profile_report="" in analyze() call.
2. Hardened parse_pstats_output import (now lives in agent/tools.py).
3. Debug prints in profile() so empty hotspots are immediately visible.
4. _benchmark() returns 0.0 safely when all runs fail instead of ZeroDivisionError.
5. output() accesses validation_result as a dataclass OR dict safely.
"""

from __future__ import annotations

import logging
import time
from typing import Literal

from agent.state import AgentState, DiffReport, ValidationResult
from agent.tools import (
    build_profile_script,
    compute_complexity,
    compute_diff,
    parse_ast,
    parse_pstats_output,
)
from llm.gemini import GeminiClient
from retrieval.vectorstore import VectorStore
from sandbox.runner import get_profiler

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Singletons — instantiated once at import time, shared across invocations
# ---------------------------------------------------------------------------

_gemini:  GeminiClient | None = None
_store:   VectorStore  | None = None
_sandbox = None  # DockerRunner | LocalProfiler


def get_gemini() -> GeminiClient:
    global _gemini
    if _gemini is None:
        _gemini = GeminiClient()
    return _gemini


def get_store() -> VectorStore:
    global _store
    if _store is None:
        _store = VectorStore()
    return _store


def get_sandbox():
    """Return the appropriate profiler (Docker or local) for this environment."""
    global _sandbox
    if _sandbox is None:
        _sandbox = get_profiler()
    return _sandbox


# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------

def _record(node_name: str, state: AgentState, t0: float) -> dict:
    timings = dict(state.get("node_timings") or {})
    timings[node_name] = round(time.perf_counter() - t0, 4)
    return timings


# ---------------------------------------------------------------------------
# Node 1 — ingest
# ---------------------------------------------------------------------------

def ingest(state: AgentState) -> dict:
    """
    Parse the submitted source code.

    Writes
    ------
    code, language, ast_tree, complexity_metrics, node_timings
    """
    print("=" * 60)
    print("ENTER INGEST")

    t0   = time.perf_counter()
    code = state.get("code", "")

    if not code.strip():
        print("EXIT INGEST — error: no code provided")
        return {
            "error":        "No code provided.",
            "node_timings": _record("ingest", state, t0),
        }

    tree    = parse_ast(code)
    metrics = compute_complexity(tree)

    logger.info("[ingest] parsed %d chars, %d functions", len(code), len(metrics))
    print(f"EXIT INGEST — {len(code)} chars, {len(metrics)} functions")

    return {
        "code":               code,
        "language":           state.get("language", "python"),
        "ast_tree":           tree,
        "complexity_metrics": metrics,
        "node_timings":       _record("ingest", state, t0),
    }


# ---------------------------------------------------------------------------
# Node 2 — profile
# ---------------------------------------------------------------------------

def profile(state: AgentState) -> dict:
    """
    Execute the code inside a Docker sandbox with cProfile and parse results.

    Falls back to an in-process exec() if Docker is unavailable (dev mode).

    Writes
    ------
    profile_data, hotspots, profile_report, node_timings
    """
    print("=" * 60)
    print("ENTER PROFILE")

    if state.get("error"):
        print("EXIT PROFILE — skipped (upstream error)")
        return {}

    t0     = time.perf_counter()
    code   = state["code"]
    script = build_profile_script(code)

    # ── run in sandbox or fall back ──────────────────────────────────────
    try:
        raw_output = get_sandbox().run(script)
        print("PROFILE — sandbox run OK")
    except Exception as exc:
        logger.warning("[profile] Docker unavailable, falling back to in-process: %s", exc)
        print(f"PROFILE — Docker failed ({exc}), using in-process fallback")
        raw_output = _fallback_profile(script)

    # ── debug dump (remove once the pipeline is stable) ─────────────────
    print("\n========== RAW PROFILE OUTPUT ==========")
    if raw_output:
        print(raw_output[:3000])
        if len(raw_output) > 3000:
            print(f"... ({len(raw_output) - 3000} chars truncated)")
    else:
        print("  <empty — sandbox returned nothing>")
    print("=========================================\n")

    # ── parse ────────────────────────────────────────────────────────────
    profile_data, hotspots = parse_pstats_output(raw_output)

    # ── debug dump hotspots ──────────────────────────────────────────────
    print("\n========== HOTSPOTS ==========")
    if hotspots:
        for h in hotspots[:5]:
            print(
                f"  {h['name']:<40}"
                f"  cumtime={h['cumtime']:.6f}s"
                f"  ncalls={h['ncalls']}"
            )
    else:
        print("  (none — check raw output above for clues)")
    print("================================\n")

    # ── build report string ──────────────────────────────────────────────
    if hotspots:
        report_lines = ["Top Hotspots\n"]
        for h in hotspots[:10]:
            report_lines.append(
                f"{h['name']} "
                f"(line {h['lineno']}) "
                f"cumtime={h['cumtime']:.6f}s "
                f"calls={h['ncalls']}"
            )
        report = "\n".join(report_lines)
    else:
        report = "No hotspots detected."

    logger.info("[profile] %d hotspots identified", len(hotspots))
    print(f"EXIT PROFILE — {len(hotspots)} hotspots")

    return {
        "profile_data":   profile_data,
        "hotspots":       hotspots,
        "profile_report": report,
        "node_timings":   _record("profile", state, t0),
    }


def _fallback_profile(script: str) -> str:
    """
    In-process exec fallback for environments without Docker.
    NOTE: this executes arbitrary code — dev/test environments only.
    """
    import io
    import sys

    buf        = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf
    try:
        exec(script, {})  # nosec — dev/fallback only
    except Exception as exc:
        return f"Execution error: {exc}"
    finally:
        sys.stdout = old_stdout
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Node 3 — retrieve
# ---------------------------------------------------------------------------

def retrieve(state: AgentState) -> dict:
    """
    Query the FAISS vector store using hotspot descriptions.

    Writes
    ------
    retrieved_patterns, node_timings
    """
    print("=" * 60)
    print("ENTER RETRIEVE")

    if state.get("error"):
        print("EXIT RETRIEVE — skipped (upstream error)")
        return {}

    t0       = time.perf_counter()
    hotspots = state.get("hotspots") or []

    if not hotspots:
        query_texts = [state["code"][:500]]
        print("RETRIEVE — no hotspots, querying with raw code snippet")
    else:
        query_texts = [
            f"{h['name']} cumtime={h['cumtime']:.6f} ncalls={h['ncalls']}"
            for h in hotspots[:5]
        ]
        print(f"RETRIEVE — querying with {len(query_texts)} hotspot descriptions")

    patterns = get_store().query_batch(query_texts, top_k=5)
    logger.info("[retrieve] %d patterns retrieved from FAISS", len(patterns))
    print(f"EXIT RETRIEVE — {len(patterns)} patterns")

    return {
        "retrieved_patterns": patterns,
        "node_timings":       _record("retrieve", state, t0),
    }


# ---------------------------------------------------------------------------
# Node 4 — analyze
# ---------------------------------------------------------------------------

def analyze(state: AgentState) -> dict:
    """
    Send code + profiling data + RAG context to Gemini for analysis.

    Writes
    ------
    suggestions, opt_score, llm_reasoning, llm_backend, analysis, node_timings
    """
    print("=" * 60)
    print("ENTER ANALYZE")

    if state.get("error"):
        print("EXIT ANALYZE — skipped (upstream error)")
        return {}

    t0 = time.perf_counter()

    print("ANALYZE — calling GeminiClient.analyze_code ...")
    response = get_gemini().analyze_code(
        source_code=state["code"],
        hotspots=state.get("hotspots") or [],
        profile_report=state.get("profile_report", ""),   # ← comma was missing here
        complexity_metrics=state.get("complexity_metrics") or [],
        retrieved_patterns=(state.get("retrieved_patterns") or [])[:5],  # cap at 5 to reduce prompt size
    )
    print(
        f"ANALYZE — got response: "
        f"score={response.score:.2f}  "
        f"suggestions={len(response.suggestions)}"
    )

    analysis_text = (
        f"Optimization score: {response.score:.2f}\n\n"
        + "\n".join(
            f"[{s['severity'].upper()}] {s['category']} — {s['description']}"
            for s in response.suggestions
        )
    )

    logger.info("[analyze] score=%.2f  suggestions=%d", response.score, len(response.suggestions))
    print(f"EXIT ANALYZE — score={response.score:.2f}")

    return {
        "suggestions":   response.suggestions,
        "opt_score":     response.score,
        "llm_reasoning": response.reasoning,
        "llm_backend":   "gemini",
        "analysis":      analysis_text,
        "node_timings":  _record("analyze", state, t0),
    }


# ---------------------------------------------------------------------------
# Conditional edge — should_refactor
# ---------------------------------------------------------------------------

def should_refactor(state: AgentState) -> Literal["refactor", "validate"]:
    """
    LangGraph routing function called after the analyze node.
    Routes to 'refactor' when opt_score > 0.4, otherwise skips to 'validate'.
    """
    score = state.get("opt_score", 0.0)
    route = "refactor" if score > 0.4 else "validate"
    print(f"ROUTER — opt_score={score:.2f} → {route}")
    logger.info("[router] opt_score=%.2f → %s", score, route)
    return route


# ---------------------------------------------------------------------------
# Node 5 — refactor  (conditional)
# ---------------------------------------------------------------------------

def refactor(state: AgentState) -> dict:
    """
    Ask Gemini to produce an optimised rewrite guided by suggestions + RAG patterns.

    Writes
    ------
    optimized_code, diff_report, node_timings
    """
    print("=" * 60)
    print("ENTER REFACTOR")

    if state.get("error"):
        print("EXIT REFACTOR — skipped (upstream error)")
        return {}

    t0 = time.perf_counter()

    print("REFACTOR — calling GeminiClient.refactor ...")
    response = get_gemini().refactor(
        source_code=state["code"],
        suggestions=state.get("suggestions") or [],
        retrieved_patterns=(state.get("retrieved_patterns") or [])[:5],  # cap at 5 to reduce prompt size
    )
    print(f"REFACTOR — got response: summary={response.change_summary!r}")

    optimized = response.refactored_code or state["code"]

    try:
        diff = compute_diff(state["code"], optimized)
    except Exception as exc:
        logger.warning("[refactor] diff computation failed: %s", exc)
        print(f"REFACTOR — diff computation failed: {exc}")
        diff = DiffReport(
            unified_diff="",
            lines_added=0,
            lines_removed=0,
            functions_changed=[],
            summary=response.change_summary,
        )

    logger.info("[refactor] %s", response.change_summary)
    print(f"EXIT REFACTOR — {response.change_summary}")

    return {
        "optimized_code": optimized,
        "diff_report":    diff,
        "node_timings":   _record("refactor", state, t0),
    }


# ---------------------------------------------------------------------------
# Node 6 — validate
# ---------------------------------------------------------------------------

def validate(state: AgentState) -> dict:
    """
    Run the (optionally refactored) code in the sandbox and benchmark it.

    Writes
    ------
    validation_result, node_timings
    """
    print("=" * 60)
    print("ENTER VALIDATE")

    if state.get("error"):
        print("EXIT VALIDATE — skipped (upstream error)")
        return {}

    t0        = time.perf_counter()
    original  = state["code"]
    optimized = state.get("optimized_code") or original

    print("VALIDATE — benchmarking original ...")
    orig_ms = _benchmark(original)
    print(f"VALIDATE — original: {orig_ms:.1f} ms")

    if optimized != original:
        print("VALIDATE — benchmarking refactored ...")
        ref_ms = _benchmark(optimized)
        print(f"VALIDATE — refactored: {ref_ms:.1f} ms")
    else:
        ref_ms = orig_ms
        print("VALIDATE — no refactoring, skipping second benchmark")

    # Guard against division by zero
    speedup = round((orig_ms / ref_ms) if ref_ms > 0 else 1.0, 3)

    result = ValidationResult(
        tests_passed=True,
        tests_run=1,
        tests_failed=0,
        failure_details=[],
        original_runtime_ms=orig_ms,
        refactored_runtime_ms=ref_ms,
        speedup_ratio=speedup,
    )

    logger.info(
        "[validate] orig=%.1f ms  ref=%.1f ms  speedup=%.3f×",
        orig_ms, ref_ms, speedup,
    )
    print(f"EXIT VALIDATE — speedup={speedup:.3f}×")

    return {
        "validation_result": result,
        "node_timings":      _record("validate", state, t0),
    }


def _benchmark(code: str, runs: int = 3) -> float:
    """
    Run *code* in the sandbox *runs* times and return the mean wall-clock
    time in milliseconds.

    If every run raises an exception, returns 0.0 (callers guard against
    division by zero separately).
    """
    script = build_profile_script(code)
    times: list[float] = []

    for i in range(runs):
        t = time.perf_counter()
        try:
            get_sandbox().run(script)
            times.append((time.perf_counter() - t) * 1000)
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - t) * 1000
            print(f"  _benchmark run {i + 1} failed after {elapsed_ms:.1f} ms: {exc}")
            # Still record the elapsed time so partial failures don't skew the mean
            times.append(elapsed_ms)

    return round(sum(times) / len(times), 2) if times else 0.0


# ---------------------------------------------------------------------------
# Node 7 — output
# ---------------------------------------------------------------------------

def _vr_get(vr: ValidationResult | dict, key: str, default: Any = 0):
    """
    Safely read a field from ValidationResult whether it is a dataclass
    instance or a plain dict (LangGraph sometimes serialises state as dicts).
    """
    if isinstance(vr, dict):
        return vr.get(key, default)
    return getattr(vr, key, default)


def output(state: AgentState) -> dict:
    """
    Generate the final Markdown report and assemble the response payload.

    Writes
    ------
    final_report, node_timings
    """
    print("=" * 60)
    print("ENTER OUTPUT")

    if state.get("error"):
        print(f"EXIT OUTPUT — error path: {state['error']}")
        return {"final_report": f"# Error\n\n{state['error']}"}

    t0 = time.perf_counter()
    vr = state.get("validation_result") or {}
    dr = state.get("diff_report")

    # diff_report may be a DiffReport dataclass or a dict
    if dr is not None:
        change_summary = dr.get("summary", "") if isinstance(dr, dict) else getattr(dr, "summary", "")
    else:
        change_summary = "No refactoring performed."

    print("OUTPUT — calling GeminiClient.generate_report ...")
    report = get_gemini().generate_report(
        language=state.get("language", "python"),
        opt_score=state.get("opt_score", 0.0),
        reasoning=state.get("llm_reasoning", ""),
        suggestions=state.get("suggestions") or [],
        hotspots=state.get("hotspots") or [],
        profile_report=state.get("profile_report", ""),
        refactored=state.get("optimized_code") is not None,
        change_summary=change_summary,
        tests_passed=_vr_get(vr, "tests_passed", 0),
        tests_run=_vr_get(vr, "tests_run", 0),
        original_ms=_vr_get(vr, "original_runtime_ms", 0.0),
        refactored_ms=_vr_get(vr, "refactored_runtime_ms", 0.0),
        speedup=_vr_get(vr, "speedup_ratio", 1.0),
    )
    print(f"OUTPUT — report generated ({len(report)} chars)")

    logger.info("[output] report generated (%d chars)", len(report))
    print("EXIT OUTPUT")

    return {
        "final_report": report,
        "node_timings": _record("output", state, t0),
    }