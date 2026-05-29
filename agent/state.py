"""
agent/state.py
--------------
AgentState and all supporting types for the LangGraph pipeline.

Python 3.12 compatibility
--------------------------
AgentState is defined as a @dataclass instead of TypedDict.
Python 3.12 removed isinstance() / issubclass() support for TypedDict,
which caused LangGraph to raise:

    TypeError: TypedDict does not support instance and class checks

All types that llm/gemini.py imports are defined here:
    - OptimizationSuggestion  (TypedDict — used as plain dict by the LLM layer)
    - RetrievedPattern        (TypedDict — used as plain dict by the retrieval layer)
    - DiffReport              (dataclass)
    - ValidationResult        (dataclass)
    - AgentState              (dataclass — the LangGraph state schema)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional
from typing_extensions import TypedDict   # typing_extensions works on 3.12


# ---------------------------------------------------------------------------
# TypedDicts used as structured dict types by gemini.py / vectorstore.py
# (NOT passed to isinstance() so they are safe on Python 3.12)
# ---------------------------------------------------------------------------

class OptimizationSuggestion(TypedDict, total=False):
    """
    One optimisation suggestion produced by the LLM analysis step.

    Fields
    ------
    category        : algorithm | data-structure | caching | io |
                      concurrency | vectorisation | string | memory | other
    severity        : low | medium | high
    target_function : name of the function to change, or '__module__'
    description     : one-sentence description of the problem
    example_fix     : short illustrative code snippet, or None
    """
    category:        str
    severity:        str
    target_function: str
    description:     str
    example_fix:     Optional[str]


class RetrievedPattern(TypedDict, total=False):
    """
    One RAG pattern retrieved from the FAISS vector store.

    Fields
    ------
    pattern_id  : unique identifier string
    title       : short human-readable name
    description : explanation of the pattern
    tags        : list of keyword tags
    before      : example code showing the anti-pattern
    after       : example code showing the optimised version
    score       : cosine similarity score from the vector search
    """
    pattern_id:  str
    title:       str
    description: str
    tags:        list[str]
    before:      str
    after:       str
    score:       float


# ---------------------------------------------------------------------------
# Supporting dataclasses
# ---------------------------------------------------------------------------

@dataclass
class DiffReport:
    unified_diff:      str
    lines_added:       int
    lines_removed:     int
    functions_changed: list[str] = field(default_factory=list)
    summary:           str       = ""


@dataclass
class ValidationResult:
    tests_passed:          bool
    tests_run:             int
    tests_failed:          int
    failure_details:       list[str]
    original_runtime_ms:   float
    refactored_runtime_ms: float
    speedup_ratio:         float


# ---------------------------------------------------------------------------
# AgentState — @dataclass so LangGraph never calls isinstance(x, TypedDict)
# ---------------------------------------------------------------------------

@dataclass
class AgentState:
    """
    Shared mutable state threaded through every LangGraph node.

    Every field has a default value so the graph can be initialised with
    only the fields the caller supplies (typically just `code` + `language`).
    LangGraph merges the dict returned by each node back into this object
    via the .update() method below.
    """

    # ── inputs ───────────────────────────────────────────────────────────
    code:     str = ""
    language: str = "python"

    # ── ingest ───────────────────────────────────────────────────────────
    ast_tree:           Any                         = None
    complexity_metrics: list[dict[str, Any]]        = field(default_factory=list)

    # ── profile ──────────────────────────────────────────────────────────
    profile_data:   dict[str, Any]       = field(default_factory=dict)
    hotspots:       list[dict[str, Any]] = field(default_factory=list)
    profile_report: str                  = ""

    # ── retrieve ─────────────────────────────────────────────────────────
    retrieved_patterns: list[RetrievedPattern] = field(default_factory=list)

    # ── analyze ──────────────────────────────────────────────────────────
    suggestions:   list[OptimizationSuggestion] = field(default_factory=list)
    opt_score:     float                         = 0.0
    llm_reasoning: str                           = ""
    llm_backend:   str                           = ""
    analysis:      str                           = ""

    # ── refactor ─────────────────────────────────────────────────────────
    optimized_code: Optional[str]        = None
    diff_report:    Optional[DiffReport] = None

    # ── validate ─────────────────────────────────────────────────────────
    validation_result: Optional[ValidationResult] = None

    # ── output ───────────────────────────────────────────────────────────
    final_report: str = ""

    # ── meta ─────────────────────────────────────────────────────────────
    error:        str              = ""
    node_timings: dict[str, float] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # dict-like helpers — lets nodes.py use state.get() / state["key"]
    # without any changes
    # ------------------------------------------------------------------

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def __getitem__(self, key: str) -> Any:
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key)

    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self, key, value)

    def __contains__(self, key: object) -> bool:
        return hasattr(self, key)

    def update(self, data: dict[str, Any]) -> None:
        """Merge a node's return dict back into state (called by the graph runner)."""
        for k, v in data.items():
            if hasattr(self, k):
                setattr(self, k, v)
            else:
                raise AttributeError(
                    f"AgentState has no field '{k}'. "
                    "Add it to the dataclass definition in agent/state.py."
                )