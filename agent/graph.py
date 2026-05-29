"""
agent/graph.py
--------------
Builds and compiles the LangGraph StateGraph for the code-optimisation pipeline.

Python 3.12 / LangGraph compatibility
--------------------------------------
We pass the AgentState *dataclass* directly to StateGraph().  LangGraph >= 0.1
supports dataclasses as state schemas and does NOT call isinstance() against
them, avoiding the Python 3.12 TypedDict breakage.

Node wiring
-----------
    ingest
      │
    profile
      │
    retrieve
      │
    analyze
      │
    should_refactor  ──(score > 0.4)──► refactor ─┐
      │                                            │
      └──────────────────────────────────────────► validate
                                                   │
                                                 output
"""

from __future__ import annotations

from langgraph.graph import StateGraph, END

from agent.state import AgentState
from agent.nodes import (
    ingest,
    profile,
    retrieve,
    analyze,
    refactor,
    validate,
    output,
    should_refactor,
)


def build_graph() -> StateGraph:
    """
    Construct and compile the optimisation pipeline graph.

    Returns a compiled graph whose .invoke() method accepts an AgentState
    (or a plain dict with the same keys) and returns the final AgentState.
    """
    # Pass the dataclass — NOT a TypedDict — to avoid the Python 3.12 error
    graph = StateGraph(AgentState)

    # ── register nodes ───────────────────────────────────────────────────
    graph.add_node("ingest",   ingest)
    graph.add_node("profile",  profile)
    graph.add_node("retrieve", retrieve)
    graph.add_node("analyze",  analyze)
    graph.add_node("refactor", refactor)
    graph.add_node("validate", validate)
    graph.add_node("output",   output)

    # ── linear edges ─────────────────────────────────────────────────────
    graph.add_edge("ingest",   "profile")
    graph.add_edge("profile",  "retrieve")
    graph.add_edge("retrieve", "analyze")
    graph.add_edge("refactor", "validate")
    graph.add_edge("validate", "output")
    graph.add_edge("output",   END)

    # ── conditional edge after analyze ───────────────────────────────────
    graph.add_conditional_edges(
        "analyze",
        should_refactor,          # returns "refactor" or "validate"
        {
            "refactor": "refactor",
            "validate": "validate",
        },
    )

    # ── entry point ───────────────────────────────────────────────────────
    graph.set_entry_point("ingest")

    return graph.compile()


# Module-level compiled graph — import this in app.py
compiled_graph = build_graph()