"""
api/routes.py
-------------
Flask Blueprint exposing the code optimiser endpoints.
"""

from __future__ import annotations

import traceback

from flask import Blueprint, jsonify, request

from agent.graph import app
from agent.state import initial_state

api = Blueprint("api", __name__)


@api.route("/")
def home():
    return jsonify({"message": "AI Code Optimizer Running"})


@api.route("/health")
def health():
    return jsonify({"status": "ok"})


@api.route("/optimize", methods=["POST"])
def optimize():
    payload = request.get_json(silent=True)
    if not payload:
        return jsonify({"error": "Request body must be JSON"}), 400

    code = payload.get("code", "").strip()
    if not code:
        return jsonify({"error": "Field 'code' is required and must not be empty"}), 400

    state = initial_state(code)

    print("=" * 60)
    print("BEFORE INVOKE")
    print(f"  code length : {len(code)} chars")
    print(f"  code preview: {code[:120]!r}")
    print("=" * 60)

    try:
        result = app.invoke(state)
    except Exception as exc:
        print("=" * 60)
        print("INVOKE FAILED")
        traceback.print_exc()
        print("=" * 60)
        return jsonify({"error": str(exc)}), 500

    print("=" * 60)
    print("AFTER INVOKE")
    print(f"  opt_score     : {result.get('opt_score')}")
    print(f"  analysis chars: {len(result.get('analysis') or '')}")
    print(f"  report chars  : {len(result.get('final_report') or '')}")
    print(f"  optimized     : {'yes' if result.get('optimized_code') else 'no'}")
    print(f"  node_timings  : {result.get('node_timings')}")
    print("=" * 60)

    # Serialise diff_report — it may be a dataclass or a dict
    diff = result.get("diff_report")
    if diff is not None and hasattr(diff, "__dict__"):
        diff = diff.__dict__

    return jsonify(
        {
            "analysis":       result.get("analysis"),
            "optimized_code": result.get("optimized_code"),
            "diff":           diff,
            "final_report":   result.get("final_report"),
            "opt_score":      result.get("opt_score"),
            "node_timings":   result.get("node_timings"),
        }
    )