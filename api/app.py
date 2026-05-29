"""
app.py
------
Flask HTTP entrypoint for the Autonomous AI Code Optimisation Agent.

Endpoints
---------
POST /optimize
    Body   : {"code": "<python source>", "language": "python"}
    Returns: JSON result with final_report, analysis, suggestions, etc.

GET /health
    Returns: {"status": "ok"}

CORS
----
Allowed origins are read from the CORS_ORIGINS environment variable
(comma-separated).  The Vite dev server (http://localhost:5173) is always
included so local development works without any extra config.

    CORS_ORIGINS=https://myapp.example.com        # production
    CORS_ORIGINS=http://localhost:5173,http://localhost:4173   # override
"""

from __future__ import annotations

import logging
import os
import traceback
from typing import Any

from flask import Flask, Response, jsonify, request
from flask_cors import CORS

from agent.graph import compiled_graph
from agent.state import AgentState

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------
# Always permit the Vite dev server; merge with any extra origins from env.
_DEFAULT_DEV_ORIGINS = [
    "http://localhost:5173",   # Vite default
    "http://localhost:4173",   # Vite preview
    "http://127.0.0.1:5173",
    "http://127.0.0.1:4173",
]

def _allowed_origins() -> list[str]:
    env_val = os.environ.get("CORS_ORIGINS", "").strip()
    extra   = [o.strip() for o in env_val.split(",") if o.strip()]
    # Deduplicate while preserving order
    seen, result = set(), []
    for origin in _DEFAULT_DEV_ORIGINS + extra:
        if origin not in seen:
            seen.add(origin)
            result.append(origin)
    return result

CORS(
    app,
    origins=_allowed_origins(),
    methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
    supports_credentials=False,
)

logger.info("CORS enabled for origins: %s", _allowed_origins())

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _state_to_dict(state: AgentState | dict) -> dict[str, Any]:
    """
    Serialise an AgentState (dataclass) or plain dict into a JSON-safe dict.
    DiffReport and ValidationResult are also converted to dicts so Flask's
    jsonify() does not choke on dataclass instances.
    """
    raw = state if isinstance(state, dict) else vars(state)

    result: dict[str, Any] = {}
    for key, value in raw.items():
        if key == "ast_tree":          # not JSON-serialisable, not useful to callers
            continue
        if hasattr(value, "__dataclass_fields__"):
            result[key] = vars(value)  # DiffReport / ValidationResult → dict
        else:
            result[key] = value
    return result


def _error_response(message: str, status: int = 500) -> tuple[Response, int]:
    logger.error("Returning error %d: %s", status, message)
    return jsonify({"error": message}), status


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/health", methods=["GET"])
def health() -> Response:
    return jsonify({"status": "ok"})


@app.route("/optimize", methods=["POST"])
def optimize() -> Response:
    """
    Run the full optimisation pipeline on the submitted code.

    Expected JSON body
    ------------------
    {
        "code":     "<python source string>",   # required
        "language": "python"                    # optional, default "python"
    }
    """
    body = request.get_json(silent=True)
    if not body:
        return _error_response("Request body must be JSON.", status=400)

    code = body.get("code", "").strip()
    if not code:
        return _error_response("'code' field is required and must not be empty.", status=400)

    language = body.get("language", "python")
    logger.info("POST /optimize — language=%s  code_length=%d", language, len(code))

    initial_state: dict[str, Any] = {"code": code, "language": language}

    try:
        result = compiled_graph.invoke(initial_state)
    except Exception as exc:
        logger.error("Graph invocation failed:\n%s", traceback.format_exc())
        return _error_response(f"Pipeline error: {exc}", status=500)

    try:
        payload = _state_to_dict(result)
    except Exception as exc:
        logger.error("State serialisation failed: %s", exc)
        return _error_response(f"Serialisation error: {exc}", status=500)

    if payload.get("error"):
        return _error_response(payload["error"], status=422)

    vr = payload.get("validation_result") or {}
    speedup = vr.get("speedup_ratio", 1.0) if isinstance(vr, dict) else getattr(vr, "speedup_ratio", 1.0)
    logger.info("POST /optimize — done  score=%.2f  speedup=%.3f×", payload.get("opt_score", 0.0), speedup)

    return jsonify(payload), 200


# ---------------------------------------------------------------------------
# Dev server entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)