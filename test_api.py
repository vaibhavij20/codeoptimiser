"""
test_api.py  (project root — quick smoke-test suite)
------------------------------------------------------
Uses Flask's built-in test client so the server does NOT need to be running.
All tests are self-contained and work with plain `pytest` or `python -m pytest`.

For the full integration test suite see tests/test_api.py.

Run
---
    pytest test_api.py -v
    pytest test_api.py -v -k "test_health"     # single test
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from api.app import app as flask_app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def client():
    """Return a Flask test client with testing mode enabled."""
    flask_app.config["TESTING"] = True
    with flask_app.test_client() as c:
        yield c


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_status_ok(self, client):
        data = resp = client.get("/health").get_json()
        assert data == {"status": "ok"}


# ---------------------------------------------------------------------------
# /optimize — request validation
# ---------------------------------------------------------------------------

class TestOptimizeValidation:
    def test_empty_body_returns_400(self, client):
        resp = client.post("/optimize", data="not json", content_type="text/plain")
        assert resp.status_code == 400

    def test_missing_code_field_returns_400(self, client):
        resp = client.post(
            "/optimize",
            data=json.dumps({"language": "python"}),
            content_type="application/json",
        )
        assert resp.status_code == 400

    def test_empty_code_returns_400(self, client):
        resp = client.post(
            "/optimize",
            data=json.dumps({"code": "   "}),
            content_type="application/json",
        )
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# /optimize — pipeline (graph mocked so no LLM/Docker calls are made)
# ---------------------------------------------------------------------------

# Minimal valid graph result that satisfies _state_to_dict() in app.py
_MOCK_RESULT = {
    "code":              "x = 1",
    "language":          "python",
    "analysis":          "No major issues found.",
    "opt_score":         0.1,
    "optimized_code":    None,
    "diff_report":       None,
    "final_report":      "# Report\n\nCode looks good.",
    "validation_result": {"tests_passed": True, "tests_run": 1,
                          "tests_failed": 0,    "failure_details": [],
                          "original_runtime_ms": 5.0,
                          "refactored_runtime_ms": 5.0,
                          "speedup_ratio": 1.0},
    "node_timings":      {"ingest": 0.01, "profile": 0.02},
    "error":             "",
    "suggestions":       [],
    "hotspots":          [],
    "profile_report":    "",
    "retrieved_patterns": [],
    "llm_reasoning":     "",
    "llm_backend":       "gemini",
    "ast_tree":          None,       # stripped by _state_to_dict
    "complexity_metrics": [],
    "profile_data":      {},
}


@pytest.fixture()
def mock_graph():
    """Patch compiled_graph.invoke to return _MOCK_RESULT without side-effects."""
    with patch("api.app.compiled_graph") as mock_cg:
        mock_cg.invoke.return_value = _MOCK_RESULT
        yield mock_cg


class TestOptimizePipeline:
    def test_valid_request_returns_200(self, client, mock_graph):
        resp = client.post(
            "/optimize",
            data=json.dumps({"code": "print('hello')"}),
            content_type="application/json",
        )
        assert resp.status_code == 200

    def test_response_contains_required_keys(self, client, mock_graph):
        resp = client.post(
            "/optimize",
            data=json.dumps({"code": "x = 1 + 1"}),
            content_type="application/json",
        )
        body = resp.get_json()
        for key in ("analysis", "opt_score", "final_report"):
            assert key in body, f"Missing key: {key}"

    def test_graph_receives_code_and_language(self, client, mock_graph):
        client.post(
            "/optimize",
            data=json.dumps({"code": "y = 2", "language": "python"}),
            content_type="application/json",
        )
        call_kwargs = mock_graph.invoke.call_args[0][0]
        assert call_kwargs["code"] == "y = 2"
        assert call_kwargs["language"] == "python"

    def test_language_defaults_to_python(self, client, mock_graph):
        client.post(
            "/optimize",
            data=json.dumps({"code": "z = 3"}),
            content_type="application/json",
        )
        call_kwargs = mock_graph.invoke.call_args[0][0]
        assert call_kwargs["language"] == "python"

    def test_graph_exception_returns_500(self, client):
        with patch("api.app.compiled_graph") as mock_cg:
            mock_cg.invoke.side_effect = RuntimeError("sandbox exploded")
            resp = client.post(
                "/optimize",
                data=json.dumps({"code": "import this"}),
                content_type="application/json",
            )
        assert resp.status_code == 500
        assert "error" in resp.get_json()