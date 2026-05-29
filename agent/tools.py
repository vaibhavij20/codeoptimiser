"""
agent/tools.py
--------------
Helper utilities used by the LangGraph nodes:
  - build_profile_script   : wraps user code in a cProfile harness
  - parse_pstats_output    : parses pstats text → (profile_data, hotspots)
  - parse_ast              : builds an AST tree from source
  - compute_complexity     : cyclomatic-complexity metrics per function
  - compute_diff           : unified diff between original and optimised code
"""

from __future__ import annotations

import ast
import difflib
import io
import textwrap
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# build_profile_script
# ---------------------------------------------------------------------------

def build_profile_script(code: str) -> str:
    """
    Wrap *code* in a cProfile harness that writes pstats text to stdout.

    The produced script:
      1. Enables cProfile around the user code.
      2. Captures pstats output (sorted by cumulative time, top-20 entries).
      3. Prints raw pstats text to stdout so the sandbox can return it.

    Indentation safety: each line of the user code is indented by exactly
    4 spaces so it becomes the body of the try-block.
    """
    # Dedent first to normalise any pre-existing indentation
    safe_code = textwrap.dedent(code)
    indented   = textwrap.indent(safe_code, "    ")

    script = f"""\
import cProfile as _cProfile
import pstats    as _pstats
import io        as _io
import sys       as _sys

_pr = _cProfile.Profile()
_pr.enable()

try:
{indented}
except Exception as _exc:
    print(f"[profile-script] user-code exception: {{_exc}}", file=_sys.stderr)
finally:
    _pr.disable()

_buf = _io.StringIO()
_ps  = _pstats.Stats(_pr, stream=_buf)
_ps.sort_stats("cumulative")
_ps.print_stats(20)
print(_buf.getvalue())
"""
    return script


# ---------------------------------------------------------------------------
# parse_pstats_output
# ---------------------------------------------------------------------------

def parse_pstats_output(raw_output: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """
    Parse the plain-text output produced by ``pstats.Stats.print_stats()``.

    Returns
    -------
    profile_data : dict
        Raw data dict (currently stores the full text under "raw").
    hotspots : list[dict]
        Up to 20 entries, each with keys:
          name, filename, lineno, ncalls, tottime, cumtime
        Sorted by cumtime descending.

    pstats line format
    ------------------
    ::

        ncalls  tottime  percall  cumtime  percall filename:lineno(function)
           100    0.001    0.000    0.005    0.000 mymodule.py:42(my_func)

    Recursive calls look like ``200/100`` — we take the *outer* call count.
    Built-in entries look like ``{built-in method builtins.len}``; we keep
    them with filename="builtin".
    """
    profile_data: dict[str, Any]       = {"raw": raw_output or ""}
    hotspots:     list[dict[str, Any]] = []

    if not raw_output or not raw_output.strip():
        return profile_data, hotspots

    in_stats = False

    for raw_line in raw_output.splitlines():
        line = raw_line.strip()

        # ── locate the stats header ──────────────────────────────────────
        if not in_stats:
            if "ncalls" in line and "tottime" in line and "cumtime" in line:
                in_stats = True
            continue

        # blank line inside the stats block → keep going (pstats emits one
        # blank line between the header and the data rows)
        if not line:
            continue

        # ── parse a data row ─────────────────────────────────────────────
        # Split on whitespace — the location token is always last
        parts = line.split(None, 5)   # at most 6 tokens
        if len(parts) < 6:
            continue

        try:
            # ncalls may be "200/100" for recursive functions
            ncalls = int(parts[0].split("/")[0])
            tottime = float(parts[1])
            # parts[2] = percall (tottime), skip
            cumtime = float(parts[3])
            # parts[4] = percall (cumtime), skip
            location = parts[5].strip()   # "filename:lineno(funcname)"

            # ── decode location ──────────────────────────────────────────
            if "(" in location and location.endswith(")"):
                func_start = location.rfind("(")
                name       = location[func_start + 1 : -1]
                file_part  = location[:func_start]

                if ":" in file_part:
                    # split from the right so Windows paths (C:\…) survive
                    filename, lineno_str = file_part.rsplit(":", 1)
                    lineno = int(lineno_str) if lineno_str.isdigit() else 0
                else:
                    filename, lineno = file_part, 0
            else:
                # e.g. "{built-in method …}"
                name      = location
                filename  = "builtin"
                lineno    = 0

            hotspots.append({
                "name":     name,
                "filename": filename.strip(),
                "lineno":   lineno,
                "ncalls":   ncalls,
                "tottime":  round(tottime, 6),
                "cumtime":  round(cumtime, 6),
            })

        except (ValueError, IndexError):
            # malformed line — skip silently
            continue

    # Sort by cumulative time so the worst offenders come first
    hotspots.sort(key=lambda h: h["cumtime"], reverse=True)
    return profile_data, hotspots


# ---------------------------------------------------------------------------
# parse_ast
# ---------------------------------------------------------------------------

def parse_ast(code: str) -> ast.Module | None:
    """
    Parse *code* into an AST.  Returns ``None`` on SyntaxError so callers
    can handle it gracefully without try/except at the call site.
    """
    try:
        return ast.parse(code)
    except SyntaxError:
        return None


# ---------------------------------------------------------------------------
# compute_complexity
# ---------------------------------------------------------------------------

def compute_complexity(tree: ast.Module | None) -> list[dict[str, Any]]:
    """
    Return a list of per-function complexity metrics.

    Each entry has:
      name, lineno, cyclomatic_complexity, num_args, num_returns, num_branches

    Cyclomatic complexity = 1  +  number of decision points
    (if / elif / for / while / except / assert / with / comprehension guards).
    """
    if tree is None:
        return []

    metrics: list[dict[str, Any]] = []

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        branches   = 0
        returns    = 0
        num_args   = len(node.args.args)

        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.For, ast.While,
                                   ast.ExceptHandler, ast.With,
                                   ast.Assert)):
                branches += 1
            elif isinstance(child, (ast.ListComp, ast.SetComp,
                                     ast.DictComp, ast.GeneratorExp)):
                branches += 1          # comprehension guard = branch
            elif isinstance(child, ast.Return):
                returns += 1

        metrics.append({
            "name":                  node.name,
            "lineno":                node.lineno,
            "cyclomatic_complexity": 1 + branches,
            "num_args":              num_args,
            "num_returns":           returns,
            "num_branches":          branches,
        })

    # Highest complexity first
    metrics.sort(key=lambda m: m["cyclomatic_complexity"], reverse=True)
    return metrics


# ---------------------------------------------------------------------------
# DiffReport dataclass
# ---------------------------------------------------------------------------

@dataclass
class DiffReport:
    unified_diff:      str
    lines_added:       int
    lines_removed:     int
    functions_changed: list[str] = field(default_factory=list)
    summary:           str       = ""


# ---------------------------------------------------------------------------
# compute_diff
# ---------------------------------------------------------------------------

def compute_diff(original: str, optimized: str) -> DiffReport:
    """
    Compute a unified diff between *original* and *optimized* source strings.

    Also returns line-level add/remove counts and a list of top-level
    function names whose definitions changed.
    """
    orig_lines = original.splitlines(keepends=True)
    opt_lines  = optimized.splitlines(keepends=True)

    diff_lines = list(
        difflib.unified_diff(
            orig_lines,
            opt_lines,
            fromfile="original.py",
            tofile="optimized.py",
            lineterm="",
        )
    )

    unified_diff  = "".join(diff_lines)
    lines_added   = sum(1 for l in diff_lines if l.startswith("+") and not l.startswith("+++"))
    lines_removed = sum(1 for l in diff_lines if l.startswith("-") and not l.startswith("---"))

    # Detect which top-level function names differ
    functions_changed: list[str] = []
    try:
        orig_tree = ast.parse(original)
        opt_tree  = ast.parse(optimized)

        orig_funcs = {
            n.name: ast.unparse(n)
            for n in ast.walk(orig_tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        opt_funcs = {
            n.name: ast.unparse(n)
            for n in ast.walk(opt_tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        }

        for name, body in orig_funcs.items():
            if name in opt_funcs and opt_funcs[name] != body:
                functions_changed.append(name)
        for name in opt_funcs:
            if name not in orig_funcs:
                functions_changed.append(f"{name} (new)")

    except SyntaxError:
        pass

    summary = (
        f"{lines_added} lines added, {lines_removed} lines removed"
        + (f", functions changed: {', '.join(functions_changed)}" if functions_changed else "")
    )

    return DiffReport(
        unified_diff=unified_diff,
        lines_added=lines_added,
        lines_removed=lines_removed,
        functions_changed=functions_changed,
        summary=summary,
    )