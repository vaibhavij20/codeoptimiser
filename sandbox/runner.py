"""
sandbox/runner.py
-----------------
DockerRunner — executes arbitrary Python scripts inside an isolated Docker
container and returns their stdout as a string.

Fixes over the original
-----------------------
1. command=f'python -c "{script}"'  BREAKS on any code that contains double
   quotes, newlines, or shell metacharacters.  Fixed by writing the script to
   a temp file and bind-mounting it into the container, then running
   `python /sandbox/script.py`.  This is safe for any script content.

2. detach=True + remove=True means the container is gone before .logs() is
   called on the handle — logs() on a removed container raises NotFound.
   Fixed by using detach=False (blocking run) which returns logs directly, OR
   by waiting for the container explicitly before reading logs (see below).
   We use the blocking approach for simplicity and correctness.

3. No timeout — a runaway script hangs forever.  Fixed with a configurable
   timeout (default 30 s); the container is killed and removed on expiry.

4. No stderr capture — profiler errors printed to stderr were silently lost,
   making empty-hotspot debugging impossible.  Fixed by capturing both stdout
   and stderr (stdout=True, stderr=True on .logs()).

5. No exit-code check — a script that crashes returned empty string with no
   indication of failure.  Fixed by inspecting container exit code and
   including stderr in the return value when non-zero.

6. remove=True with detach=True is racy.  We now manage removal explicitly
   inside a finally block so cleanup always happens even on timeout or error.
"""

from __future__ import annotations

import io
import os
import sys
import tempfile
import textwrap
import logging
from typing import Optional, Union

import docker
import docker.errors

logger = logging.getLogger(__name__)

# Default resource limits — override via constructor kwargs
DEFAULT_MEM_LIMIT    = "512m"
DEFAULT_TIMEOUT_SECS = 30
SANDBOX_IMAGE        = os.environ.get("SANDBOX_IMAGE", "code-optimizer-sandbox")


class DockerRunner:
    """
    Runs a Python script string inside a Docker sandbox container and returns
    the combined stdout+stderr output as a string.

    Parameters
    ----------
    image : str
        Docker image name.  Must have Python 3 available as ``python``.
    mem_limit : str
        Docker memory limit string, e.g. ``"512m"``.
    timeout : int
        Seconds to wait before killing the container.  Default 30.
    """

    def __init__(
        self,
        image:     str = SANDBOX_IMAGE,
        mem_limit: str = DEFAULT_MEM_LIMIT,
        timeout:   int = DEFAULT_TIMEOUT_SECS,
    ) -> None:
        self.image     = image
        self.mem_limit = mem_limit
        self.timeout   = timeout

        try:
            self.client = docker.from_env()
            self.client.ping()
            logger.info("[DockerRunner] connected to Docker daemon")
        except Exception as exc:
            logger.warning("[DockerRunner] cannot connect to Docker: %s", exc)
            raise RuntimeError(
                f"Docker daemon is not reachable: {exc}\n"
                "Start Docker Desktop / the Docker daemon and try again, "
                "or implement a fallback runner."
            ) from exc

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, script: str) -> str:
        """
        Execute *script* inside the sandbox container.

        The script is written to a temporary file on the host and
        bind-mounted read-only into the container at ``/sandbox/script.py``.
        This avoids all shell-quoting and newline problems.

        Returns
        -------
        str
            Combined stdout + stderr from the container.

        Raises
        ------
        RuntimeError
            If the container exits with a non-zero code (the output is
            included in the exception message so callers can log it).
        TimeoutError
            If the container does not finish within ``self.timeout`` seconds.
        """
        # Write script to a temp file — Docker bind-mount is the safest way
        # to pass arbitrary multi-line code into a container.
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            prefix="sandbox_",
            delete=False,
            encoding="utf-8",
        ) as tmp:
            tmp.write(textwrap.dedent(script))
            host_script_path = tmp.name

        container = None
        try:
            logger.debug("[DockerRunner] starting container for script %s", host_script_path)

            container = self.client.containers.run(
                image=self.image,
                command=["python", "/sandbox/script.py"],
                volumes={
                    host_script_path: {
                        "bind": "/sandbox/script.py",
                        "mode": "ro",          # read-only — container cannot modify it
                    }
                },
                # Resource & security constraints
                mem_limit=self.mem_limit,
                network_disabled=True,
                read_only=True,                # immutable root fs
                tmpfs={"/tmp": "size=64m"},    # writable /tmp for pstats temp files
                # Execution control
                detach=True,                   # we manage waiting ourselves for timeout support
                remove=False,                  # we remove in the finally block
                stdout=True,
                stderr=True,
            )

            # Wait with timeout
            result = container.wait(timeout=self.timeout)
            exit_code: int = result.get("StatusCode", -1)

            # Collect logs AFTER the container has finished
            raw_logs: bytes = container.logs(stdout=True, stderr=True)
            output: str = raw_logs.decode("utf-8", errors="replace")

            if exit_code != 0:
                # Non-zero exit — raise so callers can fall back or log
                raise RuntimeError(
                    f"Container exited with code {exit_code}.\n"
                    f"Output:\n{output}"
                )

            logger.debug(
                "[DockerRunner] container finished OK, %d chars output", len(output)
            )
            return output

        except docker.errors.NotFound as exc:
            raise RuntimeError(f"Sandbox image '{self.image}' not found: {exc}") from exc

        except Exception as exc:
            # Re-raise TimeoutError from container.wait as TimeoutError
            if "timed out" in str(exc).lower() or "timeout" in type(exc).__name__.lower():
                logger.error("[DockerRunner] container timed out after %ds", self.timeout)
                if container is not None:
                    try:
                        container.kill()
                    except Exception:
                        pass
                raise TimeoutError(
                    f"Script did not finish within {self.timeout} seconds."
                ) from exc
            raise

        finally:
            # Always clean up — even on timeout, error, or KeyboardInterrupt
            _remove_container(container)
            _remove_tempfile(host_script_path)

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    def image_exists(self) -> bool:
        """Return True if the sandbox image is present on the local daemon."""
        try:
            self.client.images.get(self.image)
            return True
        except docker.errors.ImageNotFound:
            return False

    def __repr__(self) -> str:
        return (
            f"DockerRunner(image={self.image!r}, "
            f"mem_limit={self.mem_limit!r}, "
            f"timeout={self.timeout}s)"
        )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _remove_container(container: Optional[docker.models.containers.Container]) -> None:
    if container is None:
        return
    try:
        container.remove(force=True)
        logger.debug("[DockerRunner] container removed")
    except Exception as exc:
        logger.warning("[DockerRunner] could not remove container: %s", exc)


def _remove_tempfile(path: Optional[str]) -> None:
    if not path:
        return
    try:
        os.unlink(path)
    except OSError as exc:
        logger.warning("[DockerRunner] could not remove temp file %s: %s", path, exc)


# ---------------------------------------------------------------------------
# LocalProfiler — in-process fallback (no Docker required)
# ---------------------------------------------------------------------------

class LocalProfiler:
    """
    Executes a profiling script in-process using exec().

    Used automatically on Railway (and any environment where Docker is
    unavailable) via ``get_profiler()``.

    WARNING: executes arbitrary code — suitable for trusted input only.
    """

    def run(self, script: str) -> str:
        """
        Execute *script* in-process and return its stdout as a string.

        Returns
        -------
        str
            Captured stdout from the script execution.
        """
        buf = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf
        try:
            exec(textwrap.dedent(script), {})  # nosec — trusted input only
        except Exception as exc:
            logger.warning("[LocalProfiler] execution error: %s", exc)
            return f"Execution error: {exc}"
        finally:
            sys.stdout = old_stdout
        return buf.getvalue()

    def __repr__(self) -> str:
        return "LocalProfiler()"


# ---------------------------------------------------------------------------
# Factory — selects the right profiler for the current environment
# ---------------------------------------------------------------------------

def get_profiler() -> Union[DockerRunner, LocalProfiler]:
    """
    Return the appropriate profiler for the current environment.

    On Railway (``RAILWAY_ENVIRONMENT`` is set), Docker-in-Docker is not
    available, so ``LocalProfiler`` is returned directly without attempting
    a Docker connection.  Elsewhere, ``DockerRunner`` is attempted first and
    ``LocalProfiler`` is used as a fallback if the Docker daemon is
    unreachable.
    """
    if os.getenv("RAILWAY_ENVIRONMENT"):
        logger.info("[get_profiler] Railway detected — using LocalProfiler")
        return LocalProfiler()

    try:
        runner = DockerRunner()
        logger.info("[get_profiler] Docker available — using DockerRunner")
        return runner
    except RuntimeError as exc:
        logger.warning("[get_profiler] Docker unavailable (%s) — using LocalProfiler", exc)
        return LocalProfiler()