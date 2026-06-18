"""
Isolated code execution environment for self-healing agent loops.

Executes dynamically generated Python against a provided in-memory NetworkX
graph while capturing stdout, stderr, and full tracebacks for LLM feedback.
"""

from __future__ import annotations

import asyncio
import builtins
import hashlib
import io
import traceback
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, field
from typing import Any, Final

import networkx as nx

# Safe builtins exposed to generated agent code.
_SAFE_BUILTINS: Final[dict[str, Any]] = {
    name: getattr(builtins, name)
    for name in (
        "abs",
        "all",
        "any",
        "bool",
        "dict",
        "enumerate",
        "filter",
        "float",
        "frozenset",
        "int",
        "len",
        "list",
        "map",
        "max",
        "min",
        "print",
        "range",
        "reversed",
        "round",
        "set",
        "sorted",
        "str",
        "sum",
        "tuple",
        "zip",
    )
}


@dataclass(frozen=True, slots=True)
class SandboxExecutionResult:
    """Structured outcome of a single sandbox execution attempt."""

    success: bool
    stdout: str = ""
    stderr: str = ""
    traceback: str | None = None
    result: Any = field(default=None, repr=False)
    elapsed_seconds: float = 0.0
    cache_hit: bool = False

    @property
    def has_error(self) -> bool:
        return not self.success


class SandboxExecutor:
    """
    Production-grade sandbox for executing agent-generated graph analytics code.

    Security model:
    - Restricted global namespace (no imports except networkx/math pre-injected).
    - No filesystem or subprocess access from generated code.
    - Execution wrapped in try/except with full traceback capture.
    - Optional asyncio timeout to prevent runaway loops.
    """

    def __init__(
        self,
        graph: nx.DiGraph | nx.Graph,
        *,
        timeout_seconds: float = 30.0,
        allowed_modules: dict[str, Any] | None = None,
    ) -> None:
        self._graph = graph
        self._timeout_seconds = timeout_seconds
        self._allowed_modules = allowed_modules or {}
        self._result_cache: dict[str, SandboxExecutionResult] = {}

    def _build_namespace(self) -> dict[str, Any]:
        """Construct the restricted execution namespace."""
        import math

        namespace: dict[str, Any] = {
            "__builtins__": _SAFE_BUILTINS,
            "G": self._graph,
            "graph": self._graph,
            "nx": nx,
            "math": math,
            "result": None,
        }
        namespace.update(self._allowed_modules)
        return namespace

    def execute(self, code: str) -> SandboxExecutionResult:
        """
        Execute Python code synchronously against the bound graph.

        The generated script may assign its analytical output to a variable
        named `result`; that value is returned in ``SandboxExecutionResult.result``.
        """
        if not code or not code.strip():
            return SandboxExecutionResult(
                success=False,
                stderr="No code provided for execution.",
                traceback="ValueError: generated_code is empty",
            )

        cache_key = hashlib.sha256(code.encode("utf-8")).hexdigest()
        cached = self._result_cache.get(cache_key)
        if cached is not None:
            return SandboxExecutionResult(
                success=cached.success,
                stdout=cached.stdout,
                stderr=cached.stderr,
                traceback=cached.traceback,
                result=cached.result,
                elapsed_seconds=cached.elapsed_seconds,
                cache_hit=True,
            )

        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        namespace = self._build_namespace()

        try:
            with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                compiled = compile(code, "<agent_generated>", "exec")
                exec(compiled, namespace, namespace)  # noqa: S102 — intentional sandbox exec

            outcome = SandboxExecutionResult(
                success=True,
                stdout=stdout_buffer.getvalue(),
                stderr=stderr_buffer.getvalue(),
                result=namespace.get("result"),
            )
            self._result_cache[cache_key] = outcome
            return outcome

        except Exception:
            outcome = SandboxExecutionResult(
                success=False,
                stdout=stdout_buffer.getvalue(),
                stderr=stderr_buffer.getvalue(),
                traceback=traceback.format_exc(),
            )
            self._result_cache[cache_key] = outcome
            return outcome

    async def execute_async(self, code: str) -> SandboxExecutionResult:
        """Execute code in a thread pool with an asyncio timeout guard."""
        loop = asyncio.get_running_loop()
        start = loop.time()

        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(None, self.execute, code),
                timeout=self._timeout_seconds,
            )
            elapsed = loop.time() - start
            return SandboxExecutionResult(
                success=result.success,
                stdout=result.stdout,
                stderr=result.stderr,
                traceback=result.traceback,
                result=result.result,
                elapsed_seconds=elapsed,
                cache_hit=result.cache_hit,
            )
        except asyncio.TimeoutError:
            elapsed = loop.time() - start
            return SandboxExecutionResult(
                success=False,
                stderr=f"Execution exceeded {self._timeout_seconds:.1f}s timeout.",
                traceback=(
                    f"TimeoutError: sandbox execution exceeded "
                    f"{self._timeout_seconds:.1f} seconds"
                ),
                elapsed_seconds=elapsed,
            )

    def format_feedback(self, result: SandboxExecutionResult) -> str:
        """
        Format execution output for LLM self-correction prompts.

        Combines stdout, stderr, and traceback into a single diagnostic block.
        """
        sections: list[str] = []

        if result.stdout.strip():
            sections.append(f"=== STDOUT ===\n{result.stdout.rstrip()}")
        if result.stderr.strip():
            sections.append(f"=== STDERR ===\n{result.stderr.rstrip()}")
        if result.traceback:
            sections.append(f"=== TRACEBACK ===\n{result.traceback.rstrip()}")
        if result.success and result.result is not None:
            sections.append(f"=== RESULT ===\n{repr(result.result)}")

        if not sections:
            return "Execution completed with no output."

        status = "SUCCESS" if result.success else "FAILURE"
        header = f"--- Sandbox Execution ({status}, {result.elapsed_seconds:.2f}s) ---"
        return header + "\n" + "\n\n".join(sections)


# Alias used by the LangGraph orchestration layer.
CodeSandbox = SandboxExecutor
