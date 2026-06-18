"""Unit tests for the sandbox self-correction execution loop."""

from __future__ import annotations

import networkx as nx
import pytest

from src.agents.sandbox import SandboxExecutor
from src.agents.state import AgentState


@pytest.fixture
def sample_graph() -> nx.DiGraph:
    """Minimal supply-chain graph: supplier -> manufacturer -> market."""
    graph = nx.DiGraph()
    graph.add_node("ACME_CORP", type="corporation", region="US")
    graph.add_node("TSMC", type="corporation", region="TW")
    graph.add_node("US_MARKET", type="region")
    graph.add_edge("TSMC", "ACME_CORP", relation="supply_line", weight=0.85)
    graph.add_edge("ACME_CORP", "US_MARKET", relation="distribution", weight=0.6)
    return graph


@pytest.fixture
def executor(sample_graph: nx.DiGraph) -> SandboxExecutor:
    return SandboxExecutor(sample_graph, timeout_seconds=5.0)


class TestSandboxExecutor:
    def test_successful_execution_captures_result(self, executor: SandboxExecutor) -> None:
        code = """
upstream = list(G.predecessors("ACME_CORP"))
result = upstream
print(f"Suppliers: {upstream}")
"""
        outcome = executor.execute(code)

        assert outcome.success is True
        assert outcome.traceback is None
        assert outcome.result == ["TSMC"]
        assert "Suppliers: ['TSMC']" in outcome.stdout

    def test_runtime_error_captures_traceback(self, executor: SandboxExecutor) -> None:
        code = "result = G.nodes['NONEXISTENT']['type']"
        outcome = executor.execute(code)

        assert outcome.success is False
        assert outcome.traceback is not None
        assert "KeyError" in outcome.traceback

    def test_empty_code_returns_failure(self, executor: SandboxExecutor) -> None:
        outcome = executor.execute("   ")
        assert outcome.success is False
        assert "no code" in outcome.stderr.lower()

    def test_restricted_namespace_blocks_imports(self, executor: SandboxExecutor) -> None:
        code = "import os\nresult = os.getcwd()"
        outcome = executor.execute(code)

        assert outcome.success is False
        assert outcome.traceback is not None

    def test_format_feedback_includes_traceback(self, executor: SandboxExecutor) -> None:
        outcome = executor.execute("raise ValueError('bad hop')")
        feedback = executor.format_feedback(outcome)

        assert "FAILURE" in feedback
        assert "TRACEBACK" in feedback
        assert "ValueError" in feedback

    @pytest.mark.asyncio
    async def test_async_execution_success(self, executor: SandboxExecutor) -> None:
        outcome = await executor.execute_async('result = G.number_of_nodes()')
        assert outcome.success is True
        assert outcome.result == 3
        assert outcome.elapsed_seconds >= 0.0

    @pytest.mark.asyncio
    async def test_async_execution_timeout(
        self,
        sample_graph: nx.DiGraph,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Slow synchronous work inside the executor should surface as TimeoutError."""
        import time

        short_executor = SandboxExecutor(sample_graph, timeout_seconds=0.05)

        def slow_execute(_code: str) -> SandboxExecutionResult:
            time.sleep(1.0)
            return SandboxExecutionResult(success=True)

        monkeypatch.setattr(short_executor, "execute", slow_execute)
        outcome = await short_executor.execute_async("result = None")

        assert outcome.success is False
        assert outcome.traceback is not None
        assert "Timeout" in outcome.traceback


class TestAgentState:
    def test_record_execution_appends_logs_and_increments(self) -> None:
        state = AgentState(user_query="Who supplies ACME?")

        state.record_execution(
            stdout="ok",
            stderr="",
            traceback=None,
            success=True,
        )

        assert state.iteration_count == 1
        assert state.error_traceback is None
        assert any("[stdout]" in log for log in state.execution_logs)

    def test_record_execution_stores_traceback_on_failure(self) -> None:
        state = AgentState(user_query="test")
        tb = "Traceback (most recent call last):\nKeyError: 'x'"

        state.record_execution(
            stdout="",
            stderr="err",
            traceback=tb,
            success=False,
        )

        assert state.error_traceback == tb
        assert state.iteration_count == 1

    def test_langgraph_round_trip(self) -> None:
        state = AgentState(
            user_query="Exposure path?",
            generated_code="result = list(G.nodes())",
            iteration_count=2,
        )
        restored = AgentState.from_langgraph_dict(state.to_langgraph_dict())

        assert restored.user_query == state.user_query
        assert restored.generated_code == state.generated_code
        assert restored.iteration_count == state.iteration_count
