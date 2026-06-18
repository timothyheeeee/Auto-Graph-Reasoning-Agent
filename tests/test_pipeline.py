"""Unit tests for LangGraph routing and pipeline wiring."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import networkx as nx
import pytest

from config.settings import Settings
from src.agents.state import AgentState
from src.pipeline import GraphReasoningPipeline


@pytest.fixture
def sample_graph() -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_node("TSMC", type="Fabricator", country="Taiwan")
    graph.add_node("NVIDIA", type="Fabless Designer", country="USA")
    graph.add_edge("TSMC", "NVIDIA", relation="manufactures_chips", dependency_score=0.95)
    return graph


@pytest.fixture
def pipeline_deps(sample_graph: nx.DiGraph) -> GraphReasoningPipeline:
    settings = Settings(
        openai_api_key="test-key",
        max_iterations=3,
    )
    vector_store = MagicMock()
    llm = MagicMock()
    return GraphReasoningPipeline(
        graph=sample_graph,
        vector_store=vector_store,
        settings=settings,
        llm=llm,
    )


class TestShouldContinue:
    def test_routes_to_synthesis_on_success(self, pipeline_deps: GraphReasoningPipeline) -> None:
        state = AgentState(user_query="test", error_traceback=None, iteration_count=1)
        assert pipeline_deps.should_continue(state) == "synthesize_answer"

    def test_routes_to_generate_on_recoverable_error(
        self,
        pipeline_deps: GraphReasoningPipeline,
    ) -> None:
        state = AgentState(
            user_query="test",
            error_traceback="KeyError: 'TSMC'",
            iteration_count=1,
        )
        assert pipeline_deps.should_continue(state) == "generate_code"

    def test_routes_to_synthesis_on_iteration_ceiling(
        self,
        pipeline_deps: GraphReasoningPipeline,
    ) -> None:
        state = AgentState(
            user_query="test",
            error_traceback="KeyError: 'TSMC'",
            iteration_count=3,
        )
        assert pipeline_deps.should_continue(state) == "synthesize_answer"


class TestExecuteSandboxNode:
    @pytest.mark.asyncio
    async def test_execute_node_records_success_payload(
        self,
        pipeline_deps: GraphReasoningPipeline,
    ) -> None:
        state = AgentState(
            user_query="Who supplies NVIDIA?",
            generated_code='result = list(G.predecessors("NVIDIA"))',
        )
        update = await pipeline_deps._execute_sandbox_node(state)

        assert update["error_traceback"] is None
        assert update["iteration_count"] == 1
        assert update["sandbox_result"] == "['TSMC']"
        assert any("[result]" in log for log in update["execution_logs"])

    @pytest.mark.asyncio
    async def test_execute_node_records_failure(
        self,
        pipeline_deps: GraphReasoningPipeline,
    ) -> None:
        state = AgentState(
            user_query="test",
            generated_code='result = G.nodes["MISSING"]["type"]',
        )
        update = await pipeline_deps._execute_sandbox_node(state)

        assert update["error_traceback"] is not None
        assert update["iteration_count"] == 1


class TestGenerateCodeNode:
    @pytest.mark.asyncio
    async def test_generate_node_calls_initial_path(
        self,
        pipeline_deps: GraphReasoningPipeline,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        mock_initial = AsyncMock(return_value={"generated_code": "result = 1"})
        monkeypatch.setattr("src.pipeline.graph_agent.generate_initial_code", mock_initial)

        state = AgentState(user_query="test query")
        update = await pipeline_deps._generate_code_node(state)

        mock_initial.assert_awaited_once()
        assert update["generated_code"] == "result = 1"

    @pytest.mark.asyncio
    async def test_generate_node_calls_refine_path(
        self,
        pipeline_deps: GraphReasoningPipeline,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        mock_refine = AsyncMock(return_value={"generated_code": "result = 2"})
        monkeypatch.setattr("src.pipeline.graph_agent.refine_failed_code", mock_refine)

        state = AgentState(
            user_query="test query",
            error_traceback="ValueError: bad code",
            generated_code="bad",
        )
        update = await pipeline_deps._generate_code_node(state)

        mock_refine.assert_awaited_once()
        assert update["generated_code"] == "result = 2"
