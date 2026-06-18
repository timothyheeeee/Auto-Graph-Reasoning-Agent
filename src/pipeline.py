"""LangGraph workflow: generate -> execute -> optimize loop -> synthesize."""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal

import networkx as nx
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from config.settings import Settings, get_settings
from src.agents import graph_agent
from src.agents.sandbox import CodeSandbox
from src.agents.state import AgentState
from src.agents.telemetry import TelemetryTracker
from src.database.graph_manager import GraphManager
from src.database.vector_store import VectorStore

DEFAULT_RUN_CONFIG: dict[str, Any] = {
    "configurable": {"thread_id": "production-session-001"},
}

# Thread-safe in-memory checkpoint store shared across compiled workflows.
_CHECKPOINTER = MemorySaver()
_TELEMETRY_TRACKER = TelemetryTracker()


class AgentGraphState(AgentState):
    """
    LangGraph-compatible state schema.

    Inherits the canonical Pydantic ``AgentState`` fields and adds list-append
    semantics for log/context fields via annotated reducers.
    """

    execution_logs: Annotated[list[str], operator.add]  # type: ignore[assignment]
    retrieved_context: Annotated[list[str], operator.add]  # type: ignore[assignment]


RouteDecision = Literal["generate_code", "synthesize_answer"]


class GraphReasoningPipeline:
    """Factory that binds infrastructure dependencies to LangGraph node closures."""

    def __init__(
        self,
        *,
        graph: nx.DiGraph,
        vector_store: VectorStore,
        settings: Settings | None = None,
        llm: ChatOpenAI | None = None,
        checkpointer: MemorySaver | None = None,
        telemetry: TelemetryTracker | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._graph = graph
        self._vector_store = vector_store
        self._llm = llm
        self._checkpointer = checkpointer or _CHECKPOINTER
        self._telemetry = telemetry or _TELEMETRY_TRACKER
        self._sandbox = CodeSandbox(
            graph,
            timeout_seconds=self._settings.sandbox_timeout_seconds,
        )

    def _get_llm(self) -> ChatOpenAI:
        if self._llm is None:
            self._llm = graph_agent.build_llm(self._settings)
        return self._llm

    @staticmethod
    def _coerce_state(state: AgentGraphState | dict[str, Any]) -> AgentState:
        if isinstance(state, AgentState):
            return state
        return AgentState.model_validate(state)

    async def _generate_code_node(self, state: AgentGraphState | dict[str, Any]) -> dict[str, Any]:
        node_name = "generate_code"
        agent_state = self._coerce_state(state)
        self._telemetry.start_node(node_name)

        try:
            if agent_state.error_traceback:
                update = await graph_agent.refine_failed_code(
                    agent_state,
                    llm=self._get_llm(),
                    graph=self._graph,
                )
            else:
                update = await graph_agent.generate_initial_code(
                    agent_state,
                    llm=self._get_llm(),
                    vector_store=self._vector_store,
                    graph=self._graph,
                )
        except Exception:
            self._telemetry.complete_node(
                node_name,
                iteration_count=agent_state.iteration_count,
            )
            raise

        self._telemetry.complete_node(
            node_name,
            iteration_count=agent_state.iteration_count,
        )
        return update

    async def _execute_sandbox_node(self, state: AgentGraphState | dict[str, Any]) -> dict[str, Any]:
        node_name = "execute_sandbox"
        agent_state = self._coerce_state(state)
        self._telemetry.start_node(node_name)

        try:
            outcome = await self._sandbox.execute_async(agent_state.generated_code)

            log_entries: list[str] = []
            if outcome.stdout.strip():
                log_entries.append(f"[stdout]\n{outcome.stdout.strip()}")
            if outcome.stderr.strip():
                log_entries.append(f"[stderr]\n{outcome.stderr.strip()}")
            if outcome.success and outcome.result is not None:
                log_entries.append(f"[result]\n{repr(outcome.result)}")

            sandbox_result = repr(outcome.result) if outcome.result is not None else None
            next_iteration = agent_state.iteration_count + 1

            update = {
                "execution_logs": log_entries,
                "error_traceback": outcome.traceback,
                "iteration_count": next_iteration,
                "sandbox_result": sandbox_result,
            }
        except Exception:
            self._telemetry.complete_node(
                node_name,
                iteration_count=agent_state.iteration_count + 1,
                sandbox_cache_hit=False,
            )
            raise

        self._telemetry.complete_node(
            node_name,
            iteration_count=next_iteration,
            sandbox_cache_hit=outcome.cache_hit,
        )
        return update

    async def _synthesize_answer_node(self, state: AgentGraphState | dict[str, Any]) -> dict[str, Any]:
        node_name = "synthesize_answer"
        agent_state = self._coerce_state(state)
        self._telemetry.start_node(node_name)

        try:
            update = await graph_agent.synthesize_answer(
                agent_state,
                llm=self._get_llm(),
                vector_store=self._vector_store,
            )
        except Exception:
            self._telemetry.complete_node(
                node_name,
                iteration_count=agent_state.iteration_count,
            )
            raise

        self._telemetry.complete_node(
            node_name,
            iteration_count=agent_state.iteration_count,
        )
        return update

    def should_continue(self, state: AgentGraphState | dict[str, Any]) -> RouteDecision:
        """Route to synthesis on success or iteration ceiling; otherwise refine."""
        agent_state = self._coerce_state(state)

        if agent_state.error_traceback is None:
            return "synthesize_answer"

        if agent_state.iteration_count >= self._settings.max_iterations:
            return "synthesize_answer"

        return "generate_code"

    def compile(self) -> CompiledStateGraph:
        """Build and compile the LangGraph state machine with checkpointing."""
        workflow: StateGraph = StateGraph(AgentGraphState)

        workflow.add_node("generate_code", self._generate_code_node)
        workflow.add_node("execute_sandbox", self._execute_sandbox_node)
        workflow.add_node("synthesize_answer", self._synthesize_answer_node)

        workflow.add_edge(START, "generate_code")
        workflow.add_edge("generate_code", "execute_sandbox")
        workflow.add_conditional_edges(
            "execute_sandbox",
            self.should_continue,
            {
                "generate_code": "generate_code",
                "synthesize_answer": "synthesize_answer",
            },
        )
        workflow.add_edge("synthesize_answer", END)

        return workflow.compile(checkpointer=self._checkpointer)


def get_telemetry_tracker() -> TelemetryTracker:
    """Return the shared telemetry tracker used by pipeline node wrappers."""
    return _TELEMETRY_TRACKER


def get_checkpointer() -> MemorySaver:
    """Return the shared in-memory LangGraph checkpointer."""
    return _CHECKPOINTER


def build_pipeline(
    *,
    graph: nx.DiGraph | None = None,
    vector_store: VectorStore | None = None,
    settings: Settings | None = None,
) -> CompiledStateGraph:
    """
    Load infrastructure, bind dependencies, and return a compiled LangGraph app.
    """
    cfg = settings or get_settings()

    if graph is None:
        manager = GraphManager()
        manager.load_from_json(cfg.graph_data_path)
        graph = manager.graph

    if vector_store is None:
        vector_store = VectorStore(
            persist_directory=cfg.chroma_persist_directory,
            collection_name=cfg.chroma_collection_name,
        )

    return GraphReasoningPipeline(
        graph=graph,
        vector_store=vector_store,
        settings=cfg,
    ).compile()


# Default compiled workflow export for CLI and integration tests.
compiled_workflow = None


def get_compiled_workflow() -> CompiledStateGraph:
    """Lazy singleton accessor for the compiled pipeline graph."""
    global compiled_workflow
    if compiled_workflow is None:
        compiled_workflow = build_pipeline()
    return compiled_workflow
