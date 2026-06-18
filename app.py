"""
Streamlit command center for the Macro-Financial & Supply-Chain Graph Agent.

Run:
    streamlit run app.py
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import networkx as nx
import streamlit as st
from langgraph.graph.state import CompiledStateGraph

from config.settings import Settings, get_settings
from src.agents.state import AgentState
from src.database.graph_manager import GraphManager
from src.database.vector_store import VectorStore
from src.pipeline import (
    DEFAULT_RUN_CONFIG,
    build_pipeline,
    get_telemetry_tracker,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

QUICK_SCENARIOS: dict[str, str] = {
    "Carl Zeiss production halt": (
        "What happens if Carl Zeiss halts production? Quantify downstream exposure."
    ),
    "TSMC operational delay": (
        "What happens if TSMC encounters an operational delay? "
        "Which fabless designers are most exposed?"
    ),
    "ASML supply disruption": (
        "What is the cascade risk if ASML EUV machine shipments freeze?"
    ),
    "Tariff shock propagation": (
        "How would a 12% margin compression at TSMC cascade to AMD, NVIDIA, and Apple?"
    ),
    "Apple dependency map": (
        "Map Apple's multi-tier silicon dependencies and identify single points of failure."
    ),
}

PHASE_MESSAGES: dict[str, Callable[[dict[str, Any]], str]] = {
    "generate_code": lambda s: (
        "Step 2: Self-healing — refining Python script..."
        if s.get("iteration_count", 0) > 0
        else "Step 1: Extracting vector context & synthesizing Python script..."
    ),
    "execute_sandbox": lambda s: (
        f"Step 3: Executing sandbox validation "
        f"(iteration {s.get('iteration_count', 0)})..."
    ),
    "synthesize_answer": lambda _: "Step 4: Synthesizing macro-financial analysis...",
}


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GraphMetrics:
    """Supply-chain topology summary for the sidebar."""

    total_entities: int
    total_dependencies: int
    critical_monopoly_nodes: list[str]
    mean_dependency_score: float


@dataclass(frozen=True, slots=True)
class Infrastructure:
    """Cached runtime dependencies for the dashboard."""

    settings: Settings
    graph: nx.DiGraph
    pipeline: CompiledStateGraph
    graph_manager: GraphManager


# ---------------------------------------------------------------------------
# Infrastructure loaders
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading graph topology & agent pipeline...")
def load_infrastructure() -> Infrastructure:
    """Load graph, vector store, and compiled LangGraph workflow."""
    settings = get_settings()

    if not settings.openai_api_key.get_secret_value():
        raise RuntimeError(
            "OPENAI_API_KEY is not configured. Add your key to `.env` in the project root."
        )

    graph_path = Path(settings.graph_data_path)
    chroma_path = Path(settings.chroma_persist_directory)

    if not graph_path.is_file():
        raise FileNotFoundError(
            f"Structured graph not found at `{graph_path}`. "
            "Ensure `data/structured_graph.json` exists."
        )

    if not chroma_path.is_dir():
        raise FileNotFoundError(
            f"ChromaDB store not found at `{chroma_path}`. "
            "Run `python feed_data.py` to bootstrap vector data."
        )

    manager = GraphManager()
    manager.load_from_json(graph_path)

    # Validate ChromaDB is reachable before compiling the pipeline.
    VectorStore(
        persist_directory=settings.chroma_persist_directory,
        collection_name=settings.chroma_collection_name,
    )

    pipeline = build_pipeline(
        graph=manager.graph,
        settings=settings,
    )

    return Infrastructure(
        settings=settings,
        graph=manager.graph,
        pipeline=pipeline,
        graph_manager=manager,
    )


def compute_graph_metrics(graph: nx.DiGraph) -> GraphMetrics:
    """Derive sidebar analytics from the active NetworkX topology."""
    dependency_scores = [
        float(attrs["dependency_score"])
        for _, _, attrs in graph.edges(data=True)
        if "dependency_score" in attrs
    ]

    # Source nodes with no inbound edges are upstream monopoly chokepoints.
    monopoly_nodes = sorted(
        node_id
        for node_id in graph.nodes()
        if graph.in_degree(node_id) == 0
    )

    mean_score = (
        sum(dependency_scores) / len(dependency_scores) if dependency_scores else 0.0
    )

    return GraphMetrics(
        total_entities=graph.number_of_nodes(),
        total_dependencies=graph.number_of_edges(),
        critical_monopoly_nodes=monopoly_nodes,
        mean_dependency_score=mean_score,
    )


# ---------------------------------------------------------------------------
# Async pipeline runner
# ---------------------------------------------------------------------------


async def stream_pipeline(
    pipeline: CompiledStateGraph,
    query: str,
    on_phase: Callable[[str, dict[str, Any]], None],
    *,
    run_config: dict[str, Any] | None = None,
) -> tuple[AgentState, list[str]]:
    """Execute the LangGraph workflow asynchronously with per-node callbacks."""
    config = run_config or DEFAULT_RUN_CONFIG
    telemetry = get_telemetry_tracker()
    telemetry.begin_cycle()

    confirmations: list[str] = []
    initial = AgentState(user_query=query)
    merged: dict[str, Any] = dict(initial.to_langgraph_dict())

    async for event in pipeline.astream(
        initial.to_langgraph_dict(),
        config=config,
        stream_mode="updates",
    ):
        for node_name, node_update in event.items():
            merged.update(node_update)
            on_phase(node_name, merged)

    try:
        snapshot = await pipeline.aget_state(config)
        if snapshot and snapshot.values:
            thread_id = config.get("configurable", {}).get("thread_id", "unknown")
            confirmations.append(
                f"Checkpoint state securely saved for thread `{thread_id}`."
            )
    except Exception:
        pass

    written = telemetry.finalize_cycle()
    if written:
        ledger = telemetry.ledger_path
        confirmations.append(
            f"Telemetry appended: {len(written)} node record(s) -> `{ledger}`."
        )

    return AgentState.model_validate(merged), confirmations


def run_pipeline_sync(
    pipeline: CompiledStateGraph,
    query: str,
    on_phase: Callable[[str, dict[str, Any]], None],
    *,
    run_config: dict[str, Any] | None = None,
) -> tuple[AgentState, list[str]]:
    """Bridge asyncio execution for Streamlit's synchronous script model."""
    return asyncio.run(stream_pipeline(pipeline, query, on_phase, run_config=run_config))


# ---------------------------------------------------------------------------
# UI rendering
# ---------------------------------------------------------------------------


def inject_styles() -> None:
    """Minimal professional dashboard styling."""
    st.markdown(
        """
        <style>
            .block-container { padding-top: 1.5rem; max-width: 1100px; }
            div[data-testid="stMetric"] {
                background: #0e1117;
                border: 1px solid #262730;
                border-radius: 8px;
                padding: 0.75rem;
            }
            .final-answer {
                background: linear-gradient(135deg, #0f172a 0%, #111827 100%);
                border-left: 4px solid #22c55e;
                border-radius: 8px;
                padding: 1.25rem 1.5rem;
                margin-top: 0.5rem;
            }
            .phase-step { font-size: 0.95rem; color: #93c5fd; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(metrics: GraphMetrics, settings: Settings) -> None:
    """Sidebar supply-chain state summary."""
    st.sidebar.title("Supply Chain State")
    st.sidebar.caption("Live topology parsed from structured graph blueprint.")

    col_a, col_b = st.sidebar.columns(2)
    col_a.metric("Entities", metrics.total_entities)
    col_b.metric("Dependencies", metrics.total_dependencies)

    st.sidebar.metric(
        "Mean Edge Dependency",
        f"{metrics.mean_dependency_score:.2f}",
        help="Average `dependency_score` across all supply-line edges.",
    )

    st.sidebar.subheader("Critical Monopoly Nodes")
    if metrics.critical_monopoly_nodes:
        for node_id in metrics.critical_monopoly_nodes:
            node_attrs = st.session_state.get("_graph", {}).get(node_id, {})
            node_type = node_attrs.get("type", "Unknown")
            st.sidebar.markdown(f"- **{node_id}** · `{node_type}`")
    else:
        st.sidebar.info("No upstream monopoly nodes detected.")

    st.sidebar.divider()
    st.sidebar.subheader("Runtime Configuration")
    st.sidebar.text(f"Model: {settings.llm_model}")
    st.sidebar.text(f"Max iterations: {settings.max_iterations}")
    st.sidebar.text(f"Collection: {settings.chroma_collection_name}")


def render_load_error(exc: Exception) -> None:
    """Graceful failure panel with troubleshooting steps."""
    st.error("Infrastructure failed to load. The dashboard cannot run queries yet.")
    st.markdown(
        f"""
**Error details**
```
{exc}
```

**Troubleshooting checklist**
1. Add `OPENAI_API_KEY` to `.env` in the project root.
2. Run `python feed_data.py` to vectorize unstructured intel into ChromaDB.
3. Confirm `data/structured_graph.json` exists and is valid JSON.
4. Install dependencies: `pip install -r requirements.txt`
5. Launch from the project root: `streamlit run app.py`
        """
    )


def render_phase_status(phase_log: list[str]) -> None:
    """Display chronological agent phase transitions."""
    if not phase_log:
        return
    st.markdown("#### Execution Timeline")
    for entry in phase_log:
        st.markdown(f'<p class="phase-step">{entry}</p>', unsafe_allow_html=True)


def render_inspector(state: AgentState) -> None:
    """Self-correction inspector: code, logs, tracebacks, iterations."""
    with st.expander(
        "Agent Thought Process & Sandbox Execution Logs",
        expanded=bool(state.generated_code or state.execution_logs),
    ):
        st.markdown("##### Generated NetworkX Script")
        if state.generated_code:
            st.code(state.generated_code, language="python", line_numbers=True)
        else:
            st.caption("No code generated yet.")

        st.markdown("##### Optimization Loop Status")
        iter_cols = st.columns(3)
        iter_cols[0].metric("Iterations Completed", state.iteration_count)
        iter_cols[1].metric(
            "Last Run Status",
            "Success" if state.error_traceback is None and state.iteration_count else (
                "Failed" if state.error_traceback else "Pending"
            ),
        )
        iter_cols[2].metric(
            "Context Chunks",
            len(state.retrieved_context),
        )

        if state.error_traceback:
            st.warning(
                f"Sandbox exception intercepted (iteration {state.iteration_count}). "
                "The agent will attempt self-healing when iterations remain."
            )
            st.code(state.error_traceback, language="text")

        if state.retrieved_context:
            st.markdown("##### Retrieved Vector Context")
            for index, chunk in enumerate(state.retrieved_context, start=1):
                st.markdown(f"**Chunk {index}**")
                st.caption(chunk)

        if state.execution_logs:
            st.markdown("##### Sandbox Execution Logs")
            for log_entry in state.execution_logs:
                st.text(log_entry)

        if state.sandbox_result:
            st.markdown("##### Structured Sandbox Result")
            st.code(state.sandbox_result, language="python")


def render_final_answer(answer: str | None) -> None:
    """Deliver the synthesized macro-financial report."""
    st.markdown("---")
    st.markdown("## Final Macro-Financial Analysis")

    if not answer:
        st.info("No synthesized answer was produced for this run.")
        return

    with st.container(border=True):
        st.markdown(answer)


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(
        page_title="Graph Reasoning Command Center",
        page_icon="🌐",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_styles()

    st.title("Macro-Financial & Supply-Chain Agent")
    st.caption(
        "GraphRAG + LangGraph autonomous analytics over semiconductor dependency networks."
    )

    # ------------------------------------------------------------------ load
    try:
        infra = load_infrastructure()
    except Exception as exc:
        render_load_error(exc)
        return

    metrics = compute_graph_metrics(infra.graph)
    st.session_state["_graph"] = dict(infra.graph.nodes(data=True))
    render_sidebar(metrics, infra.settings)

    # ----------------------------------------------------------- query input
    st.markdown("### Analytical Query")

    scenario_label = st.selectbox(
        "Quick-Run Scenarios",
        options=["Custom query"] + list(QUICK_SCENARIOS.keys()),
        help="Select a predefined macro shock scenario or write your own query.",
    )

    default_query = (
        QUICK_SCENARIOS.get(scenario_label, "")
        if scenario_label != "Custom query"
        else (
            "What happens if TSMC encounters an operational delay?"
        )
    )

    query = st.text_area(
        "Enter your supply-chain analytical question",
        value=default_query,
        height=120,
        placeholder="What happens if TSMC encounters an operational delay?",
    )

    run_clicked = st.button("Run Autonomous Analysis", type="primary", use_container_width=True)

    # -------------------------------------------------------------- execute
    if run_clicked:
        if not query.strip():
            st.warning("Please enter an analytical query before running the pipeline.")
            return

        phase_log: list[str] = []
        status_box = st.status("Agent pipeline running...", expanded=True)
        progress = status_box.empty()
        inspector_placeholder = st.empty()
        answer_placeholder = st.empty()

        def on_phase(node_name: str, state_dict: dict[str, Any]) -> None:
            if node_name == "_checkpoint":
                st.success(state_dict.get("message", "Checkpoint state securely saved."))
                return
            if node_name == "_telemetry":
                st.info(state_dict.get("message", "Telemetry data appended."))
                return

            label_fn = PHASE_MESSAGES.get(node_name)
            if label_fn:
                message = label_fn(state_dict)
                phase_log.append(message)
                progress.markdown(f"**{message}**")

            live_state = AgentState.model_validate(state_dict)
            with inspector_placeholder.container():
                render_inspector(live_state)

        try:
            with st.spinner("Executing LangGraph self-optimizing workflow..."):
                final_state, confirmations = run_pipeline_sync(
                    infra.pipeline,
                    query.strip(),
                    on_phase,
                    run_config=DEFAULT_RUN_CONFIG,
                )
        except Exception as exc:
            status_box.update(label="Pipeline failed", state="error", expanded=True)
            st.exception(exc)
            return

        status_box.update(label="Analysis complete", state="complete", expanded=False)

        with status_box:
            render_phase_status(phase_log)
            for message in confirmations:
                st.success(message)

        st.session_state["last_agent_state"] = final_state

        with answer_placeholder.container():
            render_final_answer(final_state.final_answer)

        with inspector_placeholder.container():
            render_inspector(final_state)

    elif "last_agent_state" in st.session_state:
        st.markdown("---")
        st.caption("Showing results from the most recent analysis run.")
        render_final_answer(st.session_state["last_agent_state"].final_answer)
        render_inspector(st.session_state["last_agent_state"])


if __name__ == "__main__":
    main()
