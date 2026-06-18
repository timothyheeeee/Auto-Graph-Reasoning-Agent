"""LLM-backed graph reasoning nodes: code generation, refinement, and synthesis."""

from __future__ import annotations

import json
import re
from typing import Any

import networkx as nx
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from config.settings import Settings, get_settings
from src.agents.state import AgentState
from src.database.vector_store import VectorStore

_GRAPH_SCHEMA_DOC = """\
NetworkX directed graph is available as `G` (alias: `graph`).
Pre-injected modules: `nx`, `math` (no import statements allowed).

Node attributes:
  - type (str): entity classification, e.g. Supplier, Fabricator
  - country (str): headquarters / domicile region
  - market_cap_billions (float): approximate market capitalization

Edge attributes:
  - relation (str): dependency type, e.g. supplies_optics, manufactures_chips
  - dependency_score (float): 0.0-1.0 exposure / reliance weight

Sandbox contract:
  - Write pure Python that performs graph analytics over `G`.
  - Assign the final analytical payload to a variable named `result`.
  - Use only pre-injected symbols; imports are blocked.
  - Node IDs are exact strings, e.g. "Carl Zeiss", "TSMC", "NVIDIA".

Strict Normalization Contract:
  - Any code traversing `G` MUST return normalized risk coefficients bounded exactly
    between 0.0 (Zero Risk) and 1.0 (Absolute Operational Collapse).
  - Clamp or rescale all computed exposure values into [0.0, 1.0] before assignment.

Calculation Protocol:
  - Propagate risk by multiplying edge `dependency_score` values along path trajectories
    (multiplicative cascade), NOT by summing scores or multiplying raw
    `market_cap_billions` into large arbitrary numbers.

Output Constraint:
  - `result` MUST be a dict mapping entity name strings to normalized float values.
  - Example: result = {'TSMC': 0.8448, 'NVIDIA': 0.8026}
"""

_CODE_FENCE_PATTERN = re.compile(
    r"```(?:python)?\s*\n?(.*?)```",
    re.DOTALL | re.IGNORECASE,
)


def build_llm(settings: Settings | None = None) -> ChatOpenAI:
    """Construct the shared ChatOpenAI client from application settings."""
    cfg = settings or get_settings()
    api_key = cfg.openai_api_key.get_secret_value()
    if not api_key:
        msg = "OPENAI_API_KEY is not configured. Set it in your environment or .env file."
        raise ValueError(msg)

    return ChatOpenAI(
        api_key=api_key,
        model=cfg.llm_model,
        temperature=cfg.llm_temperature,
        max_tokens=cfg.llm_max_tokens,
    )


def _serialize_graph_schema(graph: nx.DiGraph) -> str:
    """Render a compact topology summary for LLM grounding."""
    nodes = [
        {
            "id": node_id,
            **{key: value for key, value in attrs.items()},
        }
        for node_id, attrs in graph.nodes(data=True)
    ]
    edges = [
        {
            "source": source,
            "target": target,
            **{key: value for key, value in attrs.items()},
        }
        for source, target, attrs in graph.edges(data=True)
    ]
    payload = {"nodes": nodes, "edges": edges}
    return json.dumps(payload, indent=2)


def _extract_python_code(response_text: str) -> str:
    """Pull executable Python from an LLM response, stripping markdown fences."""
    match = _CODE_FENCE_PATTERN.search(response_text)
    if match:
        return match.group(1).strip()
    return response_text.strip()


async def _invoke_llm(llm: ChatOpenAI, system_prompt: str, user_prompt: str) -> str:
    """Async LLM invocation wrapper."""
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]
    response = await llm.ainvoke(messages)
    content = response.content
    if isinstance(content, str):
        return content
    return str(content)


async def _retrieve_context(
    vector_store: VectorStore,
    query: str,
    *,
    n_results: int = 5,
) -> list[str]:
    """Fetch semantically relevant narrative chunks from ChromaDB."""
    raw = await vector_store.query(query, n_results=n_results)
    documents = raw.get("documents") or []
    if not documents or not documents[0]:
        return []
    return [chunk for chunk in documents[0] if chunk]


async def generate_initial_code(
    state: AgentState,
    *,
    llm: ChatOpenAI,
    vector_store: VectorStore,
    graph: nx.DiGraph,
) -> dict[str, Any]:
    """
    Retrieve GraphRAG context and ask the LLM to author the first sandbox script.
    """
    context_chunks = await _retrieve_context(vector_store, state.user_query)
    context_block = "\n\n".join(
        f"[Chunk {index + 1}]\n{chunk}" for index, chunk in enumerate(context_chunks)
    ) or "No unstructured intelligence chunks were retrieved."

    system_prompt = (
        "You are an elite quantitative systems engineer specializing in global "
        "macro-financial supply chain graph analytics. You write robust NetworkX "
        "analysis scripts that execute inside a restricted sandbox. Every script "
        "MUST obey the Strict Normalization Contract: risk coefficients in [0.0, 1.0], "
        "multiplicative `dependency_score` propagation along paths, and a `result` "
        "dict mapping entity names to normalized floats."
    )
    user_prompt = f"""\
User analytical question:
{state.user_query}

Retrieved market intelligence (ChromaDB):
{context_block}

Active graph topology (JSON):
{_serialize_graph_schema(graph)}

{_GRAPH_SCHEMA_DOC}

Mandatory enforcement checklist:
1. Traverse `G` using NetworkX path/traversal APIs as needed.
2. Multiply `dependency_score` along each dependency trajectory (never sum scores or
   multiply `market_cap_billions` into raw exposure magnitudes).
3. Normalize every entity risk coefficient to [0.0, 1.0] (0.0 = Zero Risk,
   1.0 = Absolute Operational Collapse).
4. Assign `result` as a dict[str, float], e.g. {{'TSMC': 0.8448, 'NVIDIA': 0.8026}}.

Write ONLY the Python code block. No prose outside the code.
"""

    llm_response = await _invoke_llm(llm, system_prompt, user_prompt)
    generated_code = _extract_python_code(llm_response)

    return {
        "generated_code": generated_code,
        "retrieved_context": context_chunks,
        "error_traceback": None,
        "sandbox_result": None,
    }


async def refine_failed_code(
    state: AgentState,
    *,
    llm: ChatOpenAI,
    graph: nx.DiGraph,
) -> dict[str, Any]:
    """
    Self-healing pass: diagnose sandbox failure and rewrite the execution block.
    """
    logs_block = "\n\n".join(state.execution_logs) or "No execution logs captured."

    system_prompt = (
        "You are an elite debugging engineer for a restricted NetworkX sandbox. "
        "You fix Python graph analytics scripts without introducing blocked imports "
        "or invalid node identifiers. Every corrected script MUST satisfy the Strict "
        "Normalization Contract: `result` is a dict of entity -> float in [0.0, 1.0], "
        "with multiplicative `dependency_score` path propagation. Output only corrected "
        "Python code."
    )
    user_prompt = f"""\
The following sandbox execution failed and must be repaired.

Original user question:
{state.user_query}

Failed code:
{state.generated_code}

Captured execution logs:
{logs_block}

Complete error traceback:
{state.error_traceback}

Graph topology reminder (JSON):
{_serialize_graph_schema(graph)}

{_GRAPH_SCHEMA_DOC}

Diagnose the traceback. Common issues:
- Blocked imports (only nx and math are available)
- KeyError from incorrect node ID strings
- Missing `result` assignment
- Using unsupported APIs
- Risk values outside [0.0, 1.0] or non-dict `result` payloads
- Summing dependency scores or using `market_cap_billions` as multiplicative risk

Mandatory repair checklist:
1. Multiply `dependency_score` along path trajectories only.
2. Normalize all entity coefficients to [0.0, 1.0].
3. Set `result` = {{entity_name: normalized_float, ...}}.

Rewrite the full corrected script. Output ONLY Python code.
"""

    llm_response = await _invoke_llm(llm, system_prompt, user_prompt)
    generated_code = _extract_python_code(llm_response)

    return {
        "generated_code": generated_code,
        "error_traceback": None,
    }


async def synthesize_answer(
    state: AgentState,
    *,
    llm: ChatOpenAI,
    vector_store: VectorStore,
) -> dict[str, Any]:
    """
    Produce a polished macro-financial narrative from sandbox output and intel.
    """
    context_chunks = state.retrieved_context
    if not context_chunks:
        context_chunks = await _retrieve_context(vector_store, state.user_query)

    context_block = "\n\n".join(
        f"[Intel {index + 1}]\n{chunk}" for index, chunk in enumerate(context_chunks)
    ) or "No vector context available."

    execution_summary = "\n\n".join(state.execution_logs) or "No execution logs."
    result_payload = state.sandbox_result or "No structured result was produced."

    hit_iteration_wall = (
        state.error_traceback is not None
        and state.iteration_count >= get_settings().max_iterations
    )

    system_prompt = (
        "You are a senior macro-financial research analyst. Synthesize rigorous, "
        "literate answers that combine quantitative graph findings with supply-chain "
        "risk narrative. Be precise, structured, and actionable."
    )

    if hit_iteration_wall:
        user_prompt = f"""\
The autonomous optimization loop reached its iteration ceiling without a clean execution.

User question:
{state.user_query}

Last execution logs:
{execution_summary}

Last traceback:
{state.error_traceback}

Retrieved intelligence:
{context_block}

Explain what was attempted, why optimization stalled, and provide the best
partial macro-financial assessment possible from available evidence.
"""
    else:
        user_prompt = f"""\
User question:
{state.user_query}

Sandbox structured result:
{result_payload}

Sandbox execution logs:
{execution_summary}

Retrieved market intelligence:
{context_block}

Write a polished macro-financial analysis that directly answers the question.
Cite dependency chains, risk propagation, and quantitative findings where relevant.
"""

    final_answer = await _invoke_llm(llm, system_prompt, user_prompt)
    return {"final_answer": final_answer}
