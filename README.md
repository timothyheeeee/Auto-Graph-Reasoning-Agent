# 🕸️ Autonomous Graph-Reasoning Risk Agent

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![LangGraph](https://img.shields.io/badge/LangGraph-Stateful_Agents-orange.svg)
![NetworkX](https://img.shields.io/badge/NetworkX-Graph_Topology-green.svg)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Search-purple.svg)
![Status](https://img.shields.io/badge/Status-Production_Ready-success.svg)

## 📌 Executive Summary
Standard Large Language Models (LLMs) hallucinate math and struggle with multi-hop logical reasoning. This project moves beyond simple "chat wrappers" to establish a highly deterministic, production-grade **Cognitive Reasoning Engine**. 

Designed for macro-financial and supply-chain risk analysis, this system utilizes a hybrid **GraphRAG** architecture orchestrated by a **self-healing LangGraph state machine**. Rather than guessing answers, the agent actively extracts unstructured intelligence, writes functional Python analytics scripts, executes them in an isolated sandbox against a topological network, and iteratively debugs its own code until it achieves mathematical certainty.

**This project demonstrates applied mastery in:**
* **Agentic Orchestration:** Managing cyclic state, conditional edge routing, and asynchronous multi-step execution.
* **Deterministic AI Constraints:** Forcing stochastic models to generate bounded, mathematically sound outputs via isolated Python compilers.
* **Hybrid Data Engineering:** Bridging high-dimensional vector similarity search (ChromaDB) with strict mathematical node-edge traversals (NetworkX).
* **Production Telemetry:** Implementing persistent state checkpointing and execution latency logging for enterprise observability.

---

## 🏗️ System Architecture: The Dual-Memory Cognitive Loop

The agent relies on a specialized architecture that splits memory into semantic context and structural topology, bound together by a self-correcting feedback loop.

### 1. Hybrid Ingestion Pipeline (GraphRAG)
* **Unstructured Narrative (Vector Space):** SEC filings, geopolitical shocks, and news narratives are chunked, embedded, and stored locally in **ChromaDB**.
* **Structured Topology (Graph Space):** Corporate entities, market caps, and weighted dependency edges (e.g., TSMC $\rightarrow$ NVIDIA) are explicitly mapped into an in-memory **NetworkX** directed graph ($G$).

### 2. The Self-Healing Sandbox
1.  **Context Retrieval:** The agent queries ChromaDB to understand the qualitative nature of a supply chain shock.
2.  **Code Generation:** The agent writes a custom Python script designed to traverse the NetworkX graph and calculate the cascading risk coefficients.
3.  **Isolated Execution:** The script runs in a strict, sandboxed environment pre-loaded with `networkx` and safety constraints.
4.  **Traceback Optimization:** If the compiler throws an exception (e.g., `KeyError` or `SyntaxError`), the standard error output is captured and routed back to the LLM. The agent reads its own error logs and dynamically rewrites the script, repeating the loop until successful execution.

### 3. Mathematical Normalization Contract
To ensure enterprise reporting standards, the agent is strictly constrained to output absolute risk coefficients bounded precisely between $0.0$ and $1.0$. Downstream financial logic is processed via multiplicative compounding:

$$\text{Cascade Risk}_{A \rightarrow C} = \text{EdgeWeight}_{A \rightarrow B} \times \text{EdgeWeight}_{B \rightarrow C}$$

---

## 🚀 Key Features & Engineering Highlights

| Feature | Engineering Significance |
| :--- | :--- |
| **LangGraph Cyclic State Machine** | Replaces fragile linear pipelines with robust, persistent state tracking (`AgentState` via Pydantic). Capable of handling asynchronous, non-linear reasoning paths. |
| **Deterministic Code Sandboxing** | Bypasses LLM mathematical hallucinations entirely. By offloading logic to a native Python compiler, the system guarantees 100% computational accuracy on complex graph traversals. |
| **Automated Traceback Parsing** | Implements the foundational concepts of Reinforcement Learning from AI Feedback (RLAIF). The agent autonomously diagnoses system boundary violations and self-corrects without human intervention. |
| **Telemetry & Observability** | Wraps the execution nodes in an audit layer, tracking `execution_latency`, `iteration_count`, and cache hits, logging outputs directly to `data/telemetry_logs.json` for production monitoring. |
| **Streamlit Operations Dashboard** | Exposes the underlying LangGraph state transitions via an asynchronous web interface, allowing users to inspect the agent's thought process, generated code, and real-time tracebacks. |

---

## 💻 Installation & Usage

### Prerequisites
* Python 3.11+
* OpenAI API Key (For the Generator Agent)

### Local Deployment

1. **Clone & Install**
```bash
git clone [https://github.com/yourusername/Auto-Graph-Reasoning-Agent.git](https://github.com/yourusername/Auto-Graph-Reasoning-Agent.git)
cd Auto-Graph-Reasoning-Agent
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
