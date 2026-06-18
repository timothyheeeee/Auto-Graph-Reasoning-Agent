"""Bootstrap local vector store and structured graph from project data files."""

from __future__ import annotations

from pathlib import Path

from config.settings import get_settings
from src.database.ingestion import MarketIngestionPipeline

UNSTRUCTURED_DOC = Path("data/unstructured_docs/supply_chain_intel.txt")
STRUCTURED_GRAPH = Path("data/structured_graph.json")


def bootstrap_system_data() -> None:
    """
    Verify data paths and run the dual-layer ingestion pipeline.

    Layer 1: chunk + vectorize unstructured intel into local ChromaDB.
    Layer 2: hydrate the NetworkX graph from the structured blueprint JSON.
    """
    settings = get_settings()

    print("[1/4] Verifying project data architecture paths...")
    UNSTRUCTURED_DOC.parent.mkdir(parents=True, exist_ok=True)
    Path(settings.chroma_persist_directory).mkdir(parents=True, exist_ok=True)

    if not UNSTRUCTURED_DOC.is_file() or not STRUCTURED_GRAPH.is_file():
        print(
            "Error: Missing raw files. Ensure both of the following exist:\n"
            f"  - {STRUCTURED_GRAPH}\n"
            f"  - {UNSTRUCTURED_DOC}"
        )
        return

    print("[2/4] Initializing local ChromaDB vector database instance...")
    try:
        pipeline = MarketIngestionPipeline(settings=settings)
    except Exception as exc:
        print(f"Initialization failed: {exc}")
        print("Ensure requirements.txt libraries are fully installed.")
        return

    print("[3/4] Parsing unstructured text and executing vector injection...")
    meta_blueprint = {
        "sector": "Semiconductor Infrastructure",
        "data_cycle": "2026-Q2",
        "classification": "Confidential Market Analytics",
    }

    total_chunks = pipeline.ingest_document(
        file_path=UNSTRUCTURED_DOC,
        source_metadata=meta_blueprint,
    )

    print("[4/4] Loading structured graph blueprint into NetworkX...")
    graph_manager = pipeline.load_structured_graph(STRUCTURED_GRAPH)

    print("\n=======================================================")
    print("INGESTION PIPELINE COMPLETE")
    print(f"Total text chunks vectorized and stored: {total_chunks}")
    print(f"Graph nodes loaded: {graph_manager.node_count()}")
    print(f"Graph edges loaded: {graph_manager.edge_count()}")
    print(f"ChromaDB collection: {settings.chroma_collection_name}")
    print(f"Structured graph source: {STRUCTURED_GRAPH}")
    print("=======================================================")
    print("Your agent now has structured memory and code sandboxing capabilities.")


if __name__ == "__main__":
    bootstrap_system_data()
