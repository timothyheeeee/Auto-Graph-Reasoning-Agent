"""Tests for structured graph loading and ingestion helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.database.graph_manager import GraphManager
from src.database.ingestion import MarketIngestionPipeline

STRUCTURED_GRAPH = Path("data/structured_graph.json")


class TestStructuredGraphLoading:
    def test_load_structured_blueprint(self) -> None:
        manager = GraphManager()
        manager.load_from_json(STRUCTURED_GRAPH)

        assert manager.node_count() == 6
        assert manager.edge_count() == 6
        assert manager.graph.has_edge("TSMC", "NVIDIA")
        assert manager.graph.edges["TSMC", "NVIDIA"]["dependency_score"] == 0.95


class TestIngestionChunking:
    def test_chunk_text_splits_paragraphs(self) -> None:
        text = "Paragraph one.\n\nParagraph two is longer " + ("x" * 600)
        chunks = MarketIngestionPipeline._chunk_text(
            text,
            chunk_size=500,
            chunk_overlap=50,
        )

        assert len(chunks) >= 2
        assert chunks[0] == "Paragraph one."
