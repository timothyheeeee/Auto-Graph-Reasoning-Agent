"""Dual-layer data ingestion: unstructured text into ChromaDB and structured graph loading."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from config.settings import Settings, get_settings
from src.database.graph_manager import GraphManager
from src.database.vector_store import VectorStore

# ChromaDB metadata values must be scalar types.
MetadataValue = str | int | float | bool


class MarketIngestionPipeline:
    """
    Ingests macro-financial corpus documents into ChromaDB and hydrates the
    NetworkX graph from structured blueprint files.
    """

    def __init__(
        self,
        *,
        persist_directory: str | None = None,
        collection_name: str | None = None,
        settings: Settings | None = None,
    ) -> None:
        cfg = settings or get_settings()
        self._settings = cfg
        self._vector_store = VectorStore(
            persist_directory=persist_directory or cfg.chroma_persist_directory,
            collection_name=collection_name or cfg.chroma_collection_name,
        )
        self._graph_manager = GraphManager()

    @property
    def vector_store(self) -> VectorStore:
        return self._vector_store

    @property
    def graph_manager(self) -> GraphManager:
        return self._graph_manager

    def ingest_document(
        self,
        file_path: str | Path,
        *,
        source_metadata: dict[str, MetadataValue] | None = None,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ) -> int:
        """
        Chunk a text file and upsert embeddings into the ChromaDB collection.

        Returns the number of chunks stored.
        """
        path = Path(file_path)
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            return 0

        chunks = self._chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        base_metadata: dict[str, MetadataValue] = {
            "source_file": path.name,
            **(source_metadata or {}),
        }

        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict[str, MetadataValue]] = []

        stem = path.stem
        for index, chunk in enumerate(chunks):
            ids.append(f"{stem}_chunk_{index}")
            documents.append(chunk)
            metadatas.append({**base_metadata, "chunk_index": index})

        self._vector_store.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )
        return len(chunks)

    def load_structured_graph(self, graph_path: str | Path | None = None) -> GraphManager:
        """Load the structured supply-chain blueprint into the in-memory graph."""
        path = Path(graph_path or self._settings.graph_data_path)
        self._graph_manager.load_from_json(path)
        return self._graph_manager

    @staticmethod
    def _chunk_text(text: str, *, chunk_size: int, chunk_overlap: int) -> list[str]:
        """Split text on paragraph boundaries, then by fixed size with overlap."""
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        chunks: list[str] = []

        for paragraph in paragraphs:
            if len(paragraph) <= chunk_size:
                chunks.append(paragraph)
                continue

            start = 0
            while start < len(paragraph):
                end = start + chunk_size
                chunks.append(paragraph[start:end].strip())
                if end >= len(paragraph):
                    break
                start = max(end - chunk_overlap, start + 1)

        return chunks
