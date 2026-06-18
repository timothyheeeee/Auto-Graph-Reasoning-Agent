"""ChromaDB semantic chunk retrieval."""

from __future__ import annotations

from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection


class VectorStore:
    """Thin async-friendly wrapper around a ChromaDB collection."""

    def __init__(
        self,
        persist_directory: str,
        collection_name: str,
    ) -> None:
        self._client = chromadb.PersistentClient(path=persist_directory)
        self._collection: Collection = self._client.get_or_create_collection(
            name=collection_name,
        )

    @property
    def collection(self) -> Collection:
        return self._collection

    async def query(
        self,
        query_text: str,
        n_results: int = 5,
    ) -> dict[str, Any]:
        """Retrieve semantically similar document chunks."""
        return self._collection.query(
            query_texts=[query_text],
            n_results=n_results,
        )
