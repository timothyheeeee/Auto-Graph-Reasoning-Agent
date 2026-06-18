"""NetworkX graph operations and schema loading."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import networkx as nx


class GraphManager:
    """Manages an in-memory NetworkX graph for macro/supply-chain analysis."""

    def __init__(self) -> None:
        self._graph: nx.DiGraph = nx.DiGraph()

    @property
    def graph(self) -> nx.DiGraph:
        """Return the underlying directed graph instance."""
        return self._graph

    def load_from_json(self, path: str | Path) -> None:
        """Load graph data from JSON (node-link or structured blueprint format)."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if "nodes" in data and "edges" in data and "links" not in data:
            self._graph = self._from_structured_blueprint(data)
        else:
            self._graph = nx.node_link_graph(data, directed=True)

    @staticmethod
    def _from_structured_blueprint(data: dict[str, Any]) -> nx.DiGraph:
        """Convert ``{nodes, edges}`` blueprint JSON into a directed NetworkX graph."""
        graph = nx.DiGraph()
        for node in data.get("nodes", []):
            node_id = str(node["id"])
            attrs = {key: value for key, value in node.items() if key != "id"}
            graph.add_node(node_id, **attrs)
        for edge in data.get("edges", []):
            source = str(edge["source"])
            target = str(edge["target"])
            attrs = {key: value for key, value in edge.items() if key not in {"source", "target"}}
            graph.add_edge(source, target, **attrs)
        return graph

    def to_node_link_dict(self) -> dict[str, Any]:
        """Serialize the graph to NetworkX node-link format."""
        return nx.node_link_data(self._graph)

    def node_count(self) -> int:
        return self._graph.number_of_nodes()

    def edge_count(self) -> int:
        return self._graph.number_of_edges()
