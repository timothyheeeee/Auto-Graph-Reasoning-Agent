"""Lightweight performance and audit tracking for LangGraph node execution."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class TelemetryRecord:
    """Single node execution audit record."""

    node_name: str
    execution_latency_seconds: float
    iteration_count: int
    sandbox_cache_hit: bool
    recorded_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TelemetryTracker:
    """
    Track LangGraph node latency and append structured records to a local ledger.

    Call :meth:`begin_cycle` at pipeline start, :meth:`start_node` /
    :meth:`complete_node` around each node, then :meth:`finalize_cycle` to flush.
    """

    def __init__(self, ledger_path: str | Path = "data/telemetry_logs.json") -> None:
        self._ledger_path = Path(ledger_path)
        self._cycle_records: list[TelemetryRecord] = []
        self._active_starts: dict[str, float] = {}

    @property
    def ledger_path(self) -> Path:
        return self._ledger_path

    def begin_cycle(self) -> None:
        """Reset in-memory buffers for a new pipeline execution cycle."""
        self._cycle_records.clear()
        self._active_starts.clear()

    def start_node(self, node_name: str) -> None:
        """Mark the start of a LangGraph node execution step."""
        self._active_starts[node_name] = time.perf_counter()

    def complete_node(
        self,
        node_name: str,
        *,
        iteration_count: int,
        sandbox_cache_hit: bool = False,
    ) -> TelemetryRecord:
        """Record completion metrics for a LangGraph node execution step."""
        started_at = self._active_starts.pop(node_name, None)
        latency = time.perf_counter() - started_at if started_at is not None else 0.0

        record = TelemetryRecord(
            node_name=node_name,
            execution_latency_seconds=round(latency, 6),
            iteration_count=iteration_count,
            sandbox_cache_hit=sandbox_cache_hit,
        )
        self._cycle_records.append(record)
        return record

    def finalize_cycle(self) -> list[dict[str, Any]]:
        """
        Append the current cycle's telemetry records to the monitoring ledger.

        Returns the records written during this cycle.
        """
        if not self._cycle_records:
            return []

        payload_records = [record.to_dict() for record in self._cycle_records]
        existing: list[dict[str, Any]] = []

        try:
            if self._ledger_path.is_file():
                raw = self._ledger_path.read_text(encoding="utf-8").strip()
                if raw:
                    parsed = json.loads(raw)
                    if isinstance(parsed, list):
                        existing = parsed
        except (OSError, json.JSONDecodeError):
            existing = []

        merged = [*existing, *payload_records]
        self._ledger_path.parent.mkdir(parents=True, exist_ok=True)
        self._ledger_path.write_text(
            json.dumps(merged, indent=2),
            encoding="utf-8",
        )

        written = payload_records
        self._cycle_records.clear()
        self._active_starts.clear()
        return written
