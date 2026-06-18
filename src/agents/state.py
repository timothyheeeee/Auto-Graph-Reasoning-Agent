"""
LangGraph agent state definition.

Tracks the full lifecycle of a graph-reasoning query through code generation,
sandbox execution, error recovery, and final answer synthesis.
"""

from __future__ import annotations

from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _append_execution_logs(existing: list[str], new: list[str] | str) -> list[str]:
    """Reducer: append new log lines while preserving order."""
    if isinstance(new, str):
        return [*existing, new]
    return [*existing, *new]


class AgentState(BaseModel):
    """
    Canonical state object for the autonomous graph-reasoning workflow.

  Fields mirror the LangGraph state machine transitions:
  query intake -> code generation -> sandbox execution -> optimization loop -> answer.
    """

    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",
        str_strip_whitespace=True,
    )

    user_query: str = Field(
        default="",
        description="Original natural-language analytical question from the user.",
    )
    generated_code: str = Field(
        default="",
        description="Latest Python/NetworkX script produced by the code generator agent.",
    )
    execution_logs: Annotated[list[str], _append_execution_logs] = Field(
        default_factory=list,
        description="Chronological stdout/stderr and diagnostic messages from sandbox runs.",
    )
    error_traceback: str | None = Field(
        default=None,
        description="Full traceback from the most recent failed execution, if any.",
    )
    iteration_count: int = Field(
        default=0,
        ge=0,
        description="Number of generate-execute-optimize cycles completed.",
    )
    final_answer: str | None = Field(
        default=None,
        description="Synthesized answer returned to the user once the loop converges.",
    )
    retrieved_context: list[str] = Field(
        default_factory=list,
        description="ChromaDB narrative chunks retrieved for the active query.",
    )
    sandbox_result: str | None = Field(
        default=None,
        description="Serialized sandbox `result` payload from the latest execution.",
    )

    @field_validator("execution_logs", mode="before")
    @classmethod
    def _coerce_logs(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return list(value)

    def record_execution(
        self,
        *,
        stdout: str,
        stderr: str,
        traceback: str | None,
        success: bool,
    ) -> None:
        """Append sandbox output and update error state after a run."""
        if stdout.strip():
            self.execution_logs = [*self.execution_logs, f"[stdout]\n{stdout.strip()}"]
        if stderr.strip():
            self.execution_logs = [*self.execution_logs, f"[stderr]\n{stderr.strip()}"]

        if success:
            self.error_traceback = None
        else:
            self.error_traceback = traceback

        self.iteration_count += 1

    def to_langgraph_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict for LangGraph node I/O."""
        return self.model_dump()

    @classmethod
    def from_langgraph_dict(cls, data: dict[str, Any]) -> AgentState:
        """Hydrate from a LangGraph node return payload."""
        return cls.model_validate(data)
