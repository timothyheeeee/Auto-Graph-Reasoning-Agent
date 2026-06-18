"""Production CLI for the Autonomous Graph-Reasoning Data Agent."""

from __future__ import annotations

import asyncio
from typing import Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

from config.settings import get_settings
from src.agents.state import AgentState
from src.pipeline import DEFAULT_RUN_CONFIG, build_pipeline, get_telemetry_tracker

app = typer.Typer(
    name="auto-graph-reasoning",
    help="Autonomous Graph-Reasoning Data Agent for macro-financial supply chain analysis.",
    no_args_is_help=True,
)
console = Console()


@app.callback()
def cli() -> None:
    """Autonomous Graph-Reasoning Data Agent CLI."""


def _phase(message: str) -> None:
    console.print(f"[bold cyan]>>[/bold cyan] {message}")


def _render_transition(node_name: str, state: dict[str, Any]) -> None:
    if node_name == "generate_code":
        if state.get("iteration_count", 0) > 0:
            _phase("Self-healing: refining failed sandbox code...")
        else:
            _phase("Generating initial NetworkX analytics code...")
        if state.get("generated_code"):
            preview = state["generated_code"][:120].replace("\n", " ")
            console.print(f"  [dim]code preview:[/dim] {preview}...")

    elif node_name == "execute_sandbox":
        iteration = state.get("iteration_count", 0)
        _phase(f"Executing sandbox (iteration {iteration})...")
        if state.get("error_traceback"):
            console.print("  [red]Execution failed — traceback captured for refinement.[/red]")
        else:
            console.print("  [green]Execution succeeded.[/green]")

    elif node_name == "synthesize_answer":
        _phase("Synthesizing macro-financial analysis...")


async def run_pipeline_async(query: str) -> AgentState:
    """Execute the compiled LangGraph workflow for a single analytical query."""
    settings = get_settings()
    if not settings.openai_api_key.get_secret_value():
        console.print(
            "[bold red]Error:[/bold red] OPENAI_API_KEY is not set. "
            "Add it to your environment or `.env` file before running queries."
        )
        raise typer.Exit(code=1)

    telemetry = get_telemetry_tracker()
    telemetry.begin_cycle()

    _phase("Loading graph topology and ChromaDB corpus...")
    pipeline = build_pipeline(settings=settings)

    initial_state = AgentState(user_query=query)
    _phase(f"Running pipeline for query: [italic]{query}[/italic]")
    console.print(Rule(style="dim"))

    final_state: dict[str, Any] = dict(initial_state.to_langgraph_dict())

    async for event in pipeline.astream(
        initial_state.to_langgraph_dict(),
        stream_mode="updates",
        config=DEFAULT_RUN_CONFIG,
    ):
        for node_name, node_update in event.items():
            final_state.update(node_update)
            _render_transition(node_name, final_state)

    console.print(Rule(style="dim"))

    try:
        snapshot = await pipeline.aget_state(DEFAULT_RUN_CONFIG)
        if snapshot and snapshot.values:
            thread_id = DEFAULT_RUN_CONFIG["configurable"]["thread_id"]
            console.print(
                f"[bold green]Checkpoint saved:[/bold green] state persisted for "
                f"thread '{thread_id}'."
            )
    except Exception as exc:
        console.print(f"[yellow]Checkpoint confirmation skipped:[/yellow] {exc}")

    written = telemetry.finalize_cycle()
    if written:
        console.print(
            f"[bold green]Telemetry appended:[/bold green] {len(written)} node record(s) "
            f"written to {telemetry.ledger_path}."
        )

    return AgentState.model_validate(final_state)


def _print_final_answer(answer: str) -> None:
    console.print()
    console.print(
        Panel(
            answer,
            title="[bold green]Final Analysis[/bold green]",
            border_style="green",
            padding=(1, 2),
        )
    )


@app.command()
def run(
    query: str = typer.Argument(..., help="Natural-language analytical question."),
) -> None:
    """Run the autonomous graph-reasoning pipeline for a single query."""
    try:
        final_state = asyncio.run(run_pipeline_async(query))
    except typer.Exit:
        raise
    except Exception as exc:
        console.print(f"[bold red]Pipeline failed:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc

    if not final_state.final_answer:
        console.print("[yellow]Pipeline completed without a synthesized answer.[/yellow]")
        raise typer.Exit(code=1)

    _print_final_answer(final_state.final_answer)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
