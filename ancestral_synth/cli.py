"""Command-line interface for Ancestral Synth."""

import os

from dotenv import load_dotenv

load_dotenv()  # Load .env file before any other imports that need API keys

# Map GOOGLE_AI_STUDIO_API_KEY to GEMINI_API_KEY if set (pydantic-ai expects GEMINI_API_KEY)
if os.environ.get("GOOGLE_AI_STUDIO_API_KEY") and not os.environ.get("GEMINI_API_KEY"):
    os.environ["GEMINI_API_KEY"] = os.environ["GOOGLE_AI_STUDIO_API_KEY"]

import asyncio
from pathlib import Path

import typer
from loguru import logger
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from ancestral_synth.config import settings
from ancestral_synth.persistence.database import Database
from ancestral_synth.services.genealogy_service import GenealogyService

app = typer.Typer(
    name="ancestral-synth",
    help="Generate fictional genealogical datasets using LLMs",
    no_args_is_help=True,
)
console = Console()


def configure_logging(verbose: bool = False) -> None:
    """Configure logging."""
    import sys

    logger.remove()
    level = "DEBUG" if verbose else "INFO"
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    )


# Cache for models (fetched once per session)
_openai_models_cache: set[str] | None = None
_google_models_cache: set[str] | None = None


def _fetch_openai_models() -> set[str] | None:
    """Fetch available models from OpenAI API.

    Returns:
        Set of model IDs, or None if fetching failed.
    """
    global _openai_models_cache

    if _openai_models_cache is not None:
        return _openai_models_cache

    try:
        import os

        from openai import OpenAI

        # Check if API key is available
        if not os.environ.get("OPENAI_API_KEY"):
            return None

        client = OpenAI()
        models = client.models.list()

        # Filter to chat/completion models (gpt-*, o1-*, etc.)
        chat_models = {
            m.id for m in models.data
            if m.id.startswith(("gpt-", "o1", "o3", "o4"))
        }

        _openai_models_cache = chat_models
        return chat_models

    except Exception as e:
        logger.debug(f"Failed to fetch OpenAI models: {e}")
        return None


def _fetch_google_models() -> set[str] | None:
    """Fetch available models from Google AI Studio API.

    Returns:
        Set of model IDs, or None if fetching failed.
    """
    global _google_models_cache

    if _google_models_cache is not None:
        return _google_models_cache

    try:
        from google import genai

        # Check if API key is available
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_AI_STUDIO_API_KEY")
        if not api_key:
            return None

        client = genai.Client(api_key=api_key)
        models = client.models.list()

        # Filter to Gemini models that support generateContent
        gemini_models = set()
        for model in models:
            # Model names come as "models/gemini-2.0-flash" - extract just the model part
            model_id = model.name.replace("models/", "") if model.name.startswith("models/") else model.name
            if model_id.startswith("gemini"):
                gemini_models.add(model_id)

        _google_models_cache = gemini_models
        return gemini_models

    except Exception as e:
        logger.debug(f"Failed to fetch Google models: {e}")
        return None


def _validate_model_name() -> None:
    """Warn if model name looks invalid."""
    model = settings.llm_model
    provider = settings.llm_provider

    if provider == "openai":
        known_models = _fetch_openai_models()

        if known_models is None:
            # Could not fetch models, skip validation
            return

        if model not in known_models:
            console.print(
                f"[bold yellow]⚠ Warning:[/bold yellow] "
                f"Model '{model}' is not a known OpenAI model."
            )
            # Show a subset of relevant models for readability
            relevant_models = sorted(
                m for m in known_models
                if m.startswith(("gpt-4o", "gpt-4-", "gpt-3.5", "o1", "o3", "o4"))
                and not m.startswith(("gpt-4o-realtime", "gpt-4o-audio", "gpt-4o-mini-realtime"))
            )
            if relevant_models:
                console.print(f"  Some available models: {', '.join(relevant_models[:15])}")
            console.print("  This may cause API errors and slow retries.")
            console.print()

    elif provider == "google":
        known_models = _fetch_google_models()

        if known_models is None:
            # Could not fetch models, skip validation
            return

        if model not in known_models:
            console.print(
                f"[bold yellow]⚠ Warning:[/bold yellow] "
                f"Model '{model}' is not a known Google AI Studio model."
            )
            # Show a subset of relevant models for readability
            relevant_models = sorted(
                m for m in known_models
                if m.startswith(("gemini-2", "gemini-1.5", "gemini-flash", "gemini-pro"))
            )
            if relevant_models:
                console.print(f"  Some available models: {', '.join(relevant_models[:10])}")
            console.print("  This may cause API errors and slow retries.")
            console.print()


@app.command()
def generate(
    count: int = typer.Option(1, "--count", "-n", help="Number of persons to generate"),
    db_path: Path = typer.Option(
        settings.database_path,
        "--db",
        "-d",
        help="Path to database file",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output with timing"),
) -> None:
    """Generate new persons in the genealogical dataset."""
    import time

    configure_logging(verbose)
    _validate_model_name()

    async def _generate() -> None:
        total_start = time.perf_counter()

        async with Database(db_path) as db:
            service = GenealogyService(db, verbose=verbose)

            if verbose:
                console.print()
                console.print(f"[bold]Configuration:[/bold]")
                console.print(f"  LLM: {settings.llm_provider}:{settings.llm_model}")
                console.print(f"  Biography words: ~{settings.biography_word_count}")
                console.print(f"  Rate limit: {settings.llm_requests_per_minute} req/min")
                console.print()

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
                disable=verbose,  # Disable spinner in verbose mode for cleaner output
            ) as progress:
                task = progress.add_task(f"Generating {count} person(s)...", total=count)

                for i in range(count):
                    person_start = time.perf_counter()
                    progress.update(task, description=f"Processing person {i + 1}/{count}...")

                    if verbose:
                        console.print(f"[bold cyan]Person {i + 1}/{count}[/bold cyan]")

                    # Clear timer for this person
                    service.timer.clear()

                    try:
                        person = await service.process_next()
                        person_duration = time.perf_counter() - person_start

                        if person:
                            if verbose:
                                console.print(
                                    f"  [green]✓[/green] Created: [bold]{person.full_name}[/bold] "
                                    f"(gen {person.generation}) in [cyan]{person_duration:.1f}s[/cyan]"
                                )
                                console.print()
                            else:
                                console.print(
                                    f"  [green]✓[/green] Created: {person.full_name} "
                                    f"(gen {person.generation})"
                                )
                        else:
                            console.print("  [yellow]![/yellow] No person to process")
                    except Exception as e:
                        console.print(f"  [red]✗[/red] Error: {e}")
                        if verbose:
                            logger.exception("Generation failed")

                    progress.advance(task)

            total_duration = time.perf_counter() - total_start

            # Show stats
            stats = await service.get_statistics()
            console.print()
            console.print("[bold]Dataset Statistics:[/bold]")
            console.print(f"  Total persons: {stats['total_persons']}")
            console.print(f"  Complete: {stats['complete']}")
            console.print(f"  Pending: {stats['pending']}")
            console.print(f"  Queue size: {stats['queue_size']}")

            if verbose:
                console.print()
                console.print("[bold]Timing Summary:[/bold]")
                console.print(f"  Total time: [cyan]{total_duration:.1f}s[/cyan]")
                if count > 0:
                    console.print(f"  Average per person: [cyan]{total_duration / count:.1f}s[/cyan]")

                # Display cost summary
                from ancestral_synth.utils.cost_tracker import format_cost, format_tokens

                cost_summary = service.cost_tracker.get_summary()
                console.print()
                console.print("[bold]Cost Summary:[/bold]")
                console.print(f"  Provider: {cost_summary['provider']}:{cost_summary['model']}")
                console.print(f"  Total LLM calls: {cost_summary['total_llm_calls']}")
                console.print(
                    f"  Total tokens: {format_tokens(cost_summary['total_input_tokens'])} input, "
                    f"{format_tokens(cost_summary['total_output_tokens'])} output"
                )
                console.print(f"  [bold green]Total cost: {format_cost(cost_summary['total_cost'])}[/bold green]")
                if count > 0:
                    avg_cost = cost_summary['total_cost'] / count
                    console.print(f"  Average per person: {format_cost(avg_cost)}")

    asyncio.run(_generate())


@app.command()
def stats(
    db_path: Path = typer.Option(
        settings.database_path,
        "--db",
        "-d",
        help="Path to database file",
    ),
) -> None:
    """Show statistics about the genealogical dataset."""
    from ancestral_synth.services.query_service import QueryService

    async def _stats() -> None:
        async with Database(db_path) as db:
            service = QueryService(db)
            stats = await service.get_statistics()

            table = Table(title="Genealogy Dataset Statistics")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green", justify="right")

            table.add_row("Total Persons", str(stats["total_persons"]))
            table.add_row("Complete (with biography)", str(stats["complete"]))
            table.add_row("Pending (no biography yet)", str(stats["pending"]))
            table.add_row("Queued (awaiting processing)", str(stats["queued"]))
            table.add_row("Queue Size", str(stats["queue_size"]))

            console.print(table)

    asyncio.run(_stats())


@app.command()
def list_persons(
    db_path: Path = typer.Option(
        settings.database_path,
        "--db",
        "-d",
        help="Path to database file",
    ),
    limit: int = typer.Option(20, "--limit", "-l", help="Maximum number of persons to show"),
    status: str | None = typer.Option(None, "--status", "-s", help="Filter by status"),
) -> None:
    """List persons in the dataset."""
    from ancestral_synth.domain.enums import PersonStatus
    from ancestral_synth.persistence.repositories import PersonRepository
    from sqlmodel import select
    from ancestral_synth.persistence.tables import PersonTable

    async def _list() -> None:
        async with Database(db_path) as db:
            async with db.session() as session:
                stmt = select(PersonTable).limit(limit)
                if status:
                    try:
                        status_enum = PersonStatus(status.lower())
                        stmt = stmt.where(PersonTable.status == status_enum)
                    except ValueError:
                        console.print(f"[red]Invalid status: {status}[/red]")
                        console.print(f"Valid options: {', '.join(s.value for s in PersonStatus)}")
                        return

                result = await session.exec(stmt)
                persons = list(result.all())

                if not persons:
                    console.print("[yellow]No persons found.[/yellow]")
                    return

                table = Table(title=f"Persons ({len(persons)} shown)")
                table.add_column("Name", style="cyan")
                table.add_column("Birth", style="green")
                table.add_column("Death", style="red")
                table.add_column("Gen", justify="center")
                table.add_column("Status", style="yellow")

                for person in persons:
                    birth = str(person.birth_date.year) if person.birth_date else "-"
                    death = str(person.death_date.year) if person.death_date else "-"
                    table.add_row(
                        f"{person.given_name} {person.surname}",
                        birth,
                        death,
                        str(person.generation),
                        person.status.value,
                    )

                console.print(table)

    asyncio.run(_list())


@app.command()
def show(
    name: str = typer.Argument(..., help="Name of person to show (partial match)"),
    db_path: Path = typer.Option(
        settings.database_path,
        "--db",
        "-d",
        help="Path to database file",
    ),
) -> None:
    """Show details of a specific person."""
    from sqlmodel import select
    from ancestral_synth.persistence.tables import PersonTable

    async def _show() -> None:
        async with Database(db_path) as db:
            async with db.session() as session:
                # Search by partial name match
                stmt = select(PersonTable).where(
                    PersonTable.given_name.ilike(f"%{name}%")  # type: ignore[union-attr]
                    | PersonTable.surname.ilike(f"%{name}%")  # type: ignore[union-attr]
                )
                result = await session.exec(stmt)
                persons = list(result.all())

                if not persons:
                    console.print(f"[yellow]No person found matching '{name}'[/yellow]")
                    return

                for person in persons[:3]:  # Show max 3 matches
                    console.print()
                    console.print(f"[bold cyan]{person.given_name} {person.surname}[/bold cyan]")
                    console.print(f"  ID: {person.id}")
                    console.print(f"  Gender: {person.gender.value}")
                    console.print(f"  Status: {person.status.value}")
                    console.print(f"  Generation: {person.generation}")

                    if person.birth_date:
                        console.print(f"  Birth: {person.birth_date}")
                    if person.birth_place:
                        console.print(f"  Birth Place: {person.birth_place}")
                    if person.death_date:
                        console.print(f"  Death: {person.death_date}")
                    if person.death_place:
                        console.print(f"  Death Place: {person.death_place}")

                    if person.biography:
                        console.print()
                        console.print("[bold]Biography:[/bold]")
                        # Show first 500 chars
                        bio_preview = person.biography[:500]
                        if len(person.biography) > 500:
                            bio_preview += "..."
                        console.print(f"  {bio_preview}")

    asyncio.run(_show())


@app.command()
def init(
    db_path: Path = typer.Option(
        settings.database_path,
        "--db",
        "-d",
        help="Path to database file",
    ),
) -> None:
    """Initialize a new database."""

    async def _init() -> None:
        async with Database(db_path) as db:
            console.print(f"[green]✓[/green] Database initialized at: {db_path}")

    asyncio.run(_init())


@app.command()
def config() -> None:
    """Show current configuration."""
    table = Table(title="Current Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Database Path", str(settings.database_path))
    table.add_row("LLM Provider", settings.llm_provider)
    table.add_row("LLM Model", settings.llm_model)
    table.add_row("Biography Word Count", str(settings.biography_word_count))
    table.add_row("Batch Size", str(settings.batch_size))
    table.add_row("Forest Fire Probability", str(settings.forest_fire_probability))
    table.add_row("Min Parent Age", str(settings.min_parent_age))
    table.add_row("Max Parent Age", str(settings.max_parent_age))
    table.add_row("Max Lifespan", str(settings.max_lifespan))

    console.print(table)
    console.print()
    console.print("[dim]Set environment variables with ANCESTRAL_ prefix to configure.[/dim]")
    console.print("[dim]Example: ANCESTRAL_LLM_MODEL=gpt-4o[/dim]")


# Export subcommand group
export_app = typer.Typer(help="Export data to various formats")
app.add_typer(export_app, name="export")


@export_app.command("json")
def export_json(
    db_path: Path = typer.Option(
        settings.database_path,
        "--db",
        "-d",
        help="Path to database file",
    ),
    output: Path = typer.Option(
        Path("genealogy.json"),
        "--output",
        "-o",
        help="Output file path",
    ),
) -> None:
    """Export data to JSON format."""
    from ancestral_synth.export.json_exporter import JSONExporter

    async def _export() -> None:
        async with Database(db_path) as db:
            exporter = JSONExporter(db)
            with open(output, "w") as f:
                await exporter.export(f)
            console.print(f"[green]✓[/green] Exported to: {output}")

    asyncio.run(_export())


@export_app.command("csv")
def export_csv(
    db_path: Path = typer.Option(
        settings.database_path,
        "--db",
        "-d",
        help="Path to database file",
    ),
    output_dir: Path = typer.Option(
        Path("csv_export"),
        "--output-dir",
        "-o",
        help="Output directory for CSV files",
    ),
) -> None:
    """Export data to CSV format."""
    from ancestral_synth.export.csv_exporter import CSVExporter

    async def _export() -> None:
        output_dir.mkdir(parents=True, exist_ok=True)

        async with Database(db_path) as db:
            exporter = CSVExporter(db)

            with open(output_dir / "persons.csv", "w") as f:
                await exporter.export_persons(f)

            with open(output_dir / "events.csv", "w") as f:
                await exporter.export_events(f)

            with open(output_dir / "child_links.csv", "w") as f:
                await exporter.export_child_links(f)

            console.print(f"[green]✓[/green] Exported to: {output_dir}/")
            console.print("  - persons.csv")
            console.print("  - events.csv")
            console.print("  - child_links.csv")

    asyncio.run(_export())


@export_app.command("gedcom")
def export_gedcom(
    db_path: Path = typer.Option(
        settings.database_path,
        "--db",
        "-d",
        help="Path to database file",
    ),
    output: Path = typer.Option(
        Path("genealogy.ged"),
        "--output",
        "-o",
        help="Output file path",
    ),
) -> None:
    """Export data to GEDCOM 5.5.1 format."""
    from ancestral_synth.export.gedcom_exporter import GEDCOMExporter

    async def _export() -> None:
        async with Database(db_path) as db:
            exporter = GEDCOMExporter(db)
            with open(output, "w") as f:
                await exporter.export(f)
            console.print(f"[green]✓[/green] Exported to: {output}")

    asyncio.run(_export())


@export_app.command("f8vision")
def export_f8vision(
    db_path: Path = typer.Option(
        settings.database_path,
        "--db",
        "-d",
        help="Path to database file",
    ),
    output: Path = typer.Option(
        Path("genealogy.json"),
        "--output",
        "-o",
        help="Output file path",
    ),
    title: str = typer.Option(
        None,
        "--title",
        "-t",
        help="Title for the family tree",
    ),
    center: str = typer.Option(
        None,
        "--center",
        "-c",
        help="UUID of person to center the visualization on",
    ),
    description: str = typer.Option(
        None,
        "--description",
        help="Description of the family tree",
    ),
    truncate: int = typer.Option(
        None,
        "--truncate",
        help="Truncate description fields (biography) to this many characters",
    ),
) -> None:
    """Export data to f8vision-web JSON format (ancestral-synth format).

    The f8vision-web format is a JSON file that can be imported into
    f8vision-web for interactive 3D visualization of the family tree.
    Includes persons, events, notes, and relationship links.
    """
    from uuid import UUID

    from ancestral_synth.export.f8vision_exporter import F8VisionExporter

    async def _export() -> None:
        # Parse center UUID if provided
        centered_person_id = None
        if center:
            try:
                centered_person_id = UUID(center)
            except ValueError:
                console.print(f"[red]Invalid UUID: {center}[/red]")
                raise typer.Exit(1)

        async with Database(db_path) as db:
            exporter = F8VisionExporter(db)
            with open(output, "w") as f:
                await exporter.export(
                    f,
                    title=title,
                    centered_person_id=centered_person_id,
                    description=description,
                    truncate_descriptions=truncate,
                )
            console.print(f"[green]✓[/green] Exported to: {output}")

    asyncio.run(_export())


# Query commands
@app.command()
def ancestors(
    person_id: str = typer.Argument(..., help="ID of the person"),
    db_path: Path = typer.Option(
        settings.database_path,
        "--db",
        "-d",
        help="Path to database file",
    ),
    generations: int = typer.Option(
        None,
        "--generations",
        "-g",
        help="Number of generations to traverse (default: all)",
    ),
) -> None:
    """Show ancestors of a person."""
    from uuid import UUID
    from ancestral_synth.services.query_service import QueryService

    async def _ancestors() -> None:
        async with Database(db_path) as db:
            service = QueryService(db)
            try:
                pid = UUID(person_id)
            except ValueError:
                console.print(f"[red]Invalid UUID: {person_id}[/red]")
                return

            ancestor_list = await service.get_ancestors(pid, generations)

            if not ancestor_list:
                console.print("[yellow]No ancestors found.[/yellow]")
                return

            table = Table(title=f"Ancestors ({len(ancestor_list)} found)")
            table.add_column("Name", style="cyan")
            table.add_column("Birth", style="green")
            table.add_column("Death", style="red")
            table.add_column("Gen", justify="center")

            for person in ancestor_list:
                birth = str(person.birth_year) if person.birth_year else "-"
                death = str(person.death_year) if person.death_year else "-"
                table.add_row(
                    person.full_name,
                    birth,
                    death,
                    str(person.generation),
                )

            console.print(table)

    asyncio.run(_ancestors())


@app.command()
def descendants(
    person_id: str = typer.Argument(..., help="ID of the person"),
    db_path: Path = typer.Option(
        settings.database_path,
        "--db",
        "-d",
        help="Path to database file",
    ),
    generations: int = typer.Option(
        None,
        "--generations",
        "-g",
        help="Number of generations to traverse (default: all)",
    ),
) -> None:
    """Show descendants of a person."""
    from uuid import UUID
    from ancestral_synth.services.query_service import QueryService

    async def _descendants() -> None:
        async with Database(db_path) as db:
            service = QueryService(db)
            try:
                pid = UUID(person_id)
            except ValueError:
                console.print(f"[red]Invalid UUID: {person_id}[/red]")
                return

            descendant_list = await service.get_descendants(pid, generations)

            if not descendant_list:
                console.print("[yellow]No descendants found.[/yellow]")
                return

            table = Table(title=f"Descendants ({len(descendant_list)} found)")
            table.add_column("Name", style="cyan")
            table.add_column("Birth", style="green")
            table.add_column("Death", style="red")
            table.add_column("Gen", justify="center")

            for person in descendant_list:
                birth = str(person.birth_year) if person.birth_year else "-"
                death = str(person.death_year) if person.death_year else "-"
                table.add_row(
                    person.full_name,
                    birth,
                    death,
                    str(person.generation),
                )

            console.print(table)

    asyncio.run(_descendants())


@app.command()
def genealogy(
    db_path: Path = typer.Option(
        settings.database_path,
        "--db",
        "-d",
        help="Path to database file",
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file path (prints to stdout if not specified)",
    ),
    pretty: bool = typer.Option(True, "--pretty/--compact", help="Pretty print JSON output"),
    truncate: int | None = typer.Option(
        None,
        "--truncate",
        "-t",
        help="Truncate description fields (biography, event.description, note.content) to this many characters",
    ),
) -> None:
    """Output full genealogy data, optionally truncating long description fields.

    Exports all persons, events, notes, and relationships as JSON.
    Use --truncate to limit the length of biography, event descriptions, and note content.
    """
    import json
    from datetime import date as date_type, datetime
    from typing import Any

    from sqlmodel import select

    from ancestral_synth.persistence.tables import (
        ChildLinkTable,
        EventTable,
        NoteTable,
        PersonTable,
        SpouseLinkTable,
    )

    def truncate_text(text: str | None, max_length: int | None) -> str | None:
        """Truncate text to max_length if specified."""
        if text is None or max_length is None:
            return text
        if len(text) > max_length:
            return text[:max_length] + "..."
        return text

    def json_serializer(obj: Any) -> Any:
        """Custom JSON serializer for special types."""
        if isinstance(obj, (date_type, datetime)):
            return obj.isoformat()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    async def _genealogy() -> None:
        async with Database(db_path) as db:
            async with db.session() as session:
                # Get all persons
                result = await session.exec(select(PersonTable))
                persons = []
                for p in result.all():
                    persons.append({
                        "id": str(p.id),
                        "given_name": p.given_name,
                        "surname": p.surname,
                        "maiden_name": p.maiden_name,
                        "nickname": p.nickname,
                        "gender": p.gender.value if p.gender else None,
                        "birth_date": p.birth_date.isoformat() if p.birth_date else None,
                        "birth_place": p.birth_place,
                        "death_date": p.death_date.isoformat() if p.death_date else None,
                        "death_place": p.death_place,
                        "status": p.status.value if p.status else None,
                        "generation": p.generation,
                        "biography": truncate_text(p.biography, truncate),
                    })

                # Get all events
                result = await session.exec(select(EventTable))
                events = []
                for e in result.all():
                    events.append({
                        "id": str(e.id),
                        "event_type": e.event_type.value if e.event_type else None,
                        "event_date": e.event_date.isoformat() if e.event_date else None,
                        "event_year": e.event_year,
                        "location": e.location,
                        "description": truncate_text(e.description, truncate),
                        "primary_person_id": str(e.primary_person_id),
                    })

                # Get all notes
                result = await session.exec(select(NoteTable))
                notes = []
                for n in result.all():
                    notes.append({
                        "id": str(n.id),
                        "person_id": str(n.person_id),
                        "category": n.category.value if n.category else None,
                        "content": truncate_text(n.content, truncate),
                        "source": n.source,
                    })

                # Get all child links
                result = await session.exec(select(ChildLinkTable))
                child_links = [
                    {"parent_id": str(cl.parent_id), "child_id": str(cl.child_id)}
                    for cl in result.all()
                ]

                # Get all spouse links
                result = await session.exec(select(SpouseLinkTable))
                spouse_links = [
                    {"person1_id": str(sl.person1_id), "person2_id": str(sl.person2_id)}
                    for sl in result.all()
                ]

            data = {
                "metadata": {
                    "exported_at": datetime.utcnow().isoformat(),
                    "version": "1.0",
                    "format": "ancestral-synth-json",
                    "truncated": truncate,
                },
                "persons": persons,
                "events": events,
                "notes": notes,
                "child_links": child_links,
                "spouse_links": spouse_links,
            }

            indent = 2 if pretty else None
            json_output = json.dumps(data, indent=indent, default=json_serializer)

            if output:
                with open(output, "w") as f:
                    f.write(json_output)
                console.print(f"[green]✓[/green] Exported to: {output}")
            else:
                console.print(json_output)

    asyncio.run(_genealogy())


@app.command()
def search(
    db_path: Path = typer.Option(
        settings.database_path,
        "--db",
        "-d",
        help="Path to database file",
    ),
    name: str = typer.Option(
        None,
        "--name",
        "-n",
        help="Search by name (partial match)",
    ),
    born_after: str = typer.Option(
        None,
        "--born-after",
        help="Born after date (YYYY-MM-DD)",
    ),
    born_before: str = typer.Option(
        None,
        "--born-before",
        help="Born before date (YYYY-MM-DD)",
    ),
) -> None:
    """Search for persons by various criteria."""
    from datetime import date as date_type
    from ancestral_synth.services.query_service import QueryService

    async def _search() -> None:
        async with Database(db_path) as db:
            service = QueryService(db)
            results = []

            # Search by name
            if name:
                results = await service.search_by_name(name)

            # Search by date range
            elif born_after or born_before:
                start = date_type.fromisoformat(born_after) if born_after else date_type(1, 1, 1)
                end = date_type.fromisoformat(born_before) if born_before else date_type(9999, 12, 31)
                results = await service.search_by_birth_date_range(start, end)
            else:
                console.print("[yellow]Please specify search criteria (--name or --born-after/--born-before)[/yellow]")
                return

            if not results:
                console.print("[yellow]No results found.[/yellow]")
                return

            table = Table(title=f"Search Results ({len(results)} found)")
            table.add_column("Name", style="cyan")
            table.add_column("Birth", style="green")
            table.add_column("Death", style="red")
            table.add_column("Status", style="yellow")

            for person in results:
                birth = str(person.birth_year) if person.birth_year else "-"
                death = str(person.death_year) if person.death_year else "-"
                table.add_row(
                    person.full_name,
                    birth,
                    death,
                    person.status.value,
                )

            console.print(table)

    asyncio.run(_search())


if __name__ == "__main__":
    app()
