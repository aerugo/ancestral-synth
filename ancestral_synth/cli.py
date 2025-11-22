"""Command-line interface for Ancestral Synth."""

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


@app.command()
def generate(
    count: int = typer.Option(1, "--count", "-n", help="Number of persons to generate"),
    db_path: Path = typer.Option(
        settings.database_path,
        "--db",
        "-d",
        help="Path to database file",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
) -> None:
    """Generate new persons in the genealogical dataset."""
    configure_logging(verbose)

    async def _generate() -> None:
        async with Database(db_path) as db:
            service = GenealogyService(db)

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task(f"Generating {count} person(s)...", total=count)

                for i in range(count):
                    progress.update(task, description=f"Processing person {i + 1}/{count}...")

                    try:
                        person = await service.process_next()
                        if person:
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

            # Show stats
            stats = await service.get_statistics()
            console.print()
            console.print("[bold]Dataset Statistics:[/bold]")
            console.print(f"  Total persons: {stats['total_persons']}")
            console.print(f"  Complete: {stats['complete']}")
            console.print(f"  Pending: {stats['pending']}")
            console.print(f"  Queue size: {stats['queue_size']}")

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

    async def _stats() -> None:
        async with Database(db_path) as db:
            service = GenealogyService(db)
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


if __name__ == "__main__":
    app()
