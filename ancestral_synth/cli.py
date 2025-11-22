"""Command-line interface for Ancestral Synth."""

from dotenv import load_dotenv

load_dotenv()  # Load .env file before any other imports that need API keys

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
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output with timing"),
) -> None:
    """Generate new persons in the genealogical dataset."""
    import time

    configure_logging(verbose)

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
