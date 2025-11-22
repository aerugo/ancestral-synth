"""Tests for query CLI commands."""

import asyncio
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

import pytest
from typer.testing import CliRunner

from ancestral_synth.cli import app
from ancestral_synth.domain.enums import Gender, PersonStatus
from ancestral_synth.domain.models import ChildLink, Person
from ancestral_synth.persistence.database import Database
from ancestral_synth.persistence.repositories import ChildLinkRepository, PersonRepository


runner = CliRunner()


def setup_family_tree(db_path: Path) -> tuple:
    """Create a family tree in the database and return IDs."""
    parent_id = uuid4()
    child_id = uuid4()
    grandchild_id = uuid4()

    async def _setup() -> None:
        async with Database(db_path) as db:
            async with db.session() as session:
                repo = PersonRepository(session)
                link_repo = ChildLinkRepository(session)

                parent = Person(
                    id=parent_id,
                    given_name="John",
                    surname="Smith",
                    gender=Gender.MALE,
                    birth_date=date(1930, 1, 1),
                    status=PersonStatus.COMPLETE,
                    generation=-1,
                )
                child = Person(
                    id=child_id,
                    given_name="James",
                    surname="Smith",
                    gender=Gender.MALE,
                    birth_date=date(1960, 1, 1),
                    status=PersonStatus.COMPLETE,
                    generation=0,
                )
                grandchild = Person(
                    id=grandchild_id,
                    given_name="Michael",
                    surname="Smith",
                    gender=Gender.MALE,
                    birth_date=date(1990, 1, 1),
                    status=PersonStatus.COMPLETE,
                    generation=1,
                )

                await repo.create(parent)
                await repo.create(child)
                await repo.create(grandchild)

                await link_repo.create(ChildLink(parent_id=parent_id, child_id=child_id))
                await link_repo.create(ChildLink(parent_id=child_id, child_id=grandchild_id))

    asyncio.run(_setup())
    return parent_id, child_id, grandchild_id


class TestAncestorsCommand:
    """Tests for the ancestors command."""

    def test_ancestors_shows_parents(self) -> None:
        """Should show ancestors of a person."""
        with TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            parent_id, child_id, _ = setup_family_tree(db_path)

            result = runner.invoke(app, [
                "ancestors",
                str(child_id),
                "--db", str(db_path),
            ])

            assert result.exit_code == 0
            assert "John" in result.stdout

    def test_ancestors_limits_generations(self) -> None:
        """Should limit to specified generations."""
        with TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            parent_id, child_id, grandchild_id = setup_family_tree(db_path)

            result = runner.invoke(app, [
                "ancestors",
                str(grandchild_id),
                "--generations", "1",
                "--db", str(db_path),
            ])

            assert result.exit_code == 0
            assert "James" in result.stdout  # Direct parent
            # John (grandparent) should not be shown with only 1 generation

    def test_ancestors_no_ancestors(self) -> None:
        """Should handle person with no ancestors."""
        with TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            parent_id, _, _ = setup_family_tree(db_path)

            result = runner.invoke(app, [
                "ancestors",
                str(parent_id),
                "--db", str(db_path),
            ])

            assert result.exit_code == 0
            assert "no ancestors" in result.stdout.lower()


class TestDescendantsCommand:
    """Tests for the descendants command."""

    def test_descendants_shows_children(self) -> None:
        """Should show descendants of a person."""
        with TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            parent_id, child_id, grandchild_id = setup_family_tree(db_path)

            result = runner.invoke(app, [
                "descendants",
                str(parent_id),
                "--db", str(db_path),
            ])

            assert result.exit_code == 0
            assert "James" in result.stdout

    def test_descendants_limits_generations(self) -> None:
        """Should limit to specified generations."""
        with TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            parent_id, _, _ = setup_family_tree(db_path)

            result = runner.invoke(app, [
                "descendants",
                str(parent_id),
                "--generations", "1",
                "--db", str(db_path),
            ])

            assert result.exit_code == 0
            assert "James" in result.stdout

    def test_descendants_no_descendants(self) -> None:
        """Should handle person with no descendants."""
        with TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            _, _, grandchild_id = setup_family_tree(db_path)

            result = runner.invoke(app, [
                "descendants",
                str(grandchild_id),
                "--db", str(db_path),
            ])

            assert result.exit_code == 0
            assert "no descendants" in result.stdout.lower()


class TestSearchCommand:
    """Tests for the search command."""

    def test_search_by_name(self) -> None:
        """Should find persons by name."""
        with TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            setup_family_tree(db_path)

            result = runner.invoke(app, [
                "search",
                "--name", "Smith",
                "--db", str(db_path),
            ])

            assert result.exit_code == 0
            assert "John" in result.stdout
            assert "James" in result.stdout
            assert "Michael" in result.stdout

    def test_search_by_birth_range(self) -> None:
        """Should find persons by birth date range."""
        with TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            setup_family_tree(db_path)

            result = runner.invoke(app, [
                "search",
                "--born-after", "1950-01-01",
                "--born-before", "2000-01-01",
                "--db", str(db_path),
            ])

            assert result.exit_code == 0
            assert "James" in result.stdout  # Born 1960
            assert "Michael" in result.stdout  # Born 1990

    def test_search_no_results(self) -> None:
        """Should handle no search results."""
        with TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            setup_family_tree(db_path)

            result = runner.invoke(app, [
                "search",
                "--name", "NonExistent",
                "--db", str(db_path),
            ])

            assert result.exit_code == 0
            assert "no results" in result.stdout.lower()
