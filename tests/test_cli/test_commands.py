"""Tests for CLI commands."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, patch

import pytest
from typer.testing import CliRunner

from ancestral_synth.cli import app


runner = CliRunner()


class TestInitCommand:
    """Tests for the init command."""

    def test_init_creates_database(self) -> None:
        """Should create database file."""
        with TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            result = runner.invoke(app, ["init", "--db", str(db_path)])

            assert result.exit_code == 0
            assert db_path.exists()
            assert "initialized" in result.stdout.lower()

    def test_init_creates_nested_dirs(self) -> None:
        """Should create parent directories."""
        with TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "nested" / "dir" / "test.db"
            result = runner.invoke(app, ["init", "--db", str(db_path)])

            assert result.exit_code == 0
            assert db_path.exists()


class TestConfigCommand:
    """Tests for the config command."""

    def test_config_shows_settings(self) -> None:
        """Should display current configuration."""
        result = runner.invoke(app, ["config"])

        assert result.exit_code == 0
        assert "Database Path" in result.stdout
        assert "LLM Provider" in result.stdout
        assert "LLM Model" in result.stdout


class TestStatsCommand:
    """Tests for the stats command."""

    def test_stats_empty_database(self) -> None:
        """Should show stats for empty database."""
        with TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            # Initialize first
            runner.invoke(app, ["init", "--db", str(db_path)])

            result = runner.invoke(app, ["stats", "--db", str(db_path)])

            assert result.exit_code == 0
            assert "0" in result.stdout  # Should show 0 persons

    def test_stats_with_data(self) -> None:
        """Should show correct stats with data."""
        import asyncio
        from ancestral_synth.persistence.database import Database
        from ancestral_synth.persistence.repositories import PersonRepository
        from ancestral_synth.domain.models import Person
        from ancestral_synth.domain.enums import PersonStatus

        with TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"

            # Create some test data
            async def setup_data() -> None:
                async with Database(db_path) as db:
                    async with db.session() as session:
                        repo = PersonRepository(session)
                        await repo.create(Person(
                            given_name="John",
                            surname="Smith",
                            status=PersonStatus.COMPLETE,
                        ))
                        await repo.create(Person(
                            given_name="Jane",
                            surname="Doe",
                            status=PersonStatus.PENDING,
                        ))

            asyncio.run(setup_data())

            result = runner.invoke(app, ["stats", "--db", str(db_path)])

            assert result.exit_code == 0
            assert "2" in result.stdout  # Total persons


class TestListPersonsCommand:
    """Tests for the list-persons command."""

    def test_list_empty_database(self) -> None:
        """Should handle empty database."""
        with TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            runner.invoke(app, ["init", "--db", str(db_path)])

            result = runner.invoke(app, ["list-persons", "--db", str(db_path)])

            assert result.exit_code == 0
            assert "no persons" in result.stdout.lower()

    def test_list_with_data(self) -> None:
        """Should list persons."""
        import asyncio
        from ancestral_synth.persistence.database import Database
        from ancestral_synth.persistence.repositories import PersonRepository
        from ancestral_synth.domain.models import Person

        with TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"

            async def setup_data() -> None:
                async with Database(db_path) as db:
                    async with db.session() as session:
                        repo = PersonRepository(session)
                        await repo.create(Person(given_name="John", surname="Smith"))

            asyncio.run(setup_data())

            result = runner.invoke(app, ["list-persons", "--db", str(db_path)])

            assert result.exit_code == 0
            assert "John" in result.stdout
            assert "Smith" in result.stdout

    def test_list_with_status_filter(self) -> None:
        """Should filter by status."""
        import asyncio
        from ancestral_synth.persistence.database import Database
        from ancestral_synth.persistence.repositories import PersonRepository
        from ancestral_synth.domain.models import Person
        from ancestral_synth.domain.enums import PersonStatus

        with TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"

            async def setup_data() -> None:
                async with Database(db_path) as db:
                    async with db.session() as session:
                        repo = PersonRepository(session)
                        await repo.create(Person(
                            given_name="Complete",
                            surname="Person",
                            status=PersonStatus.COMPLETE,
                        ))
                        await repo.create(Person(
                            given_name="Pending",
                            surname="Person",
                            status=PersonStatus.PENDING,
                        ))

            asyncio.run(setup_data())

            result = runner.invoke(app, [
                "list-persons",
                "--db", str(db_path),
                "--status", "complete",
            ])

            assert result.exit_code == 0
            assert "Complete" in result.stdout
            assert "Pending" not in result.stdout

    def test_list_invalid_status(self) -> None:
        """Should handle invalid status."""
        with TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            runner.invoke(app, ["init", "--db", str(db_path)])

            result = runner.invoke(app, [
                "list-persons",
                "--db", str(db_path),
                "--status", "invalid",
            ])

            assert "invalid status" in result.stdout.lower()


class TestShowCommand:
    """Tests for the show command."""

    def test_show_person_not_found(self) -> None:
        """Should handle person not found."""
        with TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            runner.invoke(app, ["init", "--db", str(db_path)])

            result = runner.invoke(app, ["show", "NonExistent", "--db", str(db_path)])

            assert "no person found" in result.stdout.lower()

    def test_show_person_found(self) -> None:
        """Should display person details."""
        import asyncio
        from ancestral_synth.persistence.database import Database
        from ancestral_synth.persistence.repositories import PersonRepository
        from ancestral_synth.domain.models import Person
        from datetime import date

        with TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"

            async def setup_data() -> None:
                async with Database(db_path) as db:
                    async with db.session() as session:
                        repo = PersonRepository(session)
                        await repo.create(Person(
                            given_name="John",
                            surname="Smith",
                            birth_date=date(1950, 6, 15),
                            biography="A test biography for John.",
                        ))

            asyncio.run(setup_data())

            result = runner.invoke(app, ["show", "John", "--db", str(db_path)])

            assert result.exit_code == 0
            assert "John Smith" in result.stdout
            assert "1950" in result.stdout


class TestGenerateCommand:
    """Tests for the generate command."""

    def test_generate_with_mock_agents(self) -> None:
        """Should generate persons with mocked agents."""
        from ancestral_synth.domain.models import Biography, ExtractedData, Person
        from ancestral_synth.domain.enums import Gender, PersonStatus

        with TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"

            # Mock the GenealogyService
            mock_person = Person(
                given_name="Generated",
                surname="Person",
                status=PersonStatus.COMPLETE,
            )

            with patch("ancestral_synth.cli.GenealogyService") as MockService:
                mock_service = AsyncMock()
                mock_service.process_next.return_value = mock_person
                mock_service.get_statistics.return_value = {
                    "total_persons": 1,
                    "complete": 1,
                    "pending": 0,
                    "queued": 0,
                    "queue_size": 0,
                }
                MockService.return_value = mock_service

                result = runner.invoke(app, [
                    "generate",
                    "-n", "1",
                    "--db", str(db_path),
                ])

                assert result.exit_code == 0
                assert "Generated Person" in result.stdout or "Created" in result.stdout

    def test_generate_verbose(self) -> None:
        """Should show verbose output."""
        from ancestral_synth.domain.models import Person
        from ancestral_synth.domain.enums import PersonStatus

        with TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"

            mock_person = Person(
                given_name="Test",
                surname="Person",
                status=PersonStatus.COMPLETE,
            )

            with patch("ancestral_synth.cli.GenealogyService") as MockService:
                mock_service = AsyncMock()
                mock_service.process_next.return_value = mock_person
                mock_service.get_statistics.return_value = {
                    "total_persons": 1,
                    "complete": 1,
                    "pending": 0,
                    "queued": 0,
                    "queue_size": 0,
                }
                # Add mock for cost_tracker (non-async MagicMock)
                from unittest.mock import MagicMock
                mock_cost_tracker = MagicMock()
                mock_cost_tracker.get_summary.return_value = {
                    "total_cost": 0.005,
                    "total_persons": 1,
                    "total_llm_calls": 2,
                    "total_input_tokens": 1000,
                    "total_output_tokens": 500,
                    "provider": "openai",
                    "model": "gpt-4o-mini",
                }
                mock_service.cost_tracker = mock_cost_tracker
                MockService.return_value = mock_service

                result = runner.invoke(app, [
                    "generate",
                    "-n", "1",
                    "--db", str(db_path),
                    "-v",
                ])

                assert result.exit_code == 0
