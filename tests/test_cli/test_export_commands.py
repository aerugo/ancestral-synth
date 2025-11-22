"""Tests for export CLI commands."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import asyncio

import pytest
from typer.testing import CliRunner

from ancestral_synth.cli import app
from ancestral_synth.domain.enums import Gender, PersonStatus
from ancestral_synth.domain.models import Person
from ancestral_synth.persistence.database import Database
from ancestral_synth.persistence.repositories import PersonRepository


runner = CliRunner()


def setup_test_data(db_path: Path) -> None:
    """Create test data in the database."""
    async def _setup() -> None:
        async with Database(db_path) as db:
            async with db.session() as session:
                repo = PersonRepository(session)
                await repo.create(Person(
                    given_name="John",
                    surname="Smith",
                    gender=Gender.MALE,
                    status=PersonStatus.COMPLETE,
                ))
                await repo.create(Person(
                    given_name="Jane",
                    surname="Doe",
                    gender=Gender.FEMALE,
                    status=PersonStatus.COMPLETE,
                ))

    asyncio.run(_setup())


class TestExportJSONCommand:
    """Tests for export json command."""

    def test_export_json_creates_file(self) -> None:
        """Should create JSON export file."""
        with TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            output_path = Path(tmpdir) / "export.json"

            setup_test_data(db_path)

            result = runner.invoke(app, [
                "export", "json",
                "--db", str(db_path),
                "--output", str(output_path),
            ])

            assert result.exit_code == 0
            assert output_path.exists()

    def test_export_json_valid_content(self) -> None:
        """Should create valid JSON content."""
        with TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            output_path = Path(tmpdir) / "export.json"

            setup_test_data(db_path)

            runner.invoke(app, [
                "export", "json",
                "--db", str(db_path),
                "--output", str(output_path),
            ])

            with open(output_path) as f:
                data = json.load(f)

            assert "persons" in data
            assert "metadata" in data
            assert len(data["persons"]) == 2

    def test_export_json_default_output(self) -> None:
        """Should use default output filename."""
        with TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"

            setup_test_data(db_path)

            # Change to tmpdir so default file is created there
            import os
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                result = runner.invoke(app, [
                    "export", "json",
                    "--db", str(db_path),
                ])

                assert result.exit_code == 0
                # Should create genealogy.json in current dir
                assert (Path(tmpdir) / "genealogy.json").exists()
            finally:
                os.chdir(old_cwd)


class TestExportCSVCommand:
    """Tests for export csv command."""

    def test_export_csv_creates_files(self) -> None:
        """Should create CSV export files."""
        with TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            output_dir = Path(tmpdir) / "csv_export"

            setup_test_data(db_path)

            result = runner.invoke(app, [
                "export", "csv",
                "--db", str(db_path),
                "--output-dir", str(output_dir),
            ])

            assert result.exit_code == 0
            assert output_dir.exists()
            assert (output_dir / "persons.csv").exists()

    def test_export_csv_valid_content(self) -> None:
        """Should create valid CSV content."""
        with TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            output_dir = Path(tmpdir) / "csv_export"

            setup_test_data(db_path)

            runner.invoke(app, [
                "export", "csv",
                "--db", str(db_path),
                "--output-dir", str(output_dir),
            ])

            import csv
            with open(output_dir / "persons.csv") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            assert len(rows) == 2
            assert any(r["given_name"] == "John" for r in rows)


class TestExportGEDCOMCommand:
    """Tests for export gedcom command."""

    def test_export_gedcom_creates_file(self) -> None:
        """Should create GEDCOM export file."""
        with TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            output_path = Path(tmpdir) / "export.ged"

            setup_test_data(db_path)

            result = runner.invoke(app, [
                "export", "gedcom",
                "--db", str(db_path),
                "--output", str(output_path),
            ])

            assert result.exit_code == 0
            assert output_path.exists()

    def test_export_gedcom_valid_content(self) -> None:
        """Should create valid GEDCOM content."""
        with TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            output_path = Path(tmpdir) / "export.ged"

            setup_test_data(db_path)

            runner.invoke(app, [
                "export", "gedcom",
                "--db", str(db_path),
                "--output", str(output_path),
            ])

            with open(output_path) as f:
                content = f.read()

            assert "0 HEAD" in content
            assert "0 TRLR" in content
            assert "INDI" in content
