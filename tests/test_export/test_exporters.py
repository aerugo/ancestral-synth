"""Tests for export functionality - TDD: Write tests first."""

import json
from datetime import date
from io import StringIO
from uuid import uuid4

import pytest

from ancestral_synth.domain.enums import EventType, Gender, NoteCategory, PersonStatus
from ancestral_synth.domain.models import ChildLink, Event, Note, Person
from ancestral_synth.persistence.database import Database
from ancestral_synth.persistence.repositories import (
    ChildLinkRepository,
    EventRepository,
    NoteRepository,
    PersonRepository,
)


class TestJSONExporter:
    """Tests for JSON export functionality."""

    @pytest.fixture
    async def populated_db(self, test_db: Database):
        """Create a database with test data."""
        async with test_db.session() as session:
            person_repo = PersonRepository(session)
            event_repo = EventRepository(session)
            note_repo = NoteRepository(session)
            child_link_repo = ChildLinkRepository(session)

            # Create persons
            parent = Person(
                id=uuid4(),
                given_name="John",
                surname="Smith",
                gender=Gender.MALE,
                birth_date=date(1920, 5, 15),
                death_date=date(1990, 3, 20),
                birth_place="Boston, MA",
                status=PersonStatus.COMPLETE,
                biography="John Smith was born in 1920...",
                generation=-1,
            )
            child = Person(
                id=uuid4(),
                given_name="James",
                surname="Smith",
                gender=Gender.MALE,
                birth_date=date(1950, 8, 10),
                birth_place="New York, NY",
                status=PersonStatus.COMPLETE,
                generation=0,
            )

            await person_repo.create(parent)
            await person_repo.create(child)
            await child_link_repo.create(ChildLink(parent_id=parent.id, child_id=child.id))

            # Create events
            birth_event = Event(
                id=uuid4(),
                event_type=EventType.BIRTH,
                event_date=date(1920, 5, 15),
                location="Boston, MA",
                description="Born at Massachusetts General Hospital",
                primary_person_id=parent.id,
            )
            await event_repo.create(birth_event)

            # Create notes
            note = Note(
                id=uuid4(),
                person_id=parent.id,
                category=NoteCategory.CAREER,
                content="Worked as a carpenter",
                source="biography",
            )
            await note_repo.create(note)

        return test_db, parent, child

    @pytest.mark.asyncio
    async def test_export_to_json(self, populated_db) -> None:
        """Should export entire dataset to JSON."""
        from ancestral_synth.export.json_exporter import JSONExporter

        db, parent, child = populated_db

        exporter = JSONExporter(db)
        output = StringIO()
        await exporter.export(output)

        output.seek(0)
        data = json.load(output)

        assert "persons" in data
        assert "events" in data
        assert "notes" in data
        assert "child_links" in data
        assert len(data["persons"]) == 2

    @pytest.mark.asyncio
    async def test_json_includes_person_details(self, populated_db) -> None:
        """Should include all person details."""
        from ancestral_synth.export.json_exporter import JSONExporter

        db, parent, child = populated_db

        exporter = JSONExporter(db)
        output = StringIO()
        await exporter.export(output)

        output.seek(0)
        data = json.load(output)

        persons_by_name = {p["given_name"]: p for p in data["persons"]}
        john = persons_by_name["John"]

        assert john["surname"] == "Smith"
        assert john["gender"] == "male"
        assert john["birth_date"] == "1920-05-15"
        assert john["birth_place"] == "Boston, MA"

    @pytest.mark.asyncio
    async def test_json_includes_relationships(self, populated_db) -> None:
        """Should include child links."""
        from ancestral_synth.export.json_exporter import JSONExporter

        db, parent, child = populated_db

        exporter = JSONExporter(db)
        output = StringIO()
        await exporter.export(output)

        output.seek(0)
        data = json.load(output)

        assert len(data["child_links"]) == 1
        link = data["child_links"][0]
        assert link["parent_id"] == str(parent.id)
        assert link["child_id"] == str(child.id)

    @pytest.mark.asyncio
    async def test_json_includes_metadata(self, populated_db) -> None:
        """Should include export metadata."""
        from ancestral_synth.export.json_exporter import JSONExporter

        db, _, _ = populated_db

        exporter = JSONExporter(db)
        output = StringIO()
        await exporter.export(output)

        output.seek(0)
        data = json.load(output)

        assert "metadata" in data
        assert "exported_at" in data["metadata"]
        assert "version" in data["metadata"]


class TestCSVExporter:
    """Tests for CSV export functionality."""

    @pytest.fixture
    async def populated_db(self, test_db: Database):
        """Create a database with test data."""
        async with test_db.session() as session:
            person_repo = PersonRepository(session)

            p1 = Person(
                given_name="John",
                surname="Smith",
                gender=Gender.MALE,
                birth_date=date(1920, 5, 15),
                status=PersonStatus.COMPLETE,
            )
            p2 = Person(
                given_name="Jane",
                surname="Doe",
                gender=Gender.FEMALE,
                birth_date=date(1925, 9, 20),
                status=PersonStatus.COMPLETE,
            )

            await person_repo.create(p1)
            await person_repo.create(p2)

        return test_db

    @pytest.mark.asyncio
    async def test_export_persons_to_csv(self, populated_db: Database) -> None:
        """Should export persons to CSV."""
        from ancestral_synth.export.csv_exporter import CSVExporter

        exporter = CSVExporter(populated_db)
        output = StringIO()
        await exporter.export_persons(output)

        output.seek(0)
        lines = output.getvalue().strip().split("\n")

        # Header + 2 data rows
        assert len(lines) == 3

        # Check header
        header = lines[0]
        assert "given_name" in header
        assert "surname" in header
        assert "birth_date" in header

    @pytest.mark.asyncio
    async def test_csv_data_format(self, populated_db: Database) -> None:
        """Should format data correctly in CSV."""
        from ancestral_synth.export.csv_exporter import CSVExporter
        import csv

        exporter = CSVExporter(populated_db)
        output = StringIO()
        await exporter.export_persons(output)

        output.seek(0)
        reader = csv.DictReader(output)
        rows = list(reader)

        # Find John
        john = next(r for r in rows if r["given_name"] == "John")
        assert john["surname"] == "Smith"
        assert john["gender"] == "male"


class TestGEDCOMExporter:
    """Tests for GEDCOM export functionality."""

    @pytest.fixture
    async def populated_db(self, test_db: Database):
        """Create a database with test data."""
        async with test_db.session() as session:
            person_repo = PersonRepository(session)
            child_link_repo = ChildLinkRepository(session)

            father = Person(
                id=uuid4(),
                given_name="John",
                surname="Smith",
                gender=Gender.MALE,
                birth_date=date(1920, 5, 15),
                death_date=date(1990, 3, 20),
                birth_place="Boston, MA",
                status=PersonStatus.COMPLETE,
            )
            mother = Person(
                id=uuid4(),
                given_name="Mary",
                surname="Johnson",
                gender=Gender.FEMALE,
                birth_date=date(1925, 9, 20),
                status=PersonStatus.COMPLETE,
            )
            child = Person(
                id=uuid4(),
                given_name="James",
                surname="Smith",
                gender=Gender.MALE,
                birth_date=date(1950, 8, 10),
                status=PersonStatus.COMPLETE,
            )

            await person_repo.create(father)
            await person_repo.create(mother)
            await person_repo.create(child)
            await child_link_repo.create(ChildLink(parent_id=father.id, child_id=child.id))
            await child_link_repo.create(ChildLink(parent_id=mother.id, child_id=child.id))

        return test_db, father, mother, child

    @pytest.mark.asyncio
    async def test_gedcom_header(self, populated_db) -> None:
        """Should include GEDCOM header."""
        from ancestral_synth.export.gedcom_exporter import GEDCOMExporter

        db, _, _, _ = populated_db

        exporter = GEDCOMExporter(db)
        output = StringIO()
        await exporter.export(output)

        output.seek(0)
        content = output.read()

        assert "0 HEAD" in content
        assert "1 GEDC" in content
        assert "2 VERS 5.5.1" in content

    @pytest.mark.asyncio
    async def test_gedcom_individuals(self, populated_db) -> None:
        """Should include individual records."""
        from ancestral_synth.export.gedcom_exporter import GEDCOMExporter

        db, father, _, _ = populated_db

        exporter = GEDCOMExporter(db)
        output = StringIO()
        await exporter.export(output)

        output.seek(0)
        content = output.read()

        # Should have INDI records
        assert "0 @" in content
        assert "@ INDI" in content
        assert "1 NAME John /Smith/" in content

    @pytest.mark.asyncio
    async def test_gedcom_birth_death_dates(self, populated_db) -> None:
        """Should include birth and death dates."""
        from ancestral_synth.export.gedcom_exporter import GEDCOMExporter

        db, _, _, _ = populated_db

        exporter = GEDCOMExporter(db)
        output = StringIO()
        await exporter.export(output)

        output.seek(0)
        content = output.read()

        assert "1 BIRT" in content
        assert "2 DATE" in content

    @pytest.mark.asyncio
    async def test_gedcom_family_records(self, populated_db) -> None:
        """Should include family records linking parents and children."""
        from ancestral_synth.export.gedcom_exporter import GEDCOMExporter

        db, _, _, _ = populated_db

        exporter = GEDCOMExporter(db)
        output = StringIO()
        await exporter.export(output)

        output.seek(0)
        content = output.read()

        # Should have FAM records
        assert "@ FAM" in content

    @pytest.mark.asyncio
    async def test_gedcom_trailer(self, populated_db) -> None:
        """Should end with TRLR."""
        from ancestral_synth.export.gedcom_exporter import GEDCOMExporter

        db, _, _, _ = populated_db

        exporter = GEDCOMExporter(db)
        output = StringIO()
        await exporter.export(output)

        output.seek(0)
        content = output.read()

        assert content.strip().endswith("0 TRLR")
