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


class TestF8VisionExporter:
    """Tests for F8Vision-web JSON export functionality (ancestral-synth format)."""

    @pytest.fixture
    async def populated_db(self, test_db: Database):
        """Create a database with test data including spouse relationships."""
        from ancestral_synth.domain.models import SpouseLink
        from ancestral_synth.persistence.repositories import (
            EventRepository,
            NoteRepository,
            SpouseLinkRepository,
        )

        async with test_db.session() as session:
            person_repo = PersonRepository(session)
            child_link_repo = ChildLinkRepository(session)
            spouse_link_repo = SpouseLinkRepository(session)
            event_repo = EventRepository(session)
            note_repo = NoteRepository(session)

            father = Person(
                id=uuid4(),
                given_name="John",
                surname="Smith",
                gender=Gender.MALE,
                birth_date=date(1920, 5, 15),
                death_date=date(1990, 3, 20),
                birth_place="Boston, MA",
                status=PersonStatus.COMPLETE,
                biography="John Smith was a carpenter who built homes for over 40 years.",
                generation=-1,
            )
            mother = Person(
                id=uuid4(),
                given_name="Mary",
                surname="Smith",
                maiden_name="Johnson",
                gender=Gender.FEMALE,
                birth_date=date(1925, 9, 20),
                status=PersonStatus.COMPLETE,
                biography="Mary was a teacher who loved gardening.",
                generation=-1,
            )
            child = Person(
                id=uuid4(),
                given_name="James",
                surname="Smith",
                gender=Gender.MALE,
                birth_date=date(1950, 8, 10),
                status=PersonStatus.COMPLETE,
                generation=0,
            )

            await person_repo.create(father)
            await person_repo.create(mother)
            await person_repo.create(child)
            await child_link_repo.create(ChildLink(parent_id=father.id, child_id=child.id))
            await child_link_repo.create(ChildLink(parent_id=mother.id, child_id=child.id))
            await spouse_link_repo.create(SpouseLink(person1_id=father.id, person2_id=mother.id))

            # Add an event for father
            birth_event = Event(
                id=uuid4(),
                event_type=EventType.BIRTH,
                event_date=date(1920, 5, 15),
                location="Boston, MA",
                description="Born at Massachusetts General Hospital",
                primary_person_id=father.id,
            )
            await event_repo.create(birth_event)

            # Add a note for father
            note = Note(
                id=uuid4(),
                person_id=father.id,
                category=NoteCategory.CAREER,
                content="Worked as a master carpenter",
                source="biography",
            )
            await note_repo.create(note)

        return test_db, father, mother, child

    @pytest.mark.asyncio
    async def test_export_to_json(self, populated_db) -> None:
        """Should export entire dataset to JSON with ancestral-synth format."""
        from ancestral_synth.export.f8vision_exporter import F8VisionExporter

        db, _, _, _ = populated_db

        exporter = F8VisionExporter(db)
        output = StringIO()
        await exporter.export(output)

        output.seek(0)
        data = json.load(output)

        assert "persons" in data
        assert len(data["persons"]) == 3
        assert "events" in data
        assert "notes" in data
        assert "child_links" in data
        assert "spouse_links" in data

    @pytest.mark.asyncio
    async def test_json_includes_metadata(self, populated_db) -> None:
        """Should include metadata section when provided."""
        from ancestral_synth.export.f8vision_exporter import F8VisionExporter

        db, father, _, _ = populated_db

        exporter = F8VisionExporter(db)
        output = StringIO()
        await exporter.export(
            output,
            title="Smith Family",
            centered_person_id=father.id,
            description="Three generations of Smiths",
        )

        output.seek(0)
        data = json.load(output)

        assert "metadata" in data
        assert data["metadata"]["title"] == "Smith Family"
        assert data["metadata"]["centeredPersonId"] == str(father.id)
        assert data["metadata"]["description"] == "Three generations of Smiths"
        assert data["metadata"]["format"] == "ancestral-synth-json"
        assert data["metadata"]["version"] == "1.0"
        assert "exported_at" in data["metadata"]

    @pytest.mark.asyncio
    async def test_json_person_format(self, populated_db) -> None:
        """Should format persons according to f8vision-web ancestral-synth spec."""
        from ancestral_synth.export.f8vision_exporter import F8VisionExporter

        db, father, _, _ = populated_db

        exporter = F8VisionExporter(db)
        output = StringIO()
        await exporter.export(output)

        output.seek(0)
        data = json.load(output)

        # Find John by name
        john = next(p for p in data["persons"] if "John" in p.get("name", ""))

        assert john["id"] == str(father.id)
        assert john["name"] == "John Smith"
        assert john["given_name"] == "John"
        assert john["surname"] == "Smith"
        assert john["birth_date"] == "1920-05-15"
        assert john["death_date"] == "1990-03-20"
        assert john["birth_place"] == "Boston, MA"
        assert john["gender"] == "male"
        assert john["status"] == "complete"
        assert john["generation"] == -1
        assert "biography" in john

    @pytest.mark.asyncio
    async def test_json_includes_child_links(self, populated_db) -> None:
        """Should include child_links array."""
        from ancestral_synth.export.f8vision_exporter import F8VisionExporter

        db, father, mother, child = populated_db

        exporter = F8VisionExporter(db)
        output = StringIO()
        await exporter.export(output)

        output.seek(0)
        data = json.load(output)

        assert len(data["child_links"]) == 2  # father->child and mother->child
        parent_ids = {link["parent_id"] for link in data["child_links"]}
        assert str(father.id) in parent_ids
        assert str(mother.id) in parent_ids
        for link in data["child_links"]:
            assert link["child_id"] == str(child.id)

    @pytest.mark.asyncio
    async def test_json_includes_spouse_links(self, populated_db) -> None:
        """Should include spouse_links array."""
        from ancestral_synth.export.f8vision_exporter import F8VisionExporter

        db, father, mother, _ = populated_db

        exporter = F8VisionExporter(db)
        output = StringIO()
        await exporter.export(output)

        output.seek(0)
        data = json.load(output)

        assert len(data["spouse_links"]) == 1
        link = data["spouse_links"][0]
        assert {link["person1_id"], link["person2_id"]} == {str(father.id), str(mother.id)}

    @pytest.mark.asyncio
    async def test_json_includes_events(self, populated_db) -> None:
        """Should include events array."""
        from ancestral_synth.export.f8vision_exporter import F8VisionExporter

        db, father, _, _ = populated_db

        exporter = F8VisionExporter(db)
        output = StringIO()
        await exporter.export(output)

        output.seek(0)
        data = json.load(output)

        assert len(data["events"]) >= 1
        birth_event = next(e for e in data["events"] if e["event_type"] == "birth")
        assert birth_event["primary_person_id"] == str(father.id)
        assert birth_event["event_date"] == "1920-05-15"
        assert birth_event["location"] == "Boston, MA"

    @pytest.mark.asyncio
    async def test_json_includes_notes(self, populated_db) -> None:
        """Should include notes array."""
        from ancestral_synth.export.f8vision_exporter import F8VisionExporter

        db, father, _, _ = populated_db

        exporter = F8VisionExporter(db)
        output = StringIO()
        await exporter.export(output)

        output.seek(0)
        data = json.load(output)

        assert len(data["notes"]) >= 1
        career_note = next(n for n in data["notes"] if n["category"] == "career")
        assert career_note["person_id"] == str(father.id)
        assert career_note["content"] == "Worked as a master carpenter"
        assert career_note["source"] == "biography"

    @pytest.mark.asyncio
    async def test_json_includes_maiden_name(self, populated_db) -> None:
        """Should include maiden_name when present."""
        from ancestral_synth.export.f8vision_exporter import F8VisionExporter

        db, _, mother, _ = populated_db

        exporter = F8VisionExporter(db)
        output = StringIO()
        await exporter.export(output)

        output.seek(0)
        data = json.load(output)

        # Find Mary
        mary = next(p for p in data["persons"] if p.get("given_name") == "Mary")
        assert mary["maiden_name"] == "Johnson"

    @pytest.mark.asyncio
    async def test_default_center_selects_most_central_person(self, populated_db) -> None:
        """Should select the most connected person as default center."""
        from ancestral_synth.export.f8vision_exporter import F8VisionExporter

        db, father, mother, child = populated_db

        # Father has: 1 spouse + 1 child = 2 connections
        # Mother has: 1 spouse + 1 child = 2 connections
        # Child has: 2 parents = 2 connections
        # All are equally central, so any could be chosen

        exporter = F8VisionExporter(db)
        output = StringIO()
        await exporter.export(output)  # No centered_person_id specified

        output.seek(0)
        data = json.load(output)

        # The centered person should be one of the three (all equally central)
        centered_id = data["metadata"]["centeredPersonId"]
        valid_ids = {str(father.id), str(mother.id), str(child.id)}
        assert centered_id in valid_ids

    @pytest.mark.asyncio
    async def test_default_center_prefers_more_connected_person(self, test_db: Database) -> None:
        """Should select the person with most connections as default center."""
        from ancestral_synth.domain.models import SpouseLink
        from ancestral_synth.export.f8vision_exporter import F8VisionExporter
        from ancestral_synth.persistence.repositories import SpouseLinkRepository

        # Create a family where the middle-generation person has most connections
        async with test_db.session() as session:
            person_repo = PersonRepository(session)
            child_link_repo = ChildLinkRepository(session)
            spouse_link_repo = SpouseLinkRepository(session)

            # Grandparent: 1 child = 1 connection
            grandparent = Person(
                id=uuid4(),
                given_name="Grand",
                surname="Parent",
                gender=Gender.MALE,
                status=PersonStatus.COMPLETE,
                generation=-2,
            )
            # Parent: 1 parent + 1 spouse + 2 children = 4 connections (most central)
            parent = Person(
                id=uuid4(),
                given_name="Middle",
                surname="Parent",
                gender=Gender.MALE,
                status=PersonStatus.COMPLETE,
                generation=-1,
            )
            # Spouse: 1 spouse + 2 children = 3 connections
            spouse = Person(
                id=uuid4(),
                given_name="Spouse",
                surname="Parent",
                gender=Gender.FEMALE,
                status=PersonStatus.COMPLETE,
                generation=-1,
            )
            # Child 1: 2 parents = 2 connections
            child1 = Person(
                id=uuid4(),
                given_name="Child",
                surname="One",
                gender=Gender.MALE,
                status=PersonStatus.COMPLETE,
                generation=0,
            )
            # Child 2: 2 parents = 2 connections
            child2 = Person(
                id=uuid4(),
                given_name="Child",
                surname="Two",
                gender=Gender.FEMALE,
                status=PersonStatus.COMPLETE,
                generation=0,
            )

            await person_repo.create(grandparent)
            await person_repo.create(parent)
            await person_repo.create(spouse)
            await person_repo.create(child1)
            await person_repo.create(child2)

            # Grandparent -> Parent
            await child_link_repo.create(ChildLink(parent_id=grandparent.id, child_id=parent.id))
            # Parent + Spouse -> Children
            await child_link_repo.create(ChildLink(parent_id=parent.id, child_id=child1.id))
            await child_link_repo.create(ChildLink(parent_id=parent.id, child_id=child2.id))
            await child_link_repo.create(ChildLink(parent_id=spouse.id, child_id=child1.id))
            await child_link_repo.create(ChildLink(parent_id=spouse.id, child_id=child2.id))
            # Spouse link
            await spouse_link_repo.create(SpouseLink(person1_id=parent.id, person2_id=spouse.id))

        exporter = F8VisionExporter(test_db)
        output = StringIO()
        await exporter.export(output)

        output.seek(0)
        data = json.load(output)

        # Parent should be selected as most central (4 connections)
        assert data["metadata"]["centeredPersonId"] == str(parent.id)

    @pytest.mark.asyncio
    async def test_truncate_descriptions(self, populated_db) -> None:
        """Should truncate biography when truncate_descriptions is set."""
        from ancestral_synth.export.f8vision_exporter import F8VisionExporter

        db, father, _, _ = populated_db

        exporter = F8VisionExporter(db)
        output = StringIO()
        await exporter.export(output, truncate_descriptions=20)

        output.seek(0)
        data = json.load(output)

        # Check that truncated is in metadata
        assert data["metadata"]["truncated"] == 20

        # Find John
        john = next(p for p in data["persons"] if p.get("given_name") == "John")
        # Biography should be truncated to 20 chars + "..."
        assert len(john["biography"]) == 23
        assert john["biography"].endswith("...")
