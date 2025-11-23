"""Tests for domain models."""

from datetime import date
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from ancestral_synth.domain.enums import (
    EventType,
    Gender,
    NoteCategory,
    PersonStatus,
    RelationshipType,
)
from ancestral_synth.domain.models import (
    Biography,
    ChildLink,
    Event,
    ExtractedData,
    ExtractedEvent,
    Note,
    Person,
    PersonReference,
    PersonSummary,
)


class TestPerson:
    """Tests for Person model."""

    def test_create_minimal_person(self) -> None:
        """Should create a person with minimal required fields."""
        person = Person(given_name="John", surname="Smith")

        assert person.given_name == "John"
        assert person.surname == "Smith"
        assert person.status == PersonStatus.PENDING
        assert person.gender == Gender.UNKNOWN
        assert isinstance(person.id, UUID)

    def test_create_full_person(self) -> None:
        """Should create a person with all fields."""
        person_id = uuid4()
        person = Person(
            id=person_id,
            status=PersonStatus.COMPLETE,
            given_name="John",
            surname="Smith",
            maiden_name=None,
            nickname="Johnny",
            gender=Gender.MALE,
            birth_date=date(1950, 6, 15),
            birth_place="Boston, MA",
            death_date=date(2020, 3, 10),
            death_place="New York, NY",
            biography="A test biography.",
            generation=0,
        )

        assert person.id == person_id
        assert person.nickname == "Johnny"
        assert person.death_place == "New York, NY"

    def test_full_name_property(self) -> None:
        """Should compute full name correctly."""
        person = Person(given_name="John", surname="Smith")
        assert person.full_name == "John Smith"

        person_long = Person(given_name="Mary Elizabeth", surname="Van Der Berg")
        assert person_long.full_name == "Mary Elizabeth Van Der Berg"

    def test_birth_year_property(self) -> None:
        """Should extract birth year from date."""
        person = Person(
            given_name="John",
            surname="Smith",
            birth_date=date(1950, 6, 15),
        )
        assert person.birth_year == 1950

    def test_birth_year_none_when_no_date(self) -> None:
        """Should return None when no birth date."""
        person = Person(given_name="John", surname="Smith")
        assert person.birth_year is None

    def test_death_year_property(self) -> None:
        """Should extract death year from date."""
        person = Person(
            given_name="John",
            surname="Smith",
            death_date=date(2020, 3, 10),
        )
        assert person.death_year == 2020

    def test_death_year_none_when_no_date(self) -> None:
        """Should return None when no death date."""
        person = Person(given_name="John", surname="Smith")
        assert person.death_year is None

    def test_generation_default(self) -> None:
        """Should default generation to 0."""
        person = Person(given_name="John", surname="Smith")
        assert person.generation == 0

    def test_negative_generation_for_ancestors(self) -> None:
        """Should allow negative generation for ancestors."""
        person = Person(given_name="John", surname="Smith", generation=-3)
        assert person.generation == -3

    def test_positive_generation_for_descendants(self) -> None:
        """Should allow positive generation for descendants."""
        person = Person(given_name="John", surname="Smith", generation=2)
        assert person.generation == 2

    def test_serialization(self) -> None:
        """Should serialize to dict correctly."""
        person = Person(
            given_name="John",
            surname="Smith",
            gender=Gender.MALE,
            birth_date=date(1950, 6, 15),
        )
        data = person.model_dump()

        assert data["given_name"] == "John"
        assert data["surname"] == "Smith"
        assert data["gender"] == "male"
        assert data["birth_date"] == date(1950, 6, 15)

    def test_deserialization(self) -> None:
        """Should deserialize from dict correctly."""
        data = {
            "given_name": "John",
            "surname": "Smith",
            "gender": "male",
            "birth_date": "1950-06-15",
        }
        person = Person.model_validate(data)

        assert person.given_name == "John"
        assert person.gender == Gender.MALE
        assert person.birth_date == date(1950, 6, 15)


class TestPersonReference:
    """Tests for PersonReference model."""

    def test_create_minimal_reference(self) -> None:
        """Should create reference with minimal fields."""
        ref = PersonReference(name="John Smith", relationship=RelationshipType.PARENT)

        assert ref.name == "John Smith"
        assert ref.relationship == RelationshipType.PARENT
        assert ref.gender == Gender.UNKNOWN
        assert ref.approximate_birth_year is None

    def test_create_full_reference(self) -> None:
        """Should create reference with all fields."""
        ref = PersonReference(
            name="Mary Johnson",
            relationship=RelationshipType.SPOUSE,
            approximate_birth_year=1952,
            gender=Gender.FEMALE,
            context="Married in 1975 in Boston",
        )

        assert ref.name == "Mary Johnson"
        assert ref.approximate_birth_year == 1952
        assert ref.context == "Married in 1975 in Boston"

    def test_empty_name_allowed(self) -> None:
        """Empty name is technically allowed by model (validated elsewhere)."""
        ref = PersonReference(name="", relationship=RelationshipType.OTHER)
        assert ref.name == ""


class TestPersonSummary:
    """Tests for PersonSummary model."""

    def test_create_summary(self) -> None:
        """Should create summary with required fields."""
        summary = PersonSummary(
            id=uuid4(),
            full_name="John Smith",
            gender=Gender.MALE,
        )

        assert summary.full_name == "John Smith"
        assert summary.gender == Gender.MALE
        assert summary.key_facts == []

    def test_summary_with_facts(self) -> None:
        """Should store key facts."""
        summary = PersonSummary(
            id=uuid4(),
            full_name="John Smith",
            gender=Gender.MALE,
            key_facts=["Born in Boston", "Carpenter by trade"],
        )

        assert len(summary.key_facts) == 2
        assert "Born in Boston" in summary.key_facts

    def test_summary_with_relation_fields(self) -> None:
        """Should store family relation fields."""
        summary = PersonSummary(
            id=uuid4(),
            full_name="John Smith",
            gender=Gender.MALE,
            parents=["George Smith", "Martha Smith"],
            children=["James Smith", "Jane Smith"],
            spouses=["Mary Jones"],
            siblings=["Bob Smith"],
            grandparents=["Old George Smith"],
            grandchildren=["Baby Smith"],
        )

        assert summary.parents == ["George Smith", "Martha Smith"]
        assert summary.children == ["James Smith", "Jane Smith"]
        assert summary.spouses == ["Mary Jones"]
        assert summary.siblings == ["Bob Smith"]
        assert summary.grandparents == ["Old George Smith"]
        assert summary.grandchildren == ["Baby Smith"]

    def test_summary_relation_fields_default_empty(self) -> None:
        """Relation fields should default to empty lists."""
        summary = PersonSummary(
            id=uuid4(),
            full_name="John Smith",
            gender=Gender.MALE,
        )

        assert summary.parents == []
        assert summary.children == []
        assert summary.spouses == []
        assert summary.siblings == []
        assert summary.grandparents == []
        assert summary.grandchildren == []

    def test_summary_with_mentioned_by(self) -> None:
        """Should store who mentioned this person."""
        summary = PersonSummary(
            id=uuid4(),
            full_name="John Smith",
            gender=Gender.MALE,
            mentioned_by="Mary Smith",
        )

        assert summary.mentioned_by == "Mary Smith"

    def test_summary_mentioned_by_default_none(self) -> None:
        """mentioned_by should default to None."""
        summary = PersonSummary(
            id=uuid4(),
            full_name="John Smith",
            gender=Gender.MALE,
        )

        assert summary.mentioned_by is None

    def test_summary_with_generation(self) -> None:
        """Should store generation number."""
        summary = PersonSummary(
            id=uuid4(),
            full_name="John Smith",
            gender=Gender.MALE,
            generation=3,
        )

        assert summary.generation == 3

    def test_summary_generation_default_none(self) -> None:
        """generation should default to None."""
        summary = PersonSummary(
            id=uuid4(),
            full_name="John Smith",
            gender=Gender.MALE,
        )

        assert summary.generation is None

    def test_summary_with_biography_snippets(self) -> None:
        """Should store biography snippets from related people."""
        summary = PersonSummary(
            id=uuid4(),
            full_name="Eleanor Smith",
            gender=Gender.FEMALE,
            biography_snippets=[
                "His wife Eleanor was known for her kindness",
                "His sister Eleanor helped on the farm",
            ],
        )

        assert len(summary.biography_snippets) == 2
        assert "wife Eleanor" in summary.biography_snippets[0]

    def test_summary_biography_snippets_default_empty(self) -> None:
        """biography_snippets should default to empty list."""
        summary = PersonSummary(
            id=uuid4(),
            full_name="John Smith",
            gender=Gender.MALE,
        )

        assert summary.biography_snippets == []


class TestEvent:
    """Tests for Event model."""

    def test_create_event(self) -> None:
        """Should create event with required fields."""
        person_id = uuid4()
        event = Event(
            event_type=EventType.BIRTH,
            description="Born at Massachusetts General Hospital",
            primary_person_id=person_id,
        )

        assert event.event_type == EventType.BIRTH
        assert event.primary_person_id == person_id
        assert event.other_person_ids == []

    def test_event_with_date(self) -> None:
        """Should store event date."""
        event = Event(
            event_type=EventType.MARRIAGE,
            event_date=date(1975, 6, 20),
            description="Wedding ceremony",
            primary_person_id=uuid4(),
        )

        assert event.event_date == date(1975, 6, 20)
        assert event.event_year is None  # Explicit date takes precedence

    def test_event_with_year_only(self) -> None:
        """Should store year when exact date unknown."""
        event = Event(
            event_type=EventType.IMMIGRATION,
            event_year=1920,
            description="Arrived at Ellis Island",
            primary_person_id=uuid4(),
        )

        assert event.event_date is None
        assert event.event_year == 1920

    def test_event_with_participants(self) -> None:
        """Should track other participants."""
        person1 = uuid4()
        person2 = uuid4()
        person3 = uuid4()

        event = Event(
            event_type=EventType.MARRIAGE,
            description="Wedding ceremony",
            primary_person_id=person1,
            other_person_ids=[person2, person3],
        )

        assert len(event.other_person_ids) == 2
        assert person2 in event.other_person_ids


class TestNote:
    """Tests for Note model."""

    def test_create_note(self) -> None:
        """Should create note with required fields."""
        person_id = uuid4()
        note = Note(
            person_id=person_id,
            category=NoteCategory.CAREER,
            content="Worked as a carpenter for 40 years",
        )

        assert note.person_id == person_id
        assert note.category == NoteCategory.CAREER
        assert note.content == "Worked as a carpenter for 40 years"

    def test_note_with_source(self) -> None:
        """Should track source of note."""
        note = Note(
            person_id=uuid4(),
            category=NoteCategory.BIOGRAPHY,
            content="Some interesting fact",
            source="biography_extraction",
        )

        assert note.source == "biography_extraction"

    def test_note_with_references(self) -> None:
        """Should track referenced people."""
        ref1 = uuid4()
        ref2 = uuid4()
        note = Note(
            person_id=uuid4(),
            category=NoteCategory.ANECDOTE,
            content="Story involving family members",
            referenced_person_ids=[ref1, ref2],
        )

        assert len(note.referenced_person_ids) == 2


class TestChildLink:
    """Tests for ChildLink model."""

    def test_create_child_link(self) -> None:
        """Should create parent-child link."""
        parent = uuid4()
        child = uuid4()
        link = ChildLink(parent_id=parent, child_id=child)

        assert link.parent_id == parent
        assert link.child_id == child

    def test_child_link_immutable(self) -> None:
        """ChildLink should be immutable (frozen)."""
        link = ChildLink(parent_id=uuid4(), child_id=uuid4())

        with pytest.raises(ValidationError):
            link.parent_id = uuid4()  # type: ignore[misc]

    def test_child_link_hashable(self) -> None:
        """ChildLink should be hashable for use in sets."""
        parent = uuid4()
        child = uuid4()
        link1 = ChildLink(parent_id=parent, child_id=child)
        link2 = ChildLink(parent_id=parent, child_id=child)

        # Same values should have same hash
        assert hash(link1) == hash(link2)

        # Can be used in sets
        links = {link1, link2}
        assert len(links) == 1


class TestBiography:
    """Tests for Biography model."""

    def test_create_biography(self) -> None:
        """Should create biography."""
        bio = Biography(
            content="A detailed life story...",
            word_count=5,
        )

        assert bio.content == "A detailed life story..."
        assert bio.word_count == 5


class TestExtractedData:
    """Tests for ExtractedData model."""

    def test_create_minimal_extracted_data(self) -> None:
        """Should create with minimal required fields."""
        data = ExtractedData(
            given_name="John",
            surname="Smith",
            gender=Gender.MALE,
        )

        assert data.given_name == "John"
        assert data.surname == "Smith"
        assert data.parents == []
        assert data.children == []

    def test_create_full_extracted_data(self) -> None:
        """Should handle all relationship types."""
        data = ExtractedData(
            given_name="John",
            surname="Smith",
            gender=Gender.MALE,
            birth_date=date(1950, 1, 1),
            parents=[
                PersonReference(name="William Smith", relationship=RelationshipType.PARENT),
            ],
            children=[
                PersonReference(name="James Smith", relationship=RelationshipType.CHILD),
            ],
            spouses=[
                PersonReference(name="Mary Johnson", relationship=RelationshipType.SPOUSE),
            ],
            siblings=[
                PersonReference(name="Robert Smith", relationship=RelationshipType.SIBLING),
            ],
            notes=["Some interesting fact"],
        )

        assert len(data.parents) == 1
        assert len(data.children) == 1
        assert len(data.spouses) == 1
        assert len(data.siblings) == 1
        assert len(data.notes) == 1

    def test_extracted_data_with_events(self) -> None:
        """Should include extracted events."""
        event = ExtractedEvent(
            event_type=EventType.BIRTH,
            description="Born in Boston",
        )

        data = ExtractedData(
            given_name="John",
            surname="Smith",
            gender=Gender.MALE,
            events=[event],
        )

        assert len(data.events) == 1
        assert data.events[0].event_type == EventType.BIRTH

    def test_birth_year_fallback(self) -> None:
        """Should allow birth_year when exact date unknown."""
        data = ExtractedData(
            given_name="John",
            surname="Smith",
            gender=Gender.MALE,
            birth_year=1950,
        )

        assert data.birth_date is None
        assert data.birth_year == 1950
