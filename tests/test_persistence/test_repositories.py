"""Tests for repository classes."""

from datetime import date
from uuid import uuid4

import pytest

from ancestral_synth.domain.enums import Gender, PersonStatus
from ancestral_synth.domain.models import ChildLink, Event, Note, Person
from ancestral_synth.domain.enums import EventType, NoteCategory
from ancestral_synth.persistence.database import Database
from ancestral_synth.persistence.repositories import (
    ChildLinkRepository,
    EventRepository,
    NoteRepository,
    PersonRepository,
    QueueRepository,
)


class TestPersonRepository:
    """Tests for PersonRepository."""

    @pytest.mark.asyncio
    async def test_create_person(self, test_db: Database) -> None:
        """Should create a person record."""
        async with test_db.session() as session:
            repo = PersonRepository(session)
            person = Person(
                given_name="John",
                surname="Smith",
                gender=Gender.MALE,
            )
            db_person = await repo.create(person)

            assert db_person.id == person.id
            assert db_person.given_name == "John"
            assert db_person.surname == "Smith"

    @pytest.mark.asyncio
    async def test_get_by_id(self, test_db: Database) -> None:
        """Should retrieve person by ID."""
        async with test_db.session() as session:
            repo = PersonRepository(session)
            person = Person(given_name="John", surname="Smith")
            await repo.create(person)

        async with test_db.session() as session:
            repo = PersonRepository(session)
            retrieved = await repo.get_by_id(person.id)

            assert retrieved is not None
            assert retrieved.given_name == "John"

    @pytest.mark.asyncio
    async def test_get_by_id_not_found(self, test_db: Database) -> None:
        """Should return None for non-existent ID."""
        async with test_db.session() as session:
            repo = PersonRepository(session)
            result = await repo.get_by_id(uuid4())

            assert result is None

    @pytest.mark.asyncio
    async def test_get_by_name(self, test_db: Database) -> None:
        """Should find people by exact name."""
        async with test_db.session() as session:
            repo = PersonRepository(session)
            await repo.create(Person(given_name="John", surname="Smith"))
            await repo.create(Person(given_name="John", surname="Doe"))
            await repo.create(Person(given_name="Jane", surname="Smith"))

        async with test_db.session() as session:
            repo = PersonRepository(session)
            results = await repo.get_by_name("John", "Smith")

            assert len(results) == 1
            assert results[0].given_name == "John"
            assert results[0].surname == "Smith"

    @pytest.mark.asyncio
    async def test_get_by_status(self, test_db: Database) -> None:
        """Should filter by status."""
        async with test_db.session() as session:
            repo = PersonRepository(session)
            await repo.create(Person(
                given_name="John",
                surname="Smith",
                status=PersonStatus.PENDING,
            ))
            await repo.create(Person(
                given_name="Jane",
                surname="Doe",
                status=PersonStatus.COMPLETE,
            ))

        async with test_db.session() as session:
            repo = PersonRepository(session)
            pending = await repo.get_by_status(PersonStatus.PENDING)
            complete = await repo.get_by_status(PersonStatus.COMPLETE)

            assert len(pending) == 1
            assert pending[0].given_name == "John"
            assert len(complete) == 1
            assert complete[0].given_name == "Jane"

    @pytest.mark.asyncio
    async def test_update_person(self, test_db: Database) -> None:
        """Should update person fields."""
        async with test_db.session() as session:
            repo = PersonRepository(session)
            person = Person(given_name="John", surname="Smith")
            await repo.create(person)

        async with test_db.session() as session:
            repo = PersonRepository(session)
            updated = await repo.update(
                person.id,
                status=PersonStatus.COMPLETE,
                biography="A test biography",
            )

            assert updated is not None
            assert updated.status == PersonStatus.COMPLETE
            assert updated.biography == "A test biography"

    @pytest.mark.asyncio
    async def test_update_non_existent(self, test_db: Database) -> None:
        """Should return None when updating non-existent person."""
        async with test_db.session() as session:
            repo = PersonRepository(session)
            result = await repo.update(uuid4(), biography="test")

            assert result is None

    @pytest.mark.asyncio
    async def test_count_all(self, test_db: Database) -> None:
        """Should count all persons."""
        async with test_db.session() as session:
            repo = PersonRepository(session)
            await repo.create(Person(given_name="John", surname="Smith"))
            await repo.create(Person(given_name="Jane", surname="Doe"))

        async with test_db.session() as session:
            repo = PersonRepository(session)
            count = await repo.count()

            assert count == 2

    @pytest.mark.asyncio
    async def test_count_by_status(self, test_db: Database) -> None:
        """Should count by status."""
        async with test_db.session() as session:
            repo = PersonRepository(session)
            await repo.create(Person(
                given_name="John",
                surname="Smith",
                status=PersonStatus.PENDING,
            ))
            await repo.create(Person(
                given_name="Jane",
                surname="Doe",
                status=PersonStatus.PENDING,
            ))
            await repo.create(Person(
                given_name="Bob",
                surname="Wilson",
                status=PersonStatus.COMPLETE,
            ))

        async with test_db.session() as session:
            repo = PersonRepository(session)
            pending_count = await repo.count(PersonStatus.PENDING)
            complete_count = await repo.count(PersonStatus.COMPLETE)

            assert pending_count == 2
            assert complete_count == 1

    @pytest.mark.asyncio
    async def test_search_similar(self, test_db: Database) -> None:
        """Should find similar names."""
        async with test_db.session() as session:
            repo = PersonRepository(session)
            await repo.create(Person(
                given_name="John",
                surname="Smith",
                birth_date=date(1950, 1, 1),
            ))
            await repo.create(Person(
                given_name="Johnny",
                surname="Smith",
                birth_date=date(1952, 1, 1),
            ))
            await repo.create(Person(
                given_name="John",
                surname="Doe",
                birth_date=date(1950, 1, 1),
            ))

        async with test_db.session() as session:
            repo = PersonRepository(session)
            # Should find partial matches with same surname
            results = await repo.search_similar("John", "Smith")

            assert len(results) >= 1
            # All results should have surname "Smith"
            for r in results:
                assert r.surname == "Smith"

    @pytest.mark.asyncio
    async def test_to_domain_conversion(self, test_db: Database) -> None:
        """Should convert table to domain model."""
        async with test_db.session() as session:
            repo = PersonRepository(session)
            person = Person(
                given_name="John",
                surname="Smith",
                gender=Gender.MALE,
                birth_date=date(1950, 6, 15),
            )
            db_person = await repo.create(person)
            domain_person = repo.to_domain(db_person)

            assert isinstance(domain_person, Person)
            assert domain_person.given_name == "John"
            assert domain_person.gender == Gender.MALE

    @pytest.mark.asyncio
    async def test_to_summary_conversion(self, test_db: Database) -> None:
        """Should convert table to summary."""
        async with test_db.session() as session:
            repo = PersonRepository(session)
            person = Person(
                given_name="John",
                surname="Smith",
                gender=Gender.MALE,
                birth_date=date(1950, 6, 15),
            )
            db_person = await repo.create(person)
            summary = repo.to_summary(db_person)

            assert summary.full_name == "John Smith"
            assert summary.birth_year == 1950


class TestEventRepository:
    """Tests for EventRepository."""

    @pytest.mark.asyncio
    async def test_create_event(self, test_db: Database) -> None:
        """Should create an event record."""
        async with test_db.session() as session:
            # First create a person
            person_repo = PersonRepository(session)
            person = Person(given_name="John", surname="Smith")
            await person_repo.create(person)

            # Then create event
            event_repo = EventRepository(session)
            event = Event(
                event_type=EventType.BIRTH,
                event_date=date(1950, 6, 15),
                location="Boston, MA",
                description="Born at MGH",
                primary_person_id=person.id,
            )
            db_event = await event_repo.create(event)

            assert db_event.id == event.id
            assert db_event.event_type == EventType.BIRTH

    @pytest.mark.asyncio
    async def test_get_events_by_person(self, test_db: Database) -> None:
        """Should retrieve events for a person."""
        async with test_db.session() as session:
            person_repo = PersonRepository(session)
            person = Person(given_name="John", surname="Smith")
            await person_repo.create(person)

            event_repo = EventRepository(session)
            await event_repo.create(Event(
                event_type=EventType.BIRTH,
                description="Birth",
                primary_person_id=person.id,
            ))
            await event_repo.create(Event(
                event_type=EventType.MARRIAGE,
                description="Marriage",
                primary_person_id=person.id,
            ))

        async with test_db.session() as session:
            event_repo = EventRepository(session)
            events = await event_repo.get_by_person(person.id)

            assert len(events) == 2

    @pytest.mark.asyncio
    async def test_event_with_participants(self, test_db: Database) -> None:
        """Should track event participants."""
        async with test_db.session() as session:
            person_repo = PersonRepository(session)
            person1 = Person(given_name="John", surname="Smith")
            person2 = Person(given_name="Jane", surname="Doe")
            await person_repo.create(person1)
            await person_repo.create(person2)

            event_repo = EventRepository(session)
            event = Event(
                event_type=EventType.MARRIAGE,
                description="Wedding",
                primary_person_id=person1.id,
                other_person_ids=[person2.id],
            )
            await event_repo.create(event)

        # Verify participants were stored
        async with test_db.session() as session:
            from sqlmodel import select
            from ancestral_synth.persistence.tables import EventParticipantTable

            stmt = select(EventParticipantTable).where(
                EventParticipantTable.event_id == event.id
            )
            result = await session.exec(stmt)
            participants = list(result.all())

            assert len(participants) == 1
            assert participants[0].person_id == person2.id


class TestNoteRepository:
    """Tests for NoteRepository."""

    @pytest.mark.asyncio
    async def test_create_note(self, test_db: Database) -> None:
        """Should create a note record."""
        async with test_db.session() as session:
            person_repo = PersonRepository(session)
            person = Person(given_name="John", surname="Smith")
            await person_repo.create(person)

            note_repo = NoteRepository(session)
            note = Note(
                person_id=person.id,
                category=NoteCategory.CAREER,
                content="Worked as a carpenter",
            )
            db_note = await note_repo.create(note)

            assert db_note.id == note.id
            assert db_note.content == "Worked as a carpenter"

    @pytest.mark.asyncio
    async def test_get_notes_by_person(self, test_db: Database) -> None:
        """Should retrieve notes for a person."""
        async with test_db.session() as session:
            person_repo = PersonRepository(session)
            person = Person(given_name="John", surname="Smith")
            await person_repo.create(person)

            note_repo = NoteRepository(session)
            await note_repo.create(Note(
                person_id=person.id,
                category=NoteCategory.CAREER,
                content="Note 1",
            ))
            await note_repo.create(Note(
                person_id=person.id,
                category=NoteCategory.EDUCATION,
                content="Note 2",
            ))

        async with test_db.session() as session:
            note_repo = NoteRepository(session)
            notes = await note_repo.get_by_person(person.id)

            assert len(notes) == 2


class TestChildLinkRepository:
    """Tests for ChildLinkRepository."""

    @pytest.mark.asyncio
    async def test_create_child_link(self, test_db: Database) -> None:
        """Should create a parent-child link."""
        async with test_db.session() as session:
            person_repo = PersonRepository(session)
            parent = Person(given_name="John", surname="Smith")
            child = Person(given_name="James", surname="Smith")
            await person_repo.create(parent)
            await person_repo.create(child)

            link_repo = ChildLinkRepository(session)
            link = ChildLink(parent_id=parent.id, child_id=child.id)
            await link_repo.create(link)

        async with test_db.session() as session:
            link_repo = ChildLinkRepository(session)
            exists = await link_repo.exists(parent.id, child.id)

            assert exists is True

    @pytest.mark.asyncio
    async def test_exists_returns_false(self, test_db: Database) -> None:
        """Should return False for non-existent link."""
        async with test_db.session() as session:
            link_repo = ChildLinkRepository(session)
            exists = await link_repo.exists(uuid4(), uuid4())

            assert exists is False

    @pytest.mark.asyncio
    async def test_get_children(self, test_db: Database) -> None:
        """Should retrieve children of a parent."""
        async with test_db.session() as session:
            person_repo = PersonRepository(session)
            parent = Person(given_name="John", surname="Smith")
            child1 = Person(given_name="James", surname="Smith")
            child2 = Person(given_name="Jane", surname="Smith")
            await person_repo.create(parent)
            await person_repo.create(child1)
            await person_repo.create(child2)

            link_repo = ChildLinkRepository(session)
            await link_repo.create(ChildLink(parent_id=parent.id, child_id=child1.id))
            await link_repo.create(ChildLink(parent_id=parent.id, child_id=child2.id))

        async with test_db.session() as session:
            link_repo = ChildLinkRepository(session)
            children = await link_repo.get_children(parent.id)

            assert len(children) == 2
            assert child1.id in children
            assert child2.id in children

    @pytest.mark.asyncio
    async def test_get_parents(self, test_db: Database) -> None:
        """Should retrieve parents of a child."""
        async with test_db.session() as session:
            person_repo = PersonRepository(session)
            father = Person(given_name="John", surname="Smith")
            mother = Person(given_name="Mary", surname="Smith")
            child = Person(given_name="James", surname="Smith")
            await person_repo.create(father)
            await person_repo.create(mother)
            await person_repo.create(child)

            link_repo = ChildLinkRepository(session)
            await link_repo.create(ChildLink(parent_id=father.id, child_id=child.id))
            await link_repo.create(ChildLink(parent_id=mother.id, child_id=child.id))

        async with test_db.session() as session:
            link_repo = ChildLinkRepository(session)
            parents = await link_repo.get_parents(child.id)

            assert len(parents) == 2
            assert father.id in parents
            assert mother.id in parents


class TestQueueRepository:
    """Tests for QueueRepository."""

    @pytest.mark.asyncio
    async def test_enqueue(self, test_db: Database) -> None:
        """Should add person to queue."""
        async with test_db.session() as session:
            person_repo = PersonRepository(session)
            person = Person(given_name="John", surname="Smith")
            await person_repo.create(person)

            queue_repo = QueueRepository(session)
            await queue_repo.enqueue(person.id)

        async with test_db.session() as session:
            queue_repo = QueueRepository(session)
            count = await queue_repo.count()

            assert count == 1

    @pytest.mark.asyncio
    async def test_enqueue_idempotent(self, test_db: Database) -> None:
        """Should not duplicate queue entries."""
        async with test_db.session() as session:
            person_repo = PersonRepository(session)
            person = Person(given_name="John", surname="Smith")
            await person_repo.create(person)

            queue_repo = QueueRepository(session)
            await queue_repo.enqueue(person.id)
            await queue_repo.enqueue(person.id)  # Duplicate

        async with test_db.session() as session:
            queue_repo = QueueRepository(session)
            count = await queue_repo.count()

            assert count == 1

    @pytest.mark.asyncio
    async def test_enqueue_updates_priority(self, test_db: Database) -> None:
        """Should update priority if higher."""
        async with test_db.session() as session:
            person_repo = PersonRepository(session)
            person = Person(given_name="John", surname="Smith")
            await person_repo.create(person)

            queue_repo = QueueRepository(session)
            await queue_repo.enqueue(person.id, priority=1)
            await queue_repo.enqueue(person.id, priority=5)  # Higher priority

        async with test_db.session() as session:
            from sqlmodel import select
            from ancestral_synth.persistence.tables import QueueEntryTable

            stmt = select(QueueEntryTable).where(
                QueueEntryTable.person_id == person.id
            )
            result = await session.exec(stmt)
            entry = result.first()

            assert entry is not None
            assert entry.priority == 5

    @pytest.mark.asyncio
    async def test_dequeue(self, test_db: Database) -> None:
        """Should remove and return next from queue."""
        async with test_db.session() as session:
            person_repo = PersonRepository(session)
            person = Person(given_name="John", surname="Smith")
            await person_repo.create(person)

            queue_repo = QueueRepository(session)
            await queue_repo.enqueue(person.id)

        async with test_db.session() as session:
            queue_repo = QueueRepository(session)
            dequeued = await queue_repo.dequeue()

            assert dequeued == person.id

        async with test_db.session() as session:
            queue_repo = QueueRepository(session)
            count = await queue_repo.count()

            assert count == 0

    @pytest.mark.asyncio
    async def test_dequeue_empty(self, test_db: Database) -> None:
        """Should return None for empty queue."""
        async with test_db.session() as session:
            queue_repo = QueueRepository(session)
            result = await queue_repo.dequeue()

            assert result is None

    @pytest.mark.asyncio
    async def test_dequeue_respects_priority(self, test_db: Database) -> None:
        """Should dequeue highest priority first."""
        async with test_db.session() as session:
            person_repo = PersonRepository(session)
            person1 = Person(given_name="Low", surname="Priority")
            person2 = Person(given_name="High", surname="Priority")
            await person_repo.create(person1)
            await person_repo.create(person2)

            queue_repo = QueueRepository(session)
            await queue_repo.enqueue(person1.id, priority=1)
            await queue_repo.enqueue(person2.id, priority=10)

        async with test_db.session() as session:
            queue_repo = QueueRepository(session)
            first = await queue_repo.dequeue()

            # Higher priority should come first
            assert first == person2.id

    @pytest.mark.asyncio
    async def test_peek(self, test_db: Database) -> None:
        """Should peek at queue without removing."""
        async with test_db.session() as session:
            person_repo = PersonRepository(session)
            person = Person(given_name="John", surname="Smith")
            await person_repo.create(person)

            queue_repo = QueueRepository(session)
            await queue_repo.enqueue(person.id)

        async with test_db.session() as session:
            queue_repo = QueueRepository(session)
            peeked = await queue_repo.peek(count=5)

            assert len(peeked) == 1
            assert peeked[0] == person.id

            # Should still be in queue
            count = await queue_repo.count()
            assert count == 1

    @pytest.mark.asyncio
    async def test_is_empty(self, test_db: Database) -> None:
        """Should check if queue is empty."""
        async with test_db.session() as session:
            queue_repo = QueueRepository(session)
            assert await queue_repo.is_empty() is True

        async with test_db.session() as session:
            person_repo = PersonRepository(session)
            person = Person(given_name="John", surname="Smith")
            await person_repo.create(person)

            queue_repo = QueueRepository(session)
            await queue_repo.enqueue(person.id)

        async with test_db.session() as session:
            queue_repo = QueueRepository(session)
            assert await queue_repo.is_empty() is False
