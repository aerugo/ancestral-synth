"""Repository classes for data access."""

from uuid import UUID

from sqlalchemy import func
from sqlmodel import col, select
from sqlmodel.ext.asyncio.session import AsyncSession

from ancestral_synth.domain.enums import PersonStatus
from ancestral_synth.domain.models import ChildLink, Event, Note, Person, PersonSummary, SpouseLink
from ancestral_synth.persistence.tables import (
    ChildLinkTable,
    EventParticipantTable,
    EventTable,
    NoteReferenceTable,
    NoteTable,
    PersonTable,
    QueueEntryTable,
    SpouseLinkTable,
)


class PersonRepository:
    """Repository for Person operations."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(self, person: Person) -> PersonTable:
        """Create a new person record."""
        db_person = PersonTable(
            id=person.id,
            status=person.status,
            given_name=person.given_name,
            surname=person.surname,
            maiden_name=person.maiden_name,
            nickname=person.nickname,
            gender=person.gender,
            birth_date=person.birth_date,
            birth_place=person.birth_place,
            death_date=person.death_date,
            death_place=person.death_place,
            biography=person.biography,
            generation=person.generation,
        )
        self._session.add(db_person)
        await self._session.flush()
        return db_person

    async def get_by_id(self, person_id: UUID) -> PersonTable | None:
        """Get a person by ID."""
        return await self._session.get(PersonTable, person_id)

    async def get_by_name(self, given_name: str, surname: str) -> list[PersonTable]:
        """Find people by name."""
        stmt = select(PersonTable).where(
            PersonTable.given_name == given_name,
            PersonTable.surname == surname,
        )
        result = await self._session.exec(stmt)
        return list(result.all())

    async def get_by_status(self, status: PersonStatus) -> list[PersonTable]:
        """Get all people with a given status."""
        stmt = select(PersonTable).where(PersonTable.status == status)
        result = await self._session.exec(stmt)
        return list(result.all())

    async def update(self, person_id: UUID, **kwargs) -> PersonTable | None:  # noqa: ANN003
        """Update a person record."""
        db_person = await self.get_by_id(person_id)
        if db_person is None:
            return None

        for key, value in kwargs.items():
            if hasattr(db_person, key):
                setattr(db_person, key, value)

        self._session.add(db_person)
        await self._session.flush()
        return db_person

    async def count(self, status: PersonStatus | None = None) -> int:
        """Count people, optionally filtered by status."""
        stmt = select(func.count(PersonTable.id))
        if status is not None:
            stmt = stmt.where(PersonTable.status == status)
        result = await self._session.exec(stmt)
        return result.one()

    async def delete(self, person_id: UUID) -> bool:
        """Delete a person record.

        Note: This does NOT delete associated links. Call link repository
        methods first to reassign or delete relationships.
        """
        db_person = await self.get_by_id(person_id)
        if db_person is None:
            return False
        await self._session.delete(db_person)
        await self._session.flush()
        return True

    async def search_similar(
        self,
        given_name: str,
        surname: str,
        birth_year: int | None = None,
        maiden_name: str | None = None,
    ) -> list[PersonTable]:
        """Search for potentially matching people.

        Searches for people matching:
        - Given name (case-insensitive partial match)
        - Surname OR maiden_name matching the provided surname
        - If maiden_name provided, also searches for that in surname/maiden_name
        - Birth year within ±3 years if provided
        """
        from sqlalchemy import or_

        # Build surname matching conditions:
        # 1. Direct surname match
        # 2. Maiden name matches provided surname
        # 3. If searching with maiden_name, also check if it matches
        surname_conditions = [
            PersonTable.surname == surname,
            PersonTable.maiden_name == surname,
        ]
        if maiden_name:
            surname_conditions.extend([
                PersonTable.surname == maiden_name,
                PersonTable.maiden_name == maiden_name,
            ])

        stmt = select(PersonTable).where(
            or_(*surname_conditions),
            PersonTable.given_name.ilike(f"%{given_name}%"),  # type: ignore[union-attr]
        )

        # If birth year provided, filter within a range
        if birth_year is not None:
            # Check if birth_date falls within a 5-year range
            stmt = stmt.where(
                func.strftime("%Y", PersonTable.birth_date).between(
                    str(birth_year - 3),
                    str(birth_year + 3),
                )
            )

        result = await self._session.exec(stmt)
        return list(result.all())

    async def search_by_given_name_and_birth_year(
        self,
        given_name: str,
        birth_year: int | None = None,
    ) -> list[PersonTable]:
        """Search for people by given name and optional birth year.

        This is a broader search that doesn't filter by surname, useful for
        finding married name variants where the person took their spouse's surname.
        """
        stmt = select(PersonTable).where(
            PersonTable.given_name.ilike(f"%{given_name}%"),  # type: ignore[union-attr]
        )

        # If birth year provided, filter within a range
        if birth_year is not None:
            stmt = stmt.where(
                func.strftime("%Y", PersonTable.birth_date).between(
                    str(birth_year - 5),
                    str(birth_year + 5),
                )
            )

        result = await self._session.exec(stmt)
        return list(result.all())

    def to_domain(self, db_person: PersonTable) -> Person:
        """Convert a database record to a domain model."""
        return Person(
            id=db_person.id,
            status=db_person.status,
            given_name=db_person.given_name,
            surname=db_person.surname,
            maiden_name=db_person.maiden_name,
            nickname=db_person.nickname,
            gender=db_person.gender,
            birth_date=db_person.birth_date,
            birth_place=db_person.birth_place,
            death_date=db_person.death_date,
            death_place=db_person.death_place,
            biography=db_person.biography,
            generation=db_person.generation,
        )

    def to_summary(
        self, db_person: PersonTable, relationship=None  # noqa: ANN001
    ) -> PersonSummary:
        """Convert a database record to a summary."""
        return PersonSummary(
            id=db_person.id,
            full_name=f"{db_person.given_name} {db_person.surname}",
            gender=db_person.gender,
            birth_year=db_person.birth_date.year if db_person.birth_date else None,
            death_year=db_person.death_date.year if db_person.death_date else None,
            birth_place=db_person.birth_place,
            relationship_to_subject=relationship,
        )


class EventRepository:
    """Repository for Event operations."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(self, event: Event) -> EventTable:
        """Create a new event record."""
        db_event = EventTable(
            id=event.id,
            event_type=event.event_type,
            event_date=event.event_date,
            event_year=event.event_year,
            location=event.location,
            description=event.description,
            primary_person_id=event.primary_person_id,
        )
        self._session.add(db_event)
        await self._session.flush()

        # Add participants
        for person_id in event.other_person_ids:
            participant = EventParticipantTable(
                event_id=event.id,
                person_id=person_id,
            )
            self._session.add(participant)

        await self._session.flush()
        return db_event

    async def get_by_person(self, person_id: UUID) -> list[EventTable]:
        """Get all events for a person."""
        stmt = select(EventTable).where(EventTable.primary_person_id == person_id)
        result = await self._session.exec(stmt)
        return list(result.all())

    def to_domain(self, db_event: EventTable) -> Event:
        """Convert a database record to a domain model."""
        participant_ids = [p.person_id for p in db_event.participants] if db_event.participants else []
        return Event(
            id=db_event.id,
            event_type=db_event.event_type,
            event_date=db_event.event_date,
            event_year=db_event.event_year,
            location=db_event.location,
            description=db_event.description,
            primary_person_id=db_event.primary_person_id,
            other_person_ids=participant_ids,
        )


class NoteRepository:
    """Repository for Note operations."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(self, note: Note) -> NoteTable:
        """Create a new note record."""
        db_note = NoteTable(
            id=note.id,
            person_id=note.person_id,
            category=note.category,
            content=note.content,
            source=note.source,
        )
        self._session.add(db_note)
        await self._session.flush()

        # Add references
        for person_id in note.referenced_person_ids:
            reference = NoteReferenceTable(
                note_id=note.id,
                person_id=person_id,
            )
            self._session.add(reference)

        await self._session.flush()
        return db_note

    async def get_by_person(self, person_id: UUID) -> list[NoteTable]:
        """Get all notes for a person."""
        stmt = select(NoteTable).where(NoteTable.person_id == person_id)
        result = await self._session.exec(stmt)
        return list(result.all())


class ChildLinkRepository:
    """Repository for ChildLink operations."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(self, link: ChildLink) -> ChildLinkTable:
        """Create a new child link."""
        db_link = ChildLinkTable(
            parent_id=link.parent_id,
            child_id=link.child_id,
        )
        self._session.add(db_link)
        await self._session.flush()
        return db_link

    async def exists(self, parent_id: UUID, child_id: UUID) -> bool:
        """Check if a link already exists."""
        stmt = select(ChildLinkTable).where(
            ChildLinkTable.parent_id == parent_id,
            ChildLinkTable.child_id == child_id,
        )
        result = await self._session.exec(stmt)
        return result.first() is not None

    async def get_children(self, parent_id: UUID) -> list[UUID]:
        """Get all children of a parent."""
        stmt = select(ChildLinkTable.child_id).where(ChildLinkTable.parent_id == parent_id)
        result = await self._session.exec(stmt)
        return list(result.all())

    async def get_parents(self, child_id: UUID) -> list[UUID]:
        """Get all parents of a child."""
        stmt = select(ChildLinkTable.parent_id).where(ChildLinkTable.child_id == child_id)
        result = await self._session.exec(stmt)
        return list(result.all())

    async def get_parent_count(self, child_id: UUID) -> int:
        """Count the number of parents for a child."""
        stmt = select(func.count(ChildLinkTable.parent_id)).where(
            ChildLinkTable.child_id == child_id
        )
        result = await self._session.exec(stmt)
        return result.one()

    async def delete(self, parent_id: UUID, child_id: UUID) -> bool:
        """Delete a parent-child link."""
        stmt = select(ChildLinkTable).where(
            ChildLinkTable.parent_id == parent_id,
            ChildLinkTable.child_id == child_id,
        )
        result = await self._session.exec(stmt)
        link = result.first()
        if link:
            await self._session.delete(link)
            await self._session.flush()
            return True
        return False

    async def reassign_parent(
        self, old_parent_id: UUID, new_parent_id: UUID, child_id: UUID
    ) -> bool:
        """Reassign a child's parent link from one parent to another.

        Used when merging duplicate parent records.
        """
        # Check if new link already exists
        if await self.exists(new_parent_id, child_id):
            # Just delete the old link
            return await self.delete(old_parent_id, child_id)

        # Update the existing link
        stmt = select(ChildLinkTable).where(
            ChildLinkTable.parent_id == old_parent_id,
            ChildLinkTable.child_id == child_id,
        )
        result = await self._session.exec(stmt)
        link = result.first()
        if link:
            # Delete old link and create new one (SQLite doesn't support UPDATE on composite PK)
            await self._session.delete(link)
            await self._session.flush()
            new_link = ChildLinkTable(parent_id=new_parent_id, child_id=child_id)
            self._session.add(new_link)
            await self._session.flush()
            return True
        return False

    async def reassign_all_parent_links(
        self, old_parent_id: UUID, new_parent_id: UUID
    ) -> int:
        """Reassign all child links where old_parent_id is the parent to new_parent_id.

        Used when merging duplicate parent records.
        Returns the number of links reassigned.
        """
        child_ids = await self.get_children(old_parent_id)
        reassigned = 0
        for child_id in child_ids:
            if await self.reassign_parent(old_parent_id, new_parent_id, child_id):
                reassigned += 1
        return reassigned

    async def reassign_all_child_links(
        self, old_child_id: UUID, new_child_id: UUID
    ) -> int:
        """Reassign all parent links where old_child_id is the child to new_child_id.

        Used when merging duplicate child records.
        Returns the number of links reassigned.
        """
        parent_ids = await self.get_parents(old_child_id)
        reassigned = 0
        for parent_id in parent_ids:
            # Check if new link already exists
            if await self.exists(parent_id, new_child_id):
                # Just delete the old link
                await self.delete(parent_id, old_child_id)
                reassigned += 1
            else:
                # Delete old link and create new one
                await self.delete(parent_id, old_child_id)
                new_link = ChildLinkTable(parent_id=parent_id, child_id=new_child_id)
                self._session.add(new_link)
                await self._session.flush()
                reassigned += 1
        return reassigned


class SpouseLinkRepository:
    """Repository for SpouseLink operations."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(self, link: SpouseLink) -> SpouseLinkTable:
        """Create a new spouse link.

        Links are stored with the smaller UUID as person1_id for consistency.
        """
        # Normalize the order to avoid duplicate entries (A-B vs B-A)
        if str(link.person1_id) < str(link.person2_id):
            p1, p2 = link.person1_id, link.person2_id
        else:
            p1, p2 = link.person2_id, link.person1_id

        db_link = SpouseLinkTable(person1_id=p1, person2_id=p2)
        self._session.add(db_link)
        await self._session.flush()
        return db_link

    async def exists(self, person1_id: UUID, person2_id: UUID) -> bool:
        """Check if a spouse link already exists (in either direction)."""
        # Normalize the order
        if str(person1_id) < str(person2_id):
            p1, p2 = person1_id, person2_id
        else:
            p1, p2 = person2_id, person1_id

        stmt = select(SpouseLinkTable).where(
            SpouseLinkTable.person1_id == p1,
            SpouseLinkTable.person2_id == p2,
        )
        result = await self._session.exec(stmt)
        return result.first() is not None

    async def get_spouses(self, person_id: UUID) -> list[UUID]:
        """Get all spouses of a person."""
        # Check both directions since person could be person1 or person2
        stmt1 = select(SpouseLinkTable.person2_id).where(
            SpouseLinkTable.person1_id == person_id
        )
        stmt2 = select(SpouseLinkTable.person1_id).where(
            SpouseLinkTable.person2_id == person_id
        )

        result1 = await self._session.exec(stmt1)
        result2 = await self._session.exec(stmt2)

        spouses = list(result1.all()) + list(result2.all())
        return spouses

    async def reassign_spouse_links(
        self, old_person_id: UUID, new_person_id: UUID
    ) -> int:
        """Reassign all spouse links from one person to another.

        Used when merging duplicate person records.
        Returns the number of links reassigned.
        """
        reassigned = 0

        # Get all spouses of the old person
        spouse_ids = await self.get_spouses(old_person_id)

        for spouse_id in spouse_ids:
            # Skip if the spouse is the new person (would be self-spouse)
            if spouse_id == new_person_id:
                continue

            # Check if link already exists with new person
            if not await self.exists(new_person_id, spouse_id):
                # Create new link
                await self.create(SpouseLink(person1_id=new_person_id, person2_id=spouse_id))
                reassigned += 1

        # Delete all old links
        await self._delete_all_for_person(old_person_id)

        return reassigned

    async def _delete_all_for_person(self, person_id: UUID) -> int:
        """Delete all spouse links for a person."""
        deleted = 0

        # Delete links where person is person1
        stmt1 = select(SpouseLinkTable).where(SpouseLinkTable.person1_id == person_id)
        result1 = await self._session.exec(stmt1)
        for link in result1.all():
            await self._session.delete(link)
            deleted += 1

        # Delete links where person is person2
        stmt2 = select(SpouseLinkTable).where(SpouseLinkTable.person2_id == person_id)
        result2 = await self._session.exec(stmt2)
        for link in result2.all():
            await self._session.delete(link)
            deleted += 1

        await self._session.flush()
        return deleted


class QueueRepository:
    """Repository for the creation queue."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def enqueue(self, person_id: UUID, priority: int = 0) -> QueueEntryTable:
        """Add a person to the queue."""
        # Check if already in queue
        stmt = select(QueueEntryTable).where(QueueEntryTable.person_id == person_id)
        result = await self._session.exec(stmt)
        existing = result.first()

        if existing is not None:
            # Update priority if higher
            if priority > existing.priority:
                existing.priority = priority
                self._session.add(existing)
                await self._session.flush()
            return existing

        entry = QueueEntryTable(person_id=person_id, priority=priority)
        self._session.add(entry)
        await self._session.flush()
        return entry

    async def dequeue(self) -> UUID | None:
        """Remove and return the next person from the queue."""
        stmt = (
            select(QueueEntryTable)
            .order_by(col(QueueEntryTable.priority).desc(), QueueEntryTable.added_at)
            .limit(1)
        )
        result = await self._session.exec(stmt)
        entry = result.first()

        if entry is None:
            return None

        person_id = entry.person_id
        await self._session.delete(entry)
        await self._session.flush()
        return person_id

    async def peek(self, count: int = 10) -> list[UUID]:
        """Peek at the next entries in the queue without removing them."""
        stmt = (
            select(QueueEntryTable.person_id)
            .order_by(col(QueueEntryTable.priority).desc(), QueueEntryTable.added_at)
            .limit(count)
        )
        result = await self._session.exec(stmt)
        return list(result.all())

    async def count(self) -> int:
        """Count entries in the queue."""
        stmt = select(func.count(QueueEntryTable.id))
        result = await self._session.exec(stmt)
        return result.one()

    async def is_empty(self) -> bool:
        """Check if the queue is empty."""
        return await self.count() == 0
