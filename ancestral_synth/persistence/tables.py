"""SQLModel table definitions for the database."""

from datetime import date, datetime
from uuid import UUID, uuid4

from sqlmodel import Field, Relationship, SQLModel

from ancestral_synth.domain.enums import (
    EventType,
    Gender,
    NoteCategory,
    PersonStatus,
)


class PersonTable(SQLModel, table=True):
    """Database table for Person records."""

    __tablename__ = "persons"  # type: ignore[assignment]

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    status: PersonStatus = Field(default=PersonStatus.PENDING, index=True)

    # Identity
    given_name: str = Field(index=True)
    surname: str = Field(index=True)
    maiden_name: str | None = None
    nickname: str | None = None

    # Demographics
    gender: Gender = Field(default=Gender.UNKNOWN)
    birth_date: date | None = None
    birth_place: str | None = None
    death_date: date | None = None
    death_place: str | None = None

    # Biography
    biography: str | None = None

    # Metadata
    generation: int = Field(default=0, index=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    # Relationships
    events: list["EventTable"] = Relationship(back_populates="primary_person")
    notes: list["NoteTable"] = Relationship(back_populates="person")


class EventTable(SQLModel, table=True):
    """Database table for Event records."""

    __tablename__ = "events"  # type: ignore[assignment]

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    event_type: EventType = Field(index=True)

    # When
    event_date: date | None = None
    event_year: int | None = Field(default=None, index=True)

    # Where
    location: str | None = None

    # What
    description: str

    # Primary person
    primary_person_id: UUID = Field(foreign_key="persons.id", index=True)
    primary_person: PersonTable | None = Relationship(back_populates="events")

    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)

    # Other participants
    participants: list["EventParticipantTable"] = Relationship(back_populates="event")


class EventParticipantTable(SQLModel, table=True):
    """Junction table for event participants."""

    __tablename__ = "event_participants"  # type: ignore[assignment]

    event_id: UUID = Field(foreign_key="events.id", primary_key=True)
    person_id: UUID = Field(foreign_key="persons.id", primary_key=True)

    event: EventTable | None = Relationship(back_populates="participants")


class NoteTable(SQLModel, table=True):
    """Database table for Note records."""

    __tablename__ = "notes"  # type: ignore[assignment]

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    person_id: UUID = Field(foreign_key="persons.id", index=True)
    category: NoteCategory
    content: str
    source: str | None = None

    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)

    # Relationships
    person: PersonTable | None = Relationship(back_populates="notes")
    references: list["NoteReferenceTable"] = Relationship(back_populates="note")


class NoteReferenceTable(SQLModel, table=True):
    """Junction table for people referenced in notes."""

    __tablename__ = "note_references"  # type: ignore[assignment]

    note_id: UUID = Field(foreign_key="notes.id", primary_key=True)
    person_id: UUID = Field(foreign_key="persons.id", primary_key=True)

    note: NoteTable | None = Relationship(back_populates="references")


class ChildLinkTable(SQLModel, table=True):
    """Database table for parent-child relationships."""

    __tablename__ = "child_links"  # type: ignore[assignment]

    parent_id: UUID = Field(foreign_key="persons.id", primary_key=True, index=True)
    child_id: UUID = Field(foreign_key="persons.id", primary_key=True, index=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class SpouseLinkTable(SQLModel, table=True):
    """Database table for spouse relationships."""

    __tablename__ = "spouse_links"  # type: ignore[assignment]

    person1_id: UUID = Field(foreign_key="persons.id", primary_key=True, index=True)
    person2_id: UUID = Field(foreign_key="persons.id", primary_key=True, index=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class QueueEntryTable(SQLModel, table=True):
    """Database table for the creation queue."""

    __tablename__ = "creation_queue"  # type: ignore[assignment]

    id: int | None = Field(default=None, primary_key=True)
    person_id: UUID = Field(foreign_key="persons.id", unique=True, index=True)
    priority: int = Field(default=0, index=True)  # Higher = more priority
    added_at: datetime = Field(default_factory=datetime.utcnow)
