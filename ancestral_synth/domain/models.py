"""Core domain models using Pydantic."""

from datetime import date
from typing import Annotated
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from ancestral_synth.domain.enums import (
    EventType,
    Gender,
    NoteCategory,
    PersonStatus,
    RelationshipType,
)


# Type aliases for clarity
PersonId = Annotated[UUID, Field(description="Unique identifier for a person")]
EventId = Annotated[UUID, Field(description="Unique identifier for an event")]
NoteId = Annotated[UUID, Field(description="Unique identifier for a note")]


class PersonReference(BaseModel):
    """A reference to another person mentioned in a biography.

    Used during extraction to identify relationships.
    """

    name: str = Field(description="Full name of the referenced person")
    relationship: RelationshipType = Field(description="Relationship to the main person")
    approximate_birth_year: int | None = Field(
        default=None,
        description="Approximate birth year if mentioned or inferable",
    )
    gender: Gender = Field(default=Gender.UNKNOWN, description="Gender if known")
    context: str | None = Field(
        default=None,
        description="Brief context about this person from the biography",
    )


class Person(BaseModel):
    """A person in the genealogical dataset."""

    id: UUID = Field(default_factory=uuid4, description="Unique identifier")
    status: PersonStatus = Field(default=PersonStatus.PENDING, description="Processing status")

    # Identity
    given_name: str = Field(description="First/given name")
    surname: str = Field(description="Family/last name")
    maiden_name: str | None = Field(default=None, description="Maiden name if applicable")
    nickname: str | None = Field(default=None, description="Known nickname or alias")

    # Demographics
    gender: Gender = Field(default=Gender.UNKNOWN, description="Gender")
    birth_date: date | None = Field(default=None, description="Date of birth")
    birth_place: str | None = Field(default=None, description="Place of birth")
    death_date: date | None = Field(default=None, description="Date of death")
    death_place: str | None = Field(default=None, description="Place of death")

    # Biography
    biography: str | None = Field(default=None, description="Full biography text")

    # Metadata
    generation: int = Field(
        default=0,
        description="Generation number (0 = seed, negative = ancestors, positive = descendants)",
    )

    @property
    def full_name(self) -> str:
        """Get the full name of the person."""
        return f"{self.given_name} {self.surname}"

    @property
    def birth_year(self) -> int | None:
        """Get the birth year if known."""
        return self.birth_date.year if self.birth_date else None

    @property
    def death_year(self) -> int | None:
        """Get the death year if known."""
        return self.death_date.year if self.death_date else None


class PersonSummary(BaseModel):
    """A summary of a person for context in biography generation.

    Contains essential facts without the full biography.
    """

    id: UUID
    full_name: str
    gender: Gender
    birth_year: int | None = None
    death_year: int | None = None
    birth_place: str | None = None
    relationship_to_subject: RelationshipType | None = None
    key_facts: list[str] = Field(default_factory=list)

    # Context about who mentioned this person
    mentioned_by: str | None = None

    # Generation info
    generation: int | None = None

    # First degree relations
    parents: list[str] = Field(default_factory=list)
    children: list[str] = Field(default_factory=list)
    spouses: list[str] = Field(default_factory=list)
    siblings: list[str] = Field(default_factory=list)

    # Second degree relations
    grandparents: list[str] = Field(default_factory=list)
    grandchildren: list[str] = Field(default_factory=list)

    # Biography snippets from related people mentioning this person's first name
    biography_snippets: list[str] = Field(default_factory=list)


class Event(BaseModel):
    """A life event associated with one or more people."""

    id: UUID = Field(default_factory=uuid4, description="Unique identifier")
    event_type: EventType = Field(description="Type of event")

    # When
    event_date: date | None = Field(default=None, description="Date of the event")
    event_year: int | None = Field(default=None, description="Year if exact date unknown")

    # Where
    location: str | None = Field(default=None, description="Location of the event")

    # What
    description: str = Field(description="Description of the event")

    # Who - IDs of people involved
    primary_person_id: UUID = Field(description="Main person this event is about")
    other_person_ids: list[UUID] = Field(
        default_factory=list,
        description="Other people involved in this event",
    )


class Note(BaseModel):
    """Additional notes or facts about a person."""

    id: UUID = Field(default_factory=uuid4, description="Unique identifier")
    person_id: UUID = Field(description="Person this note is about")
    category: NoteCategory = Field(description="Category of the note")
    content: str = Field(description="The note content")
    source: str | None = Field(
        default=None,
        description="Source of this information (e.g., biography extraction)",
    )

    # References to other people mentioned in this note
    referenced_person_ids: list[UUID] = Field(
        default_factory=list,
        description="People referenced in this note",
    )


class ChildLink(BaseModel):
    """A parent-child relationship link."""

    model_config = {"frozen": True}

    parent_id: UUID = Field(description="ID of the parent")
    child_id: UUID = Field(description="ID of the child")


class SpouseLink(BaseModel):
    """A spouse relationship link."""

    model_config = {"frozen": True}

    person1_id: UUID = Field(description="ID of the first spouse")
    person2_id: UUID = Field(description="ID of the second spouse")


class Biography(BaseModel):
    """A generated biography for a person."""

    content: str = Field(description="The full biography text (~1000 words)")
    word_count: int = Field(description="Actual word count")


class ExtractedEvent(BaseModel):
    """An event extracted from a biography (without UUIDs - those are added later).

    This model is used for LLM extraction. UUIDs are generated when converting
    to the full Event model for storage.
    """

    event_type: EventType = Field(description="Type of event")

    # When
    event_date: date | None = Field(default=None, description="Date of the event")
    event_year: int | None = Field(default=None, description="Year if exact date unknown")

    # Where
    location: str | None = Field(default=None, description="Location of the event")

    # What
    description: str = Field(description="Description of the event")

    def to_event(self, primary_person_id: UUID) -> "Event":
        """Convert to a full Event model with generated UUIDs.

        Args:
            primary_person_id: The ID of the person this event is about.

        Returns:
            A full Event model with UUIDs.
        """
        return Event(
            event_type=self.event_type,
            event_date=self.event_date,
            event_year=self.event_year,
            location=self.location,
            description=self.description,
            primary_person_id=primary_person_id,
        )


class ExtractedData(BaseModel):
    """Structured data extracted from a biography."""

    # Person details
    given_name: str = Field(description="First/given name")
    surname: str = Field(description="Family/last name")
    maiden_name: str | None = Field(default=None, description="Maiden name if applicable")
    gender: Gender = Field(description="Gender")

    birth_date: date | None = Field(default=None, description="Date of birth")
    birth_place: str | None = Field(default=None, description="Place of birth")
    birth_year: int | None = Field(default=None, description="Birth year if exact date unknown")

    death_date: date | None = Field(default=None, description="Date of death")
    death_place: str | None = Field(default=None, description="Place of death")
    death_year: int | None = Field(default=None, description="Death year if exact date unknown")

    # Related people
    parents: list[PersonReference] = Field(
        default_factory=list,
        description="Parents mentioned in the biography",
    )
    children: list[PersonReference] = Field(
        default_factory=list,
        description="Children mentioned in the biography",
    )
    spouses: list[PersonReference] = Field(
        default_factory=list,
        description="Spouses mentioned in the biography",
    )
    siblings: list[PersonReference] = Field(
        default_factory=list,
        description="Siblings mentioned in the biography",
    )
    other_relatives: list[PersonReference] = Field(
        default_factory=list,
        description="Other relatives (aunts, uncles, cousins, etc.)",
    )

    # Events (without UUIDs - those are generated when converting to Event)
    events: list[ExtractedEvent] = Field(
        default_factory=list,
        description="Life events extracted from the biography",
    )

    # Notes
    notes: list[str] = Field(
        default_factory=list,
        description="Additional interesting facts or notes",
    )
