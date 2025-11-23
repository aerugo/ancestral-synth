"""Enumerations for the genealogical domain."""

from enum import StrEnum


class PersonStatus(StrEnum):
    """Status of a person record in the system."""

    PENDING = "pending"  # Referenced but not yet processed
    QUEUED = "queued"  # In the creation queue
    PROCESSING = "processing"  # Currently being processed
    COMPLETE = "complete"  # Fully processed with biography


class Gender(StrEnum):
    """Gender of a person."""

    MALE = "male"
    FEMALE = "female"
    UNKNOWN = "unknown"


class RelationshipType(StrEnum):
    """Types of relationships between people."""

    PARENT = "parent"
    CHILD = "child"
    SPOUSE = "spouse"
    SIBLING = "sibling"
    GRANDPARENT = "grandparent"
    GRANDCHILD = "grandchild"
    UNCLE = "uncle"
    AUNT = "aunt"
    COUSIN = "cousin"
    NIECE = "niece"
    NEPHEW = "nephew"
    OTHER = "other"


class EventType(StrEnum):
    """Types of life events."""

    BIRTH = "birth"
    DEATH = "death"
    MARRIAGE = "marriage"
    DIVORCE = "divorce"
    BAPTISM = "baptism"
    GRADUATION = "graduation"
    IMMIGRATION = "immigration"
    EMIGRATION = "emigration"
    MILITARY_SERVICE = "military_service"
    OCCUPATION = "occupation"
    RESIDENCE = "residence"
    RETIREMENT = "retirement"
    OTHER = "other"


class NoteCategory(StrEnum):
    """Categories for notes about a person."""

    BIOGRAPHY = "biography"
    HEALTH = "health"
    EDUCATION = "education"
    CAREER = "career"
    PERSONALITY = "personality"
    ACHIEVEMENT = "achievement"
    ANECDOTE = "anecdote"
    HISTORICAL_CONTEXT = "historical_context"
    CROSS_BIOGRAPHY = "cross_biography"  # Context discovered from another person's biography
    OTHER = "other"
