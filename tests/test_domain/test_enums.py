"""Tests for domain enums."""

import pytest

from ancestral_synth.domain.enums import (
    EventType,
    Gender,
    NoteCategory,
    PersonStatus,
    RelationshipType,
)


class TestPersonStatus:
    """Tests for PersonStatus enum."""

    def test_all_statuses_exist(self) -> None:
        """All expected statuses should exist."""
        assert PersonStatus.PENDING == "pending"
        assert PersonStatus.QUEUED == "queued"
        assert PersonStatus.PROCESSING == "processing"
        assert PersonStatus.COMPLETE == "complete"

    def test_status_count(self) -> None:
        """Should have exactly 4 statuses."""
        assert len(PersonStatus) == 4

    def test_string_conversion(self) -> None:
        """Should convert to string correctly."""
        assert str(PersonStatus.PENDING) == "pending"
        assert str(PersonStatus.COMPLETE) == "complete"

    def test_from_string(self) -> None:
        """Should create from string value."""
        assert PersonStatus("pending") == PersonStatus.PENDING
        assert PersonStatus("complete") == PersonStatus.COMPLETE

    def test_invalid_status_raises(self) -> None:
        """Should raise ValueError for invalid status."""
        with pytest.raises(ValueError):
            PersonStatus("invalid")


class TestGender:
    """Tests for Gender enum."""

    def test_all_genders_exist(self) -> None:
        """All expected genders should exist."""
        assert Gender.MALE == "male"
        assert Gender.FEMALE == "female"
        assert Gender.UNKNOWN == "unknown"

    def test_gender_count(self) -> None:
        """Should have exactly 3 genders."""
        assert len(Gender) == 3

    def test_from_string(self) -> None:
        """Should create from string value."""
        assert Gender("male") == Gender.MALE
        assert Gender("female") == Gender.FEMALE
        assert Gender("unknown") == Gender.UNKNOWN


class TestRelationshipType:
    """Tests for RelationshipType enum."""

    def test_all_types_exist(self) -> None:
        """All expected relationship types should exist."""
        assert RelationshipType.PARENT == "parent"
        assert RelationshipType.CHILD == "child"
        assert RelationshipType.SPOUSE == "spouse"
        assert RelationshipType.SIBLING == "sibling"
        assert RelationshipType.OTHER == "other"

    def test_relationship_count(self) -> None:
        """Should have exactly 12 relationship types."""
        assert len(RelationshipType) == 12

    def test_extended_relationship_types_exist(self) -> None:
        """Extended relationship types should exist."""
        assert RelationshipType.GRANDPARENT == "grandparent"
        assert RelationshipType.GRANDCHILD == "grandchild"
        assert RelationshipType.UNCLE == "uncle"
        assert RelationshipType.AUNT == "aunt"
        assert RelationshipType.COUSIN == "cousin"
        assert RelationshipType.NIECE == "niece"
        assert RelationshipType.NEPHEW == "nephew"


class TestEventType:
    """Tests for EventType enum."""

    def test_key_event_types_exist(self) -> None:
        """Key event types should exist."""
        assert EventType.BIRTH == "birth"
        assert EventType.DEATH == "death"
        assert EventType.MARRIAGE == "marriage"
        assert EventType.DIVORCE == "divorce"

    def test_has_other(self) -> None:
        """Should have an OTHER type for extensibility."""
        assert EventType.OTHER == "other"

    def test_event_type_count(self) -> None:
        """Should have expected number of event types."""
        assert len(EventType) >= 10  # At least 10 event types


class TestNoteCategory:
    """Tests for NoteCategory enum."""

    def test_key_categories_exist(self) -> None:
        """Key note categories should exist."""
        assert NoteCategory.BIOGRAPHY == "biography"
        assert NoteCategory.HEALTH == "health"
        assert NoteCategory.EDUCATION == "education"
        assert NoteCategory.CAREER == "career"

    def test_has_other(self) -> None:
        """Should have an OTHER category for extensibility."""
        assert NoteCategory.OTHER == "other"


class TestEnumSerialization:
    """Tests for enum JSON serialization."""

    def test_enum_json_serialization(self) -> None:
        """Enums should serialize to their string values."""
        import json

        # StrEnum values serialize directly
        assert json.dumps(PersonStatus.PENDING.value) == '"pending"'
        assert json.dumps(Gender.MALE.value) == '"male"'

    def test_enum_in_pydantic_model(self) -> None:
        """Enums should work in Pydantic models."""
        from pydantic import BaseModel

        class TestModel(BaseModel):
            status: PersonStatus
            gender: Gender

        model = TestModel(status=PersonStatus.PENDING, gender=Gender.MALE)
        data = model.model_dump()

        assert data["status"] == "pending"
        assert data["gender"] == "male"

    def test_enum_from_pydantic_json(self) -> None:
        """Enums should deserialize from JSON strings."""
        from pydantic import BaseModel

        class TestModel(BaseModel):
            status: PersonStatus
            gender: Gender

        model = TestModel.model_validate({"status": "pending", "gender": "male"})

        assert model.status == PersonStatus.PENDING
        assert model.gender == Gender.MALE
