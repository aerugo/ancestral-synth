"""Tests for the validation service."""

from datetime import date

import pytest

from ancestral_synth.domain.enums import Gender, RelationshipType, EventType
from ancestral_synth.domain.models import ExtractedData, ExtractedEvent, PersonReference
from ancestral_synth.services.validation import ValidationResult, Validator


class TestValidatorBasic:
    """Basic validation tests."""

    def test_validator_default_settings(self) -> None:
        """Should use default settings from config."""
        validator = Validator()

        assert validator.min_parent_age == 14
        assert validator.max_parent_age == 60
        assert validator.max_lifespan == 120

    def test_validator_custom_settings(self) -> None:
        """Should accept custom settings."""
        validator = Validator(
            min_parent_age=16,
            max_parent_age=50,
            max_lifespan=100,
        )

        assert validator.min_parent_age == 16
        assert validator.max_parent_age == 50
        assert validator.max_lifespan == 100


class TestValidateExtractedData:
    """Tests for validate_extracted_data method."""

    @pytest.fixture
    def validator(self) -> Validator:
        """Provide a validator instance."""
        return Validator()

    def test_valid_minimal_data(self, validator: Validator) -> None:
        """Should pass for minimal valid data."""
        data = ExtractedData(
            given_name="John",
            surname="Smith",
            gender=Gender.MALE,
        )
        result = validator.validate_extracted_data(data)

        assert result.is_valid is True
        assert result.errors == []

    def test_valid_full_data(self, validator: Validator) -> None:
        """Should pass for complete valid data."""
        data = ExtractedData(
            given_name="John",
            surname="Smith",
            gender=Gender.MALE,
            birth_date=date(1950, 6, 15),
            death_date=date(2020, 3, 10),
            parents=[
                PersonReference(
                    name="William Smith",
                    relationship=RelationshipType.PARENT,
                    approximate_birth_year=1920,
                    gender=Gender.MALE,
                ),
            ],
            children=[
                PersonReference(
                    name="James Smith",
                    relationship=RelationshipType.CHILD,
                    approximate_birth_year=1980,
                    gender=Gender.MALE,
                ),
            ],
        )
        result = validator.validate_extracted_data(data)

        assert result.is_valid is True

    def test_death_before_birth_error(self, validator: Validator) -> None:
        """Should error when death is before birth."""
        data = ExtractedData(
            given_name="John",
            surname="Smith",
            gender=Gender.MALE,
            birth_date=date(1950, 1, 1),
            death_date=date(1940, 1, 1),  # Before birth!
        )
        result = validator.validate_extracted_data(data)

        assert result.is_valid is False
        assert any("before birth" in e.lower() for e in result.errors)

    def test_excessive_lifespan_warning(self, validator: Validator) -> None:
        """Should warn for unusually long lifespan."""
        data = ExtractedData(
            given_name="John",
            surname="Smith",
            gender=Gender.MALE,
            birth_date=date(1800, 1, 1),
            death_date=date(1950, 1, 1),  # 150 years
        )
        result = validator.validate_extracted_data(data)

        assert result.is_valid is True  # Warning, not error
        assert any("lifespan" in w.lower() for w in result.warnings)

    def test_short_lifespan_warning(self, validator: Validator) -> None:
        """Should warn for very short lifespan."""
        data = ExtractedData(
            given_name="John",
            surname="Smith",
            gender=Gender.MALE,
            birth_date=date(1950, 1, 1),
            death_date=date(1950, 6, 1),  # 0 years (same year)
        )
        result = validator.validate_extracted_data(data)

        assert result.is_valid is True
        assert any("short lifespan" in w.lower() for w in result.warnings)

    def test_parent_too_young_error(self, validator: Validator) -> None:
        """Should error when parent is too young at child's birth."""
        data = ExtractedData(
            given_name="John",
            surname="Smith",
            gender=Gender.MALE,
            birth_date=date(1950, 1, 1),
            parents=[
                PersonReference(
                    name="William Smith",
                    relationship=RelationshipType.PARENT,
                    approximate_birth_year=1945,  # Would be 5 years old!
                    gender=Gender.MALE,
                ),
            ],
        )
        result = validator.validate_extracted_data(data)

        assert result.is_valid is False
        assert any("too young" in e.lower() for e in result.errors)

    def test_parent_too_old_warning(self, validator: Validator) -> None:
        """Should warn when parent is very old at child's birth."""
        data = ExtractedData(
            given_name="John",
            surname="Smith",
            gender=Gender.MALE,
            birth_date=date(1950, 1, 1),
            parents=[
                PersonReference(
                    name="William Smith",
                    relationship=RelationshipType.PARENT,
                    approximate_birth_year=1850,  # Would be 100 years old!
                    gender=Gender.MALE,
                ),
            ],
        )
        result = validator.validate_extracted_data(data)

        assert any("very old" in w.lower() for w in result.warnings)

    def test_child_too_young_parent_error(self, validator: Validator) -> None:
        """Should error when subject is too young to be parent."""
        data = ExtractedData(
            given_name="John",
            surname="Smith",
            gender=Gender.MALE,
            birth_date=date(1950, 1, 1),
            children=[
                PersonReference(
                    name="James Smith",
                    relationship=RelationshipType.CHILD,
                    approximate_birth_year=1960,  # John would be 10
                    gender=Gender.MALE,
                ),
            ],
        )
        result = validator.validate_extracted_data(data)

        assert result.is_valid is False
        assert any("too young" in e.lower() for e in result.errors)

    def test_child_born_after_parent_death_error(self, validator: Validator) -> None:
        """Should error when child born after parent's death."""
        data = ExtractedData(
            given_name="John",
            surname="Smith",
            gender=Gender.MALE,
            birth_date=date(1950, 1, 1),
            death_date=date(1980, 1, 1),
            children=[
                PersonReference(
                    name="James Smith",
                    relationship=RelationshipType.CHILD,
                    approximate_birth_year=1985,  # 5 years after death
                    gender=Gender.MALE,
                ),
            ],
        )
        result = validator.validate_extracted_data(data)

        assert result.is_valid is False
        assert any("died" in e.lower() for e in result.errors)

    def test_posthumous_birth_allowed(self, validator: Validator) -> None:
        """Should allow birth within 1 year of parent's death."""
        data = ExtractedData(
            given_name="John",
            surname="Smith",
            gender=Gender.MALE,
            birth_date=date(1920, 1, 1),
            death_date=date(1980, 6, 1),
            children=[
                PersonReference(
                    name="James Smith",
                    relationship=RelationshipType.CHILD,
                    approximate_birth_year=1981,  # Within 1 year
                    gender=Gender.MALE,
                ),
            ],
        )
        result = validator.validate_extracted_data(data)

        # Should not have error about child born after death
        assert not any("died" in e.lower() for e in result.errors)

    def test_large_spouse_age_gap_warning(self, validator: Validator) -> None:
        """Should warn for large age gap between spouses."""
        data = ExtractedData(
            given_name="John",
            surname="Smith",
            gender=Gender.MALE,
            birth_date=date(1950, 1, 1),
            spouses=[
                PersonReference(
                    name="Mary Johnson",
                    relationship=RelationshipType.SPOUSE,
                    approximate_birth_year=1910,  # 40 year gap
                    gender=Gender.FEMALE,
                ),
            ],
        )
        result = validator.validate_extracted_data(data)

        assert any("age gap" in w.lower() for w in result.warnings)

    def test_event_before_birth_error(self, validator: Validator) -> None:
        """Should error when event occurs before birth."""
        data = ExtractedData(
            given_name="John",
            surname="Smith",
            gender=Gender.MALE,
            birth_date=date(1950, 1, 1),
            events=[
                ExtractedEvent(
                    event_type=EventType.GRADUATION,
                    event_date=date(1940, 6, 1),  # Before birth
                    description="Graduated college",
                ),
            ],
        )
        result = validator.validate_extracted_data(data)

        assert result.is_valid is False
        assert any("before birth" in e.lower() for e in result.errors)

    def test_event_after_death_error(self, validator: Validator) -> None:
        """Should error when event occurs after death."""
        data = ExtractedData(
            given_name="John",
            surname="Smith",
            gender=Gender.MALE,
            birth_date=date(1950, 1, 1),
            death_date=date(2000, 1, 1),
            events=[
                ExtractedEvent(
                    event_type=EventType.RETIREMENT,
                    event_date=date(2010, 6, 1),  # After death
                    description="Retired",
                ),
            ],
        )
        result = validator.validate_extracted_data(data)

        assert result.is_valid is False
        assert any("after death" in e.lower() for e in result.errors)

    def test_event_with_year_only(self, validator: Validator) -> None:
        """Should validate events with only year specified."""
        data = ExtractedData(
            given_name="John",
            surname="Smith",
            gender=Gender.MALE,
            birth_date=date(1950, 1, 1),
            events=[
                ExtractedEvent(
                    event_type=EventType.IMMIGRATION,
                    event_year=1940,  # Before birth
                    description="Immigrated",
                ),
            ],
        )
        result = validator.validate_extracted_data(data)

        assert result.is_valid is False


class TestValidatePersonReference:
    """Tests for validate_person_reference method."""

    @pytest.fixture
    def validator(self) -> Validator:
        """Provide a validator instance."""
        return Validator()

    def test_valid_reference(self, validator: Validator) -> None:
        """Should pass for valid reference."""
        ref = PersonReference(
            name="John Smith",
            relationship=RelationshipType.PARENT,
            approximate_birth_year=1920,
        )
        result = validator.validate_person_reference(ref)

        assert result.is_valid is True

    def test_empty_name_error(self, validator: Validator) -> None:
        """Should error for empty name."""
        ref = PersonReference(
            name="",
            relationship=RelationshipType.PARENT,
        )
        result = validator.validate_person_reference(ref)

        assert result.is_valid is False
        assert any("empty name" in e.lower() for e in result.errors)

    def test_whitespace_only_name_error(self, validator: Validator) -> None:
        """Should error for whitespace-only name."""
        ref = PersonReference(
            name="   ",
            relationship=RelationshipType.PARENT,
        )
        result = validator.validate_person_reference(ref)

        assert result.is_valid is False

    def test_future_birth_year_error(self, validator: Validator) -> None:
        """Should error for future birth year."""
        ref = PersonReference(
            name="John Smith",
            relationship=RelationshipType.PARENT,
            approximate_birth_year=2100,  # Future
        )
        result = validator.validate_person_reference(ref)

        assert result.is_valid is False
        assert any("future" in e.lower() for e in result.errors)

    def test_very_old_birth_year_warning(self, validator: Validator) -> None:
        """Should warn for very old birth year."""
        ref = PersonReference(
            name="John Smith",
            relationship=RelationshipType.PARENT,
            approximate_birth_year=1400,  # Very old
        )
        result = validator.validate_person_reference(ref)

        assert result.is_valid is True  # Warning, not error
        assert any("very old" in w.lower() for w in result.warnings)


class TestValidationResult:
    """Tests for ValidationResult class."""

    def test_is_valid_with_no_errors(self) -> None:
        """Should be valid when no errors."""
        result = ValidationResult(is_valid=True)

        assert result.is_valid is True
        assert result.errors == []
        assert result.warnings == []

    def test_is_invalid_with_errors(self) -> None:
        """Should be invalid when errors present."""
        result = ValidationResult(
            is_valid=False,
            errors=["Error 1", "Error 2"],
        )

        assert result.is_valid is False
        assert len(result.errors) == 2

    def test_warnings_dont_affect_validity(self) -> None:
        """Warnings should not affect validity."""
        result = ValidationResult(
            is_valid=True,
            warnings=["Warning 1", "Warning 2"],
        )

        assert result.is_valid is True
        assert len(result.warnings) == 2
