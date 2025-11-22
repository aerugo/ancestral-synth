"""Genealogical validation service."""

from dataclasses import dataclass, field
from datetime import date

from ancestral_synth.config import settings
from ancestral_synth.domain.models import ExtractedData, PersonReference


@dataclass
class ValidationResult:
    """Result of genealogical validation."""

    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class Validator:
    """Validates genealogical data for plausibility."""

    def __init__(
        self,
        min_parent_age: int | None = None,
        max_parent_age: int | None = None,
        max_lifespan: int | None = None,
    ) -> None:
        """Initialize the validator.

        Args:
            min_parent_age: Minimum age to become a parent.
            max_parent_age: Maximum age to become a parent (for mothers).
            max_lifespan: Maximum realistic lifespan.
        """
        self.min_parent_age = min_parent_age or settings.min_parent_age
        self.max_parent_age = max_parent_age or settings.max_parent_age
        self.max_lifespan = max_lifespan or settings.max_lifespan

    def validate_extracted_data(self, data: ExtractedData) -> ValidationResult:
        """Validate extracted data for genealogical plausibility.

        Args:
            data: The extracted data to validate.

        Returns:
            Validation result with any errors or warnings.
        """
        errors: list[str] = []
        warnings: list[str] = []

        # Get subject's birth and death years
        birth_year = self._get_year(data.birth_date, data.birth_year)
        death_year = self._get_year(data.death_date, data.death_year)

        # Validate lifespan
        if birth_year and death_year:
            lifespan = death_year - birth_year
            if lifespan < 0:
                errors.append(f"Death year ({death_year}) before birth year ({birth_year})")
            elif lifespan > self.max_lifespan:
                warnings.append(f"Unusually long lifespan: {lifespan} years")
            elif lifespan < 1:
                warnings.append(f"Very short lifespan: {lifespan} years")

        # Validate parent relationships
        for parent in data.parents:
            self._validate_parent_child(
                parent_name=parent.name,
                parent_birth_year=parent.approximate_birth_year,
                child_birth_year=birth_year,
                errors=errors,
                warnings=warnings,
            )

        # Validate children
        for child in data.children:
            self._validate_parent_child(
                parent_name=f"{data.given_name} {data.surname}",
                parent_birth_year=birth_year,
                child_birth_year=child.approximate_birth_year,
                errors=errors,
                warnings=warnings,
                parent_death_year=death_year,
            )

        # Validate spouses have reasonable age gaps
        for spouse in data.spouses:
            if birth_year and spouse.approximate_birth_year:
                age_gap = abs(birth_year - spouse.approximate_birth_year)
                if age_gap > 30:
                    warnings.append(
                        f"Large age gap with spouse {spouse.name}: {age_gap} years"
                    )

        # Validate event dates
        for event in data.events:
            event_year = self._get_year(event.event_date, event.event_year)
            if event_year and birth_year:
                if event_year < birth_year:
                    errors.append(
                        f"Event '{event.event_type}' in {event_year} before birth ({birth_year})"
                    )
                if death_year and event_year > death_year:
                    errors.append(
                        f"Event '{event.event_type}' in {event_year} after death ({death_year})"
                    )

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )

    def _validate_parent_child(
        self,
        parent_name: str,
        parent_birth_year: int | None,
        child_birth_year: int | None,
        errors: list[str],
        warnings: list[str],
        parent_death_year: int | None = None,
    ) -> None:
        """Validate a parent-child relationship."""
        if parent_birth_year is None or child_birth_year is None:
            return

        parent_age_at_birth = child_birth_year - parent_birth_year

        if parent_age_at_birth < self.min_parent_age:
            errors.append(
                f"Parent {parent_name} too young at child's birth: "
                f"age {parent_age_at_birth} (min: {self.min_parent_age})"
            )
        elif parent_age_at_birth > self.max_parent_age + 20:  # Allow older fathers
            warnings.append(
                f"Parent {parent_name} very old at child's birth: age {parent_age_at_birth}"
            )

        # Check if child born after parent's death
        if parent_death_year is not None:
            if child_birth_year > parent_death_year + 1:  # Allow 1 year for posthumous
                errors.append(
                    f"Child born in {child_birth_year}, but parent {parent_name} "
                    f"died in {parent_death_year}"
                )

    def _get_year(self, date_val: date | None, year_val: int | None) -> int | None:
        """Get year from date or year value."""
        if date_val:
            return date_val.year
        return year_val

    def validate_person_reference(
        self,
        reference: PersonReference,
        subject_birth_year: int | None = None,
    ) -> ValidationResult:
        """Validate a person reference for basic plausibility.

        Args:
            reference: The person reference to validate.
            subject_birth_year: Birth year of the subject (for relative age checks).

        Returns:
            Validation result.
        """
        errors: list[str] = []
        warnings: list[str] = []

        # Check name is not empty
        if not reference.name or not reference.name.strip():
            errors.append("Person reference has empty name")

        # Check birth year is reasonable
        if reference.approximate_birth_year:
            current_year = date.today().year
            if reference.approximate_birth_year > current_year:
                errors.append(f"Birth year {reference.approximate_birth_year} is in the future")
            if reference.approximate_birth_year < 1500:
                warnings.append(f"Very old birth year: {reference.approximate_birth_year}")

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )
