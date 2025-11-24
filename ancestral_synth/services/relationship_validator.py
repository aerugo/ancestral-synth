"""Relationship validation service for genealogical data integrity.

This module provides post-processing validation of relationships after
a person has been processed. It checks for inconsistencies such as:
- People with more than two parents (likely duplicates)
- Children born after parent's death
- Children born before parent was old enough

The validator is designed to be extensible - new validation rules can be
registered without modifying the core class.
"""

from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from uuid import UUID

from loguru import logger
from sqlmodel.ext.asyncio.session import AsyncSession

from ancestral_synth.config import settings
from ancestral_synth.persistence.repositories import (
    ChildLinkRepository,
    PersonRepository,
    SpouseLinkRepository,
)


class ValidationSeverity(Enum):
    """Severity level of a validation issue."""

    WARNING = "warning"
    ERROR = "error"


class ValidationAction(Enum):
    """Recommended action for a validation issue."""

    LOG = "log"  # Just log the issue
    MERGE = "merge"  # Suggests duplicate records should be merged
    REVIEW = "review"  # Needs manual review


@dataclass
class RelationshipValidationIssue:
    """A single validation issue found during relationship validation."""

    person_id: UUID
    severity: ValidationSeverity
    message: str
    action: ValidationAction
    related_person_ids: list[UUID] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)


@dataclass
class RelationshipValidationResult:
    """Result of relationship validation for one or more persons."""

    issues: list[RelationshipValidationIssue] = field(default_factory=list)

    @property
    def has_issues(self) -> bool:
        """Check if there are any validation issues."""
        return len(self.issues) > 0

    @property
    def has_errors(self) -> bool:
        """Check if there are any error-level issues."""
        return any(i.severity == ValidationSeverity.ERROR for i in self.issues)

    @property
    def has_warnings(self) -> bool:
        """Check if there are any warning-level issues."""
        return any(i.severity == ValidationSeverity.WARNING for i in self.issues)


# Type alias for validation functions
ValidationFunc = Callable[
    [AsyncSession, UUID, set[UUID]],
    Coroutine[Any, Any, list[RelationshipValidationIssue]],
]


class RelationshipValidator:
    """Validates genealogical relationships for consistency and plausibility.

    This validator checks relationships between people in the database,
    identifying issues that may indicate data quality problems or duplicate
    records that need to be merged.

    The validator is extensible - additional validation rules can be registered
    using the register_validation() method.

    Attributes:
        min_parent_age: Minimum age to become a parent.
        max_parent_age: Maximum age to become a parent (for mothers).
        posthumous_allowance: Years after death a child can be born (for fathers).
    """

    def __init__(
        self,
        min_parent_age: int | None = None,
        max_parent_age: int | None = None,
        posthumous_allowance: int | None = None,
    ) -> None:
        """Initialize the relationship validator.

        Args:
            min_parent_age: Minimum age to become a parent.
            max_parent_age: Maximum age to become a parent.
            posthumous_allowance: Years after death a child can be born.
        """
        self.min_parent_age = min_parent_age or settings.min_parent_age
        self.max_parent_age = max_parent_age or settings.max_parent_age
        self.posthumous_allowance = posthumous_allowance or 1

        # Built-in validations
        self._validations: list[ValidationFunc] = [
            self._validate_too_many_parents,
            self._validate_parent_child_temporal,
        ]

    def register_validation(self, validation_func: ValidationFunc) -> None:
        """Register a custom validation function.

        The function should have the signature:
            async def validate(
                session: AsyncSession,
                person_id: UUID,
                person_ids_to_validate: set[UUID]
            ) -> list[RelationshipValidationIssue]

        Args:
            validation_func: The validation function to register.
        """
        self._validations.append(validation_func)

    async def validate_person_relationships(
        self,
        session: AsyncSession,
        person_id: UUID,
        include_first_degree: bool = True,
    ) -> RelationshipValidationResult:
        """Validate relationships for a person and optionally their first-degree relations.

        Args:
            session: Database session.
            person_id: ID of the person to validate.
            include_first_degree: If True, also validate parents, children, and spouses.

        Returns:
            Validation result containing any issues found.
        """
        # Gather all person IDs to validate
        person_ids_to_validate = await self._gather_persons_to_validate(
            session, person_id, include_first_degree
        )

        # Run all validations for each person to validate
        all_issues: list[RelationshipValidationIssue] = []
        validated_persons: set[UUID] = set()

        for pid in person_ids_to_validate:
            if pid in validated_persons:
                continue
            validated_persons.add(pid)

            for validation_func in self._validations:
                try:
                    issues = await validation_func(session, pid, person_ids_to_validate)
                    all_issues.extend(issues)
                except Exception as e:
                    logger.error(f"Validation failed for person {pid}: {e}")
                    # Continue with other validations

        return RelationshipValidationResult(issues=all_issues)

    async def validate_updated_persons(
        self,
        session: AsyncSession,
        updated_person_ids: list[UUID],
    ) -> RelationshipValidationResult:
        """Validate all updated persons and their first-degree relations.

        This is the main entry point for post-processing validation after
        one or more persons have been updated.

        Args:
            session: Database session.
            updated_person_ids: IDs of persons that were updated.

        Returns:
            Validation result containing all issues found.
        """
        # Gather all unique person IDs to validate (including first-degree relations)
        all_person_ids: set[UUID] = set()

        for person_id in updated_person_ids:
            person_ids = await self._gather_persons_to_validate(
                session, person_id, include_first_degree=True
            )
            all_person_ids.update(person_ids)

        # Run all validations for each person, avoiding duplicate validations
        all_issues: list[RelationshipValidationIssue] = []
        validated_persons: set[UUID] = set()

        for person_id in all_person_ids:
            if person_id in validated_persons:
                continue

            validated_persons.add(person_id)

            for validation_func in self._validations:
                try:
                    issues = await validation_func(
                        session, person_id, all_person_ids
                    )
                    all_issues.extend(issues)
                except Exception as e:
                    logger.error(f"Validation failed for person {person_id}: {e}")

        return RelationshipValidationResult(issues=all_issues)

    async def _gather_persons_to_validate(
        self,
        session: AsyncSession,
        person_id: UUID,
        include_first_degree: bool,
    ) -> set[UUID]:
        """Gather all person IDs that should be validated.

        Args:
            session: Database session.
            person_id: The main person ID.
            include_first_degree: Whether to include first-degree relations.

        Returns:
            Set of person IDs to validate.
        """
        person_ids: set[UUID] = {person_id}

        if not include_first_degree:
            return person_ids

        child_link_repo = ChildLinkRepository(session)
        spouse_link_repo = SpouseLinkRepository(session)

        # Get parents
        parent_ids = await child_link_repo.get_parents(person_id)
        person_ids.update(parent_ids)

        # Get children
        child_ids = await child_link_repo.get_children(person_id)
        person_ids.update(child_ids)

        # Get spouses
        spouse_ids = await spouse_link_repo.get_spouses(person_id)
        person_ids.update(spouse_ids)

        return person_ids

    async def _validate_too_many_parents(
        self,
        session: AsyncSession,
        person_id: UUID,
        person_ids_to_validate: set[UUID],
    ) -> list[RelationshipValidationIssue]:
        """Validate that a person doesn't have more than two parents.

        Having more than two parents indicates duplicate parent records
        that should be merged.

        Args:
            session: Database session.
            person_id: ID of the person to validate.
            person_ids_to_validate: Set of all person IDs being validated (unused).

        Returns:
            List of validation issues (empty if valid).
        """
        child_link_repo = ChildLinkRepository(session)
        person_repo = PersonRepository(session)

        parent_count = await child_link_repo.get_parent_count(person_id)

        if parent_count > 2:
            # Get the person's name for the message
            person = await person_repo.get_by_id(person_id)
            person_name = f"{person.given_name} {person.surname}" if person else str(person_id)

            # Get parent IDs for context
            parent_ids = await child_link_repo.get_parents(person_id)

            return [
                RelationshipValidationIssue(
                    person_id=person_id,
                    severity=ValidationSeverity.ERROR,
                    message=f"{person_name} has {parent_count} parents (expected max 2). "
                    f"This likely indicates duplicate parent records that should be merged.",
                    action=ValidationAction.MERGE,
                    related_person_ids=parent_ids,
                    context={"parent_count": parent_count},
                )
            ]

        return []

    async def _validate_parent_child_temporal(
        self,
        session: AsyncSession,
        person_id: UUID,
        person_ids_to_validate: set[UUID],
    ) -> list[RelationshipValidationIssue]:
        """Validate temporal consistency of parent-child relationships.

        Checks:
        - Children not born more than posthumous_allowance years after parent's death
        - Parent was at least min_parent_age when child was born

        Args:
            session: Database session.
            person_id: ID of the person to validate.
            person_ids_to_validate: Set of all person IDs being validated.

        Returns:
            List of validation issues.
        """
        issues: list[RelationshipValidationIssue] = []

        child_link_repo = ChildLinkRepository(session)
        person_repo = PersonRepository(session)

        # Get the person
        person = await person_repo.get_by_id(person_id)
        if person is None:
            return issues

        # Skip if person doesn't have dates we can validate
        parent_birth_year = person.birth_date.year if person.birth_date else None
        parent_death_year = person.death_date.year if person.death_date else None

        if parent_birth_year is None and parent_death_year is None:
            return issues

        # Get children and validate temporal relationships
        child_ids = await child_link_repo.get_children(person_id)

        for child_id in child_ids:
            child = await person_repo.get_by_id(child_id)
            if child is None or child.birth_date is None:
                continue

            child_birth_year = child.birth_date.year
            child_name = f"{child.given_name} {child.surname}"
            parent_name = f"{person.given_name} {person.surname}"

            # Check if child born after parent death (beyond allowance)
            if parent_death_year is not None:
                years_after_death = child_birth_year - parent_death_year
                if years_after_death > self.posthumous_allowance:
                    issues.append(
                        RelationshipValidationIssue(
                            person_id=person_id,
                            severity=ValidationSeverity.ERROR,
                            message=f"Child {child_name} was born after parent {parent_name} died: "
                            f"born in {child_birth_year}, parent died in {parent_death_year} "
                            f"({years_after_death} years after death). "
                            f"This is beyond the {self.posthumous_allowance}-year posthumous allowance.",
                            action=ValidationAction.REVIEW,
                            related_person_ids=[child_id],
                            context={
                                "child_birth_year": child_birth_year,
                                "parent_death_year": parent_death_year,
                                "years_after_death": years_after_death,
                            },
                        )
                    )

            # Check if parent was old enough
            if parent_birth_year is not None:
                parent_age_at_birth = child_birth_year - parent_birth_year
                if parent_age_at_birth < self.min_parent_age:
                    issues.append(
                        RelationshipValidationIssue(
                            person_id=person_id,
                            severity=ValidationSeverity.ERROR,
                            message=f"Parent {parent_name} was too young at child's birth: "
                            f"age {parent_age_at_birth} (minimum: {self.min_parent_age}). "
                            f"Child {child_name} was born in {child_birth_year}.",
                            action=ValidationAction.REVIEW,
                            related_person_ids=[child_id],
                            context={
                                "parent_age_at_birth": parent_age_at_birth,
                                "min_parent_age": self.min_parent_age,
                                "child_birth_year": child_birth_year,
                            },
                        )
                    )

        return issues
