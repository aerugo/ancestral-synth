"""Tests for the relationship validation service."""

from datetime import date
from uuid import uuid4

import pytest

from ancestral_synth.domain.enums import Gender, PersonStatus
from ancestral_synth.domain.models import ChildLink, Person, SpouseLink
from ancestral_synth.persistence.database import Database
from ancestral_synth.persistence.repositories import (
    ChildLinkRepository,
    PersonRepository,
    SpouseLinkRepository,
)
from ancestral_synth.services.relationship_validator import (
    RelationshipValidationIssue,
    RelationshipValidationResult,
    RelationshipValidator,
    ValidationAction,
    ValidationSeverity,
)


class TestRelationshipValidatorBasic:
    """Basic relationship validator tests."""

    def test_validator_default_settings(self) -> None:
        """Should use default settings from config."""
        validator = RelationshipValidator()

        assert validator.min_parent_age == 14
        assert validator.max_parent_age == 60
        assert validator.posthumous_allowance == 1

    def test_validator_custom_settings(self) -> None:
        """Should accept custom settings."""
        validator = RelationshipValidator(
            min_parent_age=16,
            max_parent_age=50,
            posthumous_allowance=2,
        )

        assert validator.min_parent_age == 16
        assert validator.max_parent_age == 50
        assert validator.posthumous_allowance == 2


class TestRelationshipValidationResult:
    """Tests for RelationshipValidationResult class."""

    def test_has_issues_empty(self) -> None:
        """Result with no issues should report no issues."""
        result = RelationshipValidationResult(issues=[])

        assert result.has_issues is False
        assert result.has_errors is False
        assert result.has_warnings is False

    def test_has_issues_with_warning(self) -> None:
        """Result with warning should report has_issues and has_warnings."""
        issue = RelationshipValidationIssue(
            person_id=uuid4(),
            severity=ValidationSeverity.WARNING,
            message="Test warning",
            action=ValidationAction.LOG,
        )
        result = RelationshipValidationResult(issues=[issue])

        assert result.has_issues is True
        assert result.has_warnings is True
        assert result.has_errors is False

    def test_has_issues_with_error(self) -> None:
        """Result with error should report has_issues and has_errors."""
        issue = RelationshipValidationIssue(
            person_id=uuid4(),
            severity=ValidationSeverity.ERROR,
            message="Test error",
            action=ValidationAction.MERGE,
        )
        result = RelationshipValidationResult(issues=[issue])

        assert result.has_issues is True
        assert result.has_errors is True
        assert result.has_warnings is False


class TestTooManyParentsValidation:
    """Tests for detecting people with more than two parents."""

    @pytest.mark.asyncio
    async def test_person_with_two_parents_valid(self, test_db: Database) -> None:
        """Person with exactly two parents should pass validation."""
        async with test_db.session() as session:
            person_repo = PersonRepository(session)
            child_link_repo = ChildLinkRepository(session)

            father = Person(given_name="John", surname="Smith", gender=Gender.MALE)
            mother = Person(given_name="Mary", surname="Smith", gender=Gender.FEMALE)
            child = Person(given_name="James", surname="Smith", gender=Gender.MALE)

            await person_repo.create(father)
            await person_repo.create(mother)
            await person_repo.create(child)

            await child_link_repo.create(ChildLink(parent_id=father.id, child_id=child.id))
            await child_link_repo.create(ChildLink(parent_id=mother.id, child_id=child.id))

        validator = RelationshipValidator()

        async with test_db.session() as session:
            result = await validator.validate_person_relationships(
                session, child.id, include_first_degree=False
            )

            assert result.has_issues is False

    @pytest.mark.asyncio
    async def test_person_with_three_parents_error(self, test_db: Database) -> None:
        """Person with three parents should trigger an error with merge action."""
        async with test_db.session() as session:
            person_repo = PersonRepository(session)
            child_link_repo = ChildLinkRepository(session)

            parent1 = Person(given_name="John", surname="Smith", gender=Gender.MALE)
            parent2 = Person(given_name="Mary", surname="Smith", gender=Gender.FEMALE)
            parent3 = Person(given_name="Jane", surname="Doe", gender=Gender.FEMALE)
            child = Person(given_name="James", surname="Smith", gender=Gender.MALE)

            await person_repo.create(parent1)
            await person_repo.create(parent2)
            await person_repo.create(parent3)
            await person_repo.create(child)

            await child_link_repo.create(ChildLink(parent_id=parent1.id, child_id=child.id))
            await child_link_repo.create(ChildLink(parent_id=parent2.id, child_id=child.id))
            await child_link_repo.create(ChildLink(parent_id=parent3.id, child_id=child.id))

        validator = RelationshipValidator()

        async with test_db.session() as session:
            result = await validator.validate_person_relationships(
                session, child.id, include_first_degree=False
            )

            assert result.has_errors is True
            assert len(result.issues) == 1
            issue = result.issues[0]
            assert issue.person_id == child.id
            assert issue.severity == ValidationSeverity.ERROR
            assert "3 parents" in issue.message
            assert issue.action == ValidationAction.MERGE

    @pytest.mark.asyncio
    async def test_person_with_one_parent_valid(self, test_db: Database) -> None:
        """Person with only one parent should pass validation."""
        async with test_db.session() as session:
            person_repo = PersonRepository(session)
            child_link_repo = ChildLinkRepository(session)

            parent = Person(given_name="John", surname="Smith", gender=Gender.MALE)
            child = Person(given_name="James", surname="Smith", gender=Gender.MALE)

            await person_repo.create(parent)
            await person_repo.create(child)

            await child_link_repo.create(ChildLink(parent_id=parent.id, child_id=child.id))

        validator = RelationshipValidator()

        async with test_db.session() as session:
            result = await validator.validate_person_relationships(
                session, child.id, include_first_degree=False
            )

            assert result.has_issues is False


class TestChildBornAfterParentDeathValidation:
    """Tests for detecting children born after parent death."""

    @pytest.mark.asyncio
    async def test_child_born_before_parent_death_valid(self, test_db: Database) -> None:
        """Child born before parent's death should pass validation."""
        async with test_db.session() as session:
            person_repo = PersonRepository(session)
            child_link_repo = ChildLinkRepository(session)

            parent = Person(
                given_name="John",
                surname="Smith",
                gender=Gender.MALE,
                birth_date=date(1950, 1, 1),
                death_date=date(2000, 1, 1),
            )
            child = Person(
                given_name="James",
                surname="Smith",
                gender=Gender.MALE,
                birth_date=date(1980, 1, 1),
            )

            await person_repo.create(parent)
            await person_repo.create(child)

            await child_link_repo.create(ChildLink(parent_id=parent.id, child_id=child.id))

        validator = RelationshipValidator()

        async with test_db.session() as session:
            result = await validator.validate_person_relationships(
                session, parent.id, include_first_degree=True
            )

            # Check there are no posthumous birth issues
            posthumous_issues = [
                i for i in result.issues if "born after" in i.message.lower()
            ]
            assert len(posthumous_issues) == 0

    @pytest.mark.asyncio
    async def test_child_born_after_parent_death_error(self, test_db: Database) -> None:
        """Child born more than 1 year after parent's death should trigger error."""
        async with test_db.session() as session:
            person_repo = PersonRepository(session)
            child_link_repo = ChildLinkRepository(session)

            parent = Person(
                given_name="John",
                surname="Smith",
                gender=Gender.MALE,
                birth_date=date(1950, 1, 1),
                death_date=date(2000, 1, 1),
            )
            child = Person(
                given_name="James",
                surname="Smith",
                gender=Gender.MALE,
                birth_date=date(2005, 1, 1),  # 5 years after parent death
            )

            await person_repo.create(parent)
            await person_repo.create(child)

            await child_link_repo.create(ChildLink(parent_id=parent.id, child_id=child.id))

        validator = RelationshipValidator()

        async with test_db.session() as session:
            result = await validator.validate_person_relationships(
                session, parent.id, include_first_degree=True
            )

            # Check there is a posthumous birth issue
            posthumous_issues = [
                i for i in result.issues if "born after" in i.message.lower()
            ]
            assert len(posthumous_issues) == 1
            assert posthumous_issues[0].severity == ValidationSeverity.ERROR
            assert "James Smith" in posthumous_issues[0].message

    @pytest.mark.asyncio
    async def test_posthumous_birth_within_allowance_valid(self, test_db: Database) -> None:
        """Child born within posthumous allowance period should pass validation."""
        async with test_db.session() as session:
            person_repo = PersonRepository(session)
            child_link_repo = ChildLinkRepository(session)

            parent = Person(
                given_name="John",
                surname="Smith",
                gender=Gender.MALE,
                birth_date=date(1950, 1, 1),
                death_date=date(2000, 6, 1),
            )
            child = Person(
                given_name="James",
                surname="Smith",
                gender=Gender.MALE,
                birth_date=date(2001, 1, 1),  # Within 1 year of death
            )

            await person_repo.create(parent)
            await person_repo.create(child)

            await child_link_repo.create(ChildLink(parent_id=parent.id, child_id=child.id))

        validator = RelationshipValidator()

        async with test_db.session() as session:
            result = await validator.validate_person_relationships(
                session, parent.id, include_first_degree=True
            )

            # Check there are no posthumous birth errors
            posthumous_issues = [
                i for i in result.issues
                if "born after" in i.message.lower() and i.severity == ValidationSeverity.ERROR
            ]
            assert len(posthumous_issues) == 0


class TestChildBornBeforeParentOldEnough:
    """Tests for detecting children born before parent was old enough."""

    @pytest.mark.asyncio
    async def test_parent_old_enough_valid(self, test_db: Database) -> None:
        """Parent old enough at child's birth should pass validation."""
        async with test_db.session() as session:
            person_repo = PersonRepository(session)
            child_link_repo = ChildLinkRepository(session)

            parent = Person(
                given_name="John",
                surname="Smith",
                gender=Gender.MALE,
                birth_date=date(1950, 1, 1),
            )
            child = Person(
                given_name="James",
                surname="Smith",
                gender=Gender.MALE,
                birth_date=date(1980, 1, 1),  # Parent is 30
            )

            await person_repo.create(parent)
            await person_repo.create(child)

            await child_link_repo.create(ChildLink(parent_id=parent.id, child_id=child.id))

        validator = RelationshipValidator()

        async with test_db.session() as session:
            result = await validator.validate_person_relationships(
                session, parent.id, include_first_degree=True
            )

            # Check there are no "too young" issues
            too_young_issues = [
                i for i in result.issues if "too young" in i.message.lower()
            ]
            assert len(too_young_issues) == 0

    @pytest.mark.asyncio
    async def test_parent_too_young_error(self, test_db: Database) -> None:
        """Parent too young at child's birth should trigger error."""
        async with test_db.session() as session:
            person_repo = PersonRepository(session)
            child_link_repo = ChildLinkRepository(session)

            parent = Person(
                given_name="John",
                surname="Smith",
                gender=Gender.MALE,
                birth_date=date(1990, 1, 1),
            )
            child = Person(
                given_name="James",
                surname="Smith",
                gender=Gender.MALE,
                birth_date=date(2000, 1, 1),  # Parent would be 10 years old
            )

            await person_repo.create(parent)
            await person_repo.create(child)

            await child_link_repo.create(ChildLink(parent_id=parent.id, child_id=child.id))

        validator = RelationshipValidator()

        async with test_db.session() as session:
            result = await validator.validate_person_relationships(
                session, parent.id, include_first_degree=True
            )

            # Check there is a "too young" issue
            too_young_issues = [
                i for i in result.issues if "too young" in i.message.lower()
            ]
            assert len(too_young_issues) == 1
            assert too_young_issues[0].severity == ValidationSeverity.ERROR

    @pytest.mark.asyncio
    async def test_parent_exactly_min_age_valid(self, test_db: Database) -> None:
        """Parent exactly at minimum age should pass validation."""
        async with test_db.session() as session:
            person_repo = PersonRepository(session)
            child_link_repo = ChildLinkRepository(session)

            parent = Person(
                given_name="John",
                surname="Smith",
                gender=Gender.MALE,
                birth_date=date(1986, 1, 1),
            )
            child = Person(
                given_name="James",
                surname="Smith",
                gender=Gender.MALE,
                birth_date=date(2000, 1, 1),  # Parent is exactly 14
            )

            await person_repo.create(parent)
            await person_repo.create(child)

            await child_link_repo.create(ChildLink(parent_id=parent.id, child_id=child.id))

        validator = RelationshipValidator()

        async with test_db.session() as session:
            result = await validator.validate_person_relationships(
                session, parent.id, include_first_degree=True
            )

            # Check there are no "too young" errors (boundary case)
            too_young_issues = [
                i for i in result.issues
                if "too young" in i.message.lower() and i.severity == ValidationSeverity.ERROR
            ]
            assert len(too_young_issues) == 0


class TestFirstDegreeRelationValidation:
    """Tests for validating first-degree relations (parents, children, spouses)."""

    @pytest.mark.asyncio
    async def test_validate_includes_children(self, test_db: Database) -> None:
        """Validation should include children when include_first_degree is True."""
        async with test_db.session() as session:
            person_repo = PersonRepository(session)
            child_link_repo = ChildLinkRepository(session)

            parent = Person(
                given_name="John",
                surname="Smith",
                gender=Gender.MALE,
                birth_date=date(1990, 1, 1),  # Too young
            )
            child = Person(
                given_name="James",
                surname="Smith",
                gender=Gender.MALE,
                birth_date=date(2000, 1, 1),
            )

            await person_repo.create(parent)
            await person_repo.create(child)

            await child_link_repo.create(ChildLink(parent_id=parent.id, child_id=child.id))

        validator = RelationshipValidator()

        # Validate from child's perspective with first_degree
        async with test_db.session() as session:
            result = await validator.validate_person_relationships(
                session, child.id, include_first_degree=True
            )

            # Should find the issue with the parent being too young
            too_young_issues = [
                i for i in result.issues if "too young" in i.message.lower()
            ]
            assert len(too_young_issues) == 1

    @pytest.mark.asyncio
    async def test_validate_excludes_first_degree_when_disabled(self, test_db: Database) -> None:
        """Validation should not check first-degree relations when disabled."""
        async with test_db.session() as session:
            person_repo = PersonRepository(session)
            child_link_repo = ChildLinkRepository(session)

            parent = Person(
                given_name="John",
                surname="Smith",
                gender=Gender.MALE,
                birth_date=date(1990, 1, 1),  # Too young
            )
            child = Person(
                given_name="James",
                surname="Smith",
                gender=Gender.MALE,
                birth_date=date(2000, 1, 1),
            )

            await person_repo.create(parent)
            await person_repo.create(child)

            await child_link_repo.create(ChildLink(parent_id=parent.id, child_id=child.id))

        validator = RelationshipValidator()

        # Validate from child's perspective WITHOUT first_degree
        async with test_db.session() as session:
            result = await validator.validate_person_relationships(
                session, child.id, include_first_degree=False
            )

            # Should NOT find the issue with the parent (only checking child directly)
            too_young_issues = [
                i for i in result.issues if "too young" in i.message.lower()
            ]
            assert len(too_young_issues) == 0


class TestExtensibleValidation:
    """Tests for the extensible validation framework."""

    @pytest.mark.asyncio
    async def test_custom_validation_can_be_registered(self, test_db: Database) -> None:
        """Custom validations can be registered and will be executed."""
        custom_called = False

        async def custom_validation(
            session, person_id, person_ids_to_validate
        ) -> list[RelationshipValidationIssue]:
            nonlocal custom_called
            custom_called = True
            return [
                RelationshipValidationIssue(
                    person_id=person_id,
                    severity=ValidationSeverity.WARNING,
                    message="Custom validation triggered",
                    action=ValidationAction.LOG,
                )
            ]

        async with test_db.session() as session:
            person_repo = PersonRepository(session)
            person = Person(given_name="John", surname="Smith")
            await person_repo.create(person)

        validator = RelationshipValidator()
        validator.register_validation(custom_validation)

        async with test_db.session() as session:
            result = await validator.validate_person_relationships(
                session, person.id, include_first_degree=False
            )

            assert custom_called is True
            assert result.has_warnings is True
            assert any("Custom validation" in i.message for i in result.issues)

    @pytest.mark.asyncio
    async def test_multiple_custom_validations(self, test_db: Database) -> None:
        """Multiple custom validations can be registered and all will run."""
        call_order = []

        async def validation_a(session, person_id, person_ids_to_validate):
            call_order.append("A")
            return []

        async def validation_b(session, person_id, person_ids_to_validate):
            call_order.append("B")
            return []

        async with test_db.session() as session:
            person_repo = PersonRepository(session)
            person = Person(given_name="John", surname="Smith")
            await person_repo.create(person)

        validator = RelationshipValidator()
        validator.register_validation(validation_a)
        validator.register_validation(validation_b)

        async with test_db.session() as session:
            await validator.validate_person_relationships(
                session, person.id, include_first_degree=False
            )

            assert "A" in call_order
            assert "B" in call_order


class TestValidateUpdatedPersons:
    """Tests for validating a set of updated persons and their relations."""

    @pytest.mark.asyncio
    async def test_validate_multiple_updated_persons(self, test_db: Database) -> None:
        """Should validate all updated persons and their first-degree relations."""
        async with test_db.session() as session:
            person_repo = PersonRepository(session)
            child_link_repo = ChildLinkRepository(session)

            # Create a family where parent is too young
            parent = Person(
                given_name="John",
                surname="Smith",
                gender=Gender.MALE,
                birth_date=date(1990, 1, 1),
            )
            child1 = Person(
                given_name="James",
                surname="Smith",
                gender=Gender.MALE,
                birth_date=date(2000, 1, 1),  # Parent was 10
            )
            child2 = Person(
                given_name="Jane",
                surname="Smith",
                gender=Gender.FEMALE,
                birth_date=date(2002, 1, 1),  # Parent was 12
            )

            await person_repo.create(parent)
            await person_repo.create(child1)
            await person_repo.create(child2)

            await child_link_repo.create(ChildLink(parent_id=parent.id, child_id=child1.id))
            await child_link_repo.create(ChildLink(parent_id=parent.id, child_id=child2.id))

        validator = RelationshipValidator()

        async with test_db.session() as session:
            result = await validator.validate_updated_persons(
                session, updated_person_ids=[child1.id, child2.id]
            )

            # Should find issues for both children (parent too young)
            assert result.has_errors is True
            # Both children have the same parent issue, so we should see issues
            # related to the parent being too young for both
            too_young_issues = [
                i for i in result.issues if "too young" in i.message.lower()
            ]
            # At least 2 issues (one for each child)
            assert len(too_young_issues) >= 2

    @pytest.mark.asyncio
    async def test_validate_updated_persons_deduplicates(self, test_db: Database) -> None:
        """Should not validate the same person twice when they appear in multiple relations."""
        async with test_db.session() as session:
            person_repo = PersonRepository(session)
            child_link_repo = ChildLinkRepository(session)

            # Two parents with the same child - child appears twice in relations
            parent1 = Person(given_name="John", surname="Smith", gender=Gender.MALE)
            parent2 = Person(given_name="Mary", surname="Smith", gender=Gender.FEMALE)
            child = Person(given_name="James", surname="Smith", gender=Gender.MALE)

            await person_repo.create(parent1)
            await person_repo.create(parent2)
            await person_repo.create(child)

            await child_link_repo.create(ChildLink(parent_id=parent1.id, child_id=child.id))
            await child_link_repo.create(ChildLink(parent_id=parent2.id, child_id=child.id))

        validator = RelationshipValidator()

        validation_count = {}

        original_validate = validator._validate_too_many_parents

        async def counting_validate(session, person_id, person_ids):
            validation_count[person_id] = validation_count.get(person_id, 0) + 1
            return await original_validate(session, person_id, person_ids)

        validator._validate_too_many_parents = counting_validate

        async with test_db.session() as session:
            await validator.validate_updated_persons(
                session, updated_person_ids=[parent1.id, parent2.id]
            )

            # Each person should only be validated once despite appearing in multiple contexts
            for count in validation_count.values():
                assert count == 1
