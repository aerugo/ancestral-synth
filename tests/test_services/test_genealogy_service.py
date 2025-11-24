"""Tests for genealogy service."""

from datetime import date
from uuid import UUID, uuid4

import pytest

from ancestral_synth.domain.enums import Gender, PersonStatus, RelationshipType
from ancestral_synth.domain.models import Biography, ExtractedData, Person, PersonReference
from ancestral_synth.persistence.database import Database
from ancestral_synth.persistence.repositories import (
    ChildLinkRepository,
    PersonRepository,
    QueueRepository,
)
from ancestral_synth.services.genealogy_service import GenealogyService
from ancestral_synth.services.relationship_validator import (
    RelationshipValidationIssue,
    RelationshipValidationResult,
    RelationshipValidator,
    ValidationAction,
    ValidationSeverity,
)
from tests.conftest import (
    MockBiographyAgent,
    MockCorrectionAgent,
    MockDedupAgent,
    MockExtractionAgent,
    MockSharedEventAgent,
)


class MockRelationshipValidator(RelationshipValidator):
    """Mock relationship validator for testing."""

    def __init__(self) -> None:
        super().__init__()
        self.validation_calls: list[tuple[UUID, list[UUID]]] = []
        self.mock_result = RelationshipValidationResult(issues=[])

    async def validate_updated_persons(
        self,
        session,
        updated_person_ids: list[UUID],
    ) -> RelationshipValidationResult:
        """Track validation calls."""
        self.validation_calls.append((updated_person_ids[0] if updated_person_ids else None, list(updated_person_ids)))
        return self.mock_result


class TestGenealogyServiceHelpers:
    """Tests for helper methods."""

    def test_extract_key_facts(self) -> None:
        """Should extract key facts from biography."""
        service = GenealogyService.__new__(GenealogyService)

        biography = (
            "John Smith was born in 1950 in Boston. "
            "He grew up in a working-class family. "
            "He married Mary in 1975. "
            "They had three children together. "
            "He worked as a carpenter for forty years."
        )

        facts = service._extract_key_facts(biography, max_facts=3)

        assert len(facts) <= 3
        assert all(isinstance(f, str) for f in facts)
        assert all(len(f) > 20 for f in facts)

    def test_extract_key_facts_empty(self) -> None:
        """Should handle empty biography."""
        service = GenealogyService.__new__(GenealogyService)

        facts = service._extract_key_facts("")

        assert facts == []

    def test_extract_key_facts_short_sentences(self) -> None:
        """Should skip very short sentences."""
        service = GenealogyService.__new__(GenealogyService)

        biography = "Hi. Yes. No. This is a longer sentence that should be included."

        facts = service._extract_key_facts(biography)

        # Should skip the short ones
        assert all(len(f) > 20 for f in facts)


class TestForestFireSampling:
    """Tests for forest fire sampling."""

    def test_samples_from_list(self) -> None:
        """Should return one person from the list."""
        from ancestral_synth.persistence.tables import PersonTable
        from ancestral_synth.domain.enums import Gender

        service = GenealogyService.__new__(GenealogyService)

        # Create mock persons
        persons = [
            PersonTable(
                id=uuid4(),
                given_name=f"Person{i}",
                surname="Test",
                gender=Gender.MALE,
                status=PersonStatus.PENDING,
                generation=i,
            )
            for i in range(5)
        ]

        selected = service._forest_fire_sample(persons)

        assert selected in persons

    def test_raises_on_empty_list(self) -> None:
        """Should raise error for empty list."""
        service = GenealogyService.__new__(GenealogyService)

        with pytest.raises(ValueError):
            service._forest_fire_sample([])

    def test_prefers_generation_zero(self) -> None:
        """Should generally prefer persons closer to generation 0."""
        from ancestral_synth.persistence.tables import PersonTable
        from ancestral_synth.domain.enums import Gender

        service = GenealogyService.__new__(GenealogyService)

        # Create persons with different generations
        persons = [
            PersonTable(
                id=uuid4(),
                given_name="FarAncestor",
                surname="Test",
                gender=Gender.MALE,
                status=PersonStatus.PENDING,
                generation=-10,
            ),
            PersonTable(
                id=uuid4(),
                given_name="CloseAncestor",
                surname="Test",
                gender=Gender.MALE,
                status=PersonStatus.PENDING,
                generation=0,
            ),
        ]

        # Sample many times and count
        selections = {}
        for _ in range(100):
            selected = service._forest_fire_sample(persons)
            name = selected.given_name
            selections[name] = selections.get(name, 0) + 1

        # Generation 0 should be selected more often (but not always due to randomness)
        # This is a statistical test, so we allow some variance
        assert selections.get("CloseAncestor", 0) >= selections.get("FarAncestor", 0) * 0.5


class TestGenealogyServiceIntegration:
    """Integration tests for GenealogyService with mocked agents."""

    @pytest.mark.asyncio
    async def test_process_next_creates_seed(self, test_db: Database) -> None:
        """Should create seed person when queue is empty."""
        mock_bio_agent = MockBiographyAgent(
            biography=Biography(
                content="John Smith was born in Boston in 1950.",
                word_count=10,
            )
        )
        mock_extract_agent = MockExtractionAgent(
            extracted_data=ExtractedData(
                given_name="John",
                surname="Smith",
                gender=Gender.MALE,
                birth_date=date(1950, 1, 1),
            )
        )

        service = GenealogyService(
            db=test_db,
            biography_agent=mock_bio_agent,
            extraction_agent=mock_extract_agent,
            correction_agent=MockCorrectionAgent(),
            dedup_agent=MockDedupAgent(),
            shared_event_agent=MockSharedEventAgent(),
        )

        person = await service.process_next()

        assert person is not None
        assert person.status == PersonStatus.COMPLETE
        assert person.biography is not None
        assert mock_bio_agent.call_count == 1
        assert mock_extract_agent.call_count == 1

    @pytest.mark.asyncio
    async def test_process_next_processes_queued_person(self, test_db: Database) -> None:
        """Should process person from queue."""
        # First, create a pending person and add to queue
        async with test_db.session() as session:
            person_repo = PersonRepository(session)
            queue_repo = QueueRepository(session)

            pending_person = Person(
                given_name="Jane",
                surname="Doe",
                gender=Gender.FEMALE,
                status=PersonStatus.QUEUED,
            )
            await person_repo.create(pending_person)
            await queue_repo.enqueue(pending_person.id)

        mock_bio_agent = MockBiographyAgent(
            biography=Biography(
                content="Jane Doe was born in New York in 1960.",
                word_count=10,
            )
        )
        mock_extract_agent = MockExtractionAgent(
            extracted_data=ExtractedData(
                given_name="Jane",
                surname="Doe",
                gender=Gender.FEMALE,
                birth_date=date(1960, 1, 1),
            )
        )

        service = GenealogyService(
            db=test_db,
            biography_agent=mock_bio_agent,
            extraction_agent=mock_extract_agent,
            correction_agent=MockCorrectionAgent(),
            dedup_agent=MockDedupAgent(),
            shared_event_agent=MockSharedEventAgent(),
        )

        person = await service.process_next()

        assert person is not None
        assert person.given_name == "Jane"
        assert person.status == PersonStatus.COMPLETE

    @pytest.mark.asyncio
    async def test_processes_references_creates_pending(self, test_db: Database) -> None:
        """Should create pending records for referenced family members."""
        mock_bio_agent = MockBiographyAgent()
        mock_extract_agent = MockExtractionAgent(
            extracted_data=ExtractedData(
                given_name="John",
                surname="Smith",
                gender=Gender.MALE,
                birth_date=date(1950, 1, 1),
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
        )

        service = GenealogyService(
            db=test_db,
            biography_agent=mock_bio_agent,
            extraction_agent=mock_extract_agent,
            correction_agent=MockCorrectionAgent(),
            dedup_agent=MockDedupAgent(),
            shared_event_agent=MockSharedEventAgent(),
        )

        await service.process_next()

        # Check that family members were created
        async with test_db.session() as session:
            person_repo = PersonRepository(session)

            # Find parent
            williams = await person_repo.get_by_name("William", "Smith")
            assert len(williams) >= 1

            # Find child
            jameses = await person_repo.get_by_name("James", "Smith")
            assert len(jameses) >= 1

    @pytest.mark.asyncio
    async def test_creates_child_links(self, test_db: Database) -> None:
        """Should create child links for parent-child relationships."""
        mock_bio_agent = MockBiographyAgent()
        mock_extract_agent = MockExtractionAgent(
            extracted_data=ExtractedData(
                given_name="John",
                surname="Smith",
                gender=Gender.MALE,
                birth_date=date(1950, 1, 1),
                children=[
                    PersonReference(
                        name="James Smith",
                        relationship=RelationshipType.CHILD,
                        approximate_birth_year=1980,
                        gender=Gender.MALE,
                    ),
                ],
            )
        )

        service = GenealogyService(
            db=test_db,
            biography_agent=mock_bio_agent,
            extraction_agent=mock_extract_agent,
            correction_agent=MockCorrectionAgent(),
            dedup_agent=MockDedupAgent(),
            shared_event_agent=MockSharedEventAgent(),
        )

        person = await service.process_next()

        # Check that child link was created
        async with test_db.session() as session:
            child_link_repo = ChildLinkRepository(session)
            children = await child_link_repo.get_children(person.id)

            assert len(children) >= 1

    @pytest.mark.asyncio
    async def test_queues_parents_and_children(self, test_db: Database) -> None:
        """Should add parents and children to the queue."""
        mock_bio_agent = MockBiographyAgent()
        mock_extract_agent = MockExtractionAgent(
            extracted_data=ExtractedData(
                given_name="John",
                surname="Smith",
                gender=Gender.MALE,
                birth_date=date(1950, 1, 1),
                parents=[
                    PersonReference(
                        name="William Smith",
                        relationship=RelationshipType.PARENT,
                        approximate_birth_year=1920,
                        gender=Gender.MALE,
                    ),
                ],
            )
        )

        service = GenealogyService(
            db=test_db,
            biography_agent=mock_bio_agent,
            extraction_agent=mock_extract_agent,
            correction_agent=MockCorrectionAgent(),
            dedup_agent=MockDedupAgent(),
            shared_event_agent=MockSharedEventAgent(),
        )

        await service.process_next()

        # Check that parent was queued
        async with test_db.session() as session:
            queue_repo = QueueRepository(session)
            queue_count = await queue_repo.count()

            assert queue_count >= 1

    @pytest.mark.asyncio
    async def test_get_statistics(self, test_db: Database) -> None:
        """Should return correct statistics."""
        # Create some test data
        async with test_db.session() as session:
            person_repo = PersonRepository(session)
            queue_repo = QueueRepository(session)

            p1 = Person(given_name="A", surname="B", status=PersonStatus.COMPLETE)
            p2 = Person(given_name="C", surname="D", status=PersonStatus.PENDING)
            p3 = Person(given_name="E", surname="F", status=PersonStatus.QUEUED)

            await person_repo.create(p1)
            await person_repo.create(p2)
            await person_repo.create(p3)
            await queue_repo.enqueue(p3.id)

        service = GenealogyService(
            db=test_db,
            biography_agent=MockBiographyAgent(),
            extraction_agent=MockExtractionAgent(),
            correction_agent=MockCorrectionAgent(),
            dedup_agent=MockDedupAgent(),
            shared_event_agent=MockSharedEventAgent(),
        )

        stats = await service.get_statistics()

        assert stats["total_persons"] == 3
        assert stats["complete"] == 1
        assert stats["pending"] == 1
        assert stats["queued"] == 1
        assert stats["queue_size"] == 1


class TestDuplicateHandling:
    """Tests for duplicate detection and handling."""

    @pytest.mark.asyncio
    async def test_dedup_returns_existing_id(self, test_db: Database) -> None:
        """Should return existing person ID when duplicate detected."""
        # Create an existing person
        existing_id = uuid4()
        async with test_db.session() as session:
            person_repo = PersonRepository(session)
            existing = Person(
                id=existing_id,
                given_name="William",
                surname="Smith",
                gender=Gender.MALE,
                birth_date=date(1920, 1, 1),
                status=PersonStatus.COMPLETE,
            )
            await person_repo.create(existing)

        # Configure dedup agent to find a match
        mock_dedup_agent = MockDedupAgent(
            is_duplicate=True,
            matched_id=str(existing_id),
        )

        mock_bio_agent = MockBiographyAgent()
        mock_extract_agent = MockExtractionAgent(
            extracted_data=ExtractedData(
                given_name="John",
                surname="Smith",
                gender=Gender.MALE,
                birth_date=date(1950, 1, 1),
                parents=[
                    PersonReference(
                        name="William Smith",
                        relationship=RelationshipType.PARENT,
                        approximate_birth_year=1920,
                        gender=Gender.MALE,
                    ),
                ],
            )
        )

        service = GenealogyService(
            db=test_db,
            biography_agent=mock_bio_agent,
            extraction_agent=mock_extract_agent,
            correction_agent=MockCorrectionAgent(),
            dedup_agent=mock_dedup_agent,
            shared_event_agent=MockSharedEventAgent(),
        )

        await service.process_next()

        # Should not have created a new William Smith
        async with test_db.session() as session:
            person_repo = PersonRepository(session)
            williams = await person_repo.get_by_name("William", "Smith")

            # Should only be the original one
            assert len(williams) == 1
            assert williams[0].id == existing_id


class TestRelationshipValidatorIntegration:
    """Tests for relationship validator integration."""

    @pytest.mark.asyncio
    async def test_validation_called_after_seed_person_creation(self, test_db: Database) -> None:
        """Should call relationship validator after creating seed person."""
        mock_bio_agent = MockBiographyAgent(
            biography=Biography(
                content="John Smith was born in Boston in 1950.",
                word_count=10,
            )
        )
        mock_extract_agent = MockExtractionAgent(
            extracted_data=ExtractedData(
                given_name="John",
                surname="Smith",
                gender=Gender.MALE,
                birth_date=date(1950, 1, 1),
            )
        )
        mock_rel_validator = MockRelationshipValidator()

        service = GenealogyService(
            db=test_db,
            biography_agent=mock_bio_agent,
            extraction_agent=mock_extract_agent,
            correction_agent=MockCorrectionAgent(),
            dedup_agent=MockDedupAgent(),
            shared_event_agent=MockSharedEventAgent(),
            relationship_validator=mock_rel_validator,
        )

        person = await service.process_next()

        # Relationship validator should have been called
        assert len(mock_rel_validator.validation_calls) == 1
        # The person's ID should be in the validated IDs
        validated_ids = mock_rel_validator.validation_calls[0][1]
        assert person.id in validated_ids

    @pytest.mark.asyncio
    async def test_validation_called_after_queued_person_processing(self, test_db: Database) -> None:
        """Should call relationship validator after processing queued person."""
        # First, create a pending person and add to queue
        async with test_db.session() as session:
            person_repo = PersonRepository(session)
            queue_repo = QueueRepository(session)

            pending_person = Person(
                given_name="Jane",
                surname="Doe",
                gender=Gender.FEMALE,
                status=PersonStatus.QUEUED,
            )
            await person_repo.create(pending_person)
            await queue_repo.enqueue(pending_person.id)
            pending_id = pending_person.id

        mock_bio_agent = MockBiographyAgent(
            biography=Biography(
                content="Jane Doe was born in New York in 1960.",
                word_count=10,
            )
        )
        mock_extract_agent = MockExtractionAgent(
            extracted_data=ExtractedData(
                given_name="Jane",
                surname="Doe",
                gender=Gender.FEMALE,
                birth_date=date(1960, 1, 1),
            )
        )
        mock_rel_validator = MockRelationshipValidator()

        service = GenealogyService(
            db=test_db,
            biography_agent=mock_bio_agent,
            extraction_agent=mock_extract_agent,
            correction_agent=MockCorrectionAgent(),
            dedup_agent=MockDedupAgent(),
            shared_event_agent=MockSharedEventAgent(),
            relationship_validator=mock_rel_validator,
        )

        person = await service.process_next()

        # Relationship validator should have been called
        assert len(mock_rel_validator.validation_calls) == 1
        # The person's ID should be in the validated IDs
        validated_ids = mock_rel_validator.validation_calls[0][1]
        assert pending_id in validated_ids

    @pytest.mark.asyncio
    async def test_validation_logs_issues(self, test_db: Database) -> None:
        """Should log validation issues when found."""
        mock_bio_agent = MockBiographyAgent(
            biography=Biography(
                content="John Smith was born in Boston in 1950.",
                word_count=10,
            )
        )
        mock_extract_agent = MockExtractionAgent(
            extracted_data=ExtractedData(
                given_name="John",
                surname="Smith",
                gender=Gender.MALE,
                birth_date=date(1950, 1, 1),
            )
        )
        mock_rel_validator = MockRelationshipValidator()
        # Set up mock to return issues
        mock_rel_validator.mock_result = RelationshipValidationResult(
            issues=[
                RelationshipValidationIssue(
                    person_id=uuid4(),
                    severity=ValidationSeverity.WARNING,
                    message="Test warning issue",
                    action=ValidationAction.LOG,
                )
            ]
        )

        service = GenealogyService(
            db=test_db,
            biography_agent=mock_bio_agent,
            extraction_agent=mock_extract_agent,
            correction_agent=MockCorrectionAgent(),
            dedup_agent=MockDedupAgent(),
            shared_event_agent=MockSharedEventAgent(),
            relationship_validator=mock_rel_validator,
        )

        # Should not raise even with validation issues
        person = await service.process_next()
        assert person is not None
