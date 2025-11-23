"""End-to-end tests for the genealogy generation workflow."""

import asyncio
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

import pytest

from ancestral_synth.domain.enums import Gender, PersonStatus, RelationshipType
from ancestral_synth.domain.models import (
    Biography,
    ExtractedData,
    Person,
    PersonReference,
)
from ancestral_synth.persistence.database import Database
from ancestral_synth.persistence.repositories import (
    ChildLinkRepository,
    PersonRepository,
    QueueRepository,
)
from ancestral_synth.services.genealogy_service import GenealogyService
from ancestral_synth.utils.rate_limiter import RateLimitConfig, RateLimiter


class MockBiographyAgent:
    """Mock biography agent for E2E testing."""

    def __init__(self) -> None:
        self.call_count = 0

    async def generate(self, context):
        from ancestral_synth.agents.biography_agent import BiographyResult
        from ancestral_synth.utils.cost_tracker import TokenUsage

        self.call_count += 1
        name = f"{context.given_name} {context.surname}"
        birth_year = context.approximate_birth_year or 1950

        content = (
            f"{name} was born around {birth_year} in {context.birth_place or 'Unknown'}. "
            f"They lived a full life with many accomplishments. "
            f"Their parents were William {context.surname} and Mary {context.surname}. "
            f"They had two children: James and Sarah."
        )
        biography = Biography(content=content, word_count=len(content.split()))
        return BiographyResult(
            biography=biography,
            usage=TokenUsage(input_tokens=100, output_tokens=500),
        )


class MockExtractionAgent:
    """Mock extraction agent for E2E testing."""

    def __init__(self) -> None:
        self.call_count = 0

    async def extract(self, biography: str):
        from ancestral_synth.agents.extraction_agent import ExtractionResult
        from ancestral_synth.utils.cost_tracker import TokenUsage

        self.call_count += 1
        data = self._create_extracted_data("Test", "Person", 1950)
        return ExtractionResult(
            data=data,
            usage=TokenUsage(input_tokens=200, output_tokens=300),
        )

    async def extract_with_hints(
        self,
        biography: str,
        expected_name: str | None = None,
        expected_birth_year: int | None = None,
    ):
        from ancestral_synth.agents.extraction_agent import ExtractionResult
        from ancestral_synth.utils.cost_tracker import TokenUsage

        self.call_count += 1
        given_name = "Test"
        surname = "Person"
        if expected_name:
            parts = expected_name.split()
            given_name = parts[0]
            surname = parts[-1] if len(parts) > 1 else "Unknown"
        data = self._create_extracted_data(
            given_name, surname, expected_birth_year or 1950
        )
        return ExtractionResult(
            data=data,
            usage=TokenUsage(input_tokens=200, output_tokens=300),
        )

    def _create_extracted_data(
        self, given_name: str, surname: str, birth_year: int
    ) -> ExtractedData:
        return ExtractedData(
            given_name=given_name,
            surname=surname,
            gender=Gender.MALE,
            birth_date=date(birth_year, 6, 15),
            birth_place="Boston, MA",
            death_date=date(birth_year + 70, 3, 10),
            death_place="New York, NY",
            parents=[
                PersonReference(
                    name=f"William {surname}",
                    relationship=RelationshipType.PARENT,
                    approximate_birth_year=birth_year - 30,
                    gender=Gender.MALE,
                ),
                PersonReference(
                    name=f"Mary {surname}",
                    relationship=RelationshipType.PARENT,
                    approximate_birth_year=birth_year - 28,
                    gender=Gender.FEMALE,
                ),
            ],
            children=[
                PersonReference(
                    name=f"James {surname}",
                    relationship=RelationshipType.CHILD,
                    approximate_birth_year=birth_year + 25,
                    gender=Gender.MALE,
                ),
            ],
            spouses=[],
            siblings=[],
            other_relatives=[],
            events=[],
            notes=["A notable person in the community"],
        )


class MockDedupAgent:
    """Mock dedup agent for E2E testing."""

    def __init__(self) -> None:
        self.call_count = 0

    async def check_duplicate(self, new_person, candidates):
        from ancestral_synth.agents.dedup_agent import DedupResult, DedupResultWithUsage
        from ancestral_synth.utils.cost_tracker import TokenUsage

        self.call_count += 1
        return DedupResultWithUsage(
            result=DedupResult(
                is_duplicate=False,
                matched_person_id=None,
                confidence=0.1,
                reasoning="No duplicates found",
            ),
            usage=TokenUsage(input_tokens=150, output_tokens=50),
        )


class TestGenerationWorkflowE2E:
    """End-to-end tests for the complete generation workflow."""

    @pytest.mark.asyncio
    async def test_create_seed_person(self) -> None:
        """Should create a seed person with biography and extracted data."""
        async with Database(":memory:") as db:
            biography_agent = MockBiographyAgent()
            extraction_agent = MockExtractionAgent()
            dedup_agent = MockDedupAgent()

            # Use fast rate limiter for tests
            rate_limiter = RateLimiter(RateLimitConfig(requests_per_minute=6000))

            service = GenealogyService(
                db=db,
                biography_agent=biography_agent,
                extraction_agent=extraction_agent,
                dedup_agent=dedup_agent,
                rate_limiter=rate_limiter,
            )

            # Process should create seed since no queue/pending
            person = await service.process_next()

            assert person is not None
            assert person.status == PersonStatus.COMPLETE
            assert person.biography is not None
            assert len(person.biography) > 0
            assert biography_agent.call_count == 1
            assert extraction_agent.call_count == 1

    @pytest.mark.asyncio
    async def test_process_queued_person(self) -> None:
        """Should process a queued person with context from relatives."""
        async with Database(":memory:") as db:
            biography_agent = MockBiographyAgent()
            extraction_agent = MockExtractionAgent()
            dedup_agent = MockDedupAgent()
            rate_limiter = RateLimiter(RateLimitConfig(requests_per_minute=6000))

            # Pre-create a pending person
            person_id = uuid4()
            async with db.session() as session:
                repo = PersonRepository(session)
                await repo.create(
                    Person(
                        id=person_id,
                        status=PersonStatus.PENDING,
                        given_name="James",
                        surname="Smith",
                        gender=Gender.MALE,
                        birth_date=date(1950, 1, 1),
                        generation=0,
                    )
                )

            service = GenealogyService(
                db=db,
                biography_agent=biography_agent,
                extraction_agent=extraction_agent,
                dedup_agent=dedup_agent,
                rate_limiter=rate_limiter,
            )

            # Process the pending person
            person = await service.process_next()

            assert person is not None
            assert person.given_name == "James"
            assert person.status == PersonStatus.COMPLETE
            assert person.biography is not None
            assert biography_agent.call_count == 1
            assert extraction_agent.call_count == 1

    @pytest.mark.asyncio
    async def test_creates_pending_relatives(self) -> None:
        """Should create pending records for referenced relatives."""
        async with Database(":memory:") as db:
            biography_agent = MockBiographyAgent()
            extraction_agent = MockExtractionAgent()
            dedup_agent = MockDedupAgent()
            rate_limiter = RateLimiter(RateLimitConfig(requests_per_minute=6000))

            service = GenealogyService(
                db=db,
                biography_agent=biography_agent,
                extraction_agent=extraction_agent,
                dedup_agent=dedup_agent,
                rate_limiter=rate_limiter,
            )

            # Create seed person (which has parents and children in mock data)
            person = await service.process_next()

            # Check that pending relatives were created
            async with db.session() as session:
                repo = PersonRepository(session)
                all_persons = await repo.get_by_status(PersonStatus.PENDING)
                queued = await repo.get_by_status(PersonStatus.QUEUED)

                # Should have created parents and children as pending/queued
                total_relatives = len(all_persons) + len(queued)
                assert total_relatives >= 2  # At least parents or children

    @pytest.mark.asyncio
    async def test_creates_child_links(self) -> None:
        """Should create parent-child relationships."""
        async with Database(":memory:") as db:
            biography_agent = MockBiographyAgent()
            extraction_agent = MockExtractionAgent()
            dedup_agent = MockDedupAgent()
            rate_limiter = RateLimiter(RateLimitConfig(requests_per_minute=6000))

            service = GenealogyService(
                db=db,
                biography_agent=biography_agent,
                extraction_agent=extraction_agent,
                dedup_agent=dedup_agent,
                rate_limiter=rate_limiter,
            )

            # Create seed person
            person = await service.process_next()

            # Check for child links
            async with db.session() as session:
                link_repo = ChildLinkRepository(session)

                # Seed person should have parents linked
                parents = await link_repo.get_parents(person.id)
                children = await link_repo.get_children(person.id)

                # Should have at least one relationship
                assert len(parents) + len(children) >= 1

    @pytest.mark.asyncio
    async def test_multi_generation_workflow(self) -> None:
        """Should process multiple generations correctly."""
        async with Database(":memory:") as db:
            biography_agent = MockBiographyAgent()
            extraction_agent = MockExtractionAgent()
            dedup_agent = MockDedupAgent()
            rate_limiter = RateLimiter(RateLimitConfig(requests_per_minute=6000))

            service = GenealogyService(
                db=db,
                biography_agent=biography_agent,
                extraction_agent=extraction_agent,
                dedup_agent=dedup_agent,
                rate_limiter=rate_limiter,
            )

            # Process multiple rounds
            for _ in range(3):
                await service.process_next()

            # Check statistics
            stats = await service.get_statistics()

            assert stats["total_persons"] >= 3
            assert stats["complete"] >= 1


class TestStatisticsE2E:
    """End-to-end tests for statistics gathering."""

    @pytest.mark.asyncio
    async def test_get_statistics(self) -> None:
        """Should return accurate statistics."""
        async with Database(":memory:") as db:
            # Pre-populate with data
            async with db.session() as session:
                repo = PersonRepository(session)
                queue_repo = QueueRepository(session)

                for i in range(3):
                    person = Person(
                        id=uuid4(),
                        status=PersonStatus.COMPLETE,
                        given_name=f"Person{i}",
                        surname="Test",
                        gender=Gender.MALE,
                        generation=0,
                    )
                    await repo.create(person)

                pending = Person(
                    id=uuid4(),
                    status=PersonStatus.PENDING,
                    given_name="Pending",
                    surname="Person",
                    gender=Gender.FEMALE,
                    generation=1,
                )
                await repo.create(pending)

            biography_agent = MockBiographyAgent()
            extraction_agent = MockExtractionAgent()
            dedup_agent = MockDedupAgent()
            rate_limiter = RateLimiter(RateLimitConfig(requests_per_minute=6000))

            service = GenealogyService(
                db=db,
                biography_agent=biography_agent,
                extraction_agent=extraction_agent,
                dedup_agent=dedup_agent,
                rate_limiter=rate_limiter,
            )

            stats = await service.get_statistics()

            assert stats["total_persons"] == 4
            assert stats["complete"] == 3
            assert stats["pending"] == 1


class TestRateLimitingE2E:
    """End-to-end tests for rate limiting behavior."""

    @pytest.mark.asyncio
    async def test_rate_limiting_applied(self) -> None:
        """Should apply rate limiting to LLM calls."""
        import time

        async with Database(":memory:") as db:
            biography_agent = MockBiographyAgent()
            extraction_agent = MockExtractionAgent()
            dedup_agent = MockDedupAgent()

            # Very slow rate limiter - 1 request per second
            rate_limiter = RateLimiter(RateLimitConfig(requests_per_minute=60))

            service = GenealogyService(
                db=db,
                biography_agent=biography_agent,
                extraction_agent=extraction_agent,
                dedup_agent=dedup_agent,
                rate_limiter=rate_limiter,
            )

            # Time a single process (which makes 2 LLM calls)
            start = time.time()
            await service.process_next()
            elapsed = time.time() - start

            # With 60 RPM (1/sec), 2 calls should take ~1 second minimum
            assert elapsed >= 0.9  # Allow some tolerance
            assert biography_agent.call_count == 1
            assert extraction_agent.call_count == 1


class TestErrorHandlingE2E:
    """End-to-end tests for error handling."""

    @pytest.mark.asyncio
    async def test_handles_invalid_person_reference(self) -> None:
        """Should handle invalid person references gracefully."""

        class ExtractionAgentWithInvalidRef:
            async def extract(self, biography: str):
                from ancestral_synth.agents.extraction_agent import ExtractionResult
                from ancestral_synth.utils.cost_tracker import TokenUsage

                data = ExtractedData(
                    given_name="Test",
                    surname="Person",
                    gender=Gender.MALE,
                    parents=[
                        PersonReference(
                            name="",  # Invalid empty name
                            relationship=RelationshipType.PARENT,
                            gender=Gender.MALE,
                        ),
                    ],
                )
                return ExtractionResult(
                    data=data,
                    usage=TokenUsage(input_tokens=200, output_tokens=300),
                )

            async def extract_with_hints(self, biography, expected_name=None, expected_birth_year=None):
                return await self.extract(biography)

        async with Database(":memory:") as db:
            biography_agent = MockBiographyAgent()
            extraction_agent = ExtractionAgentWithInvalidRef()
            dedup_agent = MockDedupAgent()
            rate_limiter = RateLimiter(RateLimitConfig(requests_per_minute=6000))

            service = GenealogyService(
                db=db,
                biography_agent=biography_agent,
                extraction_agent=extraction_agent,
                dedup_agent=dedup_agent,
                rate_limiter=rate_limiter,
            )

            # Should not raise - invalid references should be skipped
            person = await service.process_next()
            assert person is not None
            assert person.status == PersonStatus.COMPLETE
