"""Shared test fixtures and configuration."""

import asyncio
from collections.abc import AsyncGenerator, Generator
from datetime import date
from typing import Any
from uuid import uuid4

import pytest
import pytest_asyncio

from ancestral_synth.domain.enums import (
    EventType,
    Gender,
    NoteCategory,
    PersonStatus,
    RelationshipType,
)
from ancestral_synth.domain.models import (
    Biography,
    ChildLink,
    Event,
    ExtractedData,
    Note,
    Person,
    PersonReference,
    PersonSummary,
)
from ancestral_synth.persistence.database import Database


# ============================================================================
# Event Loop Configuration
# ============================================================================


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ============================================================================
# Database Fixtures
# ============================================================================


@pytest_asyncio.fixture
async def test_db() -> AsyncGenerator[Database, None]:
    """Provide an in-memory database for testing."""
    db = Database(":memory:")
    await db.init_db()
    yield db
    await db.close()


@pytest_asyncio.fixture
async def test_db_with_data(test_db: Database) -> AsyncGenerator[Database, None]:
    """Provide a database pre-populated with test data."""
    from ancestral_synth.persistence.repositories import (
        ChildLinkRepository,
        PersonRepository,
    )

    async with test_db.session() as session:
        person_repo = PersonRepository(session)
        child_link_repo = ChildLinkRepository(session)

        # Create a small family tree
        grandparent = Person(
            id=uuid4(),
            status=PersonStatus.COMPLETE,
            given_name="William",
            surname="Smith",
            gender=Gender.MALE,
            birth_date=date(1900, 5, 15),
            death_date=date(1975, 3, 20),
            birth_place="Boston, MA",
            generation=-2,
            biography="William Smith was born in 1900...",
        )

        parent = Person(
            id=uuid4(),
            status=PersonStatus.COMPLETE,
            given_name="John",
            surname="Smith",
            gender=Gender.MALE,
            birth_date=date(1930, 8, 10),
            death_date=date(2010, 12, 5),
            birth_place="Boston, MA",
            generation=-1,
            biography="John Smith was born in 1930...",
        )

        child = Person(
            id=uuid4(),
            status=PersonStatus.PENDING,
            given_name="James",
            surname="Smith",
            gender=Gender.MALE,
            birth_date=date(1960, 2, 28),
            birth_place="New York, NY",
            generation=0,
        )

        await person_repo.create(grandparent)
        await person_repo.create(parent)
        await person_repo.create(child)

        await child_link_repo.create(
            ChildLink(parent_id=grandparent.id, child_id=parent.id)
        )
        await child_link_repo.create(
            ChildLink(parent_id=parent.id, child_id=child.id)
        )

    yield test_db


# ============================================================================
# Domain Model Fixtures
# ============================================================================


@pytest.fixture
def sample_person() -> Person:
    """Provide a valid Person instance."""
    return Person(
        id=uuid4(),
        status=PersonStatus.COMPLETE,
        given_name="John",
        surname="Smith",
        gender=Gender.MALE,
        birth_date=date(1950, 6, 15),
        birth_place="Boston, Massachusetts",
        death_date=date(2020, 3, 10),
        death_place="New York, New York",
        biography="John Smith was born on June 15, 1950, in Boston...",
        generation=0,
    )


@pytest.fixture
def sample_person_pending() -> Person:
    """Provide a pending Person instance."""
    return Person(
        id=uuid4(),
        status=PersonStatus.PENDING,
        given_name="Jane",
        surname="Doe",
        gender=Gender.FEMALE,
        birth_date=date(1955, 9, 20),
        generation=0,
    )


@pytest.fixture
def sample_event() -> Event:
    """Provide a valid Event instance."""
    person_id = uuid4()
    return Event(
        id=uuid4(),
        event_type=EventType.BIRTH,
        event_date=date(1950, 6, 15),
        location="Boston, Massachusetts",
        description="Born at Massachusetts General Hospital",
        primary_person_id=person_id,
    )


@pytest.fixture
def sample_note() -> Note:
    """Provide a valid Note instance."""
    person_id = uuid4()
    return Note(
        id=uuid4(),
        person_id=person_id,
        category=NoteCategory.CAREER,
        content="Worked as a carpenter for 40 years",
        source="biography_extraction",
    )


@pytest.fixture
def sample_child_link() -> ChildLink:
    """Provide a valid ChildLink instance."""
    return ChildLink(
        parent_id=uuid4(),
        child_id=uuid4(),
    )


@pytest.fixture
def sample_person_reference() -> PersonReference:
    """Provide a valid PersonReference instance."""
    return PersonReference(
        name="Mary Johnson",
        relationship=RelationshipType.SPOUSE,
        approximate_birth_year=1952,
        gender=Gender.FEMALE,
        context="Married in 1975",
    )


@pytest.fixture
def sample_person_summary() -> PersonSummary:
    """Provide a valid PersonSummary instance."""
    return PersonSummary(
        id=uuid4(),
        full_name="John Smith",
        gender=Gender.MALE,
        birth_year=1950,
        death_year=2020,
        birth_place="Boston, MA",
        relationship_to_subject=RelationshipType.PARENT,
        key_facts=["Born in Boston", "Worked as a carpenter"],
    )


@pytest.fixture
def sample_biography() -> Biography:
    """Provide a valid Biography instance."""
    return Biography(
        content="John Smith was born on June 15, 1950, in Boston, Massachusetts. "
        "He grew up in a working-class family and learned carpentry from his father. "
        "In 1975, he married Mary Johnson, and they had three children together.",
        word_count=42,
    )


@pytest.fixture
def sample_extracted_data() -> ExtractedData:
    """Provide a complete ExtractedData instance."""
    return ExtractedData(
        given_name="John",
        surname="Smith",
        maiden_name=None,
        gender=Gender.MALE,
        birth_date=date(1950, 6, 15),
        birth_place="Boston, Massachusetts",
        death_date=date(2020, 3, 10),
        death_place="New York, New York",
        parents=[
            PersonReference(
                name="William Smith",
                relationship=RelationshipType.PARENT,
                approximate_birth_year=1920,
                gender=Gender.MALE,
            ),
            PersonReference(
                name="Elizabeth Smith",
                relationship=RelationshipType.PARENT,
                approximate_birth_year=1925,
                gender=Gender.FEMALE,
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
        spouses=[
            PersonReference(
                name="Mary Johnson",
                relationship=RelationshipType.SPOUSE,
                approximate_birth_year=1952,
                gender=Gender.FEMALE,
                context="Married in 1975",
            ),
        ],
        siblings=[],
        other_relatives=[],
        events=[],
        notes=["Worked as a master carpenter for 40 years"],
    )


@pytest.fixture
def sample_extracted_data_minimal() -> ExtractedData:
    """Provide a minimal ExtractedData instance."""
    return ExtractedData(
        given_name="John",
        surname="Smith",
        gender=Gender.MALE,
    )


# ============================================================================
# Mock Agent Fixtures
# ============================================================================


class MockBiographyAgent:
    """Mock biography agent for testing."""

    def __init__(self, biography: Biography | None = None) -> None:
        self.biography = biography or Biography(
            content="This is a test biography for testing purposes. "
            "The subject was born in 1950 and lived a full life.",
            word_count=20,
        )
        self.call_count = 0
        self.last_context: Any = None

    async def generate(self, context: Any) -> Any:
        """Generate a mock biography."""
        from ancestral_synth.agents.biography_agent import BiographyResult
        from ancestral_synth.utils.cost_tracker import TokenUsage

        self.call_count += 1
        self.last_context = context
        return BiographyResult(
            biography=self.biography,
            usage=TokenUsage(input_tokens=100, output_tokens=500),
        )


class MockExtractionAgent:
    """Mock extraction agent for testing."""

    def __init__(self, extracted_data: ExtractedData | None = None) -> None:
        self.extracted_data = extracted_data or ExtractedData(
            given_name="Test",
            surname="Person",
            gender=Gender.UNKNOWN,
        )
        self.call_count = 0
        self.last_biography: str | None = None

    async def extract(self, biography: str) -> Any:
        """Extract mock data from biography."""
        from ancestral_synth.agents.extraction_agent import ExtractionResult
        from ancestral_synth.utils.cost_tracker import TokenUsage

        self.call_count += 1
        self.last_biography = biography
        return ExtractionResult(
            data=self.extracted_data,
            usage=TokenUsage(input_tokens=200, output_tokens=300),
        )

    async def extract_with_hints(
        self,
        biography: str,
        expected_name: str | None = None,
        expected_birth_year: int | None = None,
    ) -> Any:
        """Extract mock data with hints."""
        from ancestral_synth.agents.extraction_agent import ExtractionResult
        from ancestral_synth.utils.cost_tracker import TokenUsage

        self.call_count += 1
        self.last_biography = biography
        return ExtractionResult(
            data=self.extracted_data,
            usage=TokenUsage(input_tokens=200, output_tokens=300),
        )


class MockDedupAgent:
    """Mock deduplication agent for testing."""

    def __init__(self, is_duplicate: bool = False, matched_id: str | None = None) -> None:
        self.is_duplicate = is_duplicate
        self.matched_id = matched_id
        self.call_count = 0

    async def check_duplicate(
        self,
        new_person: PersonSummary,
        candidates: list[PersonSummary],
    ) -> Any:
        """Check for mock duplicates."""
        from ancestral_synth.agents.dedup_agent import DedupResult, DedupResultWithUsage
        from ancestral_synth.utils.cost_tracker import TokenUsage

        self.call_count += 1
        return DedupResultWithUsage(
            result=DedupResult(
                is_duplicate=self.is_duplicate,
                matched_person_id=self.matched_id,
                confidence=0.95 if self.is_duplicate else 0.1,
                reasoning="Mock dedup result",
            ),
            usage=TokenUsage(input_tokens=150, output_tokens=50),
        )


@pytest.fixture
def mock_biography_agent() -> MockBiographyAgent:
    """Provide a mock biography agent."""
    return MockBiographyAgent()


@pytest.fixture
def mock_extraction_agent() -> MockExtractionAgent:
    """Provide a mock extraction agent."""
    return MockExtractionAgent()


@pytest.fixture
def mock_dedup_agent() -> MockDedupAgent:
    """Provide a mock dedup agent."""
    return MockDedupAgent()


@pytest.fixture
def mock_extraction_agent_with_data(
    sample_extracted_data: ExtractedData,
) -> MockExtractionAgent:
    """Provide a mock extraction agent with sample data."""
    return MockExtractionAgent(extracted_data=sample_extracted_data)
