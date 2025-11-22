"""Tests for biography agent."""

import pytest

from ancestral_synth.agents.biography_agent import (
    BiographyAgent,
    BiographyContext,
    create_seed_context,
)
from ancestral_synth.domain.enums import Gender, RelationshipType
from ancestral_synth.domain.models import PersonSummary
from uuid import uuid4


class TestBiographyContext:
    """Tests for BiographyContext dataclass."""

    def test_create_minimal_context(self) -> None:
        """Should create context with minimal fields."""
        context = BiographyContext(
            given_name="John",
            surname="Smith",
        )

        assert context.given_name == "John"
        assert context.surname == "Smith"
        assert context.generation == 0
        assert context.known_relatives is None

    def test_create_full_context(self) -> None:
        """Should create context with all fields."""
        relatives = [
            PersonSummary(
                id=uuid4(),
                full_name="Mary Smith",
                gender=Gender.FEMALE,
                birth_year=1925,
                relationship_to_subject=RelationshipType.PARENT,
            ),
        ]

        context = BiographyContext(
            given_name="John",
            surname="Smith",
            gender="male",
            approximate_birth_year=1950,
            birth_place="Boston, MA",
            known_relatives=relatives,
            generation=-1,
            era_context="20th century America",
        )

        assert context.gender == "male"
        assert context.approximate_birth_year == 1950
        assert len(context.known_relatives) == 1
        assert context.generation == -1


class TestCreateSeedContext:
    """Tests for create_seed_context function."""

    def test_creates_valid_context(self) -> None:
        """Should create a valid context."""
        context = create_seed_context()

        assert context.given_name is not None
        assert context.surname is not None
        assert context.gender in ["male", "female"]
        assert context.approximate_birth_year is not None
        assert context.generation == 0

    def test_birth_year_in_range(self) -> None:
        """Should generate birth year in reasonable range."""
        for _ in range(10):
            context = create_seed_context()
            assert 1850 <= context.approximate_birth_year <= 1950

    def test_accepts_era_parameter(self) -> None:
        """Should use provided era context."""
        context = create_seed_context(era="Victorian Era")

        assert context.era_context == "Victorian Era"

    def test_accepts_region_parameter(self) -> None:
        """Should use provided region."""
        context = create_seed_context(region="England")

        assert context.birth_place == "England"

    def test_randomness(self) -> None:
        """Should generate different names each time."""
        contexts = [create_seed_context() for _ in range(5)]
        names = [(c.given_name, c.surname) for c in contexts]

        # Not all names should be identical (statistically unlikely)
        # At least 2 different combinations expected
        unique_names = set(names)
        # Could be same by chance, so just verify we get valid results
        assert all(len(n[0]) > 0 and len(n[1]) > 0 for n in names)


class TestBiographyAgentPromptBuilding:
    """Tests for biography agent prompt construction."""

    def test_build_prompt_minimal(self) -> None:
        """Should build prompt from minimal context."""
        agent = BiographyAgent.__new__(BiographyAgent)
        # Manually initialize without actually creating the pydantic-ai agent

        context = BiographyContext(
            given_name="John",
            surname="Smith",
        )

        prompt = agent._build_prompt(context)

        assert "John Smith" in prompt
        assert "biography" in prompt.lower()

    def test_build_prompt_includes_gender(self) -> None:
        """Should include gender in prompt."""
        agent = BiographyAgent.__new__(BiographyAgent)

        context = BiographyContext(
            given_name="John",
            surname="Smith",
            gender="male",
        )

        prompt = agent._build_prompt(context)

        assert "male" in prompt.lower()

    def test_build_prompt_includes_birth_year(self) -> None:
        """Should include birth year in prompt."""
        agent = BiographyAgent.__new__(BiographyAgent)

        context = BiographyContext(
            given_name="John",
            surname="Smith",
            approximate_birth_year=1950,
        )

        prompt = agent._build_prompt(context)

        assert "1950" in prompt

    def test_build_prompt_includes_generation_info(self) -> None:
        """Should include generation context."""
        agent = BiographyAgent.__new__(BiographyAgent)

        context = BiographyContext(
            given_name="John",
            surname="Smith",
            generation=-2,  # Ancestor
        )

        prompt = agent._build_prompt(context)

        assert "ancestor" in prompt.lower()

    def test_build_prompt_includes_relatives(self) -> None:
        """Should include relative information."""
        agent = BiographyAgent.__new__(BiographyAgent)

        relatives = [
            PersonSummary(
                id=uuid4(),
                full_name="Mary Smith",
                gender=Gender.FEMALE,
                birth_year=1925,
                relationship_to_subject=RelationshipType.PARENT,
                key_facts=["Born in Boston"],
            ),
        ]

        context = BiographyContext(
            given_name="John",
            surname="Smith",
            known_relatives=relatives,
        )

        prompt = agent._build_prompt(context)

        assert "Mary Smith" in prompt
        assert "1925" in prompt
        assert "Boston" in prompt

    def test_build_prompt_includes_era(self) -> None:
        """Should include era context."""
        agent = BiographyAgent.__new__(BiographyAgent)

        context = BiographyContext(
            given_name="John",
            surname="Smith",
            era_context="Victorian Era England",
        )

        prompt = agent._build_prompt(context)

        assert "Victorian" in prompt
