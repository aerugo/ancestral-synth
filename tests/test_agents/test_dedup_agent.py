"""Tests for deduplication agent."""

import pytest
from uuid import uuid4

from ancestral_synth.agents.dedup_agent import (
    DedupAgent,
    DedupResult,
    heuristic_match_score,
)
from ancestral_synth.domain.enums import Gender, RelationshipType
from ancestral_synth.domain.models import PersonSummary


class TestHeuristicMatchScore:
    """Tests for heuristic_match_score function."""

    def test_exact_name_match(self) -> None:
        """Exact name match should score highly."""
        score = heuristic_match_score(
            "John Smith",
            1950,
            "John Smith",
            1950,
        )

        # Exact name + exact year = 1.0
        assert score == 1.0

    def test_exact_name_no_year(self) -> None:
        """Exact name without year data."""
        score = heuristic_match_score(
            "John Smith",
            None,
            "John Smith",
            None,
        )

        # Exact name = 0.5
        assert score == 0.5

    def test_different_names(self) -> None:
        """Different names should score low."""
        score = heuristic_match_score(
            "John Smith",
            1950,
            "Mary Johnson",
            1950,
        )

        # No name overlap, but year match gives 0.5
        # Score should be exactly 0.5 (year match only)
        assert score == 0.5

    def test_partial_name_overlap(self) -> None:
        """Partial name overlap should score medium."""
        score = heuristic_match_score(
            "John Smith",
            1950,
            "John Doe",  # Same first name
            1950,
        )

        assert 0.0 < score < 1.0

    def test_year_exact_match_adds_score(self) -> None:
        """Exact year match should add to score."""
        score_with_year = heuristic_match_score(
            "John Smith",
            1950,
            "John Smith",
            1950,
        )
        score_without_year = heuristic_match_score(
            "John Smith",
            None,
            "John Smith",
            None,
        )

        assert score_with_year > score_without_year

    def test_year_close_match(self) -> None:
        """Years within 2 should still add score."""
        score = heuristic_match_score(
            "John Smith",
            1950,
            "John Smith",
            1951,  # 1 year off
        )

        # Should still be high
        assert score >= 0.8

    def test_year_within_5(self) -> None:
        """Years within 5 should add some score."""
        score = heuristic_match_score(
            "John Smith",
            1950,
            "John Smith",
            1954,  # 4 years off
        )

        assert score >= 0.6

    def test_year_far_apart(self) -> None:
        """Years far apart should not add score."""
        score = heuristic_match_score(
            "John Smith",
            1950,
            "John Smith",
            1980,  # 30 years off
        )

        # Only name match, no year bonus
        assert score == 0.5

    def test_case_insensitive(self) -> None:
        """Name matching should be case insensitive."""
        score = heuristic_match_score(
            "JOHN SMITH",
            1950,
            "john smith",
            1950,
        )

        assert score == 1.0

    def test_score_capped_at_1(self) -> None:
        """Score should never exceed 1.0."""
        score = heuristic_match_score(
            "John Smith",
            1950,
            "John Smith",
            1950,
        )

        assert score <= 1.0

    def test_empty_names(self) -> None:
        """Should handle empty names gracefully."""
        score = heuristic_match_score(
            "",
            None,
            "",
            None,
        )

        # Empty names match exactly
        assert score >= 0.0


class TestDedupResult:
    """Tests for DedupResult model."""

    def test_create_duplicate_result(self) -> None:
        """Should create result indicating duplicate."""
        result = DedupResult(
            is_duplicate=True,
            matched_person_id=str(uuid4()),
            confidence=0.95,
            reasoning="Names and dates match exactly",
        )

        assert result.is_duplicate is True
        assert result.matched_person_id is not None
        assert result.confidence == 0.95

    def test_create_no_duplicate_result(self) -> None:
        """Should create result indicating no duplicate."""
        result = DedupResult(
            is_duplicate=False,
            matched_person_id=None,
            confidence=0.1,
            reasoning="No candidates match",
        )

        assert result.is_duplicate is False
        assert result.matched_person_id is None

    def test_confidence_range(self) -> None:
        """Confidence should be between 0 and 1."""
        # Should not raise
        DedupResult(
            is_duplicate=True,
            confidence=0.0,
            reasoning="test",
        )
        DedupResult(
            is_duplicate=True,
            confidence=1.0,
            reasoning="test",
        )

        # Should raise
        with pytest.raises(ValueError):
            DedupResult(
                is_duplicate=True,
                confidence=-0.1,
                reasoning="test",
            )

        with pytest.raises(ValueError):
            DedupResult(
                is_duplicate=True,
                confidence=1.5,
                reasoning="test",
            )


class TestDedupAgentEmptyCandidates:
    """Tests for DedupAgent with empty candidates."""

    @pytest.mark.asyncio
    async def test_empty_candidates_returns_no_match(self) -> None:
        """Should return no match when no candidates."""
        # Create agent without initializing pydantic-ai (to avoid API calls)
        agent = DedupAgent.__new__(DedupAgent)

        new_person = PersonSummary(
            id=uuid4(),
            full_name="John Smith",
            gender=Gender.MALE,
        )

        result = await agent.check_duplicate(new_person, [])

        assert result.is_duplicate is False
        assert result.matched_person_id is None
        assert result.confidence == 1.0


class TestDedupAgentPromptBuilding:
    """Tests for DedupAgent prompt construction."""

    def test_build_prompt_includes_new_person(self) -> None:
        """Should include new person details in prompt."""
        agent = DedupAgent.__new__(DedupAgent)

        new_person = PersonSummary(
            id=uuid4(),
            full_name="John Smith",
            gender=Gender.MALE,
            birth_year=1950,
            birth_place="Boston",
        )

        candidates = [
            PersonSummary(
                id=uuid4(),
                full_name="John A. Smith",
                gender=Gender.MALE,
                birth_year=1951,
            ),
        ]

        prompt = agent._build_prompt(new_person, candidates)

        assert "John Smith" in prompt
        assert "1950" in prompt
        assert "Boston" in prompt

    def test_build_prompt_includes_candidates(self) -> None:
        """Should include candidate details."""
        agent = DedupAgent.__new__(DedupAgent)

        new_person = PersonSummary(
            id=uuid4(),
            full_name="John Smith",
            gender=Gender.MALE,
        )

        candidate_id = uuid4()
        candidates = [
            PersonSummary(
                id=candidate_id,
                full_name="John A. Smith",
                gender=Gender.MALE,
                birth_year=1951,
                key_facts=["Lived in Boston"],
            ),
        ]

        prompt = agent._build_prompt(new_person, candidates)

        assert "John A. Smith" in prompt
        assert "1951" in prompt
        assert str(candidate_id) in prompt
        assert "Boston" in prompt

    def test_build_prompt_includes_relationship_context(self) -> None:
        """Should include relationship context when available."""
        agent = DedupAgent.__new__(DedupAgent)

        new_person = PersonSummary(
            id=uuid4(),
            full_name="John Smith",
            gender=Gender.MALE,
            relationship_to_subject=RelationshipType.PARENT,
        )

        candidates = []

        prompt = agent._build_prompt(new_person, candidates)

        assert "parent" in prompt.lower()
