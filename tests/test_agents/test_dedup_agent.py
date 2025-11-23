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

        result_with_usage = await agent.check_duplicate(new_person, [])

        # Result is now wrapped in DedupResultWithUsage
        assert result_with_usage.result.is_duplicate is False
        assert result_with_usage.result.matched_person_id is None
        assert result_with_usage.result.confidence == 1.0
        assert result_with_usage.usage.total_tokens == 0  # No LLM call for empty candidates


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

    def test_build_prompt_includes_mentioned_by(self) -> None:
        """Should include who mentioned this person when available."""
        agent = DedupAgent.__new__(DedupAgent)

        new_person = PersonSummary(
            id=uuid4(),
            full_name="John Smith",
            gender=Gender.MALE,
            mentioned_by="Mary Smith",
        )

        candidates = []

        prompt = agent._build_prompt(new_person, candidates)

        assert "Mary Smith" in prompt
        assert "mentioned" in prompt.lower()

    def test_build_prompt_includes_generation(self) -> None:
        """Should include generation info when available."""
        agent = DedupAgent.__new__(DedupAgent)

        new_person = PersonSummary(
            id=uuid4(),
            full_name="John Smith",
            gender=Gender.MALE,
            generation=3,
        )

        candidates = []

        prompt = agent._build_prompt(new_person, candidates)

        assert "generation" in prompt.lower()
        assert "3" in prompt

    def test_build_prompt_includes_first_degree_relations_new_person(self) -> None:
        """Should include first degree relations for new person."""
        agent = DedupAgent.__new__(DedupAgent)

        new_person = PersonSummary(
            id=uuid4(),
            full_name="John Smith",
            gender=Gender.MALE,
            parents=["George Smith", "Martha Smith"],
            children=["James Smith"],
            spouses=["Mary Jones"],
            siblings=["Bob Smith"],
        )

        candidates = []

        prompt = agent._build_prompt(new_person, candidates)

        assert "George Smith" in prompt
        assert "Martha Smith" in prompt
        assert "James Smith" in prompt
        assert "Mary Jones" in prompt
        assert "Bob Smith" in prompt

    def test_build_prompt_includes_second_degree_relations_new_person(self) -> None:
        """Should include second degree relations for new person."""
        agent = DedupAgent.__new__(DedupAgent)

        new_person = PersonSummary(
            id=uuid4(),
            full_name="John Smith",
            gender=Gender.MALE,
            grandparents=["Old George Smith"],
            grandchildren=["Baby Smith"],
        )

        candidates = []

        prompt = agent._build_prompt(new_person, candidates)

        assert "Old George Smith" in prompt
        assert "Baby Smith" in prompt

    def test_build_prompt_includes_candidate_relations(self) -> None:
        """Should include relations for candidates."""
        agent = DedupAgent.__new__(DedupAgent)

        new_person = PersonSummary(
            id=uuid4(),
            full_name="John Smith",
            gender=Gender.MALE,
        )

        candidates = [
            PersonSummary(
                id=uuid4(),
                full_name="John A. Smith",
                gender=Gender.MALE,
                parents=["George Smith"],
                spouses=["Mary Jones"],
            ),
        ]

        prompt = agent._build_prompt(new_person, candidates)

        # Candidate's relations should be in prompt
        assert "George Smith" in prompt
        assert "Mary Jones" in prompt

    def test_build_prompt_includes_more_key_facts(self) -> None:
        """Should include more than 3 key facts for candidates."""
        agent = DedupAgent.__new__(DedupAgent)

        new_person = PersonSummary(
            id=uuid4(),
            full_name="John Smith",
            gender=Gender.MALE,
        )

        candidates = [
            PersonSummary(
                id=uuid4(),
                full_name="John A. Smith",
                gender=Gender.MALE,
                key_facts=[
                    "Fact 1",
                    "Fact 2",
                    "Fact 3",
                    "Fact 4",
                    "Fact 5",
                ],
            ),
        ]

        prompt = agent._build_prompt(new_person, candidates)

        # Should include all 5 facts (not just 3)
        assert "Fact 1" in prompt
        assert "Fact 2" in prompt
        assert "Fact 3" in prompt
        assert "Fact 4" in prompt
        assert "Fact 5" in prompt

    def test_build_prompt_includes_candidate_generation(self) -> None:
        """Should include generation info for candidates."""
        agent = DedupAgent.__new__(DedupAgent)

        new_person = PersonSummary(
            id=uuid4(),
            full_name="John Smith",
            gender=Gender.MALE,
        )

        candidates = [
            PersonSummary(
                id=uuid4(),
                full_name="John A. Smith",
                gender=Gender.MALE,
                generation=4,
            ),
        ]

        prompt = agent._build_prompt(new_person, candidates)

        assert "generation" in prompt.lower()
        assert "4" in prompt

    def test_build_prompt_includes_biography_snippets_for_candidates(self) -> None:
        """Should include biography snippets showing name mentions."""
        agent = DedupAgent.__new__(DedupAgent)

        new_person = PersonSummary(
            id=uuid4(),
            full_name="Eleanor Smith",
            gender=Gender.FEMALE,
        )

        candidates = [
            PersonSummary(
                id=uuid4(),
                full_name="Eleanor A. Smith",
                gender=Gender.FEMALE,
                biography_snippets=[
                    "His wife Eleanor was known for her kindness",
                    "His sister Eleanor helped on the farm",
                ],
            ),
        ]

        prompt = agent._build_prompt(new_person, candidates)

        # Biography snippets should be in prompt
        assert "wife Eleanor" in prompt
        assert "sister Eleanor" in prompt

    def test_build_prompt_no_snippets_when_empty(self) -> None:
        """Should not include snippet section when no snippets available."""
        agent = DedupAgent.__new__(DedupAgent)

        new_person = PersonSummary(
            id=uuid4(),
            full_name="John Smith",
            gender=Gender.MALE,
        )

        candidates = [
            PersonSummary(
                id=uuid4(),
                full_name="John A. Smith",
                gender=Gender.MALE,
                biography_snippets=[],
            ),
        ]

        prompt = agent._build_prompt(new_person, candidates)

        assert "Biography mentions" not in prompt
