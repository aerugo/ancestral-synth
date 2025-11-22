"""Deduplication agent for identifying duplicate person records."""

from pydantic import BaseModel, Field
from pydantic_ai import Agent

from ancestral_synth.config import settings
from ancestral_synth.domain.models import PersonSummary
from ancestral_synth.utils.retry import llm_retry


class DedupResult(BaseModel):
    """Result of a deduplication check."""

    is_duplicate: bool = Field(description="Whether the new person is a duplicate of an existing one")
    matched_person_id: str | None = Field(
        default=None,
        description="ID of the matched person if a duplicate was found",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score (0-1) in the deduplication decision",
    )
    reasoning: str = Field(description="Explanation of the deduplication decision")


DEDUP_SYSTEM_PROMPT = """You are an expert genealogist specializing in record deduplication.

Your task is to determine if a newly mentioned person is the same as an existing person in the database.

Guidelines for matching:
1. Same name with matching birth year (within 2 years) = likely match
2. Similar name (nicknames, maiden names) with matching details = possible match
3. Family relationships should be consistent
4. Consider historical naming patterns (Jr., Sr., III, nicknames)
5. Gender must match for a duplicate
6. If birth years differ by more than 5 years, probably not a match
7. Consider that the same person might be mentioned with different levels of detail

Be conservative - only mark as duplicate if you're confident it's the same person.
False merges are worse than false non-merges."""


class DedupAgent:
    """Agent for checking if a person is a duplicate."""

    def __init__(self, model: str | None = None) -> None:
        """Initialize the dedup agent.

        Args:
            model: The model to use.
        """
        model_name = model or f"{settings.llm_provider}:{settings.llm_model}"

        self._agent = Agent(
            model_name,
            output_type=DedupResult,
            system_prompt=DEDUP_SYSTEM_PROMPT,
        )

    async def check_duplicate(
        self,
        new_person: PersonSummary,
        candidates: list[PersonSummary],
    ) -> DedupResult:
        """Check if a new person matches any existing candidates.

        Args:
            new_person: The newly mentioned person.
            candidates: Existing people who might be duplicates.

        Returns:
            Result indicating if a duplicate was found.
        """
        if not candidates:
            return DedupResult(
                is_duplicate=False,
                matched_person_id=None,
                confidence=1.0,
                reasoning="No candidates to compare against",
            )

        prompt = self._build_prompt(new_person, candidates)
        return await self._run_llm(prompt)

    @llm_retry()
    async def _run_llm(self, prompt: str) -> DedupResult:
        """Run LLM with retry logic."""
        result = await self._agent.run(prompt)
        return result.data

    def _build_prompt(
        self,
        new_person: PersonSummary,
        candidates: list[PersonSummary],
    ) -> str:
        """Build the comparison prompt."""
        parts = [
            "Determine if this newly mentioned person matches any existing records.",
            "",
            "NEW PERSON:",
            f"  Name: {new_person.full_name}",
            f"  Gender: {new_person.gender}",
        ]

        if new_person.birth_year:
            parts.append(f"  Birth year: ~{new_person.birth_year}")
        if new_person.death_year:
            parts.append(f"  Death year: ~{new_person.death_year}")
        if new_person.birth_place:
            parts.append(f"  Birth place: {new_person.birth_place}")
        if new_person.relationship_to_subject:
            parts.append(f"  Relationship context: {new_person.relationship_to_subject}")
        if new_person.key_facts:
            parts.append("  Key facts:")
            for fact in new_person.key_facts:
                parts.append(f"    - {fact}")

        parts.append("")
        parts.append("EXISTING CANDIDATES:")

        for i, candidate in enumerate(candidates, 1):
            parts.append(f"\n  Candidate {i} (ID: {candidate.id}):")
            parts.append(f"    Name: {candidate.full_name}")
            parts.append(f"    Gender: {candidate.gender}")
            if candidate.birth_year:
                parts.append(f"    Birth year: {candidate.birth_year}")
            if candidate.death_year:
                parts.append(f"    Death year: {candidate.death_year}")
            if candidate.birth_place:
                parts.append(f"    Birth place: {candidate.birth_place}")
            if candidate.key_facts:
                parts.append("    Key facts:")
                for fact in candidate.key_facts[:3]:
                    parts.append(f"      - {fact}")

        parts.append("")
        parts.append("Is the new person a duplicate of any candidate? If so, which one?")

        return "\n".join(parts)


def heuristic_match_score(
    new_name: str,
    new_birth_year: int | None,
    candidate_name: str,
    candidate_birth_year: int | None,
) -> float:
    """Calculate a heuristic match score between two people.

    Args:
        new_name: Name of the new person.
        new_birth_year: Birth year of the new person.
        candidate_name: Name of the candidate.
        candidate_birth_year: Birth year of the candidate.

    Returns:
        Score from 0.0 (no match) to 1.0 (perfect match).
    """
    score = 0.0

    # Name comparison
    new_parts = set(new_name.lower().split())
    candidate_parts = set(candidate_name.lower().split())

    # Exact name match
    if new_name.lower() == candidate_name.lower():
        score += 0.5
    # Partial name overlap
    elif new_parts & candidate_parts:
        overlap = len(new_parts & candidate_parts) / max(len(new_parts), len(candidate_parts))
        score += 0.3 * overlap

    # Birth year comparison
    if new_birth_year and candidate_birth_year:
        year_diff = abs(new_birth_year - candidate_birth_year)
        if year_diff == 0:
            score += 0.5
        elif year_diff <= 2:
            score += 0.4
        elif year_diff <= 5:
            score += 0.2

    return min(score, 1.0)
