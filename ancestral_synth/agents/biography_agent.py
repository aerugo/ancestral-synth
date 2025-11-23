"""Biography generation agent using Pydantic AI."""

from dataclasses import dataclass

from pydantic_ai import Agent

from ancestral_synth.config import get_pydantic_ai_provider, settings
from ancestral_synth.domain.models import Biography, PersonSummary
from ancestral_synth.utils.cost_tracker import TokenUsage
from ancestral_synth.utils.retry import llm_retry


@dataclass
class BiographyResult:
    """Result of biography generation including token usage."""

    biography: Biography
    usage: TokenUsage


@dataclass
class BiographyContext:
    """Context for biography generation."""

    # Target person info
    given_name: str
    surname: str
    gender: str | None = None
    approximate_birth_year: int | None = None
    birth_place: str | None = None

    # Family context
    known_relatives: list[PersonSummary] | None = None

    # Generation parameters
    generation: int = 0  # 0 = seed, negative = ancestor, positive = descendant

    # Historical era hints
    era_context: str | None = None


# System prompt for biography generation
BIOGRAPHY_SYSTEM_PROMPT = """You are a creative writer specializing in detailed, historically-plausible fictional biographies.

Your task is to write a ~1000-word biography for a person in a fictional genealogical dataset.

Guidelines:
1. Create a rich, detailed life story with specific dates, places, and events
2. Include birth date, death date (if applicable), marriage(s), children, and major life events
3. Reference family members by their full names
4. Include occupations, education, migrations, and historical context appropriate to the era
5. Make the story feel authentic with period-appropriate details
6. Include interesting anecdotes and personality traits
7. Ensure genealogical plausibility (reasonable ages for marriage, childbearing, etc.)

The biography should be written in third person and read like a well-researched family history entry."""


class BiographyAgent:
    """Agent for generating detailed biographies."""

    def __init__(self, model: str | None = None) -> None:
        """Initialize the biography agent.

        Args:
            model: The model to use (e.g., "openai:gpt-4o-mini", "anthropic:claude-3-haiku").
                   Defaults to settings.llm_model.
        """
        model_name = model or f"{get_pydantic_ai_provider()}:{settings.llm_model}"

        self._agent = Agent(
            model_name,
            output_type=Biography,
            system_prompt=BIOGRAPHY_SYSTEM_PROMPT,
        )

    @llm_retry()
    async def generate(self, context: BiographyContext) -> BiographyResult:
        """Generate a biography for a person.

        Args:
            context: The context for biography generation.

        Returns:
            A BiographyResult with biography content and token usage.
        """
        import time

        from ancestral_synth.utils.timing import verbose_log

        prompt = self._build_prompt(context)
        verbose_log(f"      [biography] Prompt length: {len(prompt)} chars")

        start = time.perf_counter()
        result = await self._agent.run(prompt)
        elapsed = time.perf_counter() - start
        verbose_log(f"      [biography] pydantic_ai.run() completed in {elapsed:.1f}s")

        # Extract token usage from result
        usage_data = result.usage()
        usage = TokenUsage(
            input_tokens=usage_data.request_tokens or 0,
            output_tokens=usage_data.response_tokens or 0,
        )

        return BiographyResult(biography=result.output, usage=usage)

    def _build_prompt(self, context: BiographyContext) -> str:
        """Build the user prompt from context."""
        parts = [
            f"Write a detailed biography (~{settings.biography_word_count} words) for:",
            f"Name: {context.given_name} {context.surname}",
        ]

        if context.gender:
            parts.append(f"Gender: {context.gender}")

        if context.approximate_birth_year:
            parts.append(f"Approximate birth year: {context.approximate_birth_year}")

        if context.birth_place:
            parts.append(f"Birth place: {context.birth_place}")

        if context.era_context:
            parts.append(f"Historical era: {context.era_context}")

        if context.generation != 0:
            direction = "ancestor" if context.generation < 0 else "descendant"
            parts.append(f"This person is a {direction} (generation {context.generation})")

        if context.known_relatives:
            parts.append("\nKnown family members (ensure consistency with these facts):")
            for relative in context.known_relatives:
                rel_info = f"- {relative.full_name}"
                if relative.relationship_to_subject:
                    rel_info += f" ({relative.relationship_to_subject})"
                if relative.birth_year:
                    rel_info += f", born ~{relative.birth_year}"
                if relative.death_year:
                    rel_info += f", died ~{relative.death_year}"
                if relative.birth_place:
                    rel_info += f", from {relative.birth_place}"
                parts.append(rel_info)
                for fact in relative.key_facts[:3]:
                    parts.append(f"  • {fact}")

        parts.append("\nWrite a complete, engaging biography following the guidelines.")

        return "\n".join(parts)


# Convenience function for generating seed person context
def create_seed_context(
    era: str = "19th-20th century",
    region: str = "United States",
) -> BiographyContext:
    """Create a context for generating a seed person (no known relatives).

    Args:
        era: The historical era for the person.
        region: The geographic region.

    Returns:
        A context for seed person generation.
    """
    import random

    # Common historical names
    male_names = [
        "William", "John", "James", "George", "Charles", "Robert", "Joseph",
        "Thomas", "Henry", "Edward", "Samuel", "Benjamin", "Frederick", "Albert",
    ]
    female_names = [
        "Mary", "Elizabeth", "Margaret", "Anna", "Sarah", "Emma", "Catherine",
        "Martha", "Dorothy", "Helen", "Ruth", "Florence", "Lillian", "Grace",
    ]
    surnames = [
        "Smith", "Johnson", "Williams", "Brown", "Jones", "Miller", "Davis",
        "Wilson", "Anderson", "Taylor", "Thomas", "Moore", "Jackson", "Martin",
        "Thompson", "White", "Harris", "Clark", "Lewis", "Robinson", "Walker",
    ]

    is_male = random.random() > 0.5
    given_name = random.choice(male_names if is_male else female_names)
    surname = random.choice(surnames)

    # Random birth year in a reasonable range
    birth_year = random.randint(1850, 1950)

    return BiographyContext(
        given_name=given_name,
        surname=surname,
        gender="male" if is_male else "female",
        approximate_birth_year=birth_year,
        generation=0,
        era_context=era,
        birth_place=region,
    )
