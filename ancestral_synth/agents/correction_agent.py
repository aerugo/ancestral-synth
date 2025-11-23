"""Correction agent for fixing validation errors in extracted data."""

from dataclasses import dataclass

from pydantic_ai import Agent

from ancestral_synth.config import get_pydantic_ai_provider, settings
from ancestral_synth.domain.models import ExtractedData
from ancestral_synth.utils.cost_tracker import TokenUsage
from ancestral_synth.utils.retry import llm_retry


@dataclass
class CorrectionResult:
    """Result of data correction including token usage."""

    data: ExtractedData
    usage: TokenUsage

CORRECTION_SYSTEM_PROMPT = """You are an expert genealogist specializing in data validation and correction.

Your task is to fix inconsistencies in extracted genealogical data based on validation errors.

Guidelines:
1. Carefully analyze each validation error and determine the most likely correct value
2. Use the original biography text as the source of truth
3. When dates/years conflict, prefer values that make the genealogy internally consistent
4. Common issues to fix:
   - Events with dates before the person's birth (often misattributed events from relatives)
   - Events with dates after the person's death
   - Parent ages that are implausible (too young or too old at child's birth)
   - Death dates before birth dates
5. If an event clearly belongs to a different person mentioned in the biography, remove it
6. If a date was extracted incorrectly, correct it based on context
7. Maintain all valid data - only modify or remove data that causes validation errors
8. Be conservative - prefer removing invalid data over guessing if the correct value is unclear

Return the corrected data following the provided schema exactly."""


class CorrectionAgent:
    """Agent for correcting validation errors in extracted genealogical data."""

    def __init__(self, model: str | None = None) -> None:
        """Initialize the correction agent.

        Args:
            model: The model to use (e.g., "openai:gpt-4o-mini").
                   Defaults to settings.llm_model.
        """
        model_name = model or f"{get_pydantic_ai_provider()}:{settings.llm_model}"

        self._agent = Agent(
            model_name,
            output_type=ExtractedData,
            system_prompt=CORRECTION_SYSTEM_PROMPT,
        )

    @llm_retry()
    async def correct(
        self,
        biography: str,
        extracted_data: ExtractedData,
        validation_errors: list[str],
    ) -> CorrectionResult:
        """Correct validation errors in extracted data.

        Args:
            biography: The original biography text.
            extracted_data: The extracted data with validation errors.
            validation_errors: List of validation error messages.

        Returns:
            CorrectionResult with corrected data and token usage.
        """
        import time

        from ancestral_synth.utils.timing import verbose_log

        # Format the current extracted data as readable text
        data_summary = self._format_extracted_data(extracted_data)

        prompt = f"""Fix the validation errors in this extracted genealogical data.

ORIGINAL BIOGRAPHY:
---
{biography}
---

CURRENT EXTRACTED DATA:
{data_summary}

VALIDATION ERRORS TO FIX:
{chr(10).join(f"- {error}" for error in validation_errors)}

Analyze the biography carefully and return corrected data that resolves all validation errors.
The corrected data must be internally consistent (all events within birth-death range, etc.)."""

        verbose_log(f"      [correction] Prompt length: {len(prompt)} chars")
        verbose_log(f"      [correction] Errors to fix: {validation_errors}")

        start = time.perf_counter()
        result = await self._agent.run(prompt)
        elapsed = time.perf_counter() - start
        verbose_log(f"      [correction] pydantic_ai.run() completed in {elapsed:.1f}s")

        # Extract token usage from result
        usage_data = result.usage()
        usage = TokenUsage(
            input_tokens=usage_data.request_tokens or 0,
            output_tokens=usage_data.response_tokens or 0,
        )

        return CorrectionResult(data=result.output, usage=usage)

    def _format_extracted_data(self, data: ExtractedData) -> str:
        """Format extracted data as readable text for the prompt."""
        lines = [
            f"Name: {data.given_name} {data.surname}",
            f"Gender: {data.gender.value if data.gender else 'unknown'}",
        ]

        if data.birth_date:
            lines.append(f"Birth: {data.birth_date} in {data.birth_place or 'unknown location'}")
        elif data.birth_year:
            lines.append(f"Birth: {data.birth_year} in {data.birth_place or 'unknown location'}")

        if data.death_date:
            lines.append(f"Death: {data.death_date} in {data.death_place or 'unknown location'}")
        elif data.death_year:
            lines.append(f"Death: {data.death_year} in {data.death_place or 'unknown location'}")

        if data.events:
            lines.append("\nEvents:")
            for event in data.events:
                event_date = event.event_date or event.event_year or "unknown date"
                lines.append(f"  - {event.event_type.value}: {event_date} - {event.description}")

        if data.parents:
            lines.append("\nParents:")
            for parent in data.parents:
                lines.append(f"  - {parent.name} (b. {parent.approximate_birth_year or 'unknown'})")

        if data.children:
            lines.append("\nChildren:")
            for child in data.children:
                lines.append(f"  - {child.name} (b. {child.approximate_birth_year or 'unknown'})")

        if data.spouses:
            lines.append("\nSpouses:")
            for spouse in data.spouses:
                lines.append(f"  - {spouse.name} (b. {spouse.approximate_birth_year or 'unknown'})")

        return "\n".join(lines)
