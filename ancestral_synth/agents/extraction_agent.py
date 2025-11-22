"""Extraction agent for parsing biographies into structured data."""

from pydantic_ai import Agent

from ancestral_synth.config import settings
from ancestral_synth.domain.models import ExtractedData
from ancestral_synth.utils.retry import llm_retry


EXTRACTION_SYSTEM_PROMPT = """You are an expert genealogist and data extraction specialist.

Your task is to extract structured genealogical data from biographical text.

Guidelines:
1. Extract all dates in ISO format (YYYY-MM-DD) when possible, or just the year if that's all that's available
2. Identify all family relationships mentioned:
   - Parents (father, mother)
   - Children (sons, daughters)
   - Spouses (husband, wife)
   - Siblings (brothers, sisters)
   - Other relatives (grandparents, aunts, uncles, cousins, etc.)
3. Extract life events: birth, death, marriage, divorce, immigration, military service, education, career milestones
4. Capture locations with as much detail as provided (city, state, country)
5. Note interesting facts that don't fit other categories
6. Be precise - only extract information explicitly stated or clearly implied
7. For gender, infer from context (pronouns, titles, relationship terms)

Return structured data following the provided schema exactly."""


class ExtractionAgent:
    """Agent for extracting structured data from biographies."""

    def __init__(self, model: str | None = None) -> None:
        """Initialize the extraction agent.

        Args:
            model: The model to use (e.g., "openai:gpt-4o-mini").
                   Defaults to settings.llm_model.
        """
        model_name = model or f"{settings.llm_provider}:{settings.llm_model}"

        self._agent = Agent(
            model_name,
            output_type=ExtractedData,
            system_prompt=EXTRACTION_SYSTEM_PROMPT,
        )

    @llm_retry()
    async def extract(self, biography: str) -> ExtractedData:
        """Extract structured data from a biography.

        Args:
            biography: The biography text to extract from.

        Returns:
            Structured genealogical data.
        """
        prompt = f"""Extract all genealogical data from this biography:

---
{biography}
---

Return the extracted data as structured JSON following the schema."""

        result = await self._agent.run(prompt)
        return result.output

    @llm_retry()
    async def extract_with_hints(
        self,
        biography: str,
        expected_name: str | None = None,
        expected_birth_year: int | None = None,
    ) -> ExtractedData:
        """Extract data with hints for validation.

        Args:
            biography: The biography text.
            expected_name: The expected name of the subject.
            expected_birth_year: The expected approximate birth year.

        Returns:
            Structured genealogical data.
        """
        prompt_parts = ["Extract all genealogical data from this biography:"]

        if expected_name or expected_birth_year:
            prompt_parts.append("\nExpected subject details (for validation):")
            if expected_name:
                prompt_parts.append(f"- Name: {expected_name}")
            if expected_birth_year:
                prompt_parts.append(f"- Approximate birth year: {expected_birth_year}")

        prompt_parts.append(f"\n---\n{biography}\n---")
        prompt_parts.append("\nReturn the extracted data as structured JSON following the schema.")

        prompt = "\n".join(prompt_parts)
        result = await self._agent.run(prompt)
        return result.output
