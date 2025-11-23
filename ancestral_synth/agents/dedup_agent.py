"""Deduplication agent for identifying duplicate person records."""

import re
from dataclasses import dataclass

from pydantic import BaseModel, Field
from pydantic_ai import Agent

from ancestral_synth.config import get_pydantic_ai_provider, settings
from ancestral_synth.domain.models import PersonSummary
from ancestral_synth.utils.cost_tracker import TokenUsage
from ancestral_synth.utils.retry import llm_retry


@dataclass
class ParsedName:
    """A parsed name with components identified."""

    given_name: str
    middle_names: list[str]
    surname: str
    maiden_name: str | None = None
    suffixes: list[str] | None = None  # Jr., Sr., III, etc.

    @property
    def full_name_normalized(self) -> str:
        """Get the full name without suffixes, normalized."""
        parts = [self.given_name] + self.middle_names + [self.surname]
        return " ".join(parts).lower()

    @property
    def all_surnames(self) -> set[str]:
        """Get all possible surnames (including maiden name)."""
        surnames = {self.surname.lower()}
        if self.maiden_name:
            surnames.add(self.maiden_name.lower())
        return surnames


def parse_name(full_name: str) -> ParsedName:
    """Parse a full name into components.

    Handles patterns like:
    - "John Smith" -> given: John, surname: Smith
    - "John William Smith" -> given: John, middle: [William], surname: Smith
    - "Mary Smith (née Jones)" -> given: Mary, surname: Smith, maiden: Jones
    - "Mary Jones Smith" -> given: Mary, middle: [Jones], surname: Smith
    - "John Smith Jr." -> given: John, surname: Smith, suffix: Jr.
    - "Eleanor Mae Beaumont Harding" -> detects married name pattern

    Args:
        full_name: The full name to parse.

    Returns:
        ParsedName with identified components.
    """
    name = full_name.strip()
    maiden_name = None
    suffixes = []

    # Extract maiden name from "née" pattern
    nee_patterns = [
        r"\(née\s+([^)]+)\)",
        r"\(born\s+([^)]+)\)",
        r"\(maiden\s+name:?\s*([^)]+)\)",
        r"née\s+(\w+)",
    ]
    for pattern in nee_patterns:
        match = re.search(pattern, name, re.IGNORECASE)
        if match:
            maiden_name = match.group(1).strip()
            name = re.sub(pattern, "", name, flags=re.IGNORECASE).strip()
            break

    # Extract suffixes (Jr., Sr., III, IV, etc.)
    suffix_pattern = r"\b(Jr\.?|Sr\.?|III|IV|II|2nd|3rd)\b"
    suffix_matches = re.findall(suffix_pattern, name, re.IGNORECASE)
    if suffix_matches:
        suffixes = [s.rstrip(".") for s in suffix_matches]
        name = re.sub(suffix_pattern, "", name, flags=re.IGNORECASE).strip()

    # Clean up multiple spaces
    name = re.sub(r"\s+", " ", name).strip()

    # Split into parts
    parts = name.split()

    if len(parts) == 0:
        return ParsedName(
            given_name="Unknown",
            middle_names=[],
            surname="Unknown",
            maiden_name=maiden_name,
            suffixes=suffixes or None,
        )
    elif len(parts) == 1:
        return ParsedName(
            given_name=parts[0],
            middle_names=[],
            surname="Unknown",
            maiden_name=maiden_name,
            suffixes=suffixes or None,
        )
    elif len(parts) == 2:
        return ParsedName(
            given_name=parts[0],
            middle_names=[],
            surname=parts[1],
            maiden_name=maiden_name,
            suffixes=suffixes or None,
        )
    else:
        # 3+ parts: first is given name, last is surname, rest are middle names
        return ParsedName(
            given_name=parts[0],
            middle_names=parts[1:-1],
            surname=parts[-1],
            maiden_name=maiden_name,
            suffixes=suffixes or None,
        )


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


@dataclass
class DedupResultWithUsage:
    """Result of deduplication check including token usage."""

    result: DedupResult
    usage: TokenUsage


DEDUP_SYSTEM_PROMPT = """You are an expert genealogist specializing in record deduplication.

Your task is to determine if a newly mentioned person is the same as an existing person in the database.

CRITICAL NAMING PATTERNS TO RECOGNIZE:

1. MAIDEN vs MARRIED NAMES:
   - Women often appear with their maiden name OR married name (husband's surname)
   - "Mary Smith" and "Mary Jones" could be the same person if Mary Smith married Mr. Jones
   - "Eleanor Mae Beaumont" could be the same as "Eleanor Mae Beaumont Harding" (added married name)
   - Look for matching first names + approximate birth years to identify these cases

2. MIDDLE NAMES:
   - The same person may be recorded with or without middle names
   - "Thomas Arthur Beaumont" and "Thomas Beaumont" are likely the same person
   - Middle names might appear as initials: "Thomas A. Beaumont"

3. APPROXIMATED DATES:
   - January 1st dates (e.g., "1889-01-01") often indicate an approximate year
   - A candidate with birth date "1889-01-01" should match someone born "1889-06-18"
   - Within the SAME YEAR should be treated as a likely match

4. NAME SUFFIXES:
   - Jr., Sr., III, etc. distinguish different people in the same family
   - These should NOT be ignored - they indicate different individuals

5. FAMILY CONTEXT:
   - If both people share ANY of the same parents, children, or spouse = VERY LIKELY MATCH
   - Shared family members are strong evidence of being the same person
   - Same generation number + same family context = STRONG MATCH

6. BIOGRAPHY MENTION CONTEXT:
   - If the new person was mentioned in a family member's biography, they likely exist in the tree
   - Example: "William Brown" mentioned in "Mary Brown's" biography is likely her relative
   - READ THE BIOGRAPHY SNIPPETS CAREFULLY - they show how a name is actually used in context

CRITICAL: CONSISTENT PARENT REFERENCES = STRONG MATCH

This is VERY IMPORTANT: When multiple people reference someone as their PARENT:

1. MULTIPLE CHILDREN PATTERN:
   - If a candidate ALREADY HAS CHILDREN, and the new person is mentioned as a PARENT by someone else = STRONG MATCH
   - Example: Candidate "Thomas Miller" has children: [David Miller]
   - New person "Thomas Miller" mentioned by Susan Miller with relationship: parent
   - This is almost certainly the SAME Thomas Miller (Susan and David are likely siblings)
   - Same name + same generation + parent role + candidate already has children = MERGE

2. SAME SURNAME PARENT:
   - If new person shares surname with mentioning person and is their parent = likely real parent
   - Parents typically share surnames with their children (before marriage)

3. GENERATION ALIGNMENT:
   - A parent should be one generation above their children
   - If generations align correctly, this supports a match

CRITICAL: DETECTING DIFFERENT PEOPLE WITH THE SAME NAME

Be alert to these patterns that indicate DIFFERENT people who happen to share a first name:

1. CONFLICTING RELATIONSHIP TYPES:
   - If snippets show the name as BOTH "sister" AND "wife" of the same person = TWO DIFFERENT PEOPLE
   - Example: "His sister Eleanor" vs "His wife Eleanor" = NOT the same Eleanor
   - A sibling relationship CANNOT coexist with a spouse/coparent relationship (this would be incest)
   - A parent cannot also be a sibling of the same person

2. CONFLICTING FAMILY ROLES:
   - If a candidate is listed as someone's SIBLING but the new person is that same person's SPOUSE = DIFFERENT people
   - If biography snippets show different relationship contexts = DIFFERENT people
   - Pay close attention to words like: wife, husband, sister, brother, mother, father, daughter, son

3. MULTIPLE PEOPLE WITH SAME NAME IN A FAMILY:
   - Families often have multiple relatives with the same first name (named after each other)
   - A man might have both a sister named Eleanor AND a wife named Eleanor
   - Look at the relationship context to distinguish between them

4. SAME GENERATION, DIFFERENT ROLES:
   - Two people in the same generation with the same first name but different relationship types are DIFFERENT people
   - Example: Eleanor (sibling) and Eleanor (spouse) in generation 3 = TWO DIFFERENT PEOPLE

REASONING STEPS FOR EACH CANDIDATE:
1. Check if new person is mentioned as PARENT and candidate already has children = STRONG MATCH signal
2. Check for CONFLICTING relationships - if found, they are DIFFERENT people
3. Read biography snippets to understand the actual relationship context
4. Check if family members overlap (shared parents/children/spouse)
5. Compare names, birth years, and other attributes

MATCHING GUIDELINES (in order of strength):
1. CONFLICTING RELATIONSHIPS = DEFINITE NON-MATCH - If relationships conflict, different people
2. CONSISTENT PARENT REFERENCES = STRONG MATCH - Multiple children referencing same parent name
3. SHARED FAMILY MEMBERS with matching relationships = LIKELY MATCH
4. Same first name + same surname + same generation + same role = LIKELY MATCH
5. Same first name + same surname + same birth year (±2 years) = POSSIBLE MATCH (verify context)
6. Gender MUST match for a duplicate
7. If birth years differ by more than 5 years, probably NOT a match

BALANCE: Be conservative about CONFLICTING evidence, but confident about CONSISTENT evidence.
Multiple family members referencing the same person by name and role is strong evidence of a match."""


class DedupAgent:
    """Agent for checking if a person is a duplicate."""

    def __init__(self, model: str | None = None) -> None:
        """Initialize the dedup agent.

        Args:
            model: The model to use.
        """
        model_name = model or f"{get_pydantic_ai_provider()}:{settings.llm_model}"

        self._agent = Agent(
            model_name,
            output_type=DedupResult,
            system_prompt=DEDUP_SYSTEM_PROMPT,
        )

    async def check_duplicate(
        self,
        new_person: PersonSummary,
        candidates: list[PersonSummary],
    ) -> DedupResultWithUsage:
        """Check if a new person matches any existing candidates.

        Args:
            new_person: The newly mentioned person.
            candidates: Existing people who might be duplicates.

        Returns:
            DedupResultWithUsage with result and token usage.
        """
        if not candidates:
            return DedupResultWithUsage(
                result=DedupResult(
                    is_duplicate=False,
                    matched_person_id=None,
                    confidence=1.0,
                    reasoning="No candidates to compare against",
                ),
                usage=TokenUsage(input_tokens=0, output_tokens=0),
            )

        prompt = self._build_prompt(new_person, candidates)
        return await self._run_llm(prompt)

    @llm_retry()
    async def _run_llm(self, prompt: str) -> DedupResultWithUsage:
        """Run LLM with retry logic."""
        result = await self._agent.run(prompt)

        # Extract token usage from result
        usage_data = result.usage()
        usage = TokenUsage(
            input_tokens=usage_data.request_tokens or 0,
            output_tokens=usage_data.response_tokens or 0,
        )

        return DedupResultWithUsage(result=result.output, usage=usage)

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
        if new_person.generation is not None:
            parts.append(f"  Generation: {new_person.generation}")
        if new_person.mentioned_by:
            parts.append(f"  Mentioned in biography of: {new_person.mentioned_by}")
        if new_person.relationship_to_subject:
            parts.append(f"  Relationship context: {new_person.relationship_to_subject}")

        # Add relation context for new person
        self._add_relations_to_prompt(parts, new_person, indent="  ")

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
            if candidate.generation is not None:
                parts.append(f"    Generation: {candidate.generation}")

            # Add relation context for candidate
            self._add_relations_to_prompt(parts, candidate, indent="    ")

            if candidate.key_facts:
                parts.append("    Key facts:")
                for fact in candidate.key_facts:
                    parts.append(f"      - {fact}")

            # Add biography snippets showing how this person's name is mentioned
            if candidate.biography_snippets:
                parts.append("    Biography mentions of this name from relatives:")
                for snippet in candidate.biography_snippets:
                    # Truncate very long snippets
                    display_snippet = snippet[:400] + "..." if len(snippet) > 400 else snippet
                    parts.append(f"      \"{display_snippet}\"")

        parts.append("")
        parts.append("Is the new person a duplicate of any candidate? If so, which one?")

        return "\n".join(parts)

    def _add_relations_to_prompt(
        self,
        parts: list[str],
        person: PersonSummary,
        indent: str = "  ",
    ) -> None:
        """Add family relation info to prompt parts."""
        # First degree relations
        if person.parents:
            parts.append(f"{indent}Parents: {', '.join(person.parents)}")
        if person.children:
            parts.append(f"{indent}Children: {', '.join(person.children)}")
        if person.spouses:
            parts.append(f"{indent}Spouses: {', '.join(person.spouses)}")
        if person.siblings:
            parts.append(f"{indent}Siblings: {', '.join(person.siblings)}")

        # Second degree relations
        if person.grandparents:
            parts.append(f"{indent}Grandparents: {', '.join(person.grandparents)}")
        if person.grandchildren:
            parts.append(f"{indent}Grandchildren: {', '.join(person.grandchildren)}")


def heuristic_match_score(
    new_name: str,
    new_birth_year: int | None,
    candidate_name: str,
    candidate_birth_year: int | None,
) -> float:
    """Calculate a heuristic match score between two people.

    This function handles:
    - Exact name matches
    - Partial name overlap (middle names may be missing)
    - Maiden name vs married name patterns
    - Approximate birth years

    Args:
        new_name: Name of the new person.
        new_birth_year: Birth year of the new person.
        candidate_name: Name of the candidate.
        candidate_birth_year: Birth year of the candidate.

    Returns:
        Score from 0.0 (no match) to 1.0 (perfect match).
    """
    name_score = 0.0
    year_score = 0.0

    # Parse both names
    new_parsed = parse_name(new_name)
    candidate_parsed = parse_name(candidate_name)

    # Check for suffix mismatch (Jr. vs Sr. = different people)
    if new_parsed.suffixes and candidate_parsed.suffixes:
        if set(new_parsed.suffixes) != set(candidate_parsed.suffixes):
            return 0.0  # Different suffixes = different people

    # Given name comparison (most important)
    new_given = new_parsed.given_name.lower()
    candidate_given = candidate_parsed.given_name.lower()

    if new_given == candidate_given:
        name_score += 0.25  # First name match

        # Surname comparison (handles maiden/married names)
        new_surnames = new_parsed.all_surnames
        candidate_surnames = candidate_parsed.all_surnames

        # Check if any surname overlaps
        if new_surnames & candidate_surnames:
            name_score += 0.25  # Direct surname match
        else:
            # Check if new person's surname appears anywhere in candidate's name
            # This catches "Mary Jones" matching "Mary Jones Smith"
            candidate_all_parts = {p.lower() for p in [candidate_parsed.given_name] +
                                  candidate_parsed.middle_names + [candidate_parsed.surname]}
            if new_parsed.surname.lower() in candidate_all_parts:
                name_score += 0.2  # Surname appears in candidate's full name
            elif candidate_parsed.surname.lower() in {p.lower() for p in [new_parsed.given_name] +
                                                       new_parsed.middle_names + [new_parsed.surname]}:
                name_score += 0.2  # Candidate's surname appears in new person's name
            # No surname connection - could still be a match with married name change

        # Middle name comparison (bonus for matching, but not required)
        new_middle_set = {m.lower() for m in new_parsed.middle_names}
        candidate_middle_set = {m.lower() for m in candidate_parsed.middle_names}

        if new_middle_set and candidate_middle_set:
            if new_middle_set & candidate_middle_set:
                name_score += 0.1  # Middle name match bonus

    # Birth year comparison (independent of name matching)
    if new_birth_year and candidate_birth_year:
        year_diff = abs(new_birth_year - candidate_birth_year)
        if year_diff == 0:
            year_score = 0.5  # Exact year match
        elif year_diff <= 2:
            year_score = 0.45  # Within 2 years
        elif year_diff <= 5:
            year_score = 0.3  # Within 5 years
        # Years far apart = no year bonus (year_score stays 0)

    return max(0.0, min(name_score + year_score, 1.0))


def extract_name_mentions(
    first_name: str,
    biography: str | None,
    padding: int = 300,
) -> list[str]:
    """Extract snippets from biography around mentions of a first name.

    Uses regex to find whole-word matches of the first name and extracts
    surrounding context to help understand the relationship.

    Args:
        first_name: The first name to search for.
        biography: The biography text to search in.
        padding: Number of characters before and after to include.

    Returns:
        List of snippets containing the name with surrounding context.
    """
    if not biography or not first_name:
        return []

    # Build regex for whole-word match (case insensitive)
    pattern = re.compile(
        rf"\b{re.escape(first_name)}\b",
        re.IGNORECASE,
    )

    snippets = []
    for match in pattern.finditer(biography):
        start = max(0, match.start() - padding)
        end = min(len(biography), match.end() + padding)
        snippet = biography[start:end]
        snippets.append(snippet)

    return snippets
