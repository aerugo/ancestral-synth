"""Agent for identifying shared events and context between related people."""

from dataclasses import dataclass

from pydantic import BaseModel, Field
from pydantic_ai import Agent

from ancestral_synth.config import get_pydantic_ai_provider, settings
from ancestral_synth.domain.enums import EventType, RelationshipType
from ancestral_synth.utils.cost_tracker import TokenUsage
from ancestral_synth.utils.retry import llm_retry


class SharedEvent(BaseModel):
    """An event shared between two people that should be added to an existing person's record."""

    event_type: EventType = Field(description="Type of the shared event")
    description: str = Field(description="Description of the event from the existing person's perspective")
    event_year: int | None = Field(default=None, description="Year of the event if known")
    location: str | None = Field(default=None, description="Location of the event if known")


class DiscoveredContext(BaseModel):
    """New context about an existing person discovered from another biography."""

    content: str = Field(description="The new information discovered about the existing person")
    significance: str = Field(
        description="Why this information is significant (e.g., 'provides new detail about their career')"
    )


class SharedEventAnalysis(BaseModel):
    """Result of analyzing two biographies for shared events and context."""

    should_update: bool = Field(
        description="Whether the existing person's record should be updated with new information"
    )
    shared_events: list[SharedEvent] = Field(
        default_factory=list,
        description="Events shared between both people that should be added to the existing person's record",
    )
    discovered_context: list[DiscoveredContext] = Field(
        default_factory=list,
        description="New context about the existing person discovered from the new biography",
    )
    reasoning: str = Field(description="Explanation of the analysis and recommendations")


@dataclass
class SharedEventAnalysisResult:
    """Result of shared event analysis including token usage."""

    analysis: SharedEventAnalysis
    usage: TokenUsage


SHARED_EVENT_SYSTEM_PROMPT = """You are an expert genealogist analyzing biographical information about related people.

Your task is to identify information from a NEW biography that should be added to an EXISTING person's record.

Consider these types of shareable information:

1. SHARED EVENTS - Events both people participated in together:
   - Family gatherings, weddings, funerals they both attended
   - Joint business ventures or collaborations
   - Shared experiences (living together, traveling together, working together)
   - Historical events they both experienced (wars, disasters, migrations)

2. DISCOVERED CONTEXT - New information about the existing person:
   - Details about their personality, habits, or character as observed by the new person
   - Specific achievements or milestones mentioned from a different perspective
   - Relationships or social connections not previously recorded
   - Career details, life circumstances, or experiences mentioned in passing

Guidelines:
1. Only recommend updates for significant, factual information
2. Rephrase events from the existing person's perspective (not the new person's)
3. Avoid duplicating information the existing person's biography already contains
4. Focus on concrete facts rather than subjective opinions
5. Consider the relationship between the people when evaluating significance
6. Be conservative - only recommend updates if the information adds genuine value

Events that should NOT be added:
- Events already clearly covered in the existing biography
- Minor interactions without lasting significance
- Information that's too speculative or uncertain
- The new person's personal opinions about the existing person"""


class SharedEventAgent:
    """Agent for identifying shared events and context between related people."""

    def __init__(self, model: str | None = None) -> None:
        """Initialize the shared event agent.

        Args:
            model: The model to use.
        """
        model_name = model or f"{get_pydantic_ai_provider()}:{settings.llm_model}"

        self._agent = Agent(
            model_name,
            output_type=SharedEventAnalysis,
            system_prompt=SHARED_EVENT_SYSTEM_PROMPT,
        )

    async def analyze(
        self,
        existing_person_name: str,
        existing_person_biography: str,
        new_person_name: str,
        new_person_biography: str,
        relationship: RelationshipType,
    ) -> SharedEventAnalysisResult:
        """Analyze two biographies for shared events and discoverable context.

        Args:
            existing_person_name: Name of the existing person.
            existing_person_biography: Biography of the existing person.
            new_person_name: Name of the new person whose biography was just processed.
            new_person_biography: Biography of the new person.
            relationship: The relationship of the existing person to the new person.

        Returns:
            SharedEventAnalysisResult with analysis and token usage.
        """
        prompt = self._build_prompt(
            existing_person_name,
            existing_person_biography,
            new_person_name,
            new_person_biography,
            relationship,
        )
        return await self._run_llm(prompt)

    @llm_retry()
    async def _run_llm(self, prompt: str) -> SharedEventAnalysisResult:
        """Run LLM with retry logic."""
        result = await self._agent.run(prompt)

        # Extract token usage from result
        usage_data = result.usage()
        usage = TokenUsage(
            input_tokens=usage_data.request_tokens or 0,
            output_tokens=usage_data.response_tokens or 0,
        )

        return SharedEventAnalysisResult(analysis=result.output, usage=usage)

    def _build_prompt(
        self,
        existing_person_name: str,
        existing_person_biography: str,
        new_person_name: str,
        new_person_biography: str,
        relationship: RelationshipType,
    ) -> str:
        """Build the analysis prompt."""
        # Describe the relationship from the existing person's perspective
        relationship_desc = self._describe_relationship(relationship)

        return f"""Analyze these two biographies to identify shared events and new context.

EXISTING PERSON: {existing_person_name}
This person's biography is already in our records. We want to find new information to add.

{existing_person_biography}

---

NEW PERSON: {new_person_name}
Relationship to {existing_person_name}: {new_person_name} is the {relationship_desc} of {existing_person_name}

{new_person_biography}

---

TASK:
1. Identify any events mentioned in {new_person_name}'s biography that {existing_person_name} also participated in
2. Find any new information about {existing_person_name} that isn't in their existing biography
3. Determine if {existing_person_name}'s record should be updated

Remember:
- Rephrase shared events from {existing_person_name}'s perspective
- Only include significant, factual information
- Don't duplicate what's already in {existing_person_name}'s biography"""

    def _describe_relationship(self, relationship: RelationshipType) -> str:
        """Convert relationship type to a descriptive string."""
        descriptions = {
            RelationshipType.PARENT: "parent",
            RelationshipType.CHILD: "child",
            RelationshipType.SPOUSE: "spouse",
            RelationshipType.SIBLING: "sibling",
            RelationshipType.GRANDPARENT: "grandparent",
            RelationshipType.GRANDCHILD: "grandchild",
            RelationshipType.UNCLE: "uncle",
            RelationshipType.AUNT: "aunt",
            RelationshipType.COUSIN: "cousin",
            RelationshipType.NIECE: "niece",
            RelationshipType.NEPHEW: "nephew",
            RelationshipType.OTHER: "relative",
        }
        return descriptions.get(relationship, "relative")
