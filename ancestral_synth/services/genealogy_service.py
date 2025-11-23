"""Main genealogy service for orchestrating the generation pipeline."""

import random
from datetime import date
from uuid import UUID, uuid4

from loguru import logger

from ancestral_synth.agents.biography_agent import (
    BiographyAgent,
    BiographyContext,
    create_seed_context,
)
from ancestral_synth.agents.correction_agent import CorrectionAgent
from ancestral_synth.agents.dedup_agent import DedupAgent, heuristic_match_score
from ancestral_synth.agents.extraction_agent import ExtractionAgent
from ancestral_synth.config import settings
from ancestral_synth.utils.cost_tracker import CostTracker, format_cost, format_tokens
from ancestral_synth.utils.rate_limiter import RateLimitConfig, RateLimiter
from ancestral_synth.utils.timing import VerboseTimer, set_verbose_log_callback
from ancestral_synth.domain.enums import (
    EventType,
    Gender,
    NoteCategory,
    PersonStatus,
    RelationshipType,
)
from ancestral_synth.domain.models import (
    ChildLink,
    Event,
    ExtractedData,
    Note,
    Person,
    PersonReference,
    PersonSummary,
    SpouseLink,
)
from ancestral_synth.persistence.database import Database
from ancestral_synth.persistence.repositories import (
    ChildLinkRepository,
    EventRepository,
    NoteRepository,
    PersonRepository,
    QueueRepository,
    SpouseLinkRepository,
)
from ancestral_synth.services.validation import ValidationResult, Validator


class GenealogyService:
    """Main service for generating and managing the genealogical dataset."""

    def __init__(
        self,
        db: Database,
        biography_agent: BiographyAgent | None = None,
        extraction_agent: ExtractionAgent | None = None,
        correction_agent: CorrectionAgent | None = None,
        dedup_agent: DedupAgent | None = None,
        validator: Validator | None = None,
        rate_limiter: RateLimiter | None = None,
        verbose: bool = False,
    ) -> None:
        """Initialize the genealogy service.

        Args:
            db: Database connection.
            biography_agent: Agent for generating biographies.
            extraction_agent: Agent for extracting data.
            correction_agent: Agent for correcting validation errors.
            dedup_agent: Agent for deduplication.
            validator: Validator for genealogical plausibility.
            rate_limiter: Rate limiter for LLM API calls.
            verbose: Enable verbose output with timing information.
        """
        self._db = db
        self._biography_agent = biography_agent or BiographyAgent()
        self._extraction_agent = extraction_agent or ExtractionAgent()
        self._correction_agent = correction_agent or CorrectionAgent()
        self._dedup_agent = dedup_agent or DedupAgent()
        self._validator = validator or Validator()
        self._rate_limiter = rate_limiter or RateLimiter(
            RateLimitConfig(requests_per_minute=settings.llm_requests_per_minute)
        )
        self._timer = VerboseTimer(enabled=verbose)
        self._verbose = verbose

        # Initialize cost tracker
        self._cost_tracker = CostTracker(settings.llm_provider, settings.llm_model)

        # Set up global verbose logging callback for retry module
        if verbose:
            set_verbose_log_callback(self._timer.log)
        else:
            set_verbose_log_callback(None)

    @property
    def timer(self) -> VerboseTimer:
        """Get the verbose timer for external access."""
        return self._timer

    @property
    def cost_tracker(self) -> CostTracker:
        """Get the cost tracker for external access."""
        return self._cost_tracker

    async def _validate_and_correct(
        self,
        biography: str,
        extracted: ExtractedData,
    ) -> tuple[ExtractedData, ValidationResult]:
        """Validate extracted data and attempt to correct any errors.

        Uses an agentic loop to fix validation errors up to max_correction_attempts.

        Args:
            biography: The original biography text.
            extracted: The extracted data to validate.

        Returns:
            Tuple of (possibly corrected data, final validation result).
        """
        # Initial validation
        with self._timer.time_operation("Validation", show_start=False):
            validation = self._validator.validate_extracted_data(extracted)

        if validation.is_valid:
            return extracted, validation

        # Attempt corrections
        current_data = extracted
        for attempt in range(settings.max_correction_attempts):
            logger.warning(f"Validation errors (attempt {attempt + 1}): {validation.errors}")
            self._timer.log(f"Attempting correction ({attempt + 1}/{settings.max_correction_attempts})")

            # Use correction agent to fix errors
            await self._acquire_rate_limit("data correction")
            with self._timer.time_operation("Data correction (LLM call)"):
                correction_result = await self._correction_agent.correct(
                    biography=biography,
                    extracted_data=current_data,
                    validation_errors=validation.errors,
                )
                current_data = correction_result.data

                # Track correction cost
                cost_result = self._cost_tracker.record_correction(correction_result.usage)
                if self._verbose:
                    self._timer.log(
                        f"Correction cost: {format_cost(cost_result.total_cost)} "
                        f"({format_tokens(correction_result.usage.total_tokens)} tokens)"
                    )

            # Re-validate
            with self._timer.time_operation("Re-validation", show_start=False):
                validation = self._validator.validate_extracted_data(current_data)

            if validation.is_valid:
                logger.info(f"Correction successful after {attempt + 1} attempt(s)")
                self._timer.log("Correction successful - data is now valid")
                return current_data, validation

        # Max attempts reached, log remaining errors
        if not validation.is_valid:
            logger.warning(
                f"Could not fix all validation errors after {settings.max_correction_attempts} attempts: "
                f"{validation.errors}"
            )

        return current_data, validation

    async def process_next(self) -> Person | None:
        """Process the next person in the queue.

        Returns:
            The processed person, or None if queue is empty.
        """
        async with self._db.session() as session:
            queue_repo = QueueRepository(session)
            person_repo = PersonRepository(session)

            # Try to get next from queue
            person_id = await queue_repo.dequeue()

            if person_id is None:
                # Check if there are pending persons to enqueue
                pending = await person_repo.get_by_status(PersonStatus.PENDING)
                if pending:
                    # Use forest fire sampling to pick one
                    selected = self._forest_fire_sample(pending)
                    person_id = selected.id
                    logger.info(f"Selected pending person via forest fire: {selected.given_name} {selected.surname}")
                else:
                    # No pending persons - create a seed
                    logger.info("No pending persons, creating seed person")
                    return await self._create_seed_person()

            # Get the person record
            db_person = await person_repo.get_by_id(person_id)
            if db_person is None:
                logger.error(f"Person {person_id} not found in database")
                return None

            # Update status to processing
            await person_repo.update(person_id, status=PersonStatus.PROCESSING)

        # Process this person
        return await self._process_person(person_id)

    async def _acquire_rate_limit(self, operation: str) -> None:
        """Acquire rate limit and log if we had to wait."""
        wait_time = await self._rate_limiter.acquire()
        if wait_time > 0.1:  # Only log waits over 100ms
            self._timer.log(f"Rate limit: waited {wait_time:.1f}s before {operation}")

    async def _create_seed_person(self) -> Person:
        """Create a new seed person from scratch."""
        context = create_seed_context()

        # Start tracking costs for this person
        self._cost_tracker.start_person()

        # Generate biography (rate limited)
        logger.info(f"Generating biography for seed: {context.given_name} {context.surname}")
        self._timer.log(f"Generating ~{settings.biography_word_count}-word biography using {settings.llm_provider}:{settings.llm_model}")
        await self._acquire_rate_limit("biography generation")
        with self._timer.time_operation("Biography generation (LLM call)"):
            bio_result = await self._biography_agent.generate(context)
            biography = bio_result.biography

            # Track biography cost
            bio_cost = self._cost_tracker.record_biography(bio_result.usage)
            if self._verbose:
                self._timer.log(
                    f"Biography cost: {format_cost(bio_cost.total_cost)} "
                    f"({format_tokens(bio_result.usage.total_tokens)} tokens)"
                )

        self._timer.log(f"Generated {biography.word_count} words")

        # Extract data (rate limited)
        await self._acquire_rate_limit("data extraction")
        with self._timer.time_operation("Data extraction (LLM call)"):
            extract_result = await self._extraction_agent.extract(biography.content)
            extracted = extract_result.data

            # Track extraction cost
            extract_cost = self._cost_tracker.record_extraction(extract_result.usage)
            if self._verbose:
                self._timer.log(
                    f"Extraction cost: {format_cost(extract_cost.total_cost)} "
                    f"({format_tokens(extract_result.usage.total_tokens)} tokens)"
                )

        self._timer.log(f"Extracted {len(extracted.parents)} parents, {len(extracted.children)} children, {len(extracted.events)} events")

        # Validate and correct if needed
        extracted, validation = await self._validate_and_correct(biography.content, extracted)
        for warning in validation.warnings:
            logger.warning(f"Validation warning: {warning}")

        # Create person record
        person = Person(
            status=PersonStatus.COMPLETE,
            given_name=extracted.given_name or context.given_name,
            surname=extracted.surname or context.surname,
            maiden_name=extracted.maiden_name,
            gender=extracted.gender,
            birth_date=extracted.birth_date,
            birth_place=extracted.birth_place,
            death_date=extracted.death_date,
            death_place=extracted.death_place,
            biography=biography.content,
            generation=0,
        )

        async with self._db.session() as session:
            person_repo = PersonRepository(session)
            with self._timer.time_operation("Save to database", show_start=False):
                await person_repo.create(person)

            # Process references (family members, events, notes)
            num_refs = len(extracted.parents) + len(extracted.children) + len(extracted.spouses) + len(extracted.siblings)
            if num_refs > 0:
                self._timer.log(f"Processing {num_refs} family reference(s)")
            with self._timer.time_operation("Process references", show_start=False):
                await self._process_references(session, person, extracted)

        # Finish tracking costs for this person
        person_cost = self._cost_tracker.finish_person()
        if self._verbose and person_cost:
            self._timer.log(
                f"[bold]Person cost: {format_cost(person_cost.total_cost)}[/bold] "
                f"({person_cost.llm_call_count} LLM calls, "
                f"{format_tokens(person_cost.total_tokens.total_tokens)} tokens) | "
                f"Running total: {format_cost(self._cost_tracker.running_total)}"
            )

        logger.info(f"Created seed person: {person.full_name} (ID: {person.id})")
        return person

    async def _process_person(self, person_id: UUID) -> Person:
        """Process a queued person - generate biography and extract data."""
        async with self._db.session() as session:
            person_repo = PersonRepository(session)
            child_link_repo = ChildLinkRepository(session)

            db_person = await person_repo.get_by_id(person_id)
            if db_person is None:
                raise ValueError(f"Person {person_id} not found")

            # Gather context from known relatives
            relatives = await self._gather_relative_context(session, person_id)

            # Build biography context
            context = BiographyContext(
                given_name=db_person.given_name,
                surname=db_person.surname,
                gender=db_person.gender.value if db_person.gender else None,
                approximate_birth_year=db_person.birth_date.year if db_person.birth_date else None,
                birth_place=db_person.birth_place,
                known_relatives=relatives,
                generation=db_person.generation,
            )

        # Start tracking costs for this person
        self._cost_tracker.start_person()

        # Generate biography (rate limited)
        logger.info(f"Generating biography for: {db_person.given_name} {db_person.surname}")
        self._timer.log(f"Generating ~{settings.biography_word_count}-word biography using {settings.llm_provider}:{settings.llm_model}")
        if relatives:
            self._timer.log(f"Context includes {len(relatives)} known relative(s)")
        await self._acquire_rate_limit("biography generation")
        with self._timer.time_operation("Biography generation (LLM call)"):
            bio_result = await self._biography_agent.generate(context)
            biography = bio_result.biography

            # Track biography cost
            bio_cost = self._cost_tracker.record_biography(bio_result.usage)
            if self._verbose:
                self._timer.log(
                    f"Biography cost: {format_cost(bio_cost.total_cost)} "
                    f"({format_tokens(bio_result.usage.total_tokens)} tokens)"
                )

        self._timer.log(f"Generated {biography.word_count} words")

        # Extract data (rate limited)
        await self._acquire_rate_limit("data extraction")
        with self._timer.time_operation("Data extraction (LLM call)"):
            extract_result = await self._extraction_agent.extract_with_hints(
                biography.content,
                expected_name=f"{db_person.given_name} {db_person.surname}",
                expected_birth_year=db_person.birth_date.year if db_person.birth_date else None,
            )
            extracted = extract_result.data

            # Track extraction cost
            extract_cost = self._cost_tracker.record_extraction(extract_result.usage)
            if self._verbose:
                self._timer.log(
                    f"Extraction cost: {format_cost(extract_cost.total_cost)} "
                    f"({format_tokens(extract_result.usage.total_tokens)} tokens)"
                )

        self._timer.log(f"Extracted {len(extracted.parents)} parents, {len(extracted.children)} children, {len(extracted.events)} events")

        # Validate and correct if needed
        extracted, validation = await self._validate_and_correct(biography.content, extracted)
        for warning in validation.warnings:
            logger.warning(f"Validation warning: {warning}")

        # Update person record
        async with self._db.session() as session:
            person_repo = PersonRepository(session)

            with self._timer.time_operation("Save to database", show_start=False):
                await person_repo.update(
                    person_id,
                    status=PersonStatus.COMPLETE,
                    given_name=extracted.given_name or db_person.given_name,
                    surname=extracted.surname or db_person.surname,
                    maiden_name=extracted.maiden_name or db_person.maiden_name,
                    gender=extracted.gender,
                    birth_date=extracted.birth_date or db_person.birth_date,
                    birth_place=extracted.birth_place or db_person.birth_place,
                    death_date=extracted.death_date,
                    death_place=extracted.death_place,
                    biography=biography.content,
                )

            # Process references
            num_refs = len(extracted.parents) + len(extracted.children) + len(extracted.spouses) + len(extracted.siblings)
            if num_refs > 0:
                self._timer.log(f"Processing {num_refs} family reference(s)")
            with self._timer.time_operation("Process references", show_start=False):
                await self._process_references(session, person_repo.to_domain(db_person), extracted)

            # Finish tracking costs for this person
            person_cost = self._cost_tracker.finish_person()
            if self._verbose and person_cost:
                self._timer.log(
                    f"[bold]Person cost: {format_cost(person_cost.total_cost)}[/bold] "
                    f"({person_cost.llm_call_count} LLM calls, "
                    f"{format_tokens(person_cost.total_tokens.total_tokens)} tokens) | "
                    f"Running total: {format_cost(self._cost_tracker.running_total)}"
                )

            # Refresh person
            db_person = await person_repo.get_by_id(person_id)
            return person_repo.to_domain(db_person)  # type: ignore[arg-type]

    async def _gather_relative_context(
        self,
        session,  # noqa: ANN001
        person_id: UUID,
    ) -> list[PersonSummary]:
        """Gather context about known relatives.

        Includes: parents, children, spouses, siblings, grandparents,
        grandchildren, uncles/aunts, cousins, nieces/nephews.
        """
        person_repo = PersonRepository(session)
        child_link_repo = ChildLinkRepository(session)
        spouse_link_repo = SpouseLinkRepository(session)

        relatives: list[PersonSummary] = []
        seen_ids: set[UUID] = {person_id}  # Track seen IDs to avoid duplicates

        async def add_relative(
            rel_id: UUID, relationship: RelationshipType
        ) -> None:
            """Add a relative to the list if not already seen."""
            if rel_id in seen_ids:
                return
            seen_ids.add(rel_id)
            person = await person_repo.get_by_id(rel_id)
            if person:
                summary = person_repo.to_summary(person, relationship)
                if person.biography:
                    summary.key_facts = self._extract_key_facts(person.biography)
                relatives.append(summary)

        # Get parents
        parent_ids = await child_link_repo.get_parents(person_id)
        for parent_id in parent_ids:
            await add_relative(parent_id, RelationshipType.PARENT)

        # Get children
        child_ids = await child_link_repo.get_children(person_id)
        for child_id in child_ids:
            await add_relative(child_id, RelationshipType.CHILD)

        # Get spouses
        spouse_ids = await spouse_link_repo.get_spouses(person_id)
        for spouse_id in spouse_ids:
            await add_relative(spouse_id, RelationshipType.SPOUSE)

        # Get siblings (other children of the same parents)
        for parent_id in parent_ids:
            sibling_ids = await child_link_repo.get_children(parent_id)
            for sibling_id in sibling_ids:
                if sibling_id != person_id:
                    await add_relative(sibling_id, RelationshipType.SIBLING)

        # Get grandparents (parents of parents)
        for parent_id in parent_ids:
            grandparent_ids = await child_link_repo.get_parents(parent_id)
            for grandparent_id in grandparent_ids:
                await add_relative(grandparent_id, RelationshipType.GRANDPARENT)

        # Get grandchildren (children of children)
        for child_id in child_ids:
            grandchild_ids = await child_link_repo.get_children(child_id)
            for grandchild_id in grandchild_ids:
                await add_relative(grandchild_id, RelationshipType.GRANDCHILD)

        # Get uncles/aunts (siblings of parents)
        for parent_id in parent_ids:
            # Get grandparents to find parent's siblings
            grandparent_ids = await child_link_repo.get_parents(parent_id)
            for grandparent_id in grandparent_ids:
                parent_sibling_ids = await child_link_repo.get_children(grandparent_id)
                for parent_sibling_id in parent_sibling_ids:
                    if parent_sibling_id != parent_id:
                        # Determine if uncle or aunt based on gender
                        uncle_aunt = await person_repo.get_by_id(parent_sibling_id)
                        if uncle_aunt:
                            if uncle_aunt.gender == Gender.MALE:
                                await add_relative(parent_sibling_id, RelationshipType.UNCLE)
                            elif uncle_aunt.gender == Gender.FEMALE:
                                await add_relative(parent_sibling_id, RelationshipType.AUNT)
                            else:
                                await add_relative(parent_sibling_id, RelationshipType.UNCLE)

        # Get cousins (children of uncles/aunts)
        for parent_id in parent_ids:
            grandparent_ids = await child_link_repo.get_parents(parent_id)
            for grandparent_id in grandparent_ids:
                parent_sibling_ids = await child_link_repo.get_children(grandparent_id)
                for parent_sibling_id in parent_sibling_ids:
                    if parent_sibling_id != parent_id:
                        cousin_ids = await child_link_repo.get_children(parent_sibling_id)
                        for cousin_id in cousin_ids:
                            await add_relative(cousin_id, RelationshipType.COUSIN)

        # Get nieces/nephews (children of siblings)
        for parent_id in parent_ids:
            sibling_ids = await child_link_repo.get_children(parent_id)
            for sibling_id in sibling_ids:
                if sibling_id != person_id:
                    niece_nephew_ids = await child_link_repo.get_children(sibling_id)
                    for niece_nephew_id in niece_nephew_ids:
                        # Determine if niece or nephew based on gender
                        niece_nephew = await person_repo.get_by_id(niece_nephew_id)
                        if niece_nephew:
                            if niece_nephew.gender == Gender.MALE:
                                await add_relative(niece_nephew_id, RelationshipType.NEPHEW)
                            elif niece_nephew.gender == Gender.FEMALE:
                                await add_relative(niece_nephew_id, RelationshipType.NIECE)
                            else:
                                await add_relative(niece_nephew_id, RelationshipType.NEPHEW)

        return relatives

    def _extract_key_facts(self, biography: str, max_facts: int = 3) -> list[str]:
        """Extract key facts from a biography for context."""
        # Simple extraction - first few sentences
        sentences = biography.split(". ")
        facts = []
        for sentence in sentences[:max_facts * 2]:
            sentence = sentence.strip()
            if len(sentence) > 20 and len(sentence) < 200:
                facts.append(sentence + ".")
            if len(facts) >= max_facts:
                break
        return facts

    async def _process_references(
        self,
        session,  # noqa: ANN001
        person: Person,
        extracted: ExtractedData,
    ) -> None:
        """Process all references from extracted data."""
        person_repo = PersonRepository(session)
        event_repo = EventRepository(session)
        note_repo = NoteRepository(session)
        child_link_repo = ChildLinkRepository(session)
        spouse_link_repo = SpouseLinkRepository(session)
        queue_repo = QueueRepository(session)

        # Process parents - these get queued
        for parent_ref in extracted.parents:
            parent_id = await self._resolve_person_reference(
                session, parent_ref, person.generation - 1
            )
            if parent_id and not await child_link_repo.exists(parent_id, person.id):
                await child_link_repo.create(ChildLink(parent_id=parent_id, child_id=person.id))
                # Queue parent for biography generation
                parent = await person_repo.get_by_id(parent_id)
                if parent and parent.status == PersonStatus.PENDING:
                    await queue_repo.enqueue(parent_id, priority=1)
                    await person_repo.update(parent_id, status=PersonStatus.QUEUED)

        # Process children - these get queued
        for child_ref in extracted.children:
            child_id = await self._resolve_person_reference(
                session, child_ref, person.generation + 1
            )
            if child_id and not await child_link_repo.exists(person.id, child_id):
                await child_link_repo.create(ChildLink(parent_id=person.id, child_id=child_id))
                # Queue child for biography generation
                child = await person_repo.get_by_id(child_id)
                if child and child.status == PersonStatus.PENDING:
                    await queue_repo.enqueue(child_id, priority=1)
                    await person_repo.update(child_id, status=PersonStatus.QUEUED)

        # Process spouses - create spouse links
        for spouse_ref in extracted.spouses:
            spouse_id = await self._resolve_person_reference(
                session, spouse_ref, person.generation
            )
            if spouse_id and not await spouse_link_repo.exists(person.id, spouse_id):
                await spouse_link_repo.create(
                    SpouseLink(person1_id=person.id, person2_id=spouse_id)
                )

        # Process siblings - stay pending
        for sibling_ref in extracted.siblings:
            await self._resolve_person_reference(session, sibling_ref, person.generation)

        # Process other relatives - stay pending
        for other_ref in extracted.other_relatives:
            # Estimate generation based on relationship context
            await self._resolve_person_reference(session, other_ref, person.generation)

        # Process events (convert ExtractedEvent to Event with person's ID)
        for extracted_event in extracted.events:
            event = extracted_event.to_event(person.id)
            await event_repo.create(event)

        # Process notes
        for note_content in extracted.notes:
            note = Note(
                person_id=person.id,
                category=NoteCategory.BIOGRAPHY,
                content=note_content,
                source="biography_extraction",
            )
            await note_repo.create(note)

    async def _resolve_person_reference(
        self,
        session,  # noqa: ANN001
        reference: PersonReference,
        generation: int,
    ) -> UUID | None:
        """Resolve a person reference - find existing or create pending record."""
        person_repo = PersonRepository(session)

        # Validate reference
        validation = self._validator.validate_person_reference(reference)
        if not validation.is_valid:
            logger.warning(f"Invalid person reference: {validation.errors}")
            return None

        # Parse name
        name_parts = reference.name.strip().split()
        if len(name_parts) < 2:
            given_name = name_parts[0] if name_parts else "Unknown"
            surname = "Unknown"
        else:
            given_name = name_parts[0]
            surname = " ".join(name_parts[1:])

        # Search for existing matches
        candidates = await person_repo.search_similar(
            given_name, surname, reference.approximate_birth_year
        )

        # Heuristic filtering
        good_candidates = []
        for candidate in candidates:
            score = heuristic_match_score(
                reference.name,
                reference.approximate_birth_year,
                f"{candidate.given_name} {candidate.surname}",
                candidate.birth_date.year if candidate.birth_date else None,
            )
            if score >= 0.5:
                good_candidates.append((candidate, score))

        # If good heuristic match, use LLM to confirm
        if good_candidates:
            # Sort by score descending
            good_candidates.sort(key=lambda x: x[1], reverse=True)

            new_summary = PersonSummary(
                id=uuid4(),
                full_name=reference.name,
                gender=reference.gender,
                birth_year=reference.approximate_birth_year,
                relationship_to_subject=reference.relationship,
                key_facts=[reference.context] if reference.context else [],
            )

            candidate_summaries = [
                person_repo.to_summary(c[0]) for c in good_candidates[:5]
            ]

            # Check duplicates (rate limited)
            self._timer.log(f"Checking {len(candidate_summaries)} candidate(s) for duplicate: {reference.name}")
            await self._acquire_rate_limit("deduplication check")
            with self._timer.time_operation("Deduplication check (LLM call)"):
                dedup_result_with_usage = await self._dedup_agent.check_duplicate(
                    new_summary, candidate_summaries
                )
                dedup_result = dedup_result_with_usage.result

                # Track dedup cost
                dedup_cost = self._cost_tracker.record_dedup(dedup_result_with_usage.usage)
                if self._verbose:
                    self._timer.log(
                        f"Dedup cost: {format_cost(dedup_cost.total_cost)} "
                        f"({format_tokens(dedup_result_with_usage.usage.total_tokens)} tokens)"
                    )

            if dedup_result.is_duplicate and dedup_result.matched_person_id:
                logger.info(
                    f"Matched '{reference.name}' to existing person "
                    f"(confidence: {dedup_result.confidence:.2f})"
                )
                return UUID(dedup_result.matched_person_id)

        # No match found - create pending record
        birth_date = None
        if reference.approximate_birth_year:
            birth_date = date(reference.approximate_birth_year, 1, 1)

        new_person = Person(
            status=PersonStatus.PENDING,
            given_name=given_name,
            surname=surname,
            gender=reference.gender,
            birth_date=birth_date,
            generation=generation,
        )

        await person_repo.create(new_person)
        logger.info(f"Created pending person: {new_person.full_name} (gen {generation})")
        return new_person.id

    def _forest_fire_sample(self, persons: list) -> "PersonTable":  # type: ignore[name-defined]
        """Use forest fire sampling to select a person.

        This adds controlled randomness while favoring connected persons.
        """
        if not persons:
            raise ValueError("No persons to sample from")

        # Simple implementation: weighted random by inverse generation
        # (prefer persons closer to generation 0)
        weights = []
        for p in persons:
            gen_weight = 1.0 / (abs(p.generation) + 1)
            # Add some randomness
            if random.random() < settings.forest_fire_probability:
                gen_weight *= random.uniform(0.5, 2.0)
            weights.append(gen_weight)

        total = sum(weights)
        weights = [w / total for w in weights]

        return random.choices(persons, weights=weights, k=1)[0]

    async def get_statistics(self) -> dict:
        """Get statistics about the current dataset."""
        async with self._db.session() as session:
            person_repo = PersonRepository(session)
            queue_repo = QueueRepository(session)

            total = await person_repo.count()
            complete = await person_repo.count(PersonStatus.COMPLETE)
            pending = await person_repo.count(PersonStatus.PENDING)
            queued = await person_repo.count(PersonStatus.QUEUED)
            queue_size = await queue_repo.count()

            return {
                "total_persons": total,
                "complete": complete,
                "pending": pending,
                "queued": queued,
                "queue_size": queue_size,
            }
