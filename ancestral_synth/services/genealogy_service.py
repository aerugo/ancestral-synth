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
from ancestral_synth.agents.dedup_agent import (
    DedupAgent,
    extract_name_mentions,
    heuristic_match_score,
    parse_name,
)
from ancestral_synth.agents.extraction_agent import ExtractionAgent
from ancestral_synth.agents.shared_event_agent import SharedEventAgent
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
from ancestral_synth.persistence.relations import get_relations_summary
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
        shared_event_agent: SharedEventAgent | None = None,
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
            shared_event_agent: Agent for analyzing shared events between people.
            validator: Validator for genealogical plausibility.
            rate_limiter: Rate limiter for LLM API calls.
            verbose: Enable verbose output with timing information.
        """
        self._db = db
        self._biography_agent = biography_agent or BiographyAgent()
        self._extraction_agent = extraction_agent or ExtractionAgent()
        self._correction_agent = correction_agent or CorrectionAgent()
        self._dedup_agent = dedup_agent or DedupAgent()
        self._shared_event_agent = shared_event_agent or SharedEventAgent()
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
                session, parent_ref, person.generation - 1, new_person=person
            )
            if parent_id and not await child_link_repo.exists(parent_id, person.id):
                # Check if this would create >2 parents (potential duplicate)
                parent_count = await child_link_repo.get_parent_count(person.id)
                if parent_count >= 2:
                    # Check for duplicates and merge if found
                    parent_id = await self._check_and_merge_duplicate_parents(
                        session, person.id, parent_id
                    )
                    # Re-check if link already exists after potential merge
                    if await child_link_repo.exists(parent_id, person.id):
                        continue

                await child_link_repo.create(ChildLink(parent_id=parent_id, child_id=person.id))
                # Queue parent for biography generation
                parent = await person_repo.get_by_id(parent_id)
                if parent and parent.status == PersonStatus.PENDING:
                    await queue_repo.enqueue(parent_id, priority=1)
                    await person_repo.update(parent_id, status=PersonStatus.QUEUED)

        # Process children - these get queued
        for child_ref in extracted.children:
            child_id = await self._resolve_person_reference(
                session, child_ref, person.generation + 1, new_person=person
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
                session, spouse_ref, person.generation, new_person=person
            )
            if spouse_id and not await spouse_link_repo.exists(person.id, spouse_id):
                await spouse_link_repo.create(
                    SpouseLink(person1_id=person.id, person2_id=spouse_id)
                )

        # Process siblings - stay pending
        for sibling_ref in extracted.siblings:
            await self._resolve_person_reference(
                session, sibling_ref, person.generation, new_person=person
            )

        # Process other relatives - stay pending
        for other_ref in extracted.other_relatives:
            # Estimate generation based on relationship context
            await self._resolve_person_reference(
                session, other_ref, person.generation, new_person=person
            )

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
        new_person: Person | None = None,
    ) -> UUID | None:
        """Resolve a person reference - find existing or create pending record.

        Args:
            session: Database session.
            reference: The person reference to resolve.
            generation: The generation number for the referenced person.
            new_person: The person whose biography was just processed (for shared event analysis).

        Returns:
            The UUID of the resolved person, or None if the reference is invalid.
        """
        person_repo = PersonRepository(session)

        # Validate reference
        validation = self._validator.validate_person_reference(reference)
        if not validation.is_valid:
            logger.warning(f"Invalid person reference: {validation.errors}")
            return None

        # Parse name using the same logic as dedup agent
        parsed = parse_name(reference.name)
        given_name = parsed.given_name
        surname = parsed.surname
        maiden_name = parsed.maiden_name

        # Search for existing matches
        candidates = await person_repo.search_similar(
            given_name, surname, reference.approximate_birth_year, maiden_name
        )

        # Also search by given name + birth year only to catch married name scenarios
        # This finds people like "Eleanor Harrison" when searching for "Eleanor Maxwell"
        # if Eleanor Harrison married Robert Maxwell
        broader_candidates = await person_repo.search_by_given_name_and_birth_year(
            given_name, reference.approximate_birth_year
        )

        # Add broader candidates not already in candidates list
        candidate_ids = {c.id for c in candidates}
        for bc in broader_candidates:
            if bc.id not in candidate_ids:
                candidates.append(bc)
                candidate_ids.add(bc.id)

        # Heuristic filtering with enhanced spouse-based scoring
        good_candidates = []
        spouse_link_repo = SpouseLinkRepository(session)

        for candidate in candidates:
            base_score = heuristic_match_score(
                reference.name,
                reference.approximate_birth_year,
                f"{candidate.given_name} {candidate.surname}",
                candidate.birth_date.year if candidate.birth_date else None,
            )

            # If base score is low but given name matches, check spouse surnames
            # This catches married name scenarios
            if base_score < 0.5 and candidate.given_name.lower() == given_name.lower():
                spouse_ids = await spouse_link_repo.get_spouses(candidate.id)
                for spouse_id in spouse_ids:
                    spouse = await person_repo.get_by_id(spouse_id)
                    if spouse and spouse.surname.lower() == surname.lower():
                        # Candidate has a spouse with the surname we're looking for
                        # This is a strong indicator of a married name match
                        base_score = max(base_score, 0.6)  # Bump score to pass threshold
                        self._timer.log(
                            f"Spouse-based match: '{candidate.given_name} {candidate.surname}' "
                            f"has spouse '{spouse.given_name} {spouse.surname}' matching surname '{surname}'"
                        )
                        break

            if base_score >= 0.5:
                good_candidates.append((candidate, base_score))

        # If good heuristic match, use LLM to confirm
        if good_candidates:
            # Sort by score descending
            good_candidates.sort(key=lambda x: x[1], reverse=True)

            # Build enhanced context for deduplication
            mentioned_by_name = None
            if new_person is not None:
                mentioned_by_name = f"{new_person.given_name} {new_person.surname}"

            new_summary = PersonSummary(
                id=uuid4(),
                full_name=reference.name,
                gender=reference.gender,
                birth_year=reference.approximate_birth_year,
                relationship_to_subject=reference.relationship,
                key_facts=[reference.context] if reference.context else [],
                mentioned_by=mentioned_by_name,
                generation=generation,
            )

            # Build candidate summaries with relation context
            candidate_summaries = []
            for candidate_db, _ in good_candidates[:5]:
                summary = person_repo.to_summary(candidate_db)
                # Add relation context for each candidate
                relations = await get_relations_summary(session, candidate_db.id)
                summary.parents = relations.parents
                summary.children = relations.children
                summary.spouses = relations.spouses
                summary.siblings = relations.siblings
                summary.grandparents = relations.grandparents
                summary.grandchildren = relations.grandchildren
                summary.generation = candidate_db.generation

                # Extract biography snippets from relatives mentioning this name
                first_name = candidate_db.given_name
                biography_snippets = await self._extract_relative_biography_snippets(
                    session, candidate_db.id, first_name
                )
                summary.biography_snippets = biography_snippets

                candidate_summaries.append(summary)

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
                # Validate UUID format (LLM sometimes returns truncated UUIDs)
                try:
                    matched_id = UUID(dedup_result.matched_person_id)
                except ValueError:
                    logger.warning(
                        f"LLM returned invalid UUID '{dedup_result.matched_person_id}' "
                        f"for '{reference.name}' - treating as no match"
                    )
                    # Fall through to create new pending record
                    matched_id = None

                if matched_id is not None:
                    logger.info(
                        f"Matched '{reference.name}' to existing person "
                        f"(confidence: {dedup_result.confidence:.2f})"
                    )

                    # Evaluate shared events if the new person has a biography
                    if new_person is not None and new_person.biography is not None:
                        # Get the inverse relationship (existing person's relationship to new person)
                        inverse_relationship = self._get_inverse_relationship(reference.relationship)
                        await self._evaluate_shared_events(
                            session,
                            existing_person_id=matched_id,
                            new_person=new_person,
                            relationship=inverse_relationship,
                        )

                    return matched_id

        # No match found - create pending record
        birth_date = None
        if reference.approximate_birth_year:
            birth_date = date(reference.approximate_birth_year, 1, 1)

        new_person = Person(
            status=PersonStatus.PENDING,
            given_name=given_name,
            surname=surname,
            maiden_name=maiden_name,
            gender=reference.gender,
            birth_date=birth_date,
            generation=generation,
        )

        await person_repo.create(new_person)
        logger.info(f"Created pending person: {new_person.full_name} (gen {generation})")
        return new_person.id

    async def _extract_relative_biography_snippets(
        self,
        session,  # noqa: ANN001
        person_id: UUID,
        first_name: str,
    ) -> list[str]:
        """Extract biography snippets from relatives that mention a first name.

        This helps identify how a person is referred to in their relatives' biographies,
        which can distinguish between different people with the same first name
        (e.g., someone's sister Eleanor vs their wife Eleanor).

        Args:
            session: Database session.
            person_id: ID of the person to find mentions for.
            first_name: The first name to search for.

        Returns:
            List of biography snippets containing the name.
        """
        person_repo = PersonRepository(session)
        child_link_repo = ChildLinkRepository(session)
        spouse_link_repo = SpouseLinkRepository(session)

        snippets: list[str] = []
        checked_ids: set[UUID] = {person_id}  # Don't check person's own biography

        async def check_person_biography(rel_id: UUID) -> None:
            """Check a relative's biography for mentions of first_name."""
            if rel_id in checked_ids:
                return
            checked_ids.add(rel_id)

            person = await person_repo.get_by_id(rel_id)
            if person and person.biography:
                mentions = extract_name_mentions(first_name, person.biography, padding=150)
                # Limit to first 2 mentions per relative to keep prompt reasonable
                snippets.extend(mentions[:2])

        # Check parents' biographies
        parent_ids = await child_link_repo.get_parents(person_id)
        for parent_id in parent_ids:
            await check_person_biography(parent_id)

        # Check children's biographies
        child_ids = await child_link_repo.get_children(person_id)
        for child_id in child_ids:
            await check_person_biography(child_id)

        # Check spouses' biographies
        spouse_ids = await spouse_link_repo.get_spouses(person_id)
        for spouse_id in spouse_ids:
            await check_person_biography(spouse_id)

        # Check siblings' biographies (other children of same parents)
        for parent_id in parent_ids:
            sibling_ids = await child_link_repo.get_children(parent_id)
            for sibling_id in sibling_ids:
                if sibling_id != person_id:
                    await check_person_biography(sibling_id)

        # Limit total snippets to avoid huge prompts
        return snippets[:6]

    def _get_inverse_relationship(self, relationship: RelationshipType) -> RelationshipType:
        """Get the inverse of a relationship type.

        For example, if person A is the PARENT of person B, then person B is the CHILD of person A.

        Args:
            relationship: The relationship from one person's perspective.

        Returns:
            The relationship from the other person's perspective.
        """
        inverse_map = {
            RelationshipType.PARENT: RelationshipType.CHILD,
            RelationshipType.CHILD: RelationshipType.PARENT,
            RelationshipType.SPOUSE: RelationshipType.SPOUSE,
            RelationshipType.SIBLING: RelationshipType.SIBLING,
            RelationshipType.GRANDPARENT: RelationshipType.GRANDCHILD,
            RelationshipType.GRANDCHILD: RelationshipType.GRANDPARENT,
            RelationshipType.UNCLE: RelationshipType.NEPHEW,  # Simplified - could be niece
            RelationshipType.AUNT: RelationshipType.NIECE,  # Simplified - could be nephew
            RelationshipType.COUSIN: RelationshipType.COUSIN,
            RelationshipType.NIECE: RelationshipType.AUNT,  # Simplified - could be uncle
            RelationshipType.NEPHEW: RelationshipType.UNCLE,  # Simplified - could be aunt
            RelationshipType.OTHER: RelationshipType.OTHER,
        }
        return inverse_map.get(relationship, RelationshipType.OTHER)

    def _get_relationship_description(self, relationship: RelationshipType) -> str:
        """Convert a relationship type to a human-readable description.

        Args:
            relationship: The relationship type to describe.

        Returns:
            A human-readable description of the relationship.
        """
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

    async def _evaluate_shared_events(
        self,
        session,  # noqa: ANN001
        existing_person_id: UUID,
        new_person: Person,
        relationship: RelationshipType,
    ) -> None:
        """Evaluate if an existing person's biography should be updated with shared events.

        When a new biography references an existing person, this method analyzes both
        biographies to identify shared events or new context that should be added to
        the existing person's record.

        Args:
            session: Database session.
            existing_person_id: ID of the existing person (already has a biography).
            new_person: The person whose biography was just processed.
            relationship: The relationship of the existing person to the new person.
        """
        person_repo = PersonRepository(session)
        event_repo = EventRepository(session)
        note_repo = NoteRepository(session)

        # Get the existing person
        existing_db_person = await person_repo.get_by_id(existing_person_id)
        if existing_db_person is None or existing_db_person.biography is None:
            return

        # Only evaluate if the existing person has a complete biography
        if existing_db_person.status != PersonStatus.COMPLETE:
            return

        # Skip if the new person doesn't have a biography
        if new_person.biography is None:
            return

        existing_name = f"{existing_db_person.given_name} {existing_db_person.surname}"
        new_name = new_person.full_name

        # Get relationship description for logging
        relationship_desc = self._get_relationship_description(relationship)
        self._timer.log(
            f"Evaluating shared events between {new_name} and {existing_name} "
            f"({new_name} is {existing_name}'s {relationship_desc})"
        )

        # Call the shared event agent (rate limited)
        await self._acquire_rate_limit("shared event analysis")
        with self._timer.time_operation("Shared event analysis (LLM call)"):
            analysis_result = await self._shared_event_agent.analyze(
                existing_person_name=existing_name,
                existing_person_biography=existing_db_person.biography,
                new_person_name=new_name,
                new_person_biography=new_person.biography,
                relationship=relationship,
            )

            # Track cost
            analysis_cost = self._cost_tracker.record_dedup(analysis_result.usage)
            if self._verbose:
                self._timer.log(
                    f"Shared event analysis cost: {format_cost(analysis_cost.total_cost)} "
                    f"({format_tokens(analysis_result.usage.total_tokens)} tokens)"
                )

        analysis = analysis_result.analysis

        # Log analysis results in verbose mode
        if self._verbose:
            num_events = len(analysis.shared_events)
            num_context = len(analysis.discovered_context)
            self._timer.log(
                f"Analysis complete: found {num_events} shared event(s) "
                f"and {num_context} discovered context item(s)"
            )
            if analysis.reasoning:
                self._timer.log(f"Reasoning: {analysis.reasoning}")

        if not analysis.should_update:
            self._timer.log(f"No updates needed for {existing_name}")
            return

        # Process shared events
        events_added = 0
        for shared_event in analysis.shared_events:
            event = Event(
                event_type=shared_event.event_type,
                event_year=shared_event.event_year,
                location=shared_event.location,
                description=shared_event.description,
                primary_person_id=existing_person_id,
                other_person_ids=[new_person.id],  # Link to the person who mentioned this
            )
            await event_repo.create(event)
            events_added += 1

            # Log each event in verbose mode
            if self._verbose:
                year_str = f" ({shared_event.event_year})" if shared_event.event_year else ""
                location_str = f" at {shared_event.location}" if shared_event.location else ""
                self._timer.log(
                    f"  + Event: {shared_event.event_type.value}{year_str}{location_str}"
                )
                self._timer.log(f"    {shared_event.description}")

        # Process discovered context as notes
        notes_added = 0
        for context in analysis.discovered_context:
            note = Note(
                person_id=existing_person_id,
                category=NoteCategory.CROSS_BIOGRAPHY,
                content=f"{context.content} ({context.significance})",
                source=f"cross_biography:{new_person.id}",
                referenced_person_ids=[new_person.id],
            )
            await note_repo.create(note)
            notes_added += 1

            # Log each note in verbose mode
            if self._verbose:
                self._timer.log(f"  + Note: {context.content}")
                self._timer.log(f"    Significance: {context.significance}")

        if events_added > 0 or notes_added > 0:
            logger.info(
                f"Updated {existing_name} with {events_added} shared event(s) "
                f"and {notes_added} note(s) from {new_name}'s biography"
            )
            self._timer.log(
                f"Added {events_added} event(s) and {notes_added} note(s) to {existing_name}"
            )

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

    async def _merge_persons(
        self,
        session,  # noqa: ANN001
        keep_id: UUID,
        merge_id: UUID,
    ) -> None:
        """Merge two person records, keeping the more complete one.

        All relationships from merge_id are transferred to keep_id, then
        merge_id is deleted.

        Args:
            session: Database session.
            keep_id: ID of the person to keep.
            merge_id: ID of the person to merge into keep_id and then delete.
        """
        person_repo = PersonRepository(session)
        child_link_repo = ChildLinkRepository(session)
        spouse_link_repo = SpouseLinkRepository(session)
        queue_repo = QueueRepository(session)

        keep_person = await person_repo.get_by_id(keep_id)
        merge_person = await person_repo.get_by_id(merge_id)

        if not keep_person or not merge_person:
            logger.warning(f"Cannot merge: person(s) not found (keep={keep_id}, merge={merge_id})")
            return

        logger.info(
            f"Merging duplicate persons: keeping '{keep_person.given_name} {keep_person.surname}' "
            f"(ID: {keep_id}), merging '{merge_person.given_name} {merge_person.surname}' (ID: {merge_id})"
        )

        # Transfer all child links where merge_id is the parent
        children_transferred = await child_link_repo.reassign_all_parent_links(merge_id, keep_id)
        if children_transferred:
            logger.info(f"  Transferred {children_transferred} child link(s)")

        # Transfer all parent links where merge_id is the child
        parents_transferred = await child_link_repo.reassign_all_child_links(merge_id, keep_id)
        if parents_transferred:
            logger.info(f"  Transferred {parents_transferred} parent link(s)")

        # Transfer all spouse links
        spouses_transferred = await spouse_link_repo.reassign_spouse_links(merge_id, keep_id)
        if spouses_transferred:
            logger.info(f"  Transferred {spouses_transferred} spouse link(s)")

        # Update keep_person with any missing data from merge_person
        updates = {}
        if not keep_person.maiden_name and merge_person.maiden_name:
            updates["maiden_name"] = merge_person.maiden_name
        if not keep_person.birth_place and merge_person.birth_place:
            updates["birth_place"] = merge_person.birth_place
        if not keep_person.death_date and merge_person.death_date:
            updates["death_date"] = merge_person.death_date
        if not keep_person.death_place and merge_person.death_place:
            updates["death_place"] = merge_person.death_place
        if not keep_person.biography and merge_person.biography:
            updates["biography"] = merge_person.biography
            updates["status"] = merge_person.status

        if updates:
            await person_repo.update(keep_id, **updates)
            logger.info(f"  Updated fields: {list(updates.keys())}")

        # Remove merge_id from queue if present
        # (QueueRepository doesn't have a remove method, so we just let it fail gracefully)

        # Delete the merged person
        await person_repo.delete(merge_id)
        logger.info(f"  Deleted merged person record")

    async def _check_and_merge_duplicate_parents(
        self,
        session,  # noqa: ANN001
        child_id: UUID,
        new_parent_id: UUID,
    ) -> UUID:
        """Check if adding a new parent would result in >2 parents due to duplicates.

        If the child already has 2 parents and we're trying to add a third, this method
        checks if any of the existing parents are duplicates of the new parent. If so,
        it merges them and returns the ID of the merged parent.

        Args:
            session: Database session.
            child_id: ID of the child.
            new_parent_id: ID of the parent we want to add.

        Returns:
            The ID of the parent to use (may be different from new_parent_id if merged).
        """
        person_repo = PersonRepository(session)
        child_link_repo = ChildLinkRepository(session)
        spouse_link_repo = SpouseLinkRepository(session)

        # Get existing parents
        existing_parent_ids = await child_link_repo.get_parents(child_id)

        # If less than 2 parents, no issue
        if len(existing_parent_ids) < 2:
            return new_parent_id

        # Get the new parent info
        new_parent = await person_repo.get_by_id(new_parent_id)
        if not new_parent:
            return new_parent_id

        new_parsed = parse_name(f"{new_parent.given_name} {new_parent.surname}")

        # Check each existing parent for potential duplicate
        for existing_parent_id in existing_parent_ids:
            existing_parent = await person_repo.get_by_id(existing_parent_id)
            if not existing_parent:
                continue

            # Quick checks: same gender and similar generation
            if existing_parent.gender != new_parent.gender:
                continue

            existing_parsed = parse_name(f"{existing_parent.given_name} {existing_parent.surname}")

            # Check if given names match
            if existing_parsed.given_name.lower() != new_parsed.given_name.lower():
                continue

            # Check for surname match (including maiden name scenarios)
            surnames_match = False

            # Direct surname match
            if existing_parsed.surname.lower() == new_parsed.surname.lower():
                surnames_match = True

            # Maiden name matches
            if existing_parent.maiden_name:
                if existing_parent.maiden_name.lower() == new_parsed.surname.lower():
                    surnames_match = True
            if new_parent.maiden_name:
                if new_parent.maiden_name.lower() == existing_parsed.surname.lower():
                    surnames_match = True

            # Check if one person's surname matches the other's spouse's surname
            # (married name scenario)
            if not surnames_match:
                # Check if existing parent has a spouse with new parent's surname
                existing_spouse_ids = await spouse_link_repo.get_spouses(existing_parent_id)
                for spouse_id in existing_spouse_ids:
                    spouse = await person_repo.get_by_id(spouse_id)
                    if spouse and spouse.surname.lower() == new_parsed.surname.lower():
                        surnames_match = True
                        logger.info(
                            f"Detected married name match: '{existing_parent.given_name} {existing_parent.surname}' "
                            f"has spouse '{spouse.given_name} {spouse.surname}', matching '{new_parent.given_name} {new_parent.surname}'"
                        )
                        break

                # Check if new parent has a spouse with existing parent's surname
                if not surnames_match:
                    new_spouse_ids = await spouse_link_repo.get_spouses(new_parent_id)
                    for spouse_id in new_spouse_ids:
                        spouse = await person_repo.get_by_id(spouse_id)
                        if spouse and spouse.surname.lower() == existing_parsed.surname.lower():
                            surnames_match = True
                            logger.info(
                                f"Detected married name match: '{new_parent.given_name} {new_parent.surname}' "
                                f"has spouse '{spouse.given_name} {spouse.surname}', matching '{existing_parent.given_name} {existing_parent.surname}'"
                            )
                            break

            if surnames_match:
                # Check birth year proximity
                birth_year_match = True
                if existing_parent.birth_date and new_parent.birth_date:
                    year_diff = abs(existing_parent.birth_date.year - new_parent.birth_date.year)
                    birth_year_match = year_diff <= 5

                if birth_year_match:
                    # These are likely duplicates - merge them
                    logger.warning(
                        f"Detected duplicate parents for child {child_id}: "
                        f"'{existing_parent.given_name} {existing_parent.surname}' and "
                        f"'{new_parent.given_name} {new_parent.surname}'"
                    )

                    # Keep the one with the biography, or the existing one if neither has one
                    if new_parent.biography and not existing_parent.biography:
                        # Keep new, merge existing into it
                        await self._merge_persons(session, new_parent_id, existing_parent_id)
                        return new_parent_id
                    else:
                        # Keep existing, merge new into it
                        await self._merge_persons(session, existing_parent_id, new_parent_id)
                        return existing_parent_id

        # No duplicate found, but we have 3+ parents which is invalid
        # Log a warning but allow it (will need manual cleanup)
        logger.warning(
            f"Child {child_id} would have >2 parents but no duplicates detected. "
            f"Existing: {existing_parent_ids}, New: {new_parent_id}"
        )
        return new_parent_id

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
