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
from ancestral_synth.agents.dedup_agent import DedupAgent, heuristic_match_score
from ancestral_synth.agents.extraction_agent import ExtractionAgent
from ancestral_synth.config import settings
from ancestral_synth.utils.rate_limiter import RateLimitConfig, RateLimiter
from ancestral_synth.utils.timing import VerboseTimer, set_verbose_log_callback
from ancestral_synth.domain.enums import (
    EventType,
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
)
from ancestral_synth.persistence.database import Database
from ancestral_synth.persistence.repositories import (
    ChildLinkRepository,
    EventRepository,
    NoteRepository,
    PersonRepository,
    QueueRepository,
)
from ancestral_synth.services.validation import ValidationResult, Validator


class GenealogyService:
    """Main service for generating and managing the genealogical dataset."""

    def __init__(
        self,
        db: Database,
        biography_agent: BiographyAgent | None = None,
        extraction_agent: ExtractionAgent | None = None,
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
            dedup_agent: Agent for deduplication.
            validator: Validator for genealogical plausibility.
            rate_limiter: Rate limiter for LLM API calls.
            verbose: Enable verbose output with timing information.
        """
        self._db = db
        self._biography_agent = biography_agent or BiographyAgent()
        self._extraction_agent = extraction_agent or ExtractionAgent()
        self._dedup_agent = dedup_agent or DedupAgent()
        self._validator = validator or Validator()
        self._rate_limiter = rate_limiter or RateLimiter(
            RateLimitConfig(requests_per_minute=settings.llm_requests_per_minute)
        )
        self._timer = VerboseTimer(enabled=verbose)

        # Set up global verbose logging callback for retry module
        if verbose:
            set_verbose_log_callback(self._timer.log)
        else:
            set_verbose_log_callback(None)

    @property
    def timer(self) -> VerboseTimer:
        """Get the verbose timer for external access."""
        return self._timer

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

        # Generate biography (rate limited)
        logger.info(f"Generating biography for seed: {context.given_name} {context.surname}")
        self._timer.log(f"Generating ~{settings.biography_word_count}-word biography using {settings.llm_provider}:{settings.llm_model}")
        await self._acquire_rate_limit("biography generation")
        with self._timer.time_operation("Biography generation (LLM call)"):
            biography = await self._biography_agent.generate(context)
        self._timer.log(f"Generated {biography.word_count} words")

        # Extract data (rate limited)
        await self._acquire_rate_limit("data extraction")
        with self._timer.time_operation("Data extraction (LLM call)"):
            extracted = await self._extraction_agent.extract(biography.content)
        self._timer.log(f"Extracted {len(extracted.parents)} parents, {len(extracted.children)} children, {len(extracted.events)} events")

        # Validate
        with self._timer.time_operation("Validation", show_start=False):
            validation = self._validator.validate_extracted_data(extracted)
        if not validation.is_valid:
            logger.warning(f"Validation errors for seed: {validation.errors}")

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

        # Generate biography (rate limited)
        logger.info(f"Generating biography for: {db_person.given_name} {db_person.surname}")
        self._timer.log(f"Generating ~{settings.biography_word_count}-word biography using {settings.llm_provider}:{settings.llm_model}")
        if relatives:
            self._timer.log(f"Context includes {len(relatives)} known relative(s)")
        await self._acquire_rate_limit("biography generation")
        with self._timer.time_operation("Biography generation (LLM call)"):
            biography = await self._biography_agent.generate(context)
        self._timer.log(f"Generated {biography.word_count} words")

        # Extract data (rate limited)
        await self._acquire_rate_limit("data extraction")
        with self._timer.time_operation("Data extraction (LLM call)"):
            extracted = await self._extraction_agent.extract_with_hints(
                biography.content,
                expected_name=f"{db_person.given_name} {db_person.surname}",
                expected_birth_year=db_person.birth_date.year if db_person.birth_date else None,
            )
        self._timer.log(f"Extracted {len(extracted.parents)} parents, {len(extracted.children)} children, {len(extracted.events)} events")

        # Validate
        with self._timer.time_operation("Validation", show_start=False):
            validation = self._validator.validate_extracted_data(extracted)
        if not validation.is_valid:
            logger.warning(f"Validation errors: {validation.errors}")
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

            # Refresh person
            db_person = await person_repo.get_by_id(person_id)
            return person_repo.to_domain(db_person)  # type: ignore[arg-type]

    async def _gather_relative_context(
        self,
        session,  # noqa: ANN001
        person_id: UUID,
    ) -> list[PersonSummary]:
        """Gather context about known relatives."""
        person_repo = PersonRepository(session)
        child_link_repo = ChildLinkRepository(session)

        relatives: list[PersonSummary] = []

        # Get parents
        parent_ids = await child_link_repo.get_parents(person_id)
        for parent_id in parent_ids:
            parent = await person_repo.get_by_id(parent_id)
            if parent:
                summary = person_repo.to_summary(parent, RelationshipType.PARENT)
                if parent.biography:
                    summary.key_facts = self._extract_key_facts(parent.biography)
                relatives.append(summary)

        # Get children
        child_ids = await child_link_repo.get_children(person_id)
        for child_id in child_ids:
            child = await person_repo.get_by_id(child_id)
            if child:
                summary = person_repo.to_summary(child, RelationshipType.CHILD)
                if child.biography:
                    summary.key_facts = self._extract_key_facts(child.biography)
                relatives.append(summary)

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

        # Process spouses - these stay pending (not queued) unless they become parents
        for spouse_ref in extracted.spouses:
            await self._resolve_person_reference(session, spouse_ref, person.generation)

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
                dedup_result = await self._dedup_agent.check_duplicate(
                    new_summary, candidate_summaries
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
