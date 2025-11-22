"""Query service for searching and traversing the family tree."""

from datetime import date
from typing import Any
from uuid import UUID

from sqlmodel import select

from ancestral_synth.domain.models import Person
from ancestral_synth.persistence.database import Database
from ancestral_synth.persistence.repositories import (
    ChildLinkRepository,
    PersonRepository,
)
from ancestral_synth.persistence.tables import PersonTable


class QueryService:
    """Service for querying genealogical data."""

    def __init__(self, db: Database) -> None:
        """Initialize the query service.

        Args:
            db: Database connection.
        """
        self._db = db

    async def get_ancestors(
        self,
        person_id: UUID,
        generations: int | None = None,
    ) -> list[Person]:
        """Get ancestors of a person.

        Args:
            person_id: ID of the person.
            generations: Number of generations to traverse (None for all).

        Returns:
            List of ancestor Person objects.
        """
        ancestors: list[Person] = []

        async with self._db.session() as session:
            person_repo = PersonRepository(session)
            child_link_repo = ChildLinkRepository(session)

            # BFS to find ancestors
            current_gen = [person_id]
            gen_count = 0

            while current_gen:
                if generations is not None and gen_count >= generations:
                    break

                next_gen = []
                for pid in current_gen:
                    parent_ids = await child_link_repo.get_parents(pid)
                    for parent_id in parent_ids:
                        parent = await person_repo.get_by_id(parent_id)
                        if parent:
                            ancestors.append(person_repo.to_domain(parent))
                            next_gen.append(parent_id)

                current_gen = next_gen
                gen_count += 1

        return ancestors

    async def get_descendants(
        self,
        person_id: UUID,
        generations: int | None = None,
    ) -> list[Person]:
        """Get descendants of a person.

        Args:
            person_id: ID of the person.
            generations: Number of generations to traverse (None for all).

        Returns:
            List of descendant Person objects.
        """
        descendants: list[Person] = []

        async with self._db.session() as session:
            person_repo = PersonRepository(session)
            child_link_repo = ChildLinkRepository(session)

            # BFS to find descendants
            current_gen = [person_id]
            gen_count = 0

            while current_gen:
                if generations is not None and gen_count >= generations:
                    break

                next_gen = []
                for pid in current_gen:
                    child_ids = await child_link_repo.get_children(pid)
                    for child_id in child_ids:
                        child = await person_repo.get_by_id(child_id)
                        if child:
                            descendants.append(person_repo.to_domain(child))
                            next_gen.append(child_id)

                current_gen = next_gen
                gen_count += 1

        return descendants

    async def get_family_tree(
        self,
        person_id: UUID,
        ancestor_generations: int = 2,
        descendant_generations: int = 2,
    ) -> dict[str, Any]:
        """Get a family tree centered on a person.

        Args:
            person_id: ID of the center person.
            ancestor_generations: Generations of ancestors to include.
            descendant_generations: Generations of descendants to include.

        Returns:
            Dictionary with subject, ancestors, and descendants.
        """
        async with self._db.session() as session:
            person_repo = PersonRepository(session)
            db_person = await person_repo.get_by_id(person_id)

            if db_person is None:
                return {
                    "subject": None,
                    "ancestors": [],
                    "descendants": [],
                }

            subject = person_repo.to_domain(db_person)

        ancestors = await self.get_ancestors(person_id, ancestor_generations)
        descendants = await self.get_descendants(person_id, descendant_generations)

        return {
            "subject": subject,
            "ancestors": ancestors,
            "descendants": descendants,
        }

    async def search_by_birth_date_range(
        self,
        start_date: date,
        end_date: date,
    ) -> list[Person]:
        """Find persons born within a date range.

        Args:
            start_date: Start of the date range (inclusive).
            end_date: End of the date range (inclusive).

        Returns:
            List of matching Person objects.
        """
        async with self._db.session() as session:
            person_repo = PersonRepository(session)

            stmt = select(PersonTable).where(
                PersonTable.birth_date >= start_date,
                PersonTable.birth_date <= end_date,
            )
            result = await session.exec(stmt)
            persons = result.all()

            return [person_repo.to_domain(p) for p in persons]

    async def search_by_name(self, query: str) -> list[Person]:
        """Find persons by partial name match.

        Args:
            query: Name to search for (matches given name or surname).

        Returns:
            List of matching Person objects.
        """
        async with self._db.session() as session:
            person_repo = PersonRepository(session)

            search_pattern = f"%{query}%"
            stmt = select(PersonTable).where(
                (PersonTable.given_name.ilike(search_pattern))  # type: ignore[union-attr]
                | (PersonTable.surname.ilike(search_pattern))  # type: ignore[union-attr]
            )
            result = await session.exec(stmt)
            persons = result.all()

            return [person_repo.to_domain(p) for p in persons]

    async def get_person(self, person_id: UUID) -> Person | None:
        """Get a single person by ID.

        Args:
            person_id: ID of the person.

        Returns:
            Person object or None if not found.
        """
        async with self._db.session() as session:
            person_repo = PersonRepository(session)
            db_person = await person_repo.get_by_id(person_id)

            if db_person is None:
                return None

            return person_repo.to_domain(db_person)

    async def get_statistics(self) -> dict:
        """Get statistics about the dataset.

        Returns:
            Dictionary with dataset statistics.
        """
        from ancestral_synth.domain.enums import PersonStatus
        from ancestral_synth.persistence.repositories import QueueRepository

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
