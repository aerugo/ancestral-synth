"""Tests for query service - TDD: Write tests first."""

from datetime import date
from uuid import uuid4

import pytest

from ancestral_synth.domain.enums import Gender, PersonStatus
from ancestral_synth.domain.models import ChildLink, Person
from ancestral_synth.persistence.database import Database
from ancestral_synth.persistence.repositories import (
    ChildLinkRepository,
    PersonRepository,
)


class TestQueryService:
    """Tests for QueryService."""

    @pytest.fixture
    async def family_tree_db(self, test_db: Database):
        """Create a multi-generation family tree."""
        #
        # Family tree structure:
        #   great_grandparent (gen -3)
        #         |
        #   grandparent (gen -2)
        #         |
        #   parent (gen -1)
        #         |
        #   subject (gen 0)
        #         |
        #   child (gen 1)
        #         |
        #   grandchild (gen 2)
        #

        async with test_db.session() as session:
            person_repo = PersonRepository(session)
            child_link_repo = ChildLinkRepository(session)

            great_grandparent = Person(
                id=uuid4(),
                given_name="Augustus",
                surname="Smith",
                gender=Gender.MALE,
                birth_date=date(1870, 1, 1),
                death_date=date(1950, 1, 1),
                status=PersonStatus.COMPLETE,
                generation=-3,
            )

            grandparent = Person(
                id=uuid4(),
                given_name="William",
                surname="Smith",
                gender=Gender.MALE,
                birth_date=date(1900, 1, 1),
                death_date=date(1980, 1, 1),
                status=PersonStatus.COMPLETE,
                generation=-2,
            )

            parent = Person(
                id=uuid4(),
                given_name="John",
                surname="Smith",
                gender=Gender.MALE,
                birth_date=date(1930, 1, 1),
                death_date=date(2010, 1, 1),
                status=PersonStatus.COMPLETE,
                generation=-1,
            )

            subject = Person(
                id=uuid4(),
                given_name="James",
                surname="Smith",
                gender=Gender.MALE,
                birth_date=date(1960, 1, 1),
                status=PersonStatus.COMPLETE,
                generation=0,
            )

            child = Person(
                id=uuid4(),
                given_name="Michael",
                surname="Smith",
                gender=Gender.MALE,
                birth_date=date(1990, 1, 1),
                status=PersonStatus.COMPLETE,
                generation=1,
            )

            grandchild = Person(
                id=uuid4(),
                given_name="David",
                surname="Smith",
                gender=Gender.MALE,
                birth_date=date(2020, 1, 1),
                status=PersonStatus.PENDING,
                generation=2,
            )

            await person_repo.create(great_grandparent)
            await person_repo.create(grandparent)
            await person_repo.create(parent)
            await person_repo.create(subject)
            await person_repo.create(child)
            await person_repo.create(grandchild)

            await child_link_repo.create(ChildLink(
                parent_id=great_grandparent.id, child_id=grandparent.id
            ))
            await child_link_repo.create(ChildLink(
                parent_id=grandparent.id, child_id=parent.id
            ))
            await child_link_repo.create(ChildLink(
                parent_id=parent.id, child_id=subject.id
            ))
            await child_link_repo.create(ChildLink(
                parent_id=subject.id, child_id=child.id
            ))
            await child_link_repo.create(ChildLink(
                parent_id=child.id, child_id=grandchild.id
            ))

        return (
            test_db,
            great_grandparent,
            grandparent,
            parent,
            subject,
            child,
            grandchild,
        )

    @pytest.mark.asyncio
    async def test_get_ancestors(self, family_tree_db) -> None:
        """Should get ancestors up to n generations."""
        from ancestral_synth.services.query_service import QueryService

        db, _, grandparent, parent, subject, _, _ = family_tree_db

        service = QueryService(db)
        ancestors = await service.get_ancestors(subject.id, generations=2)

        # Should get parent and grandparent (2 generations)
        assert len(ancestors) == 2
        ancestor_ids = {a.id for a in ancestors}
        assert parent.id in ancestor_ids
        assert grandparent.id in ancestor_ids

    @pytest.mark.asyncio
    async def test_get_all_ancestors(self, family_tree_db) -> None:
        """Should get all ancestors when no generation limit."""
        from ancestral_synth.services.query_service import QueryService

        db, great_grandparent, grandparent, parent, subject, _, _ = family_tree_db

        service = QueryService(db)
        ancestors = await service.get_ancestors(subject.id, generations=None)

        # Should get all 3 ancestors
        assert len(ancestors) == 3
        ancestor_ids = {a.id for a in ancestors}
        assert parent.id in ancestor_ids
        assert grandparent.id in ancestor_ids
        assert great_grandparent.id in ancestor_ids

    @pytest.mark.asyncio
    async def test_get_descendants(self, family_tree_db) -> None:
        """Should get descendants up to n generations."""
        from ancestral_synth.services.query_service import QueryService

        db, _, _, _, subject, child, grandchild = family_tree_db

        service = QueryService(db)
        descendants = await service.get_descendants(subject.id, generations=2)

        # Should get child and grandchild
        assert len(descendants) == 2
        descendant_ids = {d.id for d in descendants}
        assert child.id in descendant_ids
        assert grandchild.id in descendant_ids

    @pytest.mark.asyncio
    async def test_get_descendants_one_generation(self, family_tree_db) -> None:
        """Should get only direct children with generations=1."""
        from ancestral_synth.services.query_service import QueryService

        db, _, _, _, subject, child, grandchild = family_tree_db

        service = QueryService(db)
        descendants = await service.get_descendants(subject.id, generations=1)

        assert len(descendants) == 1
        assert descendants[0].id == child.id

    @pytest.mark.asyncio
    async def test_get_family_tree(self, family_tree_db) -> None:
        """Should get complete family tree from a person."""
        from ancestral_synth.services.query_service import QueryService

        db, _, _, parent, subject, child, _ = family_tree_db

        service = QueryService(db)
        tree = await service.get_family_tree(
            subject.id, ancestor_generations=1, descendant_generations=1
        )

        assert "subject" in tree
        assert tree["subject"].id == subject.id
        assert "ancestors" in tree
        assert "descendants" in tree

        # Should have 1 ancestor (parent)
        assert len(tree["ancestors"]) == 1
        assert tree["ancestors"][0].id == parent.id

        # Should have 1 descendant (child)
        assert len(tree["descendants"]) == 1
        assert tree["descendants"][0].id == child.id

    @pytest.mark.asyncio
    async def test_search_by_date_range(self, family_tree_db) -> None:
        """Should find persons born within date range."""
        from ancestral_synth.services.query_service import QueryService

        db, _, _, _, subject, child, _ = family_tree_db

        service = QueryService(db)
        persons = await service.search_by_birth_date_range(
            start_date=date(1950, 1, 1),
            end_date=date(2000, 1, 1),
        )

        # Should find subject (1960) and child (1990)
        assert len(persons) == 2
        person_ids = {p.id for p in persons}
        assert subject.id in person_ids
        assert child.id in person_ids

    @pytest.mark.asyncio
    async def test_search_by_name(self, family_tree_db) -> None:
        """Should find persons by partial name match."""
        from ancestral_synth.services.query_service import QueryService

        db, _, _, _, subject, _, _ = family_tree_db

        service = QueryService(db)

        # Search by first name
        persons = await service.search_by_name("James")
        assert len(persons) >= 1
        assert any(p.given_name == "James" for p in persons)

        # Search by surname
        persons = await service.search_by_name("Smith")
        assert len(persons) >= 1  # All have surname Smith

    @pytest.mark.asyncio
    async def test_empty_results(self, family_tree_db) -> None:
        """Should return empty list when no matches."""
        from ancestral_synth.services.query_service import QueryService

        db, _, _, _, _, _, grandchild = family_tree_db

        service = QueryService(db)

        # Grandchild has no descendants
        descendants = await service.get_descendants(grandchild.id, generations=1)
        assert descendants == []

    @pytest.mark.asyncio
    async def test_person_not_found(self, family_tree_db) -> None:
        """Should return empty for non-existent person."""
        from ancestral_synth.services.query_service import QueryService

        db, _, _, _, _, _, _ = family_tree_db

        service = QueryService(db)

        ancestors = await service.get_ancestors(uuid4(), generations=1)
        assert ancestors == []
