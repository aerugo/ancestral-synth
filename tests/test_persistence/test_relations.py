"""Tests for relation-fetching functionality."""

from datetime import date
from uuid import UUID

import pytest

from ancestral_synth.domain.enums import Gender, PersonStatus
from ancestral_synth.domain.models import ChildLink, Person, SpouseLink
from ancestral_synth.persistence.database import Database
from ancestral_synth.persistence.repositories import (
    ChildLinkRepository,
    PersonRepository,
    SpouseLinkRepository,
)
from ancestral_synth.persistence.relations import (
    RelationsSummary,
    get_relations_summary,
)


class TestRelationsSummary:
    """Tests for RelationsSummary dataclass."""

    def test_create_empty_summary(self) -> None:
        """Should create empty relations summary."""
        summary = RelationsSummary()

        assert summary.parents == []
        assert summary.children == []
        assert summary.spouses == []
        assert summary.siblings == []
        assert summary.grandparents == []
        assert summary.grandchildren == []

    def test_has_relations_empty(self) -> None:
        """Empty summary should have no relations."""
        summary = RelationsSummary()

        assert summary.has_relations is False

    def test_has_relations_with_parent(self) -> None:
        """Summary with parent should have relations."""
        summary = RelationsSummary(parents=["John Smith"])

        assert summary.has_relations is True

    def test_has_relations_with_grandchild(self) -> None:
        """Summary with grandchild should have relations."""
        summary = RelationsSummary(grandchildren=["Baby Smith"])

        assert summary.has_relations is True


class TestGetRelationsSummary:
    """Tests for get_relations_summary function."""

    @pytest.mark.asyncio
    async def test_no_relations(self, test_db: Database) -> None:
        """Should return empty summary for person with no relations."""
        async with test_db.session() as session:
            person_repo = PersonRepository(session)
            person = Person(given_name="John", surname="Smith")
            await person_repo.create(person)

        async with test_db.session() as session:
            summary = await get_relations_summary(session, person.id)

            assert summary.parents == []
            assert summary.children == []
            assert summary.spouses == []
            assert summary.siblings == []
            assert summary.has_relations is False

    @pytest.mark.asyncio
    async def test_with_parents(self, test_db: Database) -> None:
        """Should include parent names."""
        async with test_db.session() as session:
            person_repo = PersonRepository(session)
            child_link_repo = ChildLinkRepository(session)

            father = Person(given_name="John", surname="Smith", gender=Gender.MALE)
            mother = Person(given_name="Mary", surname="Smith", gender=Gender.FEMALE)
            child = Person(given_name="James", surname="Smith", gender=Gender.MALE)

            await person_repo.create(father)
            await person_repo.create(mother)
            await person_repo.create(child)

            await child_link_repo.create(ChildLink(parent_id=father.id, child_id=child.id))
            await child_link_repo.create(ChildLink(parent_id=mother.id, child_id=child.id))

        async with test_db.session() as session:
            summary = await get_relations_summary(session, child.id)

            assert len(summary.parents) == 2
            assert "John Smith" in summary.parents
            assert "Mary Smith" in summary.parents
            assert summary.has_relations is True

    @pytest.mark.asyncio
    async def test_with_children(self, test_db: Database) -> None:
        """Should include children names."""
        async with test_db.session() as session:
            person_repo = PersonRepository(session)
            child_link_repo = ChildLinkRepository(session)

            parent = Person(given_name="John", surname="Smith")
            child1 = Person(given_name="James", surname="Smith")
            child2 = Person(given_name="Jane", surname="Smith")

            await person_repo.create(parent)
            await person_repo.create(child1)
            await person_repo.create(child2)

            await child_link_repo.create(ChildLink(parent_id=parent.id, child_id=child1.id))
            await child_link_repo.create(ChildLink(parent_id=parent.id, child_id=child2.id))

        async with test_db.session() as session:
            summary = await get_relations_summary(session, parent.id)

            assert len(summary.children) == 2
            assert "James Smith" in summary.children
            assert "Jane Smith" in summary.children

    @pytest.mark.asyncio
    async def test_with_spouse(self, test_db: Database) -> None:
        """Should include spouse names."""
        async with test_db.session() as session:
            person_repo = PersonRepository(session)
            spouse_link_repo = SpouseLinkRepository(session)

            person1 = Person(given_name="John", surname="Smith")
            person2 = Person(given_name="Mary", surname="Smith")

            await person_repo.create(person1)
            await person_repo.create(person2)

            await spouse_link_repo.create(SpouseLink(person1_id=person1.id, person2_id=person2.id))

        async with test_db.session() as session:
            summary = await get_relations_summary(session, person1.id)

            assert len(summary.spouses) == 1
            assert "Mary Smith" in summary.spouses

    @pytest.mark.asyncio
    async def test_with_siblings(self, test_db: Database) -> None:
        """Should include sibling names (other children of same parents)."""
        async with test_db.session() as session:
            person_repo = PersonRepository(session)
            child_link_repo = ChildLinkRepository(session)

            parent = Person(given_name="John", surname="Smith")
            child1 = Person(given_name="James", surname="Smith")
            child2 = Person(given_name="Jane", surname="Smith")
            child3 = Person(given_name="Jack", surname="Smith")

            await person_repo.create(parent)
            await person_repo.create(child1)
            await person_repo.create(child2)
            await person_repo.create(child3)

            # All three are children of the same parent
            await child_link_repo.create(ChildLink(parent_id=parent.id, child_id=child1.id))
            await child_link_repo.create(ChildLink(parent_id=parent.id, child_id=child2.id))
            await child_link_repo.create(ChildLink(parent_id=parent.id, child_id=child3.id))

        async with test_db.session() as session:
            # Get siblings of child1
            summary = await get_relations_summary(session, child1.id)

            assert len(summary.siblings) == 2
            assert "Jane Smith" in summary.siblings
            assert "Jack Smith" in summary.siblings
            # Should not include self
            assert "James Smith" not in summary.siblings

    @pytest.mark.asyncio
    async def test_with_grandparents(self, test_db: Database) -> None:
        """Should include grandparent names (parents of parents)."""
        async with test_db.session() as session:
            person_repo = PersonRepository(session)
            child_link_repo = ChildLinkRepository(session)

            grandpa = Person(given_name="George", surname="Smith")
            grandma = Person(given_name="Martha", surname="Smith")
            father = Person(given_name="John", surname="Smith")
            child = Person(given_name="James", surname="Smith")

            await person_repo.create(grandpa)
            await person_repo.create(grandma)
            await person_repo.create(father)
            await person_repo.create(child)

            # Grandparents -> Father
            await child_link_repo.create(ChildLink(parent_id=grandpa.id, child_id=father.id))
            await child_link_repo.create(ChildLink(parent_id=grandma.id, child_id=father.id))
            # Father -> Child
            await child_link_repo.create(ChildLink(parent_id=father.id, child_id=child.id))

        async with test_db.session() as session:
            summary = await get_relations_summary(session, child.id)

            assert len(summary.grandparents) == 2
            assert "George Smith" in summary.grandparents
            assert "Martha Smith" in summary.grandparents

    @pytest.mark.asyncio
    async def test_with_grandchildren(self, test_db: Database) -> None:
        """Should include grandchildren names (children of children)."""
        async with test_db.session() as session:
            person_repo = PersonRepository(session)
            child_link_repo = ChildLinkRepository(session)

            grandpa = Person(given_name="George", surname="Smith")
            father = Person(given_name="John", surname="Smith")
            grandchild1 = Person(given_name="James", surname="Smith")
            grandchild2 = Person(given_name="Jane", surname="Smith")

            await person_repo.create(grandpa)
            await person_repo.create(father)
            await person_repo.create(grandchild1)
            await person_repo.create(grandchild2)

            # Grandpa -> Father
            await child_link_repo.create(ChildLink(parent_id=grandpa.id, child_id=father.id))
            # Father -> Grandchildren
            await child_link_repo.create(ChildLink(parent_id=father.id, child_id=grandchild1.id))
            await child_link_repo.create(ChildLink(parent_id=father.id, child_id=grandchild2.id))

        async with test_db.session() as session:
            summary = await get_relations_summary(session, grandpa.id)

            assert len(summary.grandchildren) == 2
            assert "James Smith" in summary.grandchildren
            assert "Jane Smith" in summary.grandchildren

    @pytest.mark.asyncio
    async def test_complex_family(self, test_db: Database) -> None:
        """Should handle complex family with multiple relation types."""
        async with test_db.session() as session:
            person_repo = PersonRepository(session)
            child_link_repo = ChildLinkRepository(session)
            spouse_link_repo = SpouseLinkRepository(session)

            # Build a family tree
            grandpa = Person(given_name="George", surname="Smith")
            grandma = Person(given_name="Martha", surname="Smith")
            father = Person(given_name="John", surname="Smith")
            mother = Person(given_name="Mary", surname="Jones")
            uncle = Person(given_name="Bob", surname="Smith")
            subject = Person(given_name="James", surname="Smith")
            sibling = Person(given_name="Jane", surname="Smith")
            child = Person(given_name="Baby", surname="Smith")

            for p in [grandpa, grandma, father, mother, uncle, subject, sibling, child]:
                await person_repo.create(p)

            # Grandparents -> Father and Uncle
            await child_link_repo.create(ChildLink(parent_id=grandpa.id, child_id=father.id))
            await child_link_repo.create(ChildLink(parent_id=grandma.id, child_id=father.id))
            await child_link_repo.create(ChildLink(parent_id=grandpa.id, child_id=uncle.id))
            await child_link_repo.create(ChildLink(parent_id=grandma.id, child_id=uncle.id))

            # Parents -> Subject and Sibling
            await child_link_repo.create(ChildLink(parent_id=father.id, child_id=subject.id))
            await child_link_repo.create(ChildLink(parent_id=mother.id, child_id=subject.id))
            await child_link_repo.create(ChildLink(parent_id=father.id, child_id=sibling.id))
            await child_link_repo.create(ChildLink(parent_id=mother.id, child_id=sibling.id))

            # Subject -> Child
            await child_link_repo.create(ChildLink(parent_id=subject.id, child_id=child.id))

            # Father and Mother are spouses
            await spouse_link_repo.create(SpouseLink(person1_id=father.id, person2_id=mother.id))

        async with test_db.session() as session:
            summary = await get_relations_summary(session, subject.id)

            # Verify all relations
            assert len(summary.parents) == 2
            assert "John Smith" in summary.parents
            assert "Mary Jones" in summary.parents

            assert len(summary.siblings) == 1
            assert "Jane Smith" in summary.siblings

            assert len(summary.children) == 1
            assert "Baby Smith" in summary.children

            assert len(summary.grandparents) == 2
            assert "George Smith" in summary.grandparents
            assert "Martha Smith" in summary.grandparents
