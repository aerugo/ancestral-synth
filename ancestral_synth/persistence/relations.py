"""Module for fetching family relations."""

from dataclasses import dataclass, field
from uuid import UUID

from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from ancestral_synth.persistence.tables import ChildLinkTable, PersonTable, SpouseLinkTable


@dataclass
class RelationsSummary:
    """Summary of a person's family relations."""

    parents: list[str] = field(default_factory=list)
    children: list[str] = field(default_factory=list)
    spouses: list[str] = field(default_factory=list)
    siblings: list[str] = field(default_factory=list)
    grandparents: list[str] = field(default_factory=list)
    grandchildren: list[str] = field(default_factory=list)

    @property
    def has_relations(self) -> bool:
        """Check if there are any relations."""
        return bool(
            self.parents
            or self.children
            or self.spouses
            or self.siblings
            or self.grandparents
            or self.grandchildren
        )


async def get_relations_summary(session: AsyncSession, person_id: UUID) -> RelationsSummary:
    """Get a summary of all first and second degree relations for a person.

    First degree: parents, children, spouses, siblings
    Second degree: grandparents, grandchildren

    Args:
        session: Database session.
        person_id: ID of the person to get relations for.

    Returns:
        RelationsSummary with names of all related people.
    """
    summary = RelationsSummary()

    # Get parents (first degree)
    parent_ids = await _get_parent_ids(session, person_id)
    summary.parents = await _get_names_by_ids(session, parent_ids)

    # Get children (first degree)
    child_ids = await _get_child_ids(session, person_id)
    summary.children = await _get_names_by_ids(session, child_ids)

    # Get spouses (first degree)
    spouse_ids = await _get_spouse_ids(session, person_id)
    summary.spouses = await _get_names_by_ids(session, spouse_ids)

    # Get siblings (first degree) - other children of same parents
    sibling_ids: set[UUID] = set()
    for parent_id in parent_ids:
        parent_child_ids = await _get_child_ids(session, parent_id)
        sibling_ids.update(parent_child_ids)
    sibling_ids.discard(person_id)  # Remove self
    summary.siblings = await _get_names_by_ids(session, list(sibling_ids))

    # Get grandparents (second degree) - parents of parents
    grandparent_ids: set[UUID] = set()
    for parent_id in parent_ids:
        gp_ids = await _get_parent_ids(session, parent_id)
        grandparent_ids.update(gp_ids)
    summary.grandparents = await _get_names_by_ids(session, list(grandparent_ids))

    # Get grandchildren (second degree) - children of children
    grandchild_ids: set[UUID] = set()
    for child_id in child_ids:
        gc_ids = await _get_child_ids(session, child_id)
        grandchild_ids.update(gc_ids)
    summary.grandchildren = await _get_names_by_ids(session, list(grandchild_ids))

    return summary


async def _get_parent_ids(session: AsyncSession, child_id: UUID) -> list[UUID]:
    """Get parent IDs for a child."""
    stmt = select(ChildLinkTable.parent_id).where(ChildLinkTable.child_id == child_id)
    result = await session.exec(stmt)
    return list(result.all())


async def _get_child_ids(session: AsyncSession, parent_id: UUID) -> list[UUID]:
    """Get child IDs for a parent."""
    stmt = select(ChildLinkTable.child_id).where(ChildLinkTable.parent_id == parent_id)
    result = await session.exec(stmt)
    return list(result.all())


async def _get_spouse_ids(session: AsyncSession, person_id: UUID) -> list[UUID]:
    """Get spouse IDs for a person."""
    # Check both directions since person could be person1 or person2
    stmt1 = select(SpouseLinkTable.person2_id).where(SpouseLinkTable.person1_id == person_id)
    stmt2 = select(SpouseLinkTable.person1_id).where(SpouseLinkTable.person2_id == person_id)

    result1 = await session.exec(stmt1)
    result2 = await session.exec(stmt2)

    return list(result1.all()) + list(result2.all())


async def _get_names_by_ids(session: AsyncSession, person_ids: list[UUID]) -> list[str]:
    """Get full names for a list of person IDs."""
    if not person_ids:
        return []

    stmt = select(PersonTable).where(PersonTable.id.in_(person_ids))  # type: ignore[union-attr]
    result = await session.exec(stmt)
    people = result.all()

    return [f"{p.given_name} {p.surname}" for p in people]
