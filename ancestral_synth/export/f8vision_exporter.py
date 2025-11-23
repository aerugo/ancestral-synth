"""F8Vision-web YAML exporter for genealogical data."""

import random
from dataclasses import dataclass
from typing import IO, Any
from uuid import UUID

import yaml
from sqlmodel import select

from ancestral_synth.persistence.database import Database
from ancestral_synth.persistence.tables import (
    ChildLinkTable,
    PersonTable,
    SpouseLinkTable,
)


@dataclass
class SiblingGroupExport:
    """A group of siblings sharing the same parent(s) for export."""

    parent_names: tuple[str, ...]  # Names of shared parents (sorted alphabetically)
    sibling_ids: list[UUID]  # IDs of siblings in this group

    @property
    def label(self) -> str:
        """Generate a label like 'Siblings through Adam and Eve'."""
        if len(self.parent_names) == 0:
            return "Siblings"
        elif len(self.parent_names) == 1:
            return f"Siblings through {self.parent_names[0]}"
        else:
            return f"Siblings through {self.parent_names[0]} and {self.parent_names[1]}"


class F8VisionExporter:
    """Export genealogical data to f8vision-web YAML format.

    The f8vision-web format is a YAML file containing:
    - meta: Metadata about the family tree (title, centeredPersonId, description)
    - people: Array of person records with relationships

    See: https://github.com/f8vision/f8vision-web for format specification.
    """

    def __init__(self, db: Database) -> None:
        """Initialize the exporter.

        Args:
            db: Database connection.
        """
        self._db = db

    async def export(
        self,
        output: IO[str],
        title: str | None = None,
        centered_person_id: UUID | None = None,
        description: str | None = None,
    ) -> None:
        """Export all data to f8vision-web YAML format.

        Args:
            output: Output stream to write YAML to.
            title: Optional title for the family tree.
            centered_person_id: Optional ID of person to center visualization on.
            description: Optional description of the family tree.
        """
        data = await self._gather_data(title, centered_person_id, description)

        # Use safe_dump with allow_unicode for proper character handling
        yaml.safe_dump(
            data,
            output,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
            width=120,
        )

    async def _gather_data(
        self,
        title: str | None,
        centered_person_id: UUID | None,
        description: str | None,
    ) -> dict[str, Any]:
        """Gather all data for export in f8vision-web format."""
        async with self._db.session() as session:
            # Get all persons
            result = await session.exec(select(PersonTable))
            persons = list(result.all())

            # Get all child links (parent-child relationships)
            result = await session.exec(select(ChildLinkTable))
            child_links = list(result.all())

            # Get all spouse links
            result = await session.exec(select(SpouseLinkTable))
            spouse_links = list(result.all())

        # Build relationship maps
        parent_to_children = self._build_parent_to_children_map(child_links)
        child_to_parents = self._build_child_to_parents_map(child_links)
        person_to_spouses = self._build_spouse_map(spouse_links)

        # Build person ID to name map
        person_id_to_name = {p.id: f"{p.given_name} {p.surname}".strip() for p in persons}

        # Build sibling groups for each person
        person_to_sibling_groups = self._build_sibling_groups(
            persons, parent_to_children, child_to_parents, person_id_to_name
        )

        # Convert persons to f8vision format
        people = [
            self._person_to_f8vision(
                person,
                parent_to_children.get(person.id, []),
                child_to_parents.get(person.id, []),
                person_to_spouses.get(person.id, []),
                person_to_sibling_groups.get(person.id, []),
            )
            for person in persons
        ]

        # Build result
        result_data: dict[str, Any] = {}

        # Add meta section
        meta: dict[str, Any] = {}
        if title:
            meta["title"] = title
        if centered_person_id:
            meta["centeredPersonId"] = self._format_id(centered_person_id)
        elif persons:
            # Find most central person based on connection count
            most_central = self._find_most_central_person(
                persons, parent_to_children, child_to_parents, person_to_spouses
            )
            meta["centeredPersonId"] = self._format_id(most_central)
        if description:
            meta["description"] = description

        if meta:
            result_data["meta"] = meta

        result_data["people"] = people

        return result_data

    def _person_to_f8vision(
        self,
        person: PersonTable,
        child_ids: list[UUID],
        parent_ids: list[UUID],
        spouse_ids: list[UUID],
        sibling_groups: list[SiblingGroupExport],
    ) -> dict[str, Any]:
        """Convert a person to f8vision-web format.

        Args:
            person: The person database record.
            child_ids: List of this person's children's IDs.
            parent_ids: List of this person's parents' IDs.
            spouse_ids: List of this person's spouses' IDs.
            sibling_groups: List of sibling groups (grouped by shared parents).

        Returns:
            Dictionary in f8vision-web person format.
        """
        result: dict[str, Any] = {
            "id": self._format_id(person.id),
            "name": f"{person.given_name} {person.surname}".strip(),
        }

        # Add optional date fields
        if person.birth_date:
            result["birthDate"] = person.birth_date.isoformat()

        if person.death_date:
            result["deathDate"] = person.death_date.isoformat()

        # Add biography if present
        if person.biography:
            result["biography"] = person.biography

        # Add relationship arrays (only if non-empty)
        if parent_ids:
            result["parentIds"] = [self._format_id(pid) for pid in parent_ids]

        if spouse_ids:
            result["spouseIds"] = [self._format_id(sid) for sid in spouse_ids]

        if child_ids:
            result["childIds"] = [self._format_id(cid) for cid in child_ids]

        # Add sibling groups (grouped by shared parents)
        if sibling_groups:
            result["siblings"] = [
                {
                    "label": group.label,
                    "siblingIds": [self._format_id(sid) for sid in group.sibling_ids],
                }
                for group in sibling_groups
            ]

        return result

    def _format_id(self, uuid: UUID) -> str:
        """Format a UUID as a string ID for f8vision-web.

        Uses a shorter format for readability while maintaining uniqueness.

        Args:
            uuid: The UUID to format.

        Returns:
            String representation of the ID.
        """
        # Use full UUID string for guaranteed uniqueness
        return str(uuid)

    def _build_parent_to_children_map(
        self, child_links: list[ChildLinkTable]
    ) -> dict[UUID, list[UUID]]:
        """Build a map from parent IDs to their children's IDs.

        Args:
            child_links: List of parent-child relationship records.

        Returns:
            Dictionary mapping parent UUIDs to lists of child UUIDs.
        """
        result: dict[UUID, list[UUID]] = {}
        for link in child_links:
            if link.parent_id not in result:
                result[link.parent_id] = []
            result[link.parent_id].append(link.child_id)
        return result

    def _build_child_to_parents_map(
        self, child_links: list[ChildLinkTable]
    ) -> dict[UUID, list[UUID]]:
        """Build a map from child IDs to their parents' IDs.

        Args:
            child_links: List of parent-child relationship records.

        Returns:
            Dictionary mapping child UUIDs to lists of parent UUIDs.
        """
        result: dict[UUID, list[UUID]] = {}
        for link in child_links:
            if link.child_id not in result:
                result[link.child_id] = []
            result[link.child_id].append(link.parent_id)
        return result

    def _build_spouse_map(
        self, spouse_links: list[SpouseLinkTable]
    ) -> dict[UUID, list[UUID]]:
        """Build a bidirectional map of spouse relationships.

        Args:
            spouse_links: List of spouse relationship records.

        Returns:
            Dictionary mapping person UUIDs to lists of spouse UUIDs.
        """
        result: dict[UUID, list[UUID]] = {}
        for link in spouse_links:
            # Add both directions
            if link.person1_id not in result:
                result[link.person1_id] = []
            if link.person2_id not in result:
                result[link.person2_id] = []

            result[link.person1_id].append(link.person2_id)
            result[link.person2_id].append(link.person1_id)

        return result

    def _build_sibling_groups(
        self,
        persons: list[PersonTable],
        parent_to_children: dict[UUID, list[UUID]],
        child_to_parents: dict[UUID, list[UUID]],
        person_id_to_name: dict[UUID, str],
    ) -> dict[UUID, list[SiblingGroupExport]]:
        """Build sibling groups for each person, grouped by shared parents.

        Siblings are grouped by which parents they share:
        - Full siblings share both parents
        - Half-siblings share only one parent

        Args:
            persons: List of all persons.
            parent_to_children: Map of parent IDs to child IDs.
            child_to_parents: Map of child IDs to parent IDs.
            person_id_to_name: Map of person IDs to full names.

        Returns:
            Dictionary mapping person UUIDs to lists of SiblingGroupExport.
        """
        result: dict[UUID, list[SiblingGroupExport]] = {}

        for person in persons:
            parent_ids = child_to_parents.get(person.id, [])
            if not parent_ids:
                # No parents means no siblings through parents
                continue

            # Find all siblings and track which parents they share
            sibling_to_shared_parents: dict[UUID, set[UUID]] = {}

            for parent_id in parent_ids:
                children_of_parent = parent_to_children.get(parent_id, [])
                for child_id in children_of_parent:
                    if child_id != person.id:  # Exclude self
                        if child_id not in sibling_to_shared_parents:
                            sibling_to_shared_parents[child_id] = set()
                        sibling_to_shared_parents[child_id].add(parent_id)

            if not sibling_to_shared_parents:
                continue

            # Group siblings by their shared parent set
            parent_set_to_siblings: dict[frozenset[UUID], list[UUID]] = {}
            for sibling_id, shared_parent_ids in sibling_to_shared_parents.items():
                key = frozenset(shared_parent_ids)
                if key not in parent_set_to_siblings:
                    parent_set_to_siblings[key] = []
                parent_set_to_siblings[key].append(sibling_id)

            # Build SiblingGroupExport objects
            sibling_groups: list[SiblingGroupExport] = []
            for parent_set, sib_ids in parent_set_to_siblings.items():
                # Get parent names sorted alphabetically
                parent_names = tuple(sorted(
                    person_id_to_name.get(pid, "Unknown") for pid in parent_set
                ))
                sibling_groups.append(SiblingGroupExport(
                    parent_names=parent_names,
                    sibling_ids=sib_ids,
                ))

            result[person.id] = sibling_groups

        return result

    def _find_most_central_person(
        self,
        persons: list[PersonTable],
        parent_to_children: dict[UUID, list[UUID]],
        child_to_parents: dict[UUID, list[UUID]],
        person_to_spouses: dict[UUID, list[UUID]],
    ) -> UUID:
        """Find the most central person based on number of direct connections.

        Centrality is measured by degree (number of connections): parents +
        children + spouses. If multiple persons are equally central, one is
        chosen at random.

        Args:
            persons: List of all persons.
            parent_to_children: Map of parent IDs to child IDs.
            child_to_parents: Map of child IDs to parent IDs.
            person_to_spouses: Map of person IDs to spouse IDs.

        Returns:
            UUID of the most central person.
        """
        if not persons:
            raise ValueError("Cannot find central person: no persons in dataset")

        # Calculate degree centrality for each person
        person_degrees: list[tuple[UUID, int]] = []
        for person in persons:
            degree = (
                len(parent_to_children.get(person.id, []))  # children
                + len(child_to_parents.get(person.id, []))  # parents
                + len(person_to_spouses.get(person.id, []))  # spouses
            )
            person_degrees.append((person.id, degree))

        # Find the maximum degree
        max_degree = max(degree for _, degree in person_degrees)

        # Get all persons with the maximum degree
        most_central_candidates = [
            person_id for person_id, degree in person_degrees if degree == max_degree
        ]

        # Pick one at random if there are ties
        return random.choice(most_central_candidates)
