"""F8Vision-web exporter for genealogical data (Ancestral-Synth JSON format)."""

import json
import random
from datetime import datetime
from typing import IO, Any
from uuid import UUID

from sqlmodel import select

from ancestral_synth.persistence.database import Database
from ancestral_synth.persistence.tables import (
    ChildLinkTable,
    EventTable,
    NoteTable,
    PersonTable,
    SpouseLinkTable,
)


class F8VisionExporter:
    """Export genealogical data to f8vision-web format.

    Supports the Ancestral-Synth JSON format with:
    - Rich person records (all fields)
    - Events array with firefly visualization data
    - Notes array with categories and sources
    - Separate relationship links (child_links, spouse_links)

    See: https://github.com/f8vision/f8vision-web for format specification.
    """

    FORMAT_VERSION = "1.0"
    FORMAT_NAME = "ancestral-synth-json"

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
        truncate_descriptions: int | None = None,
    ) -> None:
        """Export all data to f8vision-web JSON format.

        Args:
            output: Output stream to write JSON to.
            title: Optional title for the family tree.
            centered_person_id: Optional ID of person to center visualization on.
            description: Optional description of the family tree.
            truncate_descriptions: If set, truncate biography fields to this many characters.
        """
        data = await self._gather_data(
            title, centered_person_id, description, truncate_descriptions
        )

        # Write JSON with proper formatting
        json.dump(data, output, indent=2, ensure_ascii=False)

    async def _gather_data(
        self,
        title: str | None,
        centered_person_id: UUID | None,
        description: str | None,
        truncate_descriptions: int | None = None,
    ) -> dict[str, Any]:
        """Gather all data for export in ancestral-synth JSON format."""
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

            # Get all events
            result = await session.exec(select(EventTable))
            events = list(result.all())

            # Get all notes
            result = await session.exec(select(NoteTable))
            notes = list(result.all())

        # Build relationship maps for finding central person
        parent_to_children = self._build_parent_to_children_map(child_links)
        child_to_parents = self._build_child_to_parents_map(child_links)
        person_to_spouses = self._build_spouse_map(spouse_links)

        # Build result
        result_data: dict[str, Any] = {}

        # Add metadata section
        metadata: dict[str, Any] = {
            "format": self.FORMAT_NAME,
            "version": self.FORMAT_VERSION,
            "exported_at": datetime.utcnow().isoformat() + "Z",
        }

        if title:
            metadata["title"] = title
        if description:
            metadata["description"] = description
        if truncate_descriptions:
            metadata["truncated"] = truncate_descriptions

        if centered_person_id:
            metadata["centeredPersonId"] = self._format_id(centered_person_id)
        elif persons:
            # Find most central person based on connection count
            most_central = self._find_most_central_person(
                persons, parent_to_children, child_to_parents, person_to_spouses
            )
            metadata["centeredPersonId"] = self._format_id(most_central)

        result_data["metadata"] = metadata

        # Convert persons to f8vision format (with all fields)
        result_data["persons"] = [
            self._person_to_f8vision(person, truncate_descriptions)
            for person in persons
        ]

        # Add events array
        result_data["events"] = [self._event_to_f8vision(event) for event in events]

        # Add notes array
        result_data["notes"] = [self._note_to_f8vision(note) for note in notes]

        # Add relationship links as separate arrays
        result_data["child_links"] = [
            {
                "parent_id": self._format_id(link.parent_id),
                "child_id": self._format_id(link.child_id),
            }
            for link in child_links
        ]

        result_data["spouse_links"] = [
            {
                "person1_id": self._format_id(link.person1_id),
                "person2_id": self._format_id(link.person2_id),
            }
            for link in spouse_links
        ]

        return result_data

    def _person_to_f8vision(
        self,
        person: PersonTable,
        truncate_descriptions: int | None = None,
    ) -> dict[str, Any]:
        """Convert a person to f8vision-web format with all fields.

        Args:
            person: The person database record.
            truncate_descriptions: If set, truncate biography to this many characters.

        Returns:
            Dictionary in f8vision-web person format.
        """
        result: dict[str, Any] = {
            "id": self._format_id(person.id),
        }

        # Identity fields
        if person.given_name:
            result["given_name"] = person.given_name
        if person.surname:
            result["surname"] = person.surname

        # Also include full name for compatibility
        full_name = f"{person.given_name or ''} {person.surname or ''}".strip()
        if full_name:
            result["name"] = full_name

        if person.nickname:
            result["nickname"] = person.nickname
        if person.maiden_name:
            result["maiden_name"] = person.maiden_name

        # Demographics
        if person.gender:
            # Convert enum to string value
            gender_value = person.gender.value if hasattr(person.gender, 'value') else person.gender
            result["gender"] = str(gender_value)

        # Birth info
        if person.birth_date:
            result["birth_date"] = person.birth_date.isoformat()
        if person.birth_place:
            result["birth_place"] = person.birth_place

        # Death info
        if person.death_date:
            result["death_date"] = person.death_date.isoformat()
        if person.death_place:
            result["death_place"] = person.death_place

        # Biography
        if person.biography:
            biography = person.biography
            if truncate_descriptions and len(biography) > truncate_descriptions:
                biography = biography[:truncate_descriptions] + "..."
            result["biography"] = biography

        # Status
        if person.status:
            status_value = person.status.value if hasattr(person.status, 'value') else person.status
            result["status"] = str(status_value)

        # Generation
        if person.generation is not None:
            result["generation"] = person.generation

        return result

    def _event_to_f8vision(self, event: EventTable) -> dict[str, Any]:
        """Convert an event to f8vision-web format.

        Args:
            event: The event database record.

        Returns:
            Dictionary in f8vision-web event format.
        """
        event_type_val = (
            event.event_type.value if hasattr(event.event_type, 'value') else event.event_type
        )
        result: dict[str, Any] = {
            "id": self._format_id(event.id),
            "event_type": str(event_type_val),
            "primary_person_id": self._format_id(event.primary_person_id),
        }

        if event.event_date:
            result["event_date"] = event.event_date.isoformat()
        if event.event_year is not None:
            result["event_year"] = event.event_year
        if event.location:
            result["location"] = event.location
        if event.description:
            result["description"] = event.description

        return result

    def _note_to_f8vision(self, note: NoteTable) -> dict[str, Any]:
        """Convert a note to f8vision-web format.

        Args:
            note: The note database record.

        Returns:
            Dictionary in f8vision-web note format.
        """
        result: dict[str, Any] = {
            "id": self._format_id(note.id),
            "person_id": self._format_id(note.person_id),
            "content": note.content,
        }

        if note.category:
            category_val = note.category.value if hasattr(note.category, 'value') else note.category
            result["category"] = str(category_val)
        if note.source:
            result["source"] = note.source

        return result

    def _format_id(self, uuid: UUID) -> str:
        """Format a UUID as a string ID for f8vision-web.

        Args:
            uuid: The UUID to format.

        Returns:
            String representation of the ID.
        """
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
