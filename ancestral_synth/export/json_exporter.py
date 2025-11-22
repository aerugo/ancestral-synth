"""JSON exporter for genealogical data."""

import json
from datetime import date, datetime
from typing import IO, Any

from sqlmodel import select

from ancestral_synth.persistence.database import Database
from ancestral_synth.persistence.tables import (
    ChildLinkTable,
    EventTable,
    NoteTable,
    PersonTable,
)


class JSONExporter:
    """Export genealogical data to JSON format."""

    def __init__(self, db: Database) -> None:
        """Initialize the exporter.

        Args:
            db: Database connection.
        """
        self._db = db

    async def export(self, output: IO[str]) -> None:
        """Export all data to JSON.

        Args:
            output: Output stream to write JSON to.
        """
        data = await self._gather_data()
        json.dump(data, output, indent=2, default=self._json_serializer)

    async def _gather_data(self) -> dict[str, Any]:
        """Gather all data for export."""
        async with self._db.session() as session:
            # Get all persons
            result = await session.exec(select(PersonTable))
            persons = [self._person_to_dict(p) for p in result.all()]

            # Get all events
            result = await session.exec(select(EventTable))
            events = [self._event_to_dict(e) for e in result.all()]

            # Get all notes
            result = await session.exec(select(NoteTable))
            notes = [self._note_to_dict(n) for n in result.all()]

            # Get all child links
            result = await session.exec(select(ChildLinkTable))
            child_links = [self._child_link_to_dict(cl) for cl in result.all()]

        return {
            "metadata": {
                "exported_at": datetime.utcnow().isoformat(),
                "version": "1.0",
                "format": "ancestral-synth-json",
            },
            "persons": persons,
            "events": events,
            "notes": notes,
            "child_links": child_links,
        }

    def _person_to_dict(self, person: PersonTable) -> dict[str, Any]:
        """Convert person to dictionary."""
        return {
            "id": str(person.id),
            "given_name": person.given_name,
            "surname": person.surname,
            "maiden_name": person.maiden_name,
            "nickname": person.nickname,
            "gender": person.gender.value if person.gender else None,
            "birth_date": person.birth_date.isoformat() if person.birth_date else None,
            "birth_place": person.birth_place,
            "death_date": person.death_date.isoformat() if person.death_date else None,
            "death_place": person.death_place,
            "status": person.status.value if person.status else None,
            "biography": person.biography,
            "generation": person.generation,
        }

    def _event_to_dict(self, event: EventTable) -> dict[str, Any]:
        """Convert event to dictionary."""
        return {
            "id": str(event.id),
            "event_type": event.event_type.value if event.event_type else None,
            "event_date": event.event_date.isoformat() if event.event_date else None,
            "event_year": event.event_year,
            "location": event.location,
            "description": event.description,
            "primary_person_id": str(event.primary_person_id),
        }

    def _note_to_dict(self, note: NoteTable) -> dict[str, Any]:
        """Convert note to dictionary."""
        return {
            "id": str(note.id),
            "person_id": str(note.person_id),
            "category": note.category.value if note.category else None,
            "content": note.content,
            "source": note.source,
        }

    def _child_link_to_dict(self, link: ChildLinkTable) -> dict[str, Any]:
        """Convert child link to dictionary."""
        return {
            "parent_id": str(link.parent_id),
            "child_id": str(link.child_id),
        }

    def _json_serializer(self, obj: Any) -> Any:
        """Custom JSON serializer for special types."""
        if isinstance(obj, (date, datetime)):
            return obj.isoformat()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
