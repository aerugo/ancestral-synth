"""CSV exporter for genealogical data."""

import csv
from typing import IO

from sqlmodel import select

from ancestral_synth.persistence.database import Database
from ancestral_synth.persistence.tables import (
    ChildLinkTable,
    EventTable,
    NoteTable,
    PersonTable,
)


class CSVExporter:
    """Export genealogical data to CSV format."""

    def __init__(self, db: Database) -> None:
        """Initialize the exporter.

        Args:
            db: Database connection.
        """
        self._db = db

    async def export_persons(self, output: IO[str]) -> None:
        """Export persons to CSV.

        Args:
            output: Output stream to write CSV to.
        """
        fieldnames = [
            "id",
            "given_name",
            "surname",
            "maiden_name",
            "nickname",
            "gender",
            "birth_date",
            "birth_place",
            "death_date",
            "death_place",
            "status",
            "generation",
        ]

        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()

        async with self._db.session() as session:
            result = await session.exec(select(PersonTable))
            for person in result.all():
                writer.writerow({
                    "id": str(person.id),
                    "given_name": person.given_name,
                    "surname": person.surname,
                    "maiden_name": person.maiden_name or "",
                    "nickname": person.nickname or "",
                    "gender": person.gender.value if person.gender else "",
                    "birth_date": person.birth_date.isoformat() if person.birth_date else "",
                    "birth_place": person.birth_place or "",
                    "death_date": person.death_date.isoformat() if person.death_date else "",
                    "death_place": person.death_place or "",
                    "status": person.status.value if person.status else "",
                    "generation": person.generation,
                })

    async def export_events(self, output: IO[str]) -> None:
        """Export events to CSV.

        Args:
            output: Output stream to write CSV to.
        """
        fieldnames = [
            "id",
            "event_type",
            "event_date",
            "event_year",
            "location",
            "description",
            "primary_person_id",
        ]

        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()

        async with self._db.session() as session:
            result = await session.exec(select(EventTable))
            for event in result.all():
                writer.writerow({
                    "id": str(event.id),
                    "event_type": event.event_type.value if event.event_type else "",
                    "event_date": event.event_date.isoformat() if event.event_date else "",
                    "event_year": event.event_year or "",
                    "location": event.location or "",
                    "description": event.description,
                    "primary_person_id": str(event.primary_person_id),
                })

    async def export_child_links(self, output: IO[str]) -> None:
        """Export child links to CSV.

        Args:
            output: Output stream to write CSV to.
        """
        fieldnames = ["parent_id", "child_id"]

        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()

        async with self._db.session() as session:
            result = await session.exec(select(ChildLinkTable))
            for link in result.all():
                writer.writerow({
                    "parent_id": str(link.parent_id),
                    "child_id": str(link.child_id),
                })
