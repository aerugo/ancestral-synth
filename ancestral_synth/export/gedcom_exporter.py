"""GEDCOM 5.5.1 exporter for genealogical data."""

from datetime import date, datetime
from typing import IO
from uuid import UUID

from sqlmodel import select

from ancestral_synth.domain.enums import Gender
from ancestral_synth.persistence.database import Database
from ancestral_synth.persistence.tables import ChildLinkTable, PersonTable


class GEDCOMExporter:
    """Export genealogical data to GEDCOM 5.5.1 format."""

    def __init__(self, db: Database) -> None:
        """Initialize the exporter.

        Args:
            db: Database connection.
        """
        self._db = db
        self._person_to_indi: dict[UUID, str] = {}
        self._family_counter = 0

    async def export(self, output: IO[str]) -> None:
        """Export all data to GEDCOM format.

        Args:
            output: Output stream to write GEDCOM to.
        """
        self._person_to_indi = {}
        self._family_counter = 0

        # Write header
        self._write_header(output)

        # Load all data
        async with self._db.session() as session:
            # Get all persons and assign INDI IDs
            result = await session.exec(select(PersonTable))
            persons = list(result.all())

            for i, person in enumerate(persons, 1):
                indi_id = f"I{i:04d}"
                self._person_to_indi[person.id] = indi_id

            # Write individual records
            for person in persons:
                self._write_individual(output, person)

            # Get child links and create family records
            result = await session.exec(select(ChildLinkTable))
            child_links = list(result.all())

            # Group children by parents
            families = self._group_families(child_links)
            for family in families:
                self._write_family(output, family)

        # Write trailer
        self._write_trailer(output)

    def _write_header(self, output: IO[str]) -> None:
        """Write GEDCOM header."""
        output.write("0 HEAD\n")
        output.write("1 SOUR AncestralSynth\n")
        output.write("2 VERS 1.0\n")
        output.write("2 NAME Ancestral Synth\n")
        output.write("1 GEDC\n")
        output.write("2 VERS 5.5.1\n")
        output.write("2 FORM LINEAGE-LINKED\n")
        output.write("1 CHAR UTF-8\n")
        output.write(f"1 DATE {datetime.utcnow().strftime('%d %b %Y').upper()}\n")

    def _write_individual(self, output: IO[str], person: PersonTable) -> None:
        """Write an individual record."""
        indi_id = self._person_to_indi[person.id]

        output.write(f"0 @{indi_id}@ INDI\n")
        output.write(f"1 NAME {person.given_name} /{person.surname}/\n")
        output.write(f"2 GIVN {person.given_name}\n")
        output.write(f"2 SURN {person.surname}\n")

        if person.gender:
            sex = "M" if person.gender == Gender.MALE else "F" if person.gender == Gender.FEMALE else "U"
            output.write(f"1 SEX {sex}\n")

        if person.birth_date or person.birth_place:
            output.write("1 BIRT\n")
            if person.birth_date:
                output.write(f"2 DATE {self._format_date(person.birth_date)}\n")
            if person.birth_place:
                output.write(f"2 PLAC {person.birth_place}\n")

        if person.death_date or person.death_place:
            output.write("1 DEAT\n")
            if person.death_date:
                output.write(f"2 DATE {self._format_date(person.death_date)}\n")
            if person.death_place:
                output.write(f"2 PLAC {person.death_place}\n")

    def _write_family(self, output: IO[str], family: dict) -> None:
        """Write a family record."""
        self._family_counter += 1
        fam_id = f"F{self._family_counter:04d}"

        output.write(f"0 @{fam_id}@ FAM\n")

        for parent_id in family.get("parents", []):
            indi_id = self._person_to_indi.get(parent_id)
            if indi_id:
                # Determine if husband or wife based on stored data
                # For simplicity, just use HUSB for first, WIFE for second
                output.write(f"1 HUSB @{indi_id}@\n")

        for child_id in family.get("children", []):
            indi_id = self._person_to_indi.get(child_id)
            if indi_id:
                output.write(f"1 CHIL @{indi_id}@\n")

    def _write_trailer(self, output: IO[str]) -> None:
        """Write GEDCOM trailer."""
        output.write("0 TRLR\n")

    def _format_date(self, d: date) -> str:
        """Format date in GEDCOM format."""
        return d.strftime("%d %b %Y").upper()

    def _group_families(self, child_links: list[ChildLinkTable]) -> list[dict]:
        """Group child links into family units."""
        # Simple grouping: for each unique set of parents for a child, create a family
        child_to_parents: dict[UUID, set[UUID]] = {}

        for link in child_links:
            if link.child_id not in child_to_parents:
                child_to_parents[link.child_id] = set()
            child_to_parents[link.child_id].add(link.parent_id)

        # Group children by their parent set
        parent_set_to_children: dict[frozenset[UUID], list[UUID]] = {}
        for child_id, parents in child_to_parents.items():
            key = frozenset(parents)
            if key not in parent_set_to_children:
                parent_set_to_children[key] = []
            parent_set_to_children[key].append(child_id)

        # Convert to family dicts
        families = []
        for parents, children in parent_set_to_children.items():
            families.append({
                "parents": list(parents),
                "children": children,
            })

        return families
