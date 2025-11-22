"""Tests for database connection and management."""

import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from ancestral_synth.persistence.database import Database


class TestDatabaseConnection:
    """Tests for Database connection management."""

    @pytest.mark.asyncio
    async def test_create_in_memory_database(self) -> None:
        """Should create an in-memory database."""
        db = Database(":memory:")
        await db.init_db()

        assert db._engine is not None
        await db.close()

    @pytest.mark.asyncio
    async def test_create_file_database(self) -> None:
        """Should create a file-based database."""
        with TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            db = Database(db_path)
            await db.init_db()

            assert db_path.exists()
            await db.close()

    @pytest.mark.asyncio
    async def test_create_nested_directory_database(self) -> None:
        """Should create parent directories if they don't exist."""
        with TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "nested" / "dir" / "test.db"
            db = Database(db_path)
            await db.init_db()

            assert db_path.exists()
            await db.close()

    @pytest.mark.asyncio
    async def test_close_database(self) -> None:
        """Should close the database connection."""
        db = Database(":memory:")
        await db.init_db()
        await db.close()

        assert db._engine is None

    @pytest.mark.asyncio
    async def test_close_without_init(self) -> None:
        """Should handle closing without initialization."""
        db = Database(":memory:")
        await db.close()  # Should not raise

    @pytest.mark.asyncio
    async def test_context_manager(self) -> None:
        """Should work as async context manager."""
        async with Database(":memory:") as db:
            assert db._engine is not None

        # Engine should be disposed after context
        assert db._engine is None


class TestDatabaseSession:
    """Tests for database session management."""

    @pytest.mark.asyncio
    async def test_session_provides_connection(self) -> None:
        """Session should provide a database connection."""
        async with Database(":memory:") as db:
            async with db.session() as session:
                assert session is not None

    @pytest.mark.asyncio
    async def test_session_commits_on_success(self) -> None:
        """Session should commit changes on successful exit."""
        from ancestral_synth.persistence.tables import PersonTable
        from ancestral_synth.domain.enums import Gender, PersonStatus

        async with Database(":memory:") as db:
            async with db.session() as session:
                person = PersonTable(
                    given_name="Test",
                    surname="Person",
                    gender=Gender.MALE,
                    status=PersonStatus.PENDING,
                )
                session.add(person)

            # Verify it was committed
            async with db.session() as session:
                from sqlmodel import select
                result = await session.exec(select(PersonTable))
                persons = list(result.all())
                assert len(persons) == 1

    @pytest.mark.asyncio
    async def test_session_rolls_back_on_error(self) -> None:
        """Session should rollback on error."""
        from ancestral_synth.persistence.tables import PersonTable
        from ancestral_synth.domain.enums import Gender, PersonStatus

        async with Database(":memory:") as db:
            try:
                async with db.session() as session:
                    person = PersonTable(
                        given_name="Test",
                        surname="Person",
                        gender=Gender.MALE,
                        status=PersonStatus.PENDING,
                    )
                    session.add(person)
                    raise ValueError("Simulated error")
            except ValueError:
                pass

            # Verify nothing was committed
            async with db.session() as session:
                from sqlmodel import select
                result = await session.exec(select(PersonTable))
                persons = list(result.all())
                assert len(persons) == 0

    @pytest.mark.asyncio
    async def test_multiple_sessions(self) -> None:
        """Should support multiple sequential sessions."""
        async with Database(":memory:") as db:
            async with db.session() as session1:
                pass
            async with db.session() as session2:
                pass
            # Both should work without issues


class TestDatabaseSchema:
    """Tests for database schema initialization."""

    @pytest.mark.asyncio
    async def test_creates_all_tables(self) -> None:
        """Should create all required tables."""
        from sqlalchemy import inspect

        async with Database(":memory:") as db:
            async with db.engine.connect() as conn:
                def get_tables(connection):  # noqa: ANN001, ANN202
                    inspector = inspect(connection)
                    return inspector.get_table_names()

                tables = await conn.run_sync(get_tables)

                expected_tables = [
                    "persons",
                    "events",
                    "event_participants",
                    "notes",
                    "note_references",
                    "child_links",
                    "creation_queue",
                ]

                for table in expected_tables:
                    assert table in tables, f"Missing table: {table}"

    @pytest.mark.asyncio
    async def test_init_is_idempotent(self) -> None:
        """Should be safe to call init_db multiple times."""
        db = Database(":memory:")
        await db.init_db()
        await db.init_db()  # Should not raise
        await db.close()
