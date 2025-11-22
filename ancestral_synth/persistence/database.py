"""Database connection and session management."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

from ancestral_synth.persistence.tables import (
    ChildLinkTable,
    EventParticipantTable,
    EventTable,
    NoteReferenceTable,
    NoteTable,
    PersonTable,
    QueueEntryTable,
)

# Ensure all tables are registered
_TABLES = [
    PersonTable,
    EventTable,
    EventParticipantTable,
    NoteTable,
    NoteReferenceTable,
    ChildLinkTable,
    QueueEntryTable,
]


class Database:
    """Async database connection manager."""

    def __init__(self, db_path: Path | str = "genealogy.db") -> None:
        """Initialize the database connection.

        Args:
            db_path: Path to the SQLite database file.
        """
        self.db_path = Path(db_path)
        self._engine: AsyncEngine | None = None

    @property
    def engine(self) -> AsyncEngine:
        """Get the database engine, creating it if necessary."""
        if self._engine is None:
            # Create parent directories if they don't exist
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

            # Use aiosqlite for async SQLite support
            url = f"sqlite+aiosqlite:///{self.db_path}"
            self._engine = create_async_engine(
                url,
                echo=False,
                connect_args={"check_same_thread": False},
            )
        return self._engine

    async def init_db(self) -> None:
        """Initialize the database schema."""
        async with self.engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)

    async def close(self) -> None:
        """Close the database connection."""
        if self._engine is not None:
            await self._engine.dispose()
            self._engine = None

    @asynccontextmanager
    async def session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get an async database session.

        Yields:
            An async session for database operations.
        """
        async with AsyncSession(self.engine, expire_on_commit=False) as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    async def __aenter__(self) -> "Database":
        """Async context manager entry."""
        await self.init_db()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:  # noqa: ANN001
        """Async context manager exit."""
        await self.close()
