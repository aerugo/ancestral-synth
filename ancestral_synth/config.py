"""Application configuration using Pydantic Settings."""

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="ANCESTRAL_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Database
    database_path: Path = Field(
        default=Path("genealogy.db"),
        description="Path to the SQLite database file",
    )

    # LLM Configuration
    llm_provider: Literal["openai", "anthropic", "ollama"] = Field(
        default="openai",
        description="LLM provider to use",
    )
    llm_model: str = Field(
        default="gpt-4o-mini",
        description="Model name for the LLM provider",
    )

    # Generation settings
    biography_word_count: int = Field(
        default=1000,
        description="Target word count for generated biographies",
    )
    batch_size: int = Field(
        default=10,
        description="Number of persons to process in a batch",
    )

    # Sampling
    forest_fire_probability: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Probability for forest fire sampling",
    )

    # Validation
    min_parent_age: int = Field(default=14, description="Minimum age to become a parent")
    max_parent_age: int = Field(default=60, description="Maximum age to become a parent")
    max_lifespan: int = Field(default=120, description="Maximum realistic lifespan")


settings = Settings()
