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
    llm_provider: Literal["openai", "anthropic", "ollama", "google"] = Field(
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

    # Rate limiting
    llm_requests_per_minute: int = Field(
        default=60,
        description="Maximum LLM API requests per minute",
    )

    # Retry settings
    llm_max_retries: int = Field(
        default=3,
        description="Maximum number of retry attempts for LLM calls",
    )
    llm_retry_base_delay: float = Field(
        default=2.0,
        description="Base delay in seconds for exponential backoff",
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
    max_correction_attempts: int = Field(
        default=2,
        description="Maximum attempts to correct validation errors before giving up",
    )


settings = Settings()


def get_pydantic_ai_provider() -> str:
    """Get the pydantic-ai compatible provider string.

    Maps user-friendly provider names to pydantic-ai expected values.
    For example, 'google' maps to 'google-gla' for Google AI Studio.
    """
    provider_mapping = {
        "google": "google-gla",  # Google AI Studio / Generative Language API
    }
    return provider_mapping.get(settings.llm_provider, settings.llm_provider)
