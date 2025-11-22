# Ancestral Synth

A fictional genealogical dataset generator using Large Language Models.

## Overview

Ancestral Synth generates detailed, historically-plausible fictional family trees with:

- **Biographies**: ~1000-word life stories for each person
- **Structured Data**: Names, dates, places, events, relationships
- **Family Links**: Parent-child relationships with genealogical validation
- **Deduplication**: Intelligent matching to avoid duplicate records
- **Event Tracking**: Births, deaths, marriages, and other life events

## Architecture

Built with modern Python (3.11+) best practices:

- **Pydantic**: Domain models and validation
- **Pydantic AI**: LLM integration with structured outputs
- **SQLModel**: Database ORM (SQLAlchemy + Pydantic)
- **Typer**: CLI interface
- **Rich**: Beautiful terminal output
- **Loguru**: Structured logging

### Project Structure

```
ancestral_synth/
├── domain/           # Pure Pydantic domain models
│   ├── enums.py      # Status, gender, event types
│   └── models.py     # Person, Event, Note, ChildLink
├── agents/           # Pydantic AI agents
│   ├── biography_agent.py   # Biography generation
│   ├── extraction_agent.py  # Structured data extraction
│   └── dedup_agent.py       # Deduplication checking
├── persistence/      # Database layer
│   ├── database.py   # Async database connection
│   ├── tables.py     # SQLModel table definitions
│   └── repositories.py  # Data access repositories
├── services/         # Business logic
│   ├── genealogy_service.py  # Main orchestration
│   └── validation.py         # Genealogical validation
├── config.py         # Pydantic Settings configuration
└── cli.py           # Typer CLI interface
```

## Installation

```bash
# Clone and install
git clone <repo-url>
cd ancestral-synth
pip install -e .

# Or with uv (recommended)
uv pip install -e .
```

## Configuration

Set environment variables or create a `.env` file:

```bash
# LLM Configuration
ANCESTRAL_LLM_PROVIDER=openai  # openai, anthropic, or ollama
ANCESTRAL_LLM_MODEL=gpt-4o-mini

# Database
ANCESTRAL_DATABASE_PATH=genealogy.db

# Generation settings
ANCESTRAL_BIOGRAPHY_WORD_COUNT=1000
ANCESTRAL_BATCH_SIZE=10
```

## Usage

### Initialize Database

```bash
ancestral-synth init
```

### Generate Persons

```bash
# Generate one person
ancestral-synth generate

# Generate 10 persons
ancestral-synth generate -n 10

# With verbose output
ancestral-synth generate -n 5 -v
```

### View Data

```bash
# Show statistics
ancestral-synth stats

# List persons
ancestral-synth list-persons
ancestral-synth list-persons --status complete

# Show person details
ancestral-synth show "John Smith"
```

### Configuration

```bash
# Show current configuration
ancestral-synth config
```

## How It Works

### Generation Pipeline

1. **Queue Management**: A creation queue tracks persons awaiting biography generation
2. **Biography Generation**: Pydantic AI agent generates ~1000-word biographies
3. **Data Extraction**: Another agent extracts structured data (dates, relationships, events)
4. **Validation**: Genealogical plausibility checks (ages, date ranges)
5. **Deduplication**: Heuristic + LLM matching to avoid duplicates
6. **Reference Processing**: Family members become pending records, parents/children get queued

### Forest Fire Sampling

When selecting the next person to process, the system uses forest fire sampling:
- Persons closer to generation 0 are weighted higher
- Random perturbation adds variety
- Ensures organic tree growth

### Validation Rules

- Parents must be between 14-60 years old at child's birth
- Maximum lifespan: 120 years
- Events must occur within person's lifetime
- Death cannot precede birth
- Children cannot be born after parent's death (with 1-year posthumous allowance)

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Type checking
mypy ancestral_synth

# Linting
ruff check ancestral_synth
```

## License

MIT
