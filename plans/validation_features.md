# Validation Features Plan

This document outlines missing validation features and duplicate handling improvements for the ancestral-synth project.

## Current State

### Implemented Validations (`services/validation.py`)
- Death before birth detection
- Excessive lifespan warnings (>120 years)
- Very short lifespan warnings (<1 year)
- Parent-child age gap validation (min 14, max 60+20 years)
- Posthumous birth allowance (1-year window)
- Event before birth / after death errors
- Spouse age gap warnings (>30 years)
- Future birth year errors

### Implemented Duplicate Detection (`agents/dedup_agent.py`)
- Heuristic pre-filtering (name overlap + birth year scoring)
- LLM-based matching with confidence scores
- Database search for similar candidates

---

## Missing Features

### 1. Cross-Relative Temporal Validation

#### 1.1 Sibling Age Consistency
**Priority:** High
**Location:** `services/validation.py`

Validate that siblings have reasonable age gaps:
- Warning if siblings are born more than 40 years apart
- Warning if multiple siblings share the exact same birth year (possible twins, but flag for review)
- Error if sibling is born before parent's minimum childbearing age relative to oldest sibling

**Implementation:**
```python
def validate_sibling_ages(person: Person, siblings: list[Person]) -> list[ValidationResult]:
    """Check that sibling birth years are within reasonable ranges."""
    results = []
    birth_years = [s.birth_year for s in siblings if s.birth_year]

    if person.birth_year and birth_years:
        max_gap = max(abs(person.birth_year - y) for y in birth_years)
        if max_gap > 40:
            results.append(ValidationResult(
                level=ValidationLevel.WARNING,
                message=f"Sibling age gap of {max_gap} years seems unusual"
            ))
    return results
```

#### 1.2 Multigenerational Chain Validation
**Priority:** Medium
**Location:** `services/validation.py`

Validate generational consistency across grandparent → parent → child chains:
- Error if grandchild is older than grandparent
- Warning if generational gaps are less than 28 years cumulative (14 × 2)
- Warning if generational gaps exceed 160 years cumulative (80 × 2)

**Implementation:**
```python
def validate_generational_chain(person: Person, grandparents: list[Person]) -> list[ValidationResult]:
    """Validate that generational gaps make biological sense."""
    results = []
    for grandparent in grandparents:
        if grandparent.birth_year and person.birth_year:
            gap = person.birth_year - grandparent.birth_year
            if gap < 28:
                results.append(ValidationResult(
                    level=ValidationLevel.ERROR,
                    message=f"Only {gap} years between grandparent and grandchild"
                ))
            elif gap > 160:
                results.append(ValidationResult(
                    level=ValidationLevel.WARNING,
                    message=f"Unusually large gap ({gap} years) between grandparent and grandchild"
                ))
    return results
```

---

### 2. Event Sequence Validation

#### 2.1 Divorce Before Remarriage
**Priority:** Medium
**Location:** `services/validation.py`

Validate marriage/divorce sequencing:
- Warning if person marries while previous marriage has no divorce/death end date
- Error if divorce date is before marriage date for same relationship

**Implementation:**
```python
def validate_marriage_sequence(person: Person, marriages: list[Marriage]) -> list[ValidationResult]:
    """Validate that marriages don't overlap and divorces follow marriages."""
    results = []
    sorted_marriages = sorted(marriages, key=lambda m: m.start_year or 0)

    for i, marriage in enumerate(sorted_marriages):
        # Check divorce after marriage
        if marriage.divorce_year and marriage.start_year:
            if marriage.divorce_year < marriage.start_year:
                results.append(ValidationResult(
                    level=ValidationLevel.ERROR,
                    message=f"Divorce ({marriage.divorce_year}) before marriage ({marriage.start_year})"
                ))

        # Check for overlapping marriages
        if i > 0:
            prev = sorted_marriages[i - 1]
            prev_end = prev.divorce_year or prev.spouse_death_year
            if marriage.start_year and prev_end is None:
                results.append(ValidationResult(
                    level=ValidationLevel.WARNING,
                    message=f"New marriage in {marriage.start_year} while previous marriage has no end date"
                ))
    return results
```

**Note:** Parenthood before or without marriage is explicitly allowed and should NOT be validated against.

---

### 3. Gender Consistency Validation

#### 3.1 Biological Parenthood Gender Check
**Priority:** High
**Location:** `services/validation.py`

Validate that biological parent roles align with gender:
- Error if person with gender=MALE is listed as biological "mother"
- Error if person with gender=FEMALE is listed as biological "father"
- Skip validation if gender is UNKNOWN or relationship is adoptive/step-parent

**Implementation:**
```python
def validate_parent_gender(parent: Person, relationship_type: str) -> list[ValidationResult]:
    """Validate biological parent gender consistency."""
    results = []

    if relationship_type == "biological_mother" and parent.gender == Gender.MALE:
        results.append(ValidationResult(
            level=ValidationLevel.ERROR,
            message=f"{parent.display_name} is male but listed as biological mother"
        ))
    elif relationship_type == "biological_father" and parent.gender == Gender.FEMALE:
        results.append(ValidationResult(
            level=ValidationLevel.ERROR,
            message=f"{parent.display_name} is female but listed as biological father"
        ))

    return results
```

---

### 4. Duplicate Merging and Consolidation

#### 4.1 Duplicate Merge Logic
**Priority:** High
**Location:** `persistence/repositories.py` (new methods)

When duplicates are confirmed, merge records intelligently:
- Combine relationship links from both records
- Resolve conflicting dates (prefer more precise, flag for review if different)
- Merge name variations as aliases
- Update all references to point to surviving record

**Implementation approach:**
```python
def merge_persons(primary_id: int, duplicate_id: int) -> MergeResult:
    """Merge duplicate person into primary, consolidating all data."""
    primary = get_person(primary_id)
    duplicate = get_person(duplicate_id)

    # 1. Merge attributes (prefer non-null, flag conflicts)
    conflicts = []
    if duplicate.birth_year and primary.birth_year:
        if duplicate.birth_year != primary.birth_year:
            conflicts.append(f"birth_year: {primary.birth_year} vs {duplicate.birth_year}")
    elif duplicate.birth_year:
        primary.birth_year = duplicate.birth_year

    # 2. Remap all relationships pointing to duplicate
    update_relationship_references(from_id=duplicate_id, to_id=primary_id)

    # 3. Merge name as alias if different
    if duplicate.full_name != primary.full_name:
        add_alias(primary_id, duplicate.full_name)

    # 4. Mark duplicate as merged (soft delete)
    mark_as_merged(duplicate_id, merged_into=primary_id)

    return MergeResult(primary_id=primary_id, conflicts=conflicts)
```

#### 4.2 Relationship Remapping
**Priority:** High
**Location:** `persistence/repositories.py`

Update all foreign key references when merging:
- Parent relationships
- Child relationships
- Spouse relationships
- Event participants

```python
def update_relationship_references(from_id: int, to_id: int) -> int:
    """Update all relationship foreign keys from old ID to new ID."""
    count = 0
    count += update_parent_references(from_id, to_id)
    count += update_child_references(from_id, to_id)
    count += update_spouse_references(from_id, to_id)
    return count
```

---

### 5. Enhanced Name Matching

#### 5.1 Nickname Recognition
**Priority:** Medium
**Location:** `agents/dedup_agent.py`

Add common nickname mappings for fuzzy matching:
- William ↔ Bill, Will, Billy, Willy
- Elizabeth ↔ Beth, Liz, Lizzy, Betty, Eliza
- Robert ↔ Bob, Rob, Bobby, Robbie
- Margaret ↔ Maggie, Meg, Peggy, Marge
- James ↔ Jim, Jimmy, Jamie
- etc.

**Implementation:**
```python
NICKNAME_MAP = {
    "william": ["bill", "will", "billy", "willy", "liam"],
    "elizabeth": ["beth", "liz", "lizzy", "betty", "eliza", "lisa"],
    "robert": ["bob", "rob", "bobby", "robbie", "bert"],
    "margaret": ["maggie", "meg", "peggy", "marge", "margo"],
    "james": ["jim", "jimmy", "jamie", "jem"],
    "richard": ["rick", "dick", "rich", "richie"],
    "katherine": ["kate", "kathy", "katie", "cathy", "kit"],
    "michael": ["mike", "mikey", "mick"],
    "jennifer": ["jen", "jenny", "jenna"],
    "christopher": ["chris", "kit", "topher"],
}

def names_match_with_nicknames(name1: str, name2: str) -> bool:
    """Check if names match accounting for common nicknames."""
    n1, n2 = name1.lower(), name2.lower()
    if n1 == n2:
        return True

    # Check nickname mappings
    for canonical, nicknames in NICKNAME_MAP.items():
        all_variants = [canonical] + nicknames
        if n1 in all_variants and n2 in all_variants:
            return True

    return False
```

#### 5.2 Maiden Name Handling
**Priority:** Medium
**Location:** `agents/dedup_agent.py`

Improve matching for married women with name changes:
- Store maiden name as searchable alias
- Match "Mary Smith" with "Mary Johnson (née Smith)"
- Parse common maiden name patterns: "née X", "born X", "(X)"

```python
def extract_maiden_name(full_name: str) -> str | None:
    """Extract maiden name from full name if present."""
    patterns = [
        r'\(née\s+(\w+)\)',
        r'\(born\s+(\w+)\)',
        r'née\s+(\w+)',
        r'\((\w+)\)$',  # Surname in parentheses at end
    ]
    for pattern in patterns:
        match = re.search(pattern, full_name, re.IGNORECASE)
        if match:
            return match.group(1)
    return None
```

---

### 6. Duplicate Prevention at Creation

#### 6.1 Seed Person Duplicate Check
**Priority:** Medium
**Location:** `services/genealogy_service.py`

Check for existing duplicates before creating seed persons:
- Run duplicate detection before `_create_seed_person()`
- Prompt for confirmation if high-confidence match found
- Option to use existing record instead of creating new

```python
async def create_seed_person_with_dedup(self, name: str, birth_year: int | None) -> int:
    """Create seed person, checking for duplicates first."""
    # Check for existing matches
    candidates = await self.person_repo.search_similar(
        surname=extract_surname(name),
        given_name=extract_given_name(name),
        birth_year=birth_year
    )

    if candidates:
        match = await self.dedup_agent.check_duplicate(
            reference=PersonReference(name=name, approximate_birth_year=birth_year),
            candidates=candidates
        )
        if match and match.confidence > 0.8:
            return match.person_id

    # No match found, create new
    return await self._create_seed_person(name, birth_year)
```

---

## Implementation Order

### Phase 1: High Priority
1. Gender consistency validation (3.1)
2. Sibling age consistency (1.1)
3. Duplicate merge logic (4.1)
4. Relationship remapping (4.2)

### Phase 2: Medium Priority
5. Multigenerational chain validation (1.2)
6. Divorce before remarriage validation (2.1)
7. Nickname recognition (5.1)
8. Maiden name handling (5.2)
9. Seed person duplicate check (6.1)

### Phase 3: Future Enhancements
- Location consistency validation
- Place name normalization for matching
- Historical date format handling
- Confidence scoring for merged data

---

## Testing Requirements

Each new validation should include tests for:
- Positive cases (validation passes)
- Negative cases (validation fails appropriately)
- Edge cases (unknown values, missing data)
- Integration with existing validation pipeline

Each duplicate handling feature should include tests for:
- Successful merge scenarios
- Conflict detection and flagging
- Relationship integrity after remapping
- Rollback capability for incorrect merges
