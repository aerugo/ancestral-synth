# Debugging Plan: LLM Generation Performance

## Problem Statement

Generating persons takes ~316 seconds per person on average, with some taking 900+ seconds.
This is unexpectedly slow for gpt-5-mini which should be a fast model.

### Observed Timing Data

| Person | Biography Gen | Data Extraction | Dedup Calls | Total |
|--------|---------------|-----------------|-------------|-------|
| 1 | 47.7s | 1m 7.8s | 2 × ~10s | 136.1s |
| 2 | 2m 16s | **12m 34.3s** | 2 × ~10s | **910.2s** |
| 3 | 53.0s | 1m 13.3s | 3 × ~9s | 153.5s |
| 4 | 44.4s | 1m 58.9s | 3 × ~9s | 190.3s |
| 5 | 51.3s | 2m 11.1s | 1 × ~8s | 191.0s |

**Key anomaly**: Person 2's data extraction took 12+ minutes.

---

## Phase 1: Baseline Measurement

**Goal**: Get precise timing data for each component

### 1.1 Run with new instrumentation

```bash
uv run ancestral-synth generate -n 2 --verbose 2>&1 | tee debug_run.log
```

### 1.2 Capture timing for each operation

The instrumentation now logs:
- Raw `pydantic_ai.run()` time (inside the agent)
- Total time including retry wrapper
- Per-attempt timing with success/failure status
- Rate limiter wait time
- Database operations
- Reference processing (dedup calls)

### 1.3 Expected output format

```
→ Biography generation (LLM call)...
    [attempt 1/4] Starting API call...
      [biography] Prompt length: 432 chars
      [biography] pydantic_ai.run() completed in 5.2s
    [attempt 1] API call succeeded in 5.3s
✓ Biography generation (LLM call) completed in 5.3s
```

---

## Phase 2: Isolate the LLM Layer

**Goal**: Determine if slowness is in the model API or our code

### 2.1 Direct API test

Bypass all our code to test raw OpenAI API speed:

```python
import time
import openai

client = openai.OpenAI()

# Test 1: Simple completion
start = time.perf_counter()
response = client.chat.completions.create(
    model="gpt-5-mini",
    messages=[{"role": "user", "content": "Write a 100-word biography of a fictional person."}]
)
print(f"Simple completion: {time.perf_counter() - start:.1f}s")
print(f"Tokens: {response.usage}")

# Test 2: Longer completion (~1000 words)
start = time.perf_counter()
response = client.chat.completions.create(
    model="gpt-5-mini",
    messages=[{"role": "user", "content": "Write a detailed 1000-word biography of a fictional 19th century person."}]
)
print(f"Long completion: {time.perf_counter() - start:.1f}s")
print(f"Tokens: {response.usage}")
```

### 2.2 Test structured output overhead

Compare plain text vs JSON schema mode:

```python
from pydantic import BaseModel

class SimpleOutput(BaseModel):
    name: str
    birth_year: int

# With structured output
start = time.perf_counter()
response = client.beta.chat.completions.parse(
    model="gpt-5-mini",
    messages=[{"role": "user", "content": "Generate a person's name and birth year."}],
    response_format=SimpleOutput
)
print(f"Structured output: {time.perf_counter() - start:.1f}s")
```

---

## Phase 3: Investigate Pydantic AI Internals

**Goal**: Check if pydantic_ai is doing internal retries

### 3.1 Enable pydantic_ai debug logging

```python
import logging
logging.getLogger("pydantic_ai").setLevel(logging.DEBUG)
```

### 3.2 Check for internal retries

Questions to answer:
- Does pydantic_ai retry when JSON parsing fails?
- How many actual API calls per `agent.run()`?
- Is there validation retry logic?

### 3.3 Simplify the schema

Test with minimal `ExtractedData` to isolate schema complexity:

```python
# Temporarily reduce schema complexity
class SimpleExtractedData(BaseModel):
    given_name: str
    surname: str
    birth_year: int | None = None
```

Compare timing with full schema vs simplified schema.

---

## Phase 4: Network & Rate Limiting

**Goal**: Rule out network/throttling issues

### 4.1 Check rate limiter behavior

Current config: 60 requests/minute (~1 req/sec)

Questions:
- Are we hitting the limit?
- Is there unexpected waiting?

Add logging:
```python
# In _acquire_rate_limit()
wait_time = await self._rate_limiter.acquire()
if wait_time > 0.01:  # Log any wait
    self._timer.log(f"Rate limit: waited {wait_time:.2f}s")
```

### 4.2 Monitor actual HTTP requests

```bash
# Set OpenAI debug logging
export OPENAI_LOG=debug
uv run ancestral-synth generate -n 1 --verbose 2>&1 | tee http_debug.log
```

### 4.3 Check for network latency

```bash
# Test API endpoint latency
curl -w "@curl-format.txt" -s -o /dev/null https://api.openai.com/v1/models
```

---

## Phase 5: Compare Operations

**Goal**: Find which specific operation is the bottleneck

### 5.1 Expected vs Actual Timing

| Operation | Expected | Investigate if | Current Actual |
|-----------|----------|----------------|----------------|
| Biography gen | 5-20s | >30s | 44-136s |
| Data extraction | 10-30s | >60s | 67-754s |
| Dedup check | 3-10s | >15s | 7-13s |
| DB operations | <100ms | >1s | ~10ms |
| Rate limit wait | 0-1s | >2s | ? |

### 5.2 Identify patterns

- Is extraction always slower than biography? (Yes, consistently ~2x)
- Are there specific persons that trigger slowness?
- Does context size affect timing?

---

## Phase 6: Potential Optimizations

Based on findings, consider these optimizations:

### 6.1 If LLM calls are inherently slow

```python
# Option A: Parallelize biography + extraction
async def process_parallel():
    bio_task = asyncio.create_task(generate_biography())
    # Start extraction as soon as bio is ready (streaming?)

# Option B: Batch dedup checks
async def batch_dedup(references: list[PersonReference]):
    # Check all references in one LLM call
    pass

# Option C: Use streaming to start processing earlier
async for chunk in agent.run_stream(prompt):
    process_chunk(chunk)
```

### 6.2 If structured output is slow

```python
# Option A: Simplify schema
# Remove nested lists, optional fields

# Option B: Two-stage extraction
biography = await generate_plain_text()
extracted = await parse_with_simple_schema(biography)
details = await enrich_extracted_data(extracted)

# Option C: Different output format
# Use XML or markdown instead of JSON
```

### 6.3 If rate limiting is the issue

```python
# Increase rate limit if API allows
settings.llm_requests_per_minute = 120

# Or use multiple API keys with rotation
```

### 6.4 Architecture changes

```python
# Process multiple persons concurrently
async def generate_batch(count: int):
    tasks = [process_person() for _ in range(min(count, 5))]
    results = await asyncio.gather(*tasks)
```

---

## Phase 7: Execution Checklist

- [ ] Run Phase 1: Baseline measurement with instrumentation
- [ ] Run Phase 2: Direct API tests
- [ ] Run Phase 3: Check pydantic_ai internals
- [ ] Run Phase 4: Network/rate limit investigation
- [ ] Analyze Phase 5: Compare all operations
- [ ] Implement Phase 6: Apply optimizations based on findings

---

## Notes

- The new instrumentation was added in commit `4d92dc7`
- Model validation warning added for unknown OpenAI models
- gpt-5-mini confirmed as valid model
