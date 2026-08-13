# Task 4: Pydantic Output Schema Validation

## Overview

Task 4 enforces a strict JSON output schema via Pydantic with automatic validation and retry logic.

## Schema Definition

```python
class ZeptoAssistantOutput(BaseModel):
    """Validated JSON output for Zepto Support Assistant"""
    answer: str              # The assistant's response
    sources: list[str]       # Source chunk/document IDs (empty for general_question)
    confidence: float        # Confidence level (0.0-1.0)
```

## Implementation

### 1. Deterministic Mock Mode (Graded Baseline)

Mock mode populates the schema **deterministically from code**, no LLM output to validate:

```python
# For policy_question
answer = "Based on the retrieved context: ..."
sources = [chunk_id_1, chunk_id_2, chunk_id_3]  # Retrieved chunk IDs
confidence = 1.0  # Fixed high confidence for mock

# For general_question
answer = "I can only answer questions about Zepto policies..."
sources = []  # Empty - no retrieval
confidence = 0.8  # Fixed confidence for general Q&A
```

**Key Points:**
- ✅ No LLM to fail validation (we create the schema)
- ✅ Deterministic values every run
- ✅ Sources populated from actual ChromaDB chunk IDs
- ✅ Confidence fixed per question type

### 2. Real LLM Mode with Retry Logic (Optional Extension)

Real mode validates LLM output against schema with retry:

```
Step 1: LLM generates raw answer
   ↓
Step 2: Validate against ZeptoAssistantOutput schema
   ├→ Valid? → Return validated output ✓
   └→ Invalid? → Attempt 2 of 3
   
Step 3: Apply corrective instruction, retry
   ├→ Valid? → Return validated output ✓
   └→ Invalid? → Attempt 3 of 3
   
Step 4: Final retry with corrective instruction
   ├→ Valid? → Return validated output ✓
   └→ Invalid? → Return error response with confidence=0.0
```

**Retry Logic:**
- **Attempt 1**: Initial LLM output validation
- **Attempt 2**: Fix common issues (truncate long answers, clamp confidence)
- **Attempt 3**: Final attempt with corrective prompt
- **Failure**: Return clearly marked error response

### 3. Validation Function

```python
def _validate_output(answer: str, sources: list, confidence: float, attempt: int = 1) -> ZeptoAssistantOutput:
    """
    Validate output against schema with automatic retry.
    
    Retries up to 3 times total if validation fails.
    Returns validated object or error response.
    """
```

## State Updates

The `ZeptoAssistantState` now tracks:

```python
chunk_ids: Sequence[str]      # IDs of retrieved chunks for sources
validated_output: dict        # The final validated schema dict
validation_attempts: int      # Track retry attempts
```

## Output Changes

`run_zepto_assistant()` now returns:

```json
{
  "query": "What is your refund policy?",
  "intent": "policy_question",
  "llm_used": "mock",
  "validated_output": {
    "answer": "Based on the retrieved context: Returns can be made within 30 days...",
    "sources": ["doc_02.txt_chunk_0", "doc_02.txt_chunk_1", "doc_02.txt_chunk_2"],
    "confidence": 1.0
  }
}
```

## Testing

Run the schema validation tests:

```bash
python test_schema_validation.py
```

**Test Coverage:**
1. Policy questions with sources
2. General questions with empty sources
3. Pydantic model reconstruction
4. JSON serialization round-trip
5. Schema constraint enforcement (confidence 0-1)
6. Mock vs Real mode output comparison

## Example Usage

```python
from langgraph_flow import run_zepto_assistant, ZeptoAssistantOutput

# Run query
result = run_zepto_assistant("What is your refund policy?")

# Access validated output
output = result["validated_output"]
print(f"Answer: {output['answer']}")
print(f"Sources: {output['sources']}")
print(f"Confidence: {output['confidence']}")

# Reconstruct Pydantic model
model = ZeptoAssistantOutput(**output)

# Export to JSON
json_str = model.model_dump_json(indent=2)
```

## Key Differences from Task 3

| Aspect | Task 3 | Task 4 |
|--------|--------|--------|
| Output | String answer | Validated JSON schema |
| Sources Tracking | Not tracked | Chunk IDs stored |
| Confidence | Not included | Validated 0-1 float |
| Error Handling | Basic | Retry logic (up to 3 times) |
| LLM Validation | N/A | With retries and correction |
| Schema Constraints | N/A | Enforced via Pydantic |

## Schema Constraints

The Pydantic model enforces:

```python
answer: str                              # Any text, no length limit enforced (but validated down to 500 chars in code)
sources: list[str]                       # Empty list by default
confidence: float                        # Must be 0.0 ≤ confidence ≤ 1.0 (strict validation)
```

**Constraint Violations:**
- ❌ `confidence: 1.5` → ValidationError
- ❌ `confidence: -0.1` → ValidationError
- ❌ `sources: "not_a_list"` → ValidationError
- ✅ `confidence: 0.0` → Valid
- ✅ `confidence: 0.5` → Valid
- ✅ `confidence: 1.0` → Valid
- ✅ `sources: []` → Valid
- ✅ `sources: ["doc_1", "doc_2"]` → Valid

## Production Ready Features

✅ **Type Safety**: Pydantic enforces types at runtime  
✅ **JSON Export**: Automatic JSON serialization  
✅ **Schema Validation**: All outputs must conform  
✅ **Retry Logic**: Handles LLM output failures gracefully  
✅ **Error Responses**: Clear error messages with confidence=0.0  
✅ **Documentation**: Field descriptions in schema  
✅ **Example**: Included in schema for reference  

## Summary

Task 4 implementation:
- ✅ Pydantic model with 3 required fields
- ✅ Mock mode: deterministic population (no LLM)
- ✅ Real mode: validation with up to 3 retry attempts
- ✅ Sources correctly populated for policy questions
- ✅ Confidence properly constrained (0-1)
- ✅ Clear error responses on validation failure
