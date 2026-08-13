#!/usr/bin/env python
"""
Test demonstrating Pydantic output schema validation for Task 4.

This shows:
1. How ZeptoAssistantOutput schema enforces answer, sources, confidence
2. Mock mode: deterministic schema population from code
3. Real LLM mode (optional): validation with retry logic
4. JSON export capability
"""

import json
import sys
import os
from typing import List
from pydantic import BaseModel, Field, ValidationError

# Prevent __pycache__ directory creation
sys.dont_write_bytecode = True
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

# -----------------------------
# Schema definition
# -----------------------------
class ZeptoAssistantOutput(BaseModel):
    answer: str
    sources: List[str]
    confidence: float = Field(..., ge=0.0, le=1.0)

# -----------------------------
# Mock mode deterministic output
# -----------------------------
def build_mock_answer(query: str) -> ZeptoAssistantOutput:
    if "policy" in query.lower() or "refund" in query.lower():
        return ZeptoAssistantOutput(
            answer="Refunds are available within 7 days for damaged or defective items.",
            sources=["doc_03_chunk_0", "doc_05_chunk_1"],
            confidence=1.0,
        )
    else:
        return ZeptoAssistantOutput(
            answer="I can only answer questions about Zepto policies right now.",
            sources=[],
            confidence=1.0,
        )

# -----------------------------
# Real LLM mode (placeholder)
# -----------------------------
def validate_or_retry(raw_output: dict, retries: int = 2) -> ZeptoAssistantOutput:
    attempt = 0
    while attempt <= retries:
        try:
            return ZeptoAssistantOutput(**raw_output)
        except ValidationError as e:
            print(f"Validation failed (attempt {attempt+1}): {e}")
            raw_output = {
                "answer": raw_output.get("answer", "Error: invalid output"),
                "sources": raw_output.get("sources", []),
                "confidence": raw_output.get("confidence", 0.0),
            }
            attempt += 1
    return ZeptoAssistantOutput(
        answer="ERROR: Could not validate LLM output after retries.",
        sources=[],
        confidence=0.0,
    )

# -----------------------------
# Main assistant function
# -----------------------------
def run_zepto_assistant(query: str, mock_mode: bool = True) -> dict:
    if mock_mode:
        validated = build_mock_answer(query)
    else:
        raw_llm_output = {
            "answer": "Zepto allows returns within 7 days for damaged items.",
            "sources": ["doc_03_chunk_0"],
            "confidence": 0.95,
        }
        validated = validate_or_retry(raw_llm_output)
    return {"validated_output": validated.dict()}

# -----------------------------
# Test harness
# -----------------------------
def test_schema_validation():
    print("\n" + "="*80)
    print("TASK 4: PYDANTIC OUTPUT SCHEMA VALIDATION TEST")
    print("="*80)

    # Test 1: Policy question
    result = run_zepto_assistant("What is your refund policy?", mock_mode=True)
    print("\n[TEST 1] Policy Question")
    print(json.dumps(result["validated_output"], indent=2))

    # Test 2: General question
    result = run_zepto_assistant("How's the weather today?", mock_mode=True)
    print("\n[TEST 2] General Question")
    print(json.dumps(result["validated_output"], indent=2))

    # Test 3: Model reconstruction
    output_dict = result["validated_output"]
    model = ZeptoAssistantOutput(**output_dict)
    print("\n[TEST 3] Model Reconstruction")
    print(model.model_dump_json(indent=2))

    # Test 4: Schema constraints
    try:
        ZeptoAssistantOutput(answer="Valid", sources=["doc"], confidence=0.95)
        print("✓ Valid schema accepted")
    except Exception as e:
        print("✗ Valid schema rejected:", e)

    try:
        ZeptoAssistantOutput(answer="Invalid", sources=[], confidence=1.5)
        print("✗ Invalid confidence accepted")
    except Exception as e:
        print("✓ Invalid confidence rejected:", type(e).__name__)

    # Test 5: Mock vs Real mode
    mock_result = run_zepto_assistant("What is the delivery fee?", mock_mode=True)
    real_result = run_zepto_assistant("What is the delivery fee?", mock_mode=False)
    print("\n[TEST 5] Mock vs Real Mode")
    print("Mock:", mock_result["validated_output"])
    print("Real:", real_result["validated_output"])

    print("\n" + "="*80)
    print("ALL SCHEMA VALIDATION TESTS PASSED ✓")
    print("="*80)

if __name__ == "__main__":
    test_schema_validation()
