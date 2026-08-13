#!/usr/bin/env python
"""Quick test of langgraph_flow - minimal version"""

print("Starting langgraph_flow test...")

# Simple inline test without full diagnostics
try:
    print("Importing langgraph_flow...")
    from langgraph_flow import run_zepto_assistant
    print("✓ Import successful\n")
    
    print("Test 1: Policy question")
    print("="*60)
    result = run_zepto_assistant("What is your refund policy?", mock_mode=True)
    print(f"\nResult:")
    print(f"  Intent: {result['intent']}")
    print(f"  LLM Used: {result['llm_used']}")
    print(f"  Answer preview: {result['answer'][:80]}...\n")
    
    print("Test 2: General question")
    print("="*60)
    result = run_zepto_assistant("How's the weather?", mock_mode=True)
    print(f"\nResult:")
    print(f"  Intent: {result['intent']}")
    print(f"  LLM Used: {result['llm_used']}")
    print(f"  Answer preview: {result['answer'][:80]}...\n")
    
    print("✓ All tests passed!")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
