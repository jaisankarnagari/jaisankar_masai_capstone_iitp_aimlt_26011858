#!/usr/bin/env python
"""Quick test of langgraph_flow - robust version"""

print("Starting langgraph_flow test...")

try:
    print("Importing langgraph_flow...")
    from langgraph_flow import run_zepto_assistant
    print("✓ Import successful\n")

    def run_test(query: str):
        print("=" * 60)
        print(f"Test query: {query}")
        result = run_zepto_assistant(query, mock_mode=True)
        print("\nResult keys:", list(result.keys()))
        print("Full result:", result)

        # Safely extract fields
        intent = result.get("intent", "N/A")
        llm_used = result.get("llm_used", "N/A")
        answer = result.get("answer") or result.get("response") or "<no answer field>"

        print(f"\nParsed output:")
        print(f"  Intent: {intent}")
        print(f"  LLM Used: {llm_used}")
        print(f"  Answer preview: {answer[:80]}...\n")

    # Run tests
    run_test("What is your refund policy?")
    run_test("How's the weather?")

    print("✓ All tests completed!")

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
