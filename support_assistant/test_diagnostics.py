"""Simple diagnostic script to test langgraph_flow imports and functionality"""

import sys
import os

print("="*80)
print("LANGGRAPH FLOW - DIAGNOSTIC TEST")
print("="*80)

# Test 1: Check Python version
print(f"\n1. Python Version: {sys.version}")

# Test 2: Check if module exists
print(f"\n2. Checking module existence...")
module_path = os.path.abspath("langgraph_flow.py")
print(f"   Module path: {module_path}")
print(f"   Module exists: {os.path.exists(module_path)}")

# Test 3: Try importing dependencies
print(f"\n3. Testing dependencies...")
try:
    import chromadb
    print("   ✓ chromadb imported successfully")
except ImportError as e:
    print(f"   ✗ chromadb import failed: {e}")

try:
    import langgraph
    print("   ✓ langgraph imported successfully")
except ImportError as e:
    print(f"   ✗ langgraph import failed: {e}")

try:
    from sentence_transformers import SentenceTransformer
    print("   ✓ sentence_transformers imported successfully")
except ImportError as e:
    print(f"   ✗ sentence_transformers import failed: {e}")

try:
    from prompting import ZeptoPromptTemplate
    print("   ✓ prompting module imported successfully")
except ImportError as e:
    print(f"   ✗ prompting module import failed: {e}")

# Test 4: Try importing langgraph_flow
print(f"\n4. Importing langgraph_flow...")
try:
    from langgraph_flow import run_zepto_assistant, POLICY_KEYWORDS
    print("   ✓ langgraph_flow imported successfully")
    print(f"   ✓ Found {len(POLICY_KEYWORDS)} policy keywords")
except Exception as e:
    print(f"   ✗ langgraph_flow import failed: {e}")
    print(f"   Error type: {type(e).__name__}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Run simple test
print(f"\n5. Running simple test query...")
try:
    print("   Query: 'What is your refund policy?'")
    result = run_zepto_assistant("What is your refund policy?", mock_mode=True)
    print(f"   ✓ Query executed successfully")
    print(f"   - Intent: {result.get('intent')}")
    print(f"   - LLM Used: {result.get('llm_used')}")
    print(f"   - Answer length: {len(result.get('answer', ''))}")
except Exception as e:
    print(f"   ✗ Query execution failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Test general question
print(f"\n6. Testing general question...")
try:
    print("   Query: 'How is the weather?'")
    result = run_zepto_assistant("How is the weather?", mock_mode=True)
    print(f"   ✓ Query executed successfully")
    print(f"   - Intent: {result.get('intent')}")
    print(f"   - LLM Used: {result.get('llm_used')}")
except Exception as e:
    print(f"   ✗ Query execution failed: {e}")
    sys.exit(1)

print("\n" + "="*80)
print("ALL DIAGNOSTICS PASSED ✓")
print("="*80 + "\n")
