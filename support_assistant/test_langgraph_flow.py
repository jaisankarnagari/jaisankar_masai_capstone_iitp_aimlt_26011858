"""
Test suite for langgraph_flow.py

Run tests with:
    python -m pytest test_langgraph_flow.py -v
    
Or directly:
    python test_langgraph_flow.py
"""

import os
import sys
from langgraph_flow import (
    run_zepto_assistant,
    classify_intent,
    retrieve_and_answer,
    direct_answer,
    ZeptoAssistantState,
    POLICY_KEYWORDS,
    build_zepto_graph,
)


# ============================================================================
# TEST DATA
# ============================================================================

POLICY_QUERIES = [
    "What is your refund policy?",
    "Can I return items after 30 days?",
    "How do I track my order?",
    "What is the delivery fee?",
    "Can I cancel my subscription?",
    "Do you have gift cards?",
    "What are your support hours?",
    "Tell me about membership",
]

GENERAL_QUERIES = [
    "How is the weather today?",
    "What is machine learning?",
    "Tell me about Python programming",
    "Who is the president of India?",
    "How to bake a cake?",
]


# ============================================================================
# INDIVIDUAL NODE TESTS
# ============================================================================

def test_classify_intent_mock():
    """Test intent classification in MOCK mode"""
    print("\n" + "="*80)
    print("TEST: classify_intent (MOCK mode)")
    print("="*80)
    
    state: ZeptoAssistantState = {
        "query": "What is your refund policy?",
        "intent": "",
        "retrieved_chunks": [],
        "top_chunk": "",
        "answer": "",
        "mock_mode": True,
        "llm_used": "",
    }
    
    result = classify_intent(state)
    
    assert result["intent"] in ["policy_question", "general_question"], "Invalid intent"
    assert result["intent"] == "policy_question", f"Expected policy_question, got {result['intent']}"
    print(f"✓ Policy question classified correctly: {result['intent']}")
    
    # Test general question
    state["query"] = "How is the weather?"
    result = classify_intent(state)
    assert result["intent"] == "general_question", f"Expected general_question, got {result['intent']}"
    print(f"✓ General question classified correctly: {result['intent']}")


def test_keyword_matching():
    """Test that keywords are matched correctly"""
    print("\n" + "="*80)
    print("TEST: Keyword matching")
    print("="*80)
    
    for keyword in POLICY_KEYWORDS:
        query = f"I have a question about {keyword}"
        state: ZeptoAssistantState = {
            "query": query,
            "intent": "",
            "retrieved_chunks": [],
            "top_chunk": "",
            "answer": "",
            "mock_mode": True,
            "llm_used": "",
        }
        result = classify_intent(state)
        assert result["intent"] == "policy_question", f"Failed for keyword: {keyword}"
    
    print(f"✓ All {len(POLICY_KEYWORDS)} keywords matched correctly")


def test_direct_answer_node():
    """Test direct_answer node for general questions"""
    print("\n" + "="*80)
    print("TEST: direct_answer node")
    print("="*80)
    
    state: ZeptoAssistantState = {
        "query": "How is the weather?",
        "intent": "general_question",
        "retrieved_chunks": [],
        "top_chunk": "",
        "answer": "",
        "mock_mode": True,
        "llm_used": "",
    }
    
    result = direct_answer(state)
    
    assert result["answer"], "Answer should not be empty"
    assert result["llm_used"] == "mock", f"Expected mock LLM, got {result['llm_used']}"
    assert result["retrieved_chunks"] == [], "Should have no retrieved chunks"
    print(f"✓ Direct answer node works correctly")
    print(f"  Answer: {result['answer'][:60]}...")


# ============================================================================
# END-TO-END TESTS
# ============================================================================

def test_full_workflow_policy_mock():
    """Test complete workflow with policy question in MOCK mode"""
    print("\n" + "="*80)
    print("TEST: Full workflow - Policy question (MOCK mode)")
    print("="*80)
    
    result = run_zepto_assistant("What is your refund policy?", mock_mode=True)
    
    assert result["query"] == "What is your refund policy?"
    assert result["intent"] == "policy_question"
    assert result["llm_used"] == "mock"
    assert result["answer"], "Answer should not be empty"
    print(f"✓ Policy question workflow successful")
    print(f"  Intent: {result['intent']}")
    print(f"  LLM: {result['llm_used']}")


def test_full_workflow_general_mock():
    """Test complete workflow with general question in MOCK mode"""
    print("\n" + "="*80)
    print("TEST: Full workflow - General question (MOCK mode)")
    print("="*80)
    
    result = run_zepto_assistant("How is the weather today?", mock_mode=True)
    
    assert result["query"] == "How is the weather today?"
    assert result["intent"] == "general_question"
    assert result["llm_used"] == "mock"
    assert result["answer"], "Answer should not be empty"
    print(f"✓ General question workflow successful")
    print(f"  Intent: {result['intent']}")
    print(f"  LLM: {result['llm_used']}")


def test_graph_structure():
    """Test that the graph is properly constructed"""
    print("\n" + "="*80)
    print("TEST: Graph structure")
    print("="*80)
    
    graph = build_zepto_graph()
    
    assert graph is not None, "Graph should not be None"
    print(f"✓ Graph successfully built")
    
    # Test graph execution
    initial_state = {
        "query": "What is the refund policy?",
        "intent": "",
        "retrieved_chunks": [],
        "top_chunk": "",
        "answer": "",
        "mock_mode": True,
        "llm_used": "",
    }
    
    result = graph.invoke(initial_state)
    assert result["intent"], "Intent should be classified"
    assert result["answer"], "Answer should be generated"
    print(f"✓ Graph execution successful")


# ============================================================================
# BATCH TESTS
# ============================================================================

def test_policy_queries_batch():
    """Test multiple policy questions"""
    print("\n" + "="*80)
    print("TEST: Batch - Policy questions")
    print("="*80)
    
    passed = 0
    failed = 0
    
    for query in POLICY_QUERIES[:3]:  # Test first 3
        try:
            result = run_zepto_assistant(query, mock_mode=True)
            if result["intent"] == "policy_question":
                passed += 1
                print(f"✓ {query[:50]}...")
            else:
                failed += 1
                print(f"✗ {query[:50]}... - Got {result['intent']}")
        except Exception as e:
            failed += 1
            print(f"✗ {query[:50]}... - Error: {e}")
    
    print(f"\nResults: {passed} passed, {failed} failed")


def test_general_queries_batch():
    """Test multiple general questions"""
    print("\n" + "="*80)
    print("TEST: Batch - General questions")
    print("="*80)
    
    passed = 0
    failed = 0
    
    for query in GENERAL_QUERIES[:3]:  # Test first 3
        try:
            result = run_zepto_assistant(query, mock_mode=True)
            if result["intent"] == "general_question":
                passed += 1
                print(f"✓ {query[:50]}...")
            else:
                failed += 1
                print(f"✗ {query[:50]}... - Got {result['intent']}")
        except Exception as e:
            failed += 1
            print(f"✗ {query[:50]}... - Error: {e}")
    
    print(f"\nResults: {passed} passed, {failed} failed")


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

def test_response_time():
    """Test response time for queries"""
    print("\n" + "="*80)
    print("TEST: Response time")
    print("="*80)
    
    import time
    
    queries = [
        "What is your refund policy?",
        "How is the weather?",
        "Can I cancel my order?",
    ]
    
    times = []
    for query in queries:
        start = time.time()
        result = run_zepto_assistant(query, mock_mode=True)
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"Query: {query[:40]}... Time: {elapsed:.3f}s")
    
    avg_time = sum(times) / len(times)
    print(f"\nAverage response time: {avg_time:.3f}s")
    
    if avg_time < 1.0:
        print("✓ Performance acceptable")
    else:
        print("⚠ Performance may need optimization")


# ============================================================================
# MOCK VS REAL MODE COMPARISON
# ============================================================================

def test_mock_vs_real_mode():
    """Compare MOCK and REAL mode outputs"""
    print("\n" + "="*80)
    print("TEST: MOCK vs REAL mode comparison")
    print("="*80)
    
    query = "What is your refund policy?"
    
    # Test MOCK mode
    mock_result = run_zepto_assistant(query, mock_mode=True)
    print(f"\n[MOCK Mode]")
    print(f"  Intent: {mock_result['intent']}")
    print(f"  LLM Used: {mock_result['llm_used']}")
    print(f"  Answer: {mock_result['answer'][:60]}...")
    
    # Test REAL mode (optional)
    real_result = run_zepto_assistant(query, mock_mode=False)
    print(f"\n[REAL Mode]")
    print(f"  Intent: {real_result['intent']}")
    print(f"  LLM Used: {real_result['llm_used']}")
    print(f"  Answer: {real_result['answer'][:60]}...")
    
    # Both should have same intent
    assert mock_result["intent"] == real_result["intent"], "Intent should be same"
    print(f"\n✓ Both modes produce same intent classification")


# ============================================================================
# RUN ALL TESTS
# ============================================================================

def run_all_tests():
    """Run all test functions"""
    print("\n" + "="*80)
    print("LANGGRAPH FLOW - COMPREHENSIVE TEST SUITE")
    print("="*80)
    
    test_functions = [
        test_classify_intent_mock,
        test_keyword_matching,
        test_direct_answer_node,
        test_full_workflow_policy_mock,
        test_full_workflow_general_mock,
        test_graph_structure,
        test_policy_queries_batch,
        test_general_queries_batch,
        test_response_time,
        test_mock_vs_real_mode,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in test_functions:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            failed += 1
            print(f"\n✗ FAILED: {test_func.__name__}")
            print(f"  Error: {e}")
        except Exception as e:
            failed += 1
            print(f"\n✗ ERROR: {test_func.__name__}")
            print(f"  Error: {e}")
    
    print("\n" + "="*80)
    print(f"FINAL RESULTS: {passed} passed, {failed} failed out of {len(test_functions)} tests")
    print("="*80 + "\n")
    
    return passed, failed


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

def print_usage_examples():
    """Print usage examples"""
    print("\n" + "="*80)
    print("USAGE EXAMPLES")
    print("="*80)
    
    print("""
1. Run this test file:
   python test_langgraph_flow.py

2. Run specific tests with pytest:
   pytest test_langgraph_flow.py::test_classify_intent_mock -v
   pytest test_langgraph_flow.py -k "batch" -v

3. Interactive testing:
   from langgraph_flow import run_zepto_assistant
   result = run_zepto_assistant("What is your refund policy?")
   print(result)

4. Test with real LLM mode:
   from langgraph_flow import run_zepto_assistant
   result = run_zepto_assistant("What is your refund policy?", mock_mode=False)

5. Environment variable mode:
   # MOCK mode (default)
   MOCK_LLM=1 python test_langgraph_flow.py
   
   # REAL LLM mode
   MOCK_LLM=0 python test_langgraph_flow.py
""")


if __name__ == "__main__":
    # Run all tests
    passed, failed = run_all_tests()
    
    # Print usage examples
    print_usage_examples()
    
    # Exit with appropriate code
    sys.exit(0 if failed == 0 else 1)
