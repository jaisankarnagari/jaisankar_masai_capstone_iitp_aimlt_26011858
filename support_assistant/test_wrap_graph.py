#!/usr/bin/env python
"""
Test client for wrap_graph.py FastAPI app

Run the server first:
    uvicorn wrap_graph:app --reload

Then run this script:
    python test_wrap_graph.py
"""

import requests
import json

BASE_URL = "http://127.0.0.1:8000"

def run_test(query: str):
    print("=" * 80)
    print(f"Test query: {query}")
    response = requests.post(f"{BASE_URL}/ask", json={"query": query})
    print(f"Status code: {response.status_code}")
    try:
        data = response.json()
        print("Raw JSON response:")
        print(json.dumps(data, indent=2))
    except Exception as e:
        print("Failed to parse JSON:", e)
        print("Response text:", response.text)

if __name__ == "__main__":
    # Test 1: Policy question (retrieval triggered)
    run_test("What is your refund policy?")

    # Test 2: General question (no retrieval)
    run_test("How is the weather today?")

    # Test 3: Another policy-related query
    run_test("Can I cancel my membership?")

    # Test 4: Random general query
    run_test("Tell me a joke")