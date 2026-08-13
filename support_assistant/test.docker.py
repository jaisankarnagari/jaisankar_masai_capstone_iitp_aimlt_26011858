#!/usr/bin/env python
"""
text_docker.py
Helper script to display Docker build/run/test steps for Task 6
"""

def main():
    print("="*80)
    print("🚀 Task 6: Docker Build & Run Instructions")
    print("="*80)

    print("\n1. Build the Docker image:")
    print("   docker build -t zepto-assistant .")

    print("\n2. Run the container (map port 7860):")
    print("   docker run -p 7860:7860 zepto-assistant")

    print("\n3. Test the endpoint with curl:")

    print("\n   Policy question (retrieval triggered):")
    print('   curl -X POST "http://127.0.0.1:7860/ask" \\')
    print('        -H "Content-Type: application/json" \\')
    print('        -d \'{"query": "What is your refund policy?"}\'')

    print("\n   Expected JSON response:")
    print('   {')
    print('     "answer": "Refunds are available within 7 days for damaged or defective items.",')
    print('     "sources": ["doc_03_chunk_0", "doc_05_chunk_1"],')
    print('     "confidence": 1.0')
    print('   }')

    print("\n   General question (no retrieval):")
    print('   curl -X POST "http://127.0.0.1:7860/ask" \\')
    print('        -H "Content-Type: application/json" \\')
    print('        -d \'{"query": "How is the weather today?"}\'')

    print("\n   Expected JSON response:")
    print('   {')
    print('     "answer": "I can only answer questions about Zepto policies right now.",')
    print('     "sources": [],')
    print('     "confidence": 1.0')
    print('   }')

    print("\n4. Stop the container when finished:")
    print("   docker ps")
    print("   docker stop <container_id>")

    print("\n✓ All steps complete — your FastAPI app is now containerized and testable locally.")

if __name__ == "__main__":
    main()
