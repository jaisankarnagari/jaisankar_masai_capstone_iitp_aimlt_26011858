# Minimal LangGraph orchestration wrapper. If LangGraph is available, this builds
# a tiny flow that retrieves relevant documents and then calls the (mock) LLM.
# If LangGraph isn't installed, this file is still safe to import.

try:
    from langgraph.core import Graph, Node
    LANGGRAPH_AVAILABLE = True
except Exception:
    LANGGRAPH_AVAILABLE = False


def build_flow(retriever, llm):
    if not LANGGRAPH_AVAILABLE:
        raise RuntimeError('LangGraph not available in this environment')
    # This is a placeholder; a real LangGraph flow would be more involved.
    g = Graph()
    # Nodes would wrap retrieval and LLM; left as an exercise when running with LangGraph.
    return g
