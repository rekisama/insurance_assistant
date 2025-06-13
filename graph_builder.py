from langgraph.graph import StateGraph, START, END
from state import State
from nodes import (
    pdf_summary_node,
    vector_search_node,
    chat_node,
    pdf_risk_mark_node,
    extract_formula_node,
    describe_formula_node,
    llm_free_reasoning_node,
    llm_insurance_visualize_node,
    router_node,
)
from nodes.router import route_fn

def build_graph():
    graph_builder = StateGraph(State)
    graph_builder.add_node("pdf_summary", pdf_summary_node)
    graph_builder.add_node("vector_search", vector_search_node)
    graph_builder.add_node("chat", chat_node)
    graph_builder.add_node("pdf_risk_mark", pdf_risk_mark_node)
    graph_builder.add_node("extract_formula", extract_formula_node)
    graph_builder.add_node("router", router_node)
    graph_builder.add_node("describe_formula", describe_formula_node)
    graph_builder.add_node("llm_free_reasoning", llm_free_reasoning_node)
    graph_builder.add_node("llm_insurance_visualize", llm_insurance_visualize_node)
    graph_builder.add_edge(START, "router")
    graph_builder.add_conditional_edges(
        "router",
        route_fn,
        {
            "pdf_summary": "pdf_summary",
            "vector_search": "vector_search",
            "pdf_risk_mark": "pdf_risk_mark",
            "extract_formula": "extract_formula",
            "describe_formula": "describe_formula",
            "llm_free_reasoning": "llm_free_reasoning",
            "chat": "chat",
            "llm_insurance_visualize": "llm_insurance_visualize"
        }
    )
    graph_builder.add_edge("describe_formula", END)
    graph_builder.add_edge("pdf_summary", END)
    graph_builder.add_edge("vector_search", END)
    graph_builder.add_edge("pdf_risk_mark", END)
    graph_builder.add_edge("extract_formula", END)
    graph_builder.add_edge("chat", END)
    graph_builder.add_edge("llm_free_reasoning", END)
    graph_builder.add_edge("llm_insurance_visualize", END)
    return graph_builder.compile()