from langgraph.graph import StateGraph, START, END
from weave.integrations.langchain import WeaveTracer

from src.agents import (
    AgentState,
    llm_call_router,
    knowledge_base,
    web_search,
    summarizer,
    answer_generation,
    route_decision,
)


def build_graph():
    graph_builder = StateGraph(AgentState)

    graph_builder.add_node("llm_call_router", llm_call_router)
    graph_builder.add_node("knowledge_base", knowledge_base)
    graph_builder.add_node("web_search", web_search)
    graph_builder.add_node("summarizer", summarizer)
    graph_builder.add_node("answer_generation", answer_generation)

    graph_builder.add_edge(START, "llm_call_router")
    graph_builder.add_conditional_edges(
        "llm_call_router",
        route_decision,
        {
            "knowledge_base": "knowledge_base",
            "web_search": "web_search",
            "summarizer": "summarizer",
        },
    )
    graph_builder.add_edge("knowledge_base", "answer_generation")
    graph_builder.add_edge("web_search", "answer_generation")
    graph_builder.add_edge("summarizer", "answer_generation")
    graph_builder.add_edge("answer_generation", END)

    return graph_builder.compile()


def get_weave_config():
    weave_tracer = WeaveTracer()
    return {"callbacks": [weave_tracer]}


graph = build_graph()

