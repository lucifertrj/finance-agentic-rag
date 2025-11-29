from pydantic import BaseModel, Field
from typing import TypedDict, Literal, List
from qdrant_client import models

from src.client import llm, tavily_client
from src.retrieval import db_search
from src.utils import get_target_doc_type


class Route(BaseModel):
    step: Literal["knowledge", "search", "summary"] = Field(
        None, description="The next step in the routing process"
    )


class AgentState(TypedDict):
    tool_used: str
    question: str
    context: List[str]
    response: str


router = llm.with_structured_output(Route)


def llm_call_router(state: AgentState):
    SYSTEM_PROMPT = """
    Route the input to knowledge, search, or summary based on the user's request.
    - route to knowledge_base if user query is grounded in the knowledge base and it is a QA based question within the knowledge base.
    - route to web_search if user query is not grounded in the knowledge base and needs to be fetched from web.
    - route to summarizer if user query is not grounded in the knowledge base and needs to be summarized or the question is to summarize any document i.e., policy or filing or compliance or news information from knowledge base.
    """

    decision = router.invoke(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": state["question"]}
        ]
    )

    return {"tool_used": decision.step}


def knowledge_base(state: AgentState) -> AgentState:
    query = state["question"]
    response = db_search(query, filter_condition=None)

    context_parts = []
    for point in response.points:
        payload = point.payload

        context_parts.append(
            f"Content: {payload.get('content', '')}"
            f"\nSource: {payload.get('source', 'Unknown')}"
            f"\nDocType:: {payload.get('document_type', 'Unknown')}"
            f"\nPage: {payload.get('page', 'N/A')}"
            f"\nTags: {payload.get('chunk_tags', 'None')}"
        )

    context = "\n".join(context_parts)

    return {"context": context}


def web_search(state: AgentState) -> AgentState:
    query = state['question']
    response = tavily_client.search(query, max_results=10)
    context = ""
    for result in response["results"]:
        context += result["title"] + " " + result["content"]

    return {"context": context}


def summarizer(state: AgentState) -> AgentState:
    query = state["question"]

    target_doc = get_target_doc_type(query)

    query_filter = None
    if target_doc:
        query_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="document_type",
                    match=models.MatchValue(value=target_doc)
                )
            ]
        )

    response = db_search(query, query_filter)

    context_point = []
    for point in response.points:
        context_point.append(point.payload['content'])

    context = "\n".join(context_point)

    return {"context": context}


def answer_generation(state: AgentState) -> AgentState:
    question = state["question"]
    tool_used = state["tool_used"]
    context = state["context"]

    SYSTEM_PROMPT = """
    You are an expert financial analyst specializing in SEC filings and corporate finance.
    - For knowledge_base queries: Provide precise answers with exact figures, dates, and citations [Doc Type - Source, Page X].
    - For summarizer queries: Create structured summaries organized by themes, highlighting key metrics and strategic points.
    - For web_search queries: Guide users to appropriate resources and explain available database information.
    """

    user_msg_template = {
        "knowledge_base": f"Question: {question}\n\nDocuments:\n{context}\n\nProvide precise answer with citations. If question is not from the CONTEXT, use search tool, if not say not enough information.",
        "summarizer": f"Question: {question}\n\nSummaries:\n{context}\n\nCreate comprehensive summary.",
        "web_search": f"Question: {question}\n\n{context}."
    }

    HUMAN_PROMPT = user_msg_template.get(tool_used, user_msg_template["knowledge_base"])

    prompt = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": HUMAN_PROMPT}
    ]
    response = llm.invoke(prompt)

    return {"response": response.content}


def route_decision(state: AgentState):
    if state["tool_used"] == "knowledge":
        return "knowledge_base"
    elif state["tool_used"] == "search":
        return "web_search"
    elif state["tool_used"] == "summary":
        return "summarizer"

