import weave

from src.client import weave_observer
from src.graph import graph, get_weave_config


@weave.op()
def get_response(query: str) -> str:
    result = graph.invoke({"question": query})

    current_call = weave.require_current_call()
    call_id = current_call.id
    call = weave_observer.get_call(call_id)

    call.feedback.add_note(f"Routed to: {result['tool_used']}")

    return result['response']


def run_query(query: str, use_tracing: bool = True) -> dict:
    if use_tracing:
        config = get_weave_config()
        result = graph.invoke({"question": query}, config=config)
    else:
        result = graph.invoke({"question": query})

    return result


if __name__ == "__main__":
    test_query = "Based on Netflix's most recent 10-K filing, what were the key drivers of subscriber growth on global region"
    result = run_query(test_query)
    print(f"Tool Used: {result['tool_used']}")
    print(f"Response: {result['response']}")

