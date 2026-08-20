"""
format_response.py

Final node in the graph. Takes whatever state the query ended up in
(successful generation + grounded, generation but ungrounded, or abstained)
and produces one consistent, user-facing response shape.
"""



from app.state import GraphState

def _format_grounded_answer(state: GraphState) -> str:
    return state["answer"]


def _format_ungrounded_warning(state: GraphState) -> str:
    return (
        f"{state['answer']}\n\n"
        f" Note: parts of this answer could not be fully verified against "
        f"the source documents. Please confirm with an official Home Credit "
        f"representative before relying on this."
    )


def _format_abstained(state: GraphState) -> str:
    # abstain_node in confident_gate.py already wrote a clear message into state["answer"]
    return state["answer"]


def _select_formatter(state: GraphState):
    return {
        "generated": _format_grounded_answer,
        "abstained_low_confidence": _format_abstained,
        "abstained_ungrounded": _format_ungrounded_warning,
    }[state["path"]]


def format_response(state: GraphState) -> str:
    formatter = _select_formatter(state)
    return formatter(state)


def format_response_node(state: GraphState) -> GraphState:
    return {**state, "answer": format_response(state)}