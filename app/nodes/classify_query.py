# The -> QueryClassification tells anyone reading the code—as well as your code editor and type-checking tools—that 
# when this function finishes running, it will output an object of the QueryClassification 
# type (the Pydantic model you defined earlier).


from langchain_groq import ChatGroq

from langchain_core.prompts import ChatPromptTemplate

from app.config import GROQ_API_KEY, CLASSIFIER_MODEL, CLASSIFIER_TEMPERATURE
from app.schemas import QueryClassification

from app.state import GraphState


_llm = ChatGroq(
    model=CLASSIFIER_MODEL,
    temperature=CLASSIFIER_TEMPERATURE,
    api_key=GROQ_API_KEY,
)

_structured_llm = _llm.with_structured_output(QueryClassification)

_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a query routing assistant for a RAG system over academic PDFs.
Classify the user query into exactly one of:

- specific_ref  : references a named entity - a section, figure, table, equation, author, or citation
- conceptual    : asks to explain, summarize, or compare an idea or concept
- multi_hop     : the full answer requires connecting information from multiple parts of the document

Be decisive. Do not hedge."""),
    ("human", "{query}"),
])

_classifier_chain = _prompt | _structured_llm


def classify_query(query: str) -> QueryClassification:
    """
    Takes a raw query string, returns a validated QueryClassification object.
    """
    result: QueryClassification = _classifier_chain.invoke({"query": query})
    return result


def route_by_query_type(state: GraphState) -> str:
    """
    Conditional-edge function. Reads query_type, returns the name of
    the next node to run.
    """
    return {
        "specific_ref": "retrieve_direct",
        "conceptual": "retrieve_hyde",
        "multi_hop": "retrieve_multihop",
    }[state["query_type"]]



def classify_query_node(state: GraphState) -> GraphState:
    """
    LangGraph node. Reads state['query'], writes state['query_type'].
    """
    result = classify_query(state["query"])
    print(f"[classify_query_node] type={result.query_type} | {result.reasoning}")
    return {**state, "query_type": result.query_type}
# Spreading **state first means "keep everything that was already there, and update this one field."