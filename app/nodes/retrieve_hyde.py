# This next file is where a conceptual query (like your attention-mechanisms example above) 
# gets turned into a fabricated "hypothetical answer" passage — which then gets embedded 
# instead of the raw question. The intuition: a short question like "explain attention mechanisms"
#  and a real answer chunk from a paper don't sit close together in embedding space, but two answer-shaped passages do. 
# So we fake an answer first, embed that, and search with it.


from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

from app.config import GROQ_API_KEY, HYDE_MODEL, HYDE_TEMPERATURE
from app.state import GraphState



_prompt = ChatPromptTemplate.from_messages([
    ("system", """Write a short, confident passage (3-5 sentences) that plausibly
answers the user's question, as if it were an excerpt from an academic paper.
It does not need to be factually correct - it only needs to read like real
academic prose covering the right concepts. Do not mention that this is
hypothetical. Do not add a preamble."""),
    ("human", "{query}"),
])

_llm = ChatGroq(
    model=HYDE_MODEL,
    temperature=HYDE_TEMPERATURE,
    api_key=GROQ_API_KEY,
)

_hyde_chain = _prompt | _llm | StrOutputParser()


def generate_hyde_passage(query: str) -> str:
    """
    Takes a conceptual query, returns a fabricated passage for embedding.
    Pure function - no state dict, no graph involved.
    """
    return _hyde_chain.invoke({"query": query})


def retrieve_hyde_node(state: GraphState) -> GraphState:
    """
    LangGraph node. Only reached when query_type == 'conceptual' (routing
    logic for this lives in app/graph.py, built later).
    Reads state['query'], writes state['hyde_passage'].
    """
    passage = generate_hyde_passage(state["query"])
    print(f"[retrieve_hyde_node] generated passage ({len(passage)} chars)")
    return {**state, "hyde_passage": passage}