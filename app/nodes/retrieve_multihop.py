# Why import retrieve_direct here? Once a multi-hop question is broken into sub-questions,
#  each sub-question is really just a specific_ref-style lookup. 
# No reason to build a second hybrid-retrieval function



"""
retrieve_multihop.py

Retrieval for multi_hop queries - questions that require connecting
information across multiple sections/parts of a document (e.g. "how does
the KYC policy align with the grievance redressal process?"). A single
retrieval call over the raw query usually only surfaces chunks matching
one side of the question, so we decompose into sub-queries first.
"""

from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.runnables import RunnableParallel

from app.config import GROQ_API_KEY, CLASSIFIER_MODEL, CLASSIFIER_TEMPERATURE
from app.state import GraphState
from app.nodes.retrieve_direct import retrieve_direct


from app.nodes.rerank import rerank

class SubQueries(BaseModel):
    sub_queries: list[str] = Field(
        description="2-4 independent, self-contained questions that together answer the original query"

    )



_llm = ChatGroq(
    model=CLASSIFIER_MODEL,
    temperature=CLASSIFIER_TEMPERATURE,
    api_key=GROQ_API_KEY,
).with_structured_output(SubQueries)    #This is called structured output parsing....when calling an LLM we can pass a 
                            # pydantic class and the output structure from LLM will be same as our passed pydantic class
                            # Normally we get an object with content and meta data... we do result.content to get our answer
                            # Here we can directly use all the methods in our passed pydantic class on the output.... 

_decompose_prompt = ChatPromptTemplate.from_messages([
    ("system", """Break the user's question into 2-4 independent sub-questions
that, together, cover everything needed to answer it fully. Each sub-question
must be self-contained and answerable on its own from a policy document."""),
    ("human", "{query}"),
])

_decompose_chain = _decompose_prompt | _llm   



def retrieve_multihop(query: str) -> list[Document]:
    sub_queries = _decompose_chain.invoke({"query": query}).sub_queries

    parallel_retrieval = RunnableParallel(
        **{f"sq_{i}": (lambda _, q=sq: rerank(q, retrieve_direct(q)))
           for i, sq in enumerate(sub_queries)}
    )
    results_by_subquery = parallel_retrieval.invoke({})

    chunks = [doc for docs in results_by_subquery.values() for doc in docs]
    for chunk in chunks:
        chunk.metadata["reranked_for_multihop"] = True  # explicit, unambiguous marker

    return chunks

# ---->----------------------RunnableParallel------------------
# Dictionary Comprehension: {key: value for item in list}. It loops through the sub_queries using enumerate 
# (which gives you the index i and the string sq). It creates dictionary keys like "sq_0", "sq_1", etc.

# Lambda Closure Binding (lambda q=sq:): A lambda is an anonymous, one-line function. 
# By writing lambda q=sq: retrieve_direct(q), you are telling Python: "Create a function that runs retrieve_direct. 
# Take the current value of sq in this loop iteration and lock it into the variable q." 
# If you just wrote lambda: retrieve_direct(sq), Python's "late binding" would cause every single lambda 
# to search for whatever the last sub-query in the loop was. q=sq prevents that bug.

# Kwargs Unpacking (**): RunnableParallel expects named arguments, like RunnableParallel(sq_0=func1, sq_1=func2). 
# The double asterisk (**) takes the dictionary you just built and unpacks it into those named arguments automatically.


# ----->------------------Return Statment---------------------

# The Syntax: This is a Nested List Comprehension. It reads from left to right, outside-in.

# results_by_subquery.values() grabs just the lists of documents (ignoring the "sq_0" keys).

# for chunks in ... loops over those lists.

# for doc in chunks loops over the individual documents inside each list.

# [doc ...] outputs them all into a new, flat list: [Doc1, Doc2, Doc3, Doc4].


def retrieve_multihop_node(state: GraphState) -> GraphState:
    return {**state, "retrieved_chunks": retrieve_multihop(state["query"])}