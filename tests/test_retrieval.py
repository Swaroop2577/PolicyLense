from app.nodes.retrieve_direct import retrieve_direct, retrieve_direct_node

import time

def run_tests():
    """
    Executes a series of test queries against the hybrid retrieval pipeline
    to verify BM25 and Dense integration, as well as LangGraph node functionality.
    """
    test_queries = [
        "What is the minimum and maximum number of directors on the Board?",
        "How frequently must the Audit Committee meet and what is its quorum?",
        "What are the key responsibilities of the Chief Risk Officer?"
    ]

    print("=" * 70)
    print("🚀 STARTING HYBRID RETRIEVAL PIPELINE TEST")
    print("=" * 70)

    for idx, query in enumerate(test_queries, 1):
        print(f"\n[Query {idx}]: '{query}'")
        start_time = time.time()
        
        # Mocking the GraphState expected by your LangGraph node
        initial_state = {
            "query": query, 
            "retrieved_chunks": []
        }
        
        # Execute the LangGraph node
        final_state = retrieve_direct_node(initial_state)
        
        # Extract the results from the updated state
        chunks = final_state.get("retrieved_chunks", [])
        
        elapsed_time = time.time() - start_time
        
        print(f"✅ Retrieval took {elapsed_time:.3f} seconds.")
        print(f"📄 Retrieved {len(chunks)} unique chunks via Ensemble/RRF.")
        
        # Display the top 2 results to verify semantic + keyword relevance
        for i, doc in enumerate(chunks[:2], 1):
            source = doc.metadata.get("source", "Unknown file")
            page = doc.metadata.get("page", "Unknown page")
            
            print(f"\n   --- Result {i} (Source: {source} | Page: {page}) ---")
            
            # Clean up newlines for a neater terminal output
            content_preview = doc.page_content[:300].replace('\n', ' ')
            print(f"   {content_preview}...\n")

if __name__ == "__main__":
    run_tests()