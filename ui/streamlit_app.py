"""
streamlit_app.py

Chat UI for the RAG pipeline. Imports the compiled graph via main.ask()
and adds timing instrumentation per query, since we're about to measure
latency before optimizing (caching, etc).
"""

import sys
from pathlib import Path

# Ensure project root (RAG-pipeline/) is in Python's search path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))



import time
import streamlit as st

from main import ask

st.set_page_config(page_title="Home Credit Policy Assistant", page_icon="📋")
st.title("📋 Home Credit Policy Assistant")
st.caption("Ask questions about grievance redressal, fair practices, and company policies.")


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "elapsed" in message:
            st.caption(f"⏱️ {message['elapsed']:.2f}s")




if user_query := st.chat_input("Ask about Home Credit policies..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            start = time.perf_counter()
            answer = ask(user_query)
            elapsed = time.perf_counter() - start

        st.markdown(answer)
        st.caption(f"⏱️ {elapsed:.2f}s")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "elapsed": elapsed,
    })            




