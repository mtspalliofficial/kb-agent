import sys
import time
from pathlib import Path

import streamlit as st

# ============================================================
# PROJECT PATH SETUP
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.agent import run_agent


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Enterprise AI Knowledge Assistant",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Enterprise AI Knowledge Assistant")
st.caption(
    "Ask about enterprise AI risks, policies, or use the built-in "
    "calculator, date lookup, and summarization tools."
)


# ============================================================
# SIDEBAR — EXAMPLE QUESTIONS
# ============================================================

with st.sidebar:
    st.header("Try an example")

    example_questions = [
        "What are the risks of enterprise AI?",
        "What's our policy on data privacy compliance?",
        "What's 15% of 2000?",
        "How many days until 2026-12-31?",
        "Summarise the enterprise AI overview",
        "What's the capital of France?",
    ]

    for question in example_questions:
        if st.button(question, use_container_width=True):
            st.session_state["pending_question"] = question

    st.divider()
    st.caption(
        "Routes: knowledge_qa · policy_analysis · calculator · "
        "date_time · summarize · out_of_scope"
    )


# ============================================================
# CHAT HISTORY (session state)
# ============================================================

if "history" not in st.session_state:
    st.session_state["history"] = []


# ============================================================
# RENDER EXISTING CHAT HISTORY
# ============================================================

for turn in st.session_state["history"]:
    with st.chat_message("user"):
        st.write(turn["question"])

    with st.chat_message("assistant"):
        st.write(turn["answer"])

        if turn["sources"]:
            with st.expander("Sources"):
                for source in turn["sources"]:
                    st.write(f"- {source}")

        with st.expander("Agent execution details"):
            st.write(f"**Route taken:** `{turn['intent']}`")
            st.write(f"**Response time:** {turn['elapsed']:.1f}s")


# ============================================================
# HANDLE NEW INPUT
# ============================================================

question = st.chat_input("Ask a question about enterprise AI...")

# If a sidebar example button was clicked, use that instead
if "pending_question" in st.session_state:
    question = st.session_state.pop("pending_question")

if question:

    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            start = time.time()

            try:
                result = run_agent(question, verbose=False, debug=False)
            except Exception as e:
                result = {
                    "answer": f"Something went wrong while processing that question: {e}",
                    "sources": [],
                    "intent": "error"
                }

            elapsed = time.time() - start

        st.write(result["answer"])

        if result["sources"]:
            with st.expander("Sources"):
                for source in result["sources"]:
                    st.write(f"- {source}")

        with st.expander("Agent execution details"):
            st.write(f"**Route taken:** `{result['intent']}`")
            st.write(f"**Response time:** {elapsed:.1f}s")

    st.session_state["history"].append({
        "question": question,
        "answer": result["answer"],
        "sources": result["sources"],
        "intent": result["intent"],
        "elapsed": elapsed
    })