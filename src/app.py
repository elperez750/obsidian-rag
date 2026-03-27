import sys
import os
import random

sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
from ask_local import build_retriever, answer_with_local_llm
from db import init_db, save_entry, load_history

LOADING_MESSAGES = [
    "Pondering the nature of existence...",
    "Questioning whether the question even has an answer...",
    "Consulting the cave wall shadows...",
    "Weighing the universe on a set of rusty scales...",
    "Asking what it means to truly know anything...",
    "Traversing the allegory of the cave...",
    "Contemplating the void...",
    "Reconciling free will with determinism...",
    "Seeking the form of the Good...",
    "Wondering if this answer was always inevitable...",
    "Staring into the abyss (it is staring back)...",
    "Unraveling the ship of Theseus...",
]

st.set_page_config(page_title="Obsidian RAG", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=EB+Garamond:ital,wght@0,400;0,500;1,400&display=swap');

html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    background-color: #F4ECD8;
}

[data-testid="stAppViewContainer"] {
    background-color: #F4ECD8;
}

[data-testid="stMain"] {
    background-color: #F4ECD8;
}

section[data-testid="stSidebar"] {
    background-color: #EDE3C8;
}

* {
    font-family: 'EB Garamond', Georgia, serif !important;
    color: #1C1C1C;
}

h1 {
    font-size: 2.4rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.04em;
    border-bottom: 2px solid #1C1C1C;
    padding-bottom: 0.4em;
    margin-bottom: 1.2em;
}

/* Chat messages */
[data-testid="stChatMessage"] {
    background-color: transparent !important;
    border: none !important;
    padding: 0.2em 0 !important;
}

[data-testid="stChatMessageContent"] p {
    font-size: 1.1rem !important;
    line-height: 1.85 !important;
}

/* User bubble */
[data-testid="stChatMessage"][data-testid*="user"] {
    background-color: transparent !important;
}

.user-bubble {
    background-color: #E8DCBF;
    border-left: 3px solid #5C4A2A;
    padding: 0.6em 1em;
    margin: 0.8em 0 0.2em 0;
    font-size: 1.1rem;
    line-height: 1.7;
}

.answer-bubble {
    padding: 0.4em 0 1.2em 0;
    font-size: 1.1rem;
    line-height: 1.85;
    border-bottom: 1px solid #C8B99A;
    margin-bottom: 0.4em;
}

/* Chat input bar — bottom container */
[data-testid="stBottom"],
[data-testid="stBottom"] > div {
    background-color: #F4ECD8 !important;
    border-top: none !important;
}

[data-testid="stChatInput"] {
    background-color: #ffffff !important;
    border: 1px solid #cccccc !important;
    border-radius: 8px !important;
    box-shadow: 0 0 12px rgba(0, 0, 0, 0.12) !important;
}

[data-testid="stChatInput"] textarea {
    background-color: #ffffff !important;
    font-size: 1.05rem !important;
    color: #1C1C1C !important;
}

[data-testid="stChatInput"] textarea:focus {
    box-shadow: none !important;
}

[data-testid="stChatInputSubmitButton"] button {
    background-color: #1C1C1C !important;
    border: none !important;
    transition: none !important;
}

[data-testid="stChatInputSubmitButton"] button:hover,
[data-testid="stChatInputSubmitButton"] button:focus {
    background-color: #1C1C1C !important;
    border: none !important;
    box-shadow: none !important;
}

[data-testid="stSpinner"] p {
    font-size: 1rem !important;
    font-style: italic;
    color: #5C4A2A !important;
}

[data-testid="stDecoration"] {
    display: none;
}

header[data-testid="stHeader"] {
    background-color: #F4ECD8 !important;
}

/* Sidebar */
.question-entry {
    padding: 0.5em 0;
    border-bottom: 1px solid #C8B99A;
    font-size: 0.95rem;
    font-style: italic;
    line-height: 1.4;
    color: #3B2F1E;
}

.sidebar-label {
    font-size: 0.75rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #9C8B6E !important;
    margin-bottom: 0.8em;
}

/* Welcome message */
.welcome-wrap {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 5rem 2rem 3rem 2rem;
    text-align: center;
}

.welcome-greeting {
    font-size: 2rem;
    font-weight: 500;
    letter-spacing: 0.02em;
    margin-bottom: 0.6em;
    color: #1C1C1C;
}

.welcome-sub {
    font-size: 1.15rem;
    font-style: italic;
    color: #5C4A2A;
    max-width: 480px;
    line-height: 1.8;
}
</style>
""", unsafe_allow_html=True)

# --- Session state init ---
init_db()

if "retriever" not in st.session_state:
    with st.spinner("Awakening the oracle..."):
        st.session_state.retriever = build_retriever()

if "history" not in st.session_state:
    st.session_state.history = load_history()

# --- Sidebar ---
with st.sidebar:
    st.markdown('<p class="sidebar-label">Past Questions</p>', unsafe_allow_html=True)
    if st.session_state.history:
        for entry in reversed(st.session_state.history):
            st.markdown(f'<div class="question-entry">{entry["question"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<p style="font-style: italic; color: #9C8B6E; font-size: 0.95rem;">Nothing yet. Ask something.</p>', unsafe_allow_html=True)

# --- Main ---
st.title("Obsidian RAG")

# Welcome screen — shown only when no conversation yet
if not st.session_state.history:
    st.markdown("""
    <div class="welcome-wrap">
        <div class="welcome-greeting">Welcome back, Elliott.</div>
        <div class="welcome-sub">
            "The unexamined life is not worth living." — Socrates<br><br>
            Your notes await. What would you like to explore?
        </div>
    </div>
    """, unsafe_allow_html=True)

# Render conversation history
for entry in st.session_state.history:
    st.markdown(f'<div class="user-bubble">{entry["question"]}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="answer-bubble">{entry["answer"]}</div>', unsafe_allow_html=True)

# Chat input — sticks to bottom
question = st.chat_input("Ask a question about your notes...")

if question and question.strip():
    st.markdown(f'<div class="user-bubble">{question.strip()}</div>', unsafe_allow_html=True)
    with st.spinner(random.choice(LOADING_MESSAGES)):
        answer = answer_with_local_llm(question, st.session_state.retriever)
    save_entry(question.strip(), answer)
    st.session_state.history.append({"question": question.strip(), "answer": answer})
    st.markdown(f'<div class="answer-bubble">{answer}</div>', unsafe_allow_html=True)
