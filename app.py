
"""
EditorialDesk — Streamlit RAG Application
========================================

A lightweight Retrieval-Augmented Generation (RAG) assistant
for editorial and research teams.

The application allows users to:
    1. Ask questions in natural language.
    2. Search Markdown documents for relevant information.
    3. Generate answers using the OpenAI Responses API.
    4. View retrieved passages and their similarity scores.
    5. Automatically save questions and answers to a text file.
    6. Download their question-and-answer history.

Main technologies:
    - Python
    - Streamlit
    - TF-IDF and cosine similarity
    - OpenAI Responses API
    - python-dotenv

Modules:
    rag.py:
        Handles document loading, retrieval, and generation.

    qa_history.py:
        Handles persistent question-and-answer storage.

    app.py:
        Provides the Streamlit user interface.

Run:
    py -m streamlit run app.py

Note:
    The supplied knowledge documents are fictional examples.
"""

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from qa_history import HISTORY_FILE, save_record
from rag import DOCS_DIR, Retriever, answer, load_passages


# ---------------------------------------------------------
# 1. ENVIRONMENT CONFIGURATION
# ---------------------------------------------------------

# Locate the .env file relative to this Python script.
# This allows the app to run from different working directories.

load_dotenv(
    dotenv_path=Path(__file__).resolve().parent / ".env"
)

# Configure the browser tab before rendering Streamlit UI.

st.set_page_config(
    page_title="EditorialDesk | Mini RAG",
    page_icon="📰"
)


# ---------------------------------------------------------
# 2. APPLICATION HEADER
# ---------------------------------------------------------

st.title("📰 EditorialDesk")

st.caption(
    "Mini RAG assistant for editorial teams "
    "— fictional sample policies"
)


# ---------------------------------------------------------
# 3. DOCUMENT RETRIEVAL INITIALIZATION
# ---------------------------------------------------------

@st.cache_resource
def make_retriever():
    """
    Initialize and cache the document retrieval engine.

    The function performs three operations:

        1. Reads Markdown documents from DOCS_DIR.
        2. Splits the documents into searchable passages.
        3. Creates a Retriever using TF-IDF representations.

    Streamlit caches the resulting object so that it does not
    need to rebuild the retrieval index on every interaction.

    Returns
    -------
    Retriever
        An initialized document retrieval object.

    Notes
    -----
    Cached resources are shared across Streamlit sessions.

    If the knowledge documents change, the cache may need
    to be cleared to rebuild the search index.
    """

    passages = load_passages(DOCS_DIR)

    return Retriever(passages)


# Initialize the retrieval system.

retriever = make_retriever()


# ---------------------------------------------------------
# 4. SIDEBAR: KNOWLEDGE BASE
# ---------------------------------------------------------

with st.sidebar:

    st.subheader("Knowledge base")

    # Display available Markdown documents.

    document_names = sorted(
        p.name for p in DOCS_DIR.glob("*.md")
    )

    for name in document_names:
        st.write(f"📄 {name}")

    # Show the number of searchable passages.

    st.caption(
        f"{len(retriever.passages)} searchable sections"
    )

    st.divider()

    # Check whether an OpenAI API key has been configured.

    if os.getenv("OPENAI_API_KEY"):

        st.success("Generation enabled (OpenAI API)")

    else:

        st.warning(
            "Retrieval preview only: "
            "add OPENAI_API_KEY to .env"
        )

    st.caption(
        "Try: 'How should editors correct factual errors?' "
        "or 'Can AI drafts be published without review?'"
    )

    st.divider()


    # -----------------------------------------------------
    # 5. SIDEBAR: SAVED QUESTION-AND-ANSWER HISTORY
    # -----------------------------------------------------

    st.subheader("Saved questions and answers")

    st.caption(
        "Every completed exchange is saved automatically "
        "to a local text file."
    )

    # HISTORY_FILE is defined in qa_history.py.

    if HISTORY_FILE.is_file():

        try:

            st.download_button(
                label="📥 Download Q&A history (.txt)",
                data=HISTORY_FILE.read_bytes(),
                file_name=(
                    "EditorialDesk_Questions_and_Answers.txt"
                ),
                mime="text/plain",
                use_container_width=True,
            )

        except OSError:

            st.warning(
                "Could not read the saved Q&A document."
            )

    else:

        st.caption(
            "Ask your first question to create the file."
        )

    st.caption(
        "Stored only on this computer, inside the "
        "project's `history/` folder. Keep it private."
    )


# ---------------------------------------------------------
# 6. CHAT SESSION INITIALIZATION
# ---------------------------------------------------------

# Streamlit reruns the script after user interactions.
#
# Session state preserves chat messages across reruns.
#
# Each message is stored as a dictionary containing:
#     role: "user" or "assistant"
#     text: message content
#     sources: optional retrieved document passages

if "messages" not in st.session_state:

    st.session_state.messages = []


# ---------------------------------------------------------
# 7. DISPLAY EXISTING CHAT HISTORY
# ---------------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["text"])

        # Show evidence for messages containing sources.

        if message.get("sources"):

            with st.expander("Retrieved evidence"):

                for i, (passage, score) in enumerate(
                    message["sources"], 1
                ):

                    st.markdown(
                        f"**[{i}] {passage.heading}** — "
                        f"`{passage.source}` "
                        f"(similarity {score:.2f})"
                    )

                    st.write(passage.text)


# ---------------------------------------------------------
# 8. USER QUESTION INPUT
# ---------------------------------------------------------

# st.chat_input creates the text box at the bottom.
#
# The walrus operator (:=) assigns the submitted question
# and checks whether it is non-empty.

if question := st.chat_input(
    "Ask a question about the sample editorial policies..."
):

    # Save the user's message in the current session.

    st.session_state.messages.append({
        "role": "user",
        "text": question
    })

    # Display the user's question immediately.

    with st.chat_message("user"):

        st.markdown(question)


    # -----------------------------------------------------
    # 9. RETRIEVAL STAGE
    # -----------------------------------------------------

    # Search the knowledge base for relevant passages.
    #
    # The retriever uses TF-IDF and cosine similarity
    # to rank document sections against the question.
    #
    # matches contains tuples:
    #
    #     (passage, similarity_score)

    matches = retriever.search(question)

    generation_ok = True


    # -----------------------------------------------------
    # 10. ANSWER GENERATION STAGE
    # -----------------------------------------------------

    with st.chat_message("assistant"):

        if os.getenv("OPENAI_API_KEY"):

            try:

                # The answer function sends the question
                # and retrieved passages to OpenAI.
                #
                # The model generates an answer grounded
                # in the retrieved document context.

                reply = answer(
                    question,
                    matches,
                    os.getenv(
                        "OPENAI_MODEL",
                        "gpt-4.1-mini"
                    )
                )

                # Record which processing mode was used.

                mode = (
                    "AI-generated"
                    if matches
                    else "No relevant documents found"
                )

            except Exception as exc:

                # Handle API failures without crashing
                # the application.

                reply = (
                    "Generation failed. Check your API key, "
                    "account quota, and model setting."
                )

                mode = "API error"

                generation_ok = False

                st.error(f"API error: {exc}")


        elif matches:

            # If no API key exists but relevant passages
            # were found, display retrieval-only results.

            reply = (
                "**Retrieval preview:** I found relevant "
                "sections below. Add an API key to "
                "generate a sourced answer."
            )

            mode = "Retrieval preview (no API key)"


        else:

            # No API key and no relevant document passages.

            reply = (
                "I couldn't find relevant information "
                "in the sample documents."
            )

            mode = "No relevant documents found"


        # Display the final answer or fallback message.

        st.markdown(reply)


        # -------------------------------------------------
        # 11. DISPLAY RETRIEVED EVIDENCE
        # -------------------------------------------------

        if matches:

            with st.expander(
                "Retrieved evidence",
                expanded=True
            ):

                for i, (passage, score) in enumerate(
                    matches, 1
                ):

                    st.markdown(
                        f"**[{i}] {passage.heading}** — "
                        f"`{passage.source}` "
                        f"(similarity {score:.2f})"
                    )

                    st.write(passage.text)


    # -----------------------------------------------------
    # 12. UPDATE SESSION CHAT HISTORY
    # -----------------------------------------------------

    st.session_state.messages.append({
        "role": "assistant",
        "text": reply,
        "sources": matches
    })


    # -----------------------------------------------------
    # 13. SAVE QUESTION AND ANSWER TO FILE
    # -----------------------------------------------------

    # Failed API requests should not be stored as
    # successfully generated answers.

    if generation_ok:

        try:

            save_record(
                question,
                reply,
                matches,
                mode
            )

        except OSError as exc:

            st.warning(
                "The answer appeared, but the history "
                f"file couldn't be saved: {exc}"
            )

        else:

            # Rerun Streamlit to refresh the sidebar.
            #
            # This makes the Q&A download button available
            # immediately after the first saved exchange.

            st.rerun()
