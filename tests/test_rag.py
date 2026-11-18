
"""
EditorialDesk — RAG Unit Tests
==============================

This module contains automated tests for the EditorialDesk
Retrieval-Augmented Generation (RAG) application.

The tests verify that the application can:

    1. Load documents from the knowledge base.
    2. Retrieve passages relevant to editorial questions.
    3. Rank the correct passages above less relevant ones.
    4. Handle questions outside the knowledge base.
    5. Pass retrieved evidence to an OpenAI language model.

Testing Framework
-----------------
unittest:
    Python's built-in framework for automated testing.

unittest.mock:
    Provides simulated objects that replace external services
    during testing.

Dependencies
------------
rag.py:
    Contains the document loading, retrieval, and
    answer generation functionality being tested.

Running the Tests
-----------------
From the project directory, execute:

    py -m unittest discover -s tests -v

Important
---------
The OpenAI API is mocked during testing.

This means the tests do not make real API requests,
consume API credits, or verify the quality of actual
AI-generated responses.
"""

import os
import sys
import types
import unittest

from unittest.mock import MagicMock, patch

from rag import (
    DOCS_DIR,
    FALLBACK,
    Retriever,
    answer,
    load_passages,
)


class TestMiniRag(unittest.TestCase):
    """
    Test the core retrieval and generation functionality.

    This test class checks the main components of the
    EditorialDesk RAG pipeline.

    Each test method verifies one expected behavior.

    The shared Retriever is initialized once before
    the tests run to avoid loading and indexing the
    same documents repeatedly.
    """

    # -----------------------------------------------------
    # 1. TEST SETUP
    # -----------------------------------------------------

    @classmethod
    def setUpClass(cls):
        """
        Initialize the retrieval engine for all tests.

        This method runs once before the test methods
        in this class are executed.

        Steps
        -----
        1. Read documents from DOCS_DIR.
        2. Split the documents into searchable passages.
        3. Create a Retriever instance.
        4. Store the Retriever as a class attribute.

        Notes
        -----
        Sharing the Retriever makes the test suite
        more efficient.

        The tests assume that the Retriever does not
        modify its internal state during searches.
        """

        cls.retriever = Retriever(
            load_passages(DOCS_DIR)
        )

    # -----------------------------------------------------
    # 2. TEST DOCUMENT LOADING
    # -----------------------------------------------------

    def test_load_documents(self):
        """
        Verify that documents are loaded successfully.

        The knowledge base contains Markdown documents
        divided into searchable sections.

        Expected Result
        ---------------
        The Retriever should contain at least six
        searchable passages.

        This test helps detect:
            - Missing knowledge documents.
            - Problems reading Markdown files.
            - Errors in passage extraction.
        """

        self.assertGreaterEqual(
            len(self.retriever.passages),
            6
        )

    # -----------------------------------------------------
    # 3. TEST EDITORIAL CORRECTIONS RETRIEVAL
    # -----------------------------------------------------

    def test_corrections(self):
        """
        Verify retrieval for a factual-corrections question.

        Question
        --------
        How should editors correct factual errors?

        Expected Behavior
        -----------------
        The Retriever should identify the section
        titled 'Corrections' as the most relevant.

        This checks whether TF-IDF and cosine
        similarity produce the expected ranking
        for this particular question.
        """

        question = (
            "How should editors correct factual errors?"
        )

        # Search the document collection.

        matches = self.retriever.search(question)

        # Confirm that at least one passage was found.

        self.assertTrue(matches)

        # Verify that the highest-ranked passage
        # has the expected heading.

        self.assertEqual(
            matches[0][0].heading,
            "Corrections"
        )

    # -----------------------------------------------------
    # 4. TEST AI GUIDELINES RETRIEVAL
    # -----------------------------------------------------

    def test_ai_drafts(self):
        """
        Verify retrieval of AI-assisted drafting guidelines.

        Question
        --------
        Can AI drafts be published without editorial review?

        Expected Behavior
        -----------------
        The section titled 'AI-assisted drafting'
        should be ranked first.

        This verifies that the retrieval system can
        identify the relevant policy when asked about
        AI-generated editorial content.
        """

        question = (
            "Can AI drafts be published "
            "without editorial review?"
        )

        matches = self.retriever.search(question)

        # A relevant document passage must exist.

        self.assertTrue(matches)

        # The best match must be the AI drafting section.

        self.assertEqual(
            matches[0][0].heading,
            "AI-assisted drafting"
        )

    # -----------------------------------------------------
    # 5. TEST OUT-OF-DOMAIN QUESTION HANDLING
    # -----------------------------------------------------

    def test_unknown_topic(self):
        """
        Verify fallback handling for unsupported questions.

        Question
        --------
        What is the lunar gravity on Europa?

        The knowledge base contains editorial guidelines
        rather than astronomical information.

        Expected Behavior
        -----------------
        1. No passages should pass the relevance threshold.
        2. The answer function should return FALLBACK.

        Purpose
        -------
        This test checks whether the system avoids
        generating an answer when retrieval provides
        no sufficiently relevant information.

        Limitation
        ----------
        Passing this test does not guarantee that
        the application will avoid all hallucinations.
        """

        question = (
            "What is the lunar gravity on Europa?"
        )

        matches = self.retriever.search(question)

        # The system should find no sufficiently
        # relevant passages.

        self.assertEqual(matches, [])

        # With no matches, the answer function
        # should return its predefined fallback.

        self.assertEqual(
            answer(question, matches),
            FALLBACK
        )

    # -----------------------------------------------------
    # 6. TEST LLM INPUT USING A MOCKED OPENAI API
    # -----------------------------------------------------

    def test_generation_receives_retrieved_sources(self):
        """
        Verify that retrieved passages reach the LLM.

        This test simulates an OpenAI API response
        without contacting the real OpenAI service.

        Test Process
        ------------
        1. Retrieve relevant editorial passages.
        2. Create a simulated OpenAI client.
        3. Configure a predefined model response.
        4. Temporarily supply a fake API key.
        5. Replace the OpenAI module with a mock.
        6. Call the answer-generation function.
        7. Inspect the generated API request.

        Expected Behavior
        -----------------
        - The function returns the mocked answer.
        - The answer contains citation marker [1].
        - The API input includes the retrieved heading.
        - The API input contains citation marker [1].

        Important
        ---------
        This verifies API request construction
        and response handling, not whether a real
        language model generates correct answers.
        """

        # Step 1:
        # Retrieve relevant passages from the documents.

        matches = self.retriever.search(
            "Can AI drafts be published "
            "without editorial review?"
        )

        # Step 2:
        # Create a mock OpenAI client.
        #
        # MagicMock creates objects that imitate
        # the behavior of real Python objects.

        fake_client = MagicMock()

        # Step 3:
        # Define the response the simulated API
        # should return.

        fake_client.return_value.responses.create.return_value.output_text = (
            "A human editor must review AI-generated text [1]."
        )

        # Step 4:
        # Create a temporary fake OpenAI module.

        fake_openai = types.ModuleType("openai")

        fake_openai.OpenAI = fake_client

        # Step 5:
        # Temporarily provide an API key and replace
        # the real OpenAI module.
        #
        # patch.dict restores the original values
        # when its context block finishes.

        with patch.dict(
            os.environ,
            {"OPENAI_API_KEY": "test-api-key"}
        ):

            with patch.dict(
                sys.modules,
                {"openai": fake_openai}
            ):

                # Step 6:
                # Execute the answer function.
                #
                # The function uses the mocked client
                # instead of making a real API call.

                text = answer(
                    "Can AI drafts be published without review?",
                    matches
                )

        # Step 7:
        # Check whether the returned response
        # contains the expected citation marker.

        self.assertIn("[1]", text)

        # Step 8:
        # Inspect the arguments supplied to the
        # simulated OpenAI Responses API.

        sent = (
            fake_client
            .return_value
            .responses
            .create
            .call_args
            .kwargs["input"]
        )

        # The API input should contain the heading
        # of the relevant retrieved document.

        self.assertIn(
            "AI-assisted drafting",
            sent
        )

        # The API input should also contain the
        # citation marker associated with that passage.

        self.assertIn("[1]", sent)


# ---------------------------------------------------------
# 7. TEST EXECUTION ENTRY POINT
# ---------------------------------------------------------

if __name__ == "__main__":
    """
    Run all tests when this file is executed directly.

    unittest.main() discovers and executes the test
    methods defined in the TestMiniRag class.
    """

    unittest.main()
