# EditorialDesk

**A lightweight RAG-powered knowledge assistant for editorial and research workflows.**

EditorialDesk lets users ask natural-language questions about a collection of Markdown documents. It retrieves relevant sections, uses an OpenAI language model to draft answers grounded in those sections, and displays the supporting evidence. The application also saves each completed question-and-answer exchange in a local text file.

Built with **Python, Streamlit, scikit-learn, and the OpenAI Responses API**.

> [!NOTE]
> This repository is a proof of concept. The three included editorial-policy documents are **fictional examples**, not official policies of VMN Media or any other organization.

## Features

- **Document-based Q&A:** Ask questions in a simple Streamlit chat interface.
- **Local retrieval:** Search Markdown sections with TF-IDF and cosine similarity; no vector database required.
- **Grounded answers:** Pass the top matching sections to an OpenAI model, with instructions to answer only from supplied evidence.
- **Source visibility:** Inspect document names, section headings, and retrieval similarity scores alongside the response.
- **Out-of-scope handling:** Return a fallback response when no document section meets the relevance threshold.
- **No-key mode:** Explore retrieved passages without connecting to the OpenAI API.
- **Q&A history:** Automatically append completed exchanges and source references to `history/questions_and_answers.txt`; download a copy from the sidebar.
- **Automated tests:** Run seven offline unit tests without making paid API calls.

## Demo questions

Try these after launching the app:

| Question | Expected behavior |
| --- | --- |
| Can AI-generated drafts be published without human review? | Finds the AI-assisted drafting policy and explains the review requirement. |
| How should editors correct factual errors? | Retrieves the relevant editorial standards. |
| What should analysts check before releasing a chart? | Searches the research workflow guidance. |
| What is VMN Media's revenue target for 2027? | Reports insufficient information rather than inventing an answer. |

*Answers depend on the documents in `knowledge/`. The example questions above are intended for testing, not as statements about real company policies.*

## Getting started

### Prerequisites

- Python **3.10 or newer**
- An internet connection **only if** you want LLM-generated answers
- An OpenAI API key **only if** you want LLM-generated answers (API billing is separate from ChatGPT subscriptions)

### 1. Get the project

Clone the repository or download it as a ZIP from GitHub, then open a terminal in the project directory (the folder containing `app.py`).

### 2. Install dependencies

**Windows (PowerShell)**

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Using `python -m pip` avoids invoking `pip.exe` directly, which can be blocked by some Windows application-control policies.

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### 3. Configure the API (optional)

Create `.env` from the included `.env.example` file. On Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

On macOS / Linux:

```bash
cp .env.example .env
```

For full AI-generated answers, edit `.env` and add your own key:

```dotenv
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4.1-mini
```

Leave `OPENAI_API_KEY` empty if you only want to inspect search results.

**Never commit your API key.** The `.env` file is included in `.gitignore`. OpenAI API usage may incur charges.

### 4. Run EditorialDesk

**Windows (PowerShell)**

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

**macOS / Linux**

```bash
python -m streamlit run app.py
```

Open **http://localhost:8501** if your browser does not open automatically.

## How it works

```text
Sample Markdown files (knowledge/*.md)
               |
               v
    Split into heading-based sections
               |
               v
       TF-IDF vectorization
               |
User question --> cosine similarity search
               |
               v
    Up to 3 matching sections
    (minimum similarity: 0.10)
               |
        +------+------+
        |             |
     No matches    Matches found
        |             |
        v             v
     Fallback    OpenAI Responses API
                 (if key available)
                      |
                      v
              Answer + source markers
                      |
                      v
             Streamlit chat interface
                      |
                      v
              Local Q&A history file
```

**Retrieval:** `rag.py` reads `.md` files from `knowledge/`, splits them by Markdown headings, and creates TF-IDF vectors. For each question, it compares the question vector with the stored section vectors using cosine similarity. By default it returns up to three sections scoring at least `0.10`.

**Generation:** When an API key is configured and matching sections are available, the app sends those excerpts and the question to the OpenAI Responses API. The model is instructed to stay within the excerpts and use reference markers such as `[1]`. The original retrieved excerpts remain visible in the interface.

**Fallback:** With no relevant matches, the app does not generate an unsupported answer. Without an API key, it displays a retrieval preview instead of an AI-generated response.

**Persistence:** `qa_history.py` appends completed questions, responses, timestamps, response modes, and retrieved source references to a UTF-8 file. The history is stored separately from `knowledge/`, so previous conversations are not automatically treated as source documents.

> Similarity scores are retrieval rankings, **not** measures of answer correctness. Source citations are requested from the model but are not independently verified by this prototype.

## Project structure

```text
editorialdesk-mini-rag/
├── app.py                    # Streamlit chat interface
├── rag.py                    # Document parsing, retrieval, and generation
├── qa_history.py             # Local Q&A log writer
├── knowledge/
│   ├── 01_editorial_standards.md
│   ├── 02_ai_guidelines.md
│   └── 03_research_workflow.md
├── tests/
│   ├── test_rag.py
│   └── test_qa_history.py
├── .env.example              # API configuration template
├── .gitignore                # Excludes secrets and local history
├── requirements.txt
├── PROJECT_EXPLANATION.txt   # Beginner-friendly walkthrough
└── README.md
```

The `history/` directory is created after the first saved exchange. It is not tracked in Git.

## Saving and downloading conversations

Every completed exchange is appended to:

```text
history/questions_and_answers.txt
```

Each record contains the question, answer, time, response mode, and retrieved source names (where available). You can also use **Saved questions and answers → Download Q&A history (.txt)** in the sidebar.

The history stays on your machine across restarts, but it is **not encrypted**. Failed API calls are not archived as successful Q&A. The file is excluded from Git by `.gitignore`.

## Tests

Run all offline tests from the project directory:

```bash
python -m unittest discover -s tests -v
```

On Windows, use the virtual environment interpreter: `.\.venv\Scripts\python.exe -m unittest discover -s tests -v`.

The seven tests cover document loading, retrieval behavior, unsupported questions, API-request construction with a mocked client, and Q&A history formatting/saving. **They do not validate the quality of a live LLM response.**

## Current limitations

- Uses **lexical TF-IDF retrieval**, not semantic embeddings; paraphrased questions may be harder to match.
- The similarity threshold is a heuristic, not a confidence guarantee.
- LLM citations are requested but not independently checked against the retrieved content.
- Uses a small, fixed set of Markdown files; there is no document-upload workflow.
- Has no authentication, access controls, monitoring, or production deployment setup.
- Q&A history is a local, unencrypted text file.
- When generation is enabled, questions and retrieved excerpts are sent to the OpenAI API; **do not use confidential or personal documents in this demo**.

## Possible improvements

- Replace TF-IDF with semantic embeddings and evaluate retrieval quality.
- Add PDF/Word uploads with safe text extraction and document-level permissions.
- Add an evaluation dataset for groundedness, citation accuracy, and unanswered questions.
- Introduce authentication, monitoring, and secure persistence before organizational use.
- Conduct usability testing with editorial and research staff.

## Why I built it

EditorialDesk explores a practical question: **How can AI make internal knowledge easier to access without losing traceability to the original sources?** The prototype combines a search pipeline, an LLM API, and a simple interface to demonstrate an end-to-end retrieval-augmented workflow relevant to editorial and research teams.

---

**Project scope:** Personal demonstration / proof of concept. Not affiliated with or endorsed by VMN Media.
