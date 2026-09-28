# Mini retrieval-augmented generation (RAG) application

A Streamlit application that accepts a PDF or TXT file, indexes its contents locally, and answers questions with retrieved context. It uses LangChain, Chroma, local `all-MiniLM-L6-v2` embeddings, and Google Gemini for generation.

## Architecture and workflow

```text
                         INGESTION
+----------------+     +---------------------+     +------------------+
| PDF or TXT file | --> | PyPDFLoader /       | --> | Recursive splitter|
| uploaded in UI  |     | TextLoader          |     | 800 chars / 150   |
+----------------+     +---------------------+     | char overlap      |
                                                   +--------+---------+
                                                            |
                                                            v
                    +-----------------------------------------------+
                    | Local embeddings: all-MiniLM-L6-v2 (CPU)      |
                    +-------------------------+---------------------+
                                              |
                                              v
                    +-----------------------------------------------+
                    | Chroma local vector store: ./chroma_db        |
                    +-------------------------+---------------------+

                           QUESTION-ANSWERING
+----------------+     +---------------------+     +------------------+
| User question  | --> | Similarity search   | --> | Top 3 document   |
| in Streamlit   |     | against Chroma      |     | chunks           |
+----------------+     +---------------------+     +--------+---------+
                                                            |
                                                            v
                    +-----------------------------------------------+
                    | Strict prompt: answer only from supplied      |
                    | context; otherwise return the refusal text    |
                    +-------------------------+---------------------+
                                              |
                                              v
                    +-----------------------------------------------+
                    | Gemini (`gemini-3.8-flash`, temperature 0.0)  |
                    +-------------------------+---------------------+
                                              |
                                              v
                    +-----------------------------------------------+
                    | Answer plus retrieved chunk text and metadata |
                    +-----------------------------------------------+
```

Each new upload replaces the prior local Chroma collection; it is not a multi-document knowledge base.

## Prerequisites

- Python 3.10 or later (the Docker image uses Python 3.10)
- A Google Gemini API key available as `GEMINI_API_KEY`
- Internet access on first run to download Python packages and the embedding model
- Optional: Docker Desktop and Docker Compose for containerized execution

## Local setup

Run the following commands from the directory containing `app.py` and `requirements.txt`.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Set-Content -Path .env -Value 'GEMINI_API_KEY=your_gemini_api_key'
streamlit run app.py
```

If PowerShell blocks activation, use `Set-ExecutionPolicy -Scope Process Bypass` for the current terminal, then run the activation command again.

### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
printf 'GEMINI_API_KEY=your_gemini_api_key\n' > .env
streamlit run app.py
```

Open the URL printed by Streamlit (normally `http://localhost:8501`), upload one PDF or TXT file, click **Process & Index Document**, then ask a question about that file.

## Docker setup

Create a `.env` file containing `GEMINI_API_KEY=your_gemini_api_key`, then run:

```bash
docker compose up --build
```

Open `http://localhost:8501`. The Compose configuration persists the vector database in `./chroma_db` on the host.

## Example questions and tested outputs

The repository's integration test fixture indexes this text:

```text
John Doe is a Senior Software Engineer with 8 years of experience in Python and AWS.
He graduated from Stanford University in 2016 with a Bachelor's degree in Computer Science.
```

| Question | Actual output validated by the test |
| --- | --- |
| `Where did John Doe graduate from?` | A grounded Gemini answer containing `Stanford`. |
| `What is John Doe's favorite food?` | `I cannot answer this based on the provided documents.` |

Run the same integration checks with:

```bash
python -m pytest -q
```

The second result is an exact-string assertion. The first is intentionally validated as a containment assertion because wording can vary between Gemini responses. Generated answers are not guaranteed to be byte-for-byte identical across service runs, even with temperature set to `0.0`.

You can also try these questions after uploading a document:

- `What are the main responsibilities described in this document?`
- `Which skills or technologies are mentioned?`
- `What information is not available in the document?`

## Known limitations

- The configured Gemini model and API key are external dependencies; an invalid key, unavailable model, quota limit, or network issue prevents answering.
- A new upload deletes the existing `chroma_db` collection, so previously indexed documents are lost from the app's active knowledge base.
- Retrieval uses only vector similarity (top 3 chunks). Exact names, IDs, or rare keywords may not rank as highly as desired.
- PDF extraction quality depends on the source. Scanned/image-only PDFs need OCR before their contents can be retrieved reliably.
- The app displays source filename, page, and chunk content, but does not calculate or display similarity scores.
- The guardrail is prompt-based. It reduces unsupported answers but cannot provide a formal guarantee against hallucinations.
- Tests are live integration tests: they load the embedding model and call Gemini, so they require network access, a valid API key, and may incur API usage.

## Potential improvements

- Preserve and manage multiple documents instead of replacing the collection on each upload.
- Add hybrid retrieval (BM25 keyword search plus vectors) and a reranker for stronger recall and precision.
- Add relevance-score thresholds so low-confidence retrievals return the refusal response before calling the LLM.
- Support OCR for scanned PDFs and additional formats such as DOCX, Markdown, and HTML.
- Add per-answer citations with similarity scores, document names, and page links.
- Replace live integration tests with mocked unit tests for CI, while retaining a separately marked end-to-end test suite.
- Add authentication, upload-size limits, observability, and persistent storage suitable for deployment.
