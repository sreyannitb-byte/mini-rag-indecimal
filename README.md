# Mini RAG Assignment - Indecimal (Round 1)

This project implements a **Retrieval-Augmented Generation (RAG)** chatbot for a construction marketplace use case.

It satisfies the mandatory requirements:
- Document chunking and vectorization
- Local vector search using cosine similarity
- LLM answer generation constrained to retrieved context
- Transparent output showing both retrieved chunks and final answer
- Custom frontend chatbot interface

## Tech Stack

- **Backend:** FastAPI
- **Embedding approach:** TF-IDF (`scikit-learn`)
- **Vector retrieval:** cosine similarity over local vectors
- **LLM provider:** OpenRouter (free model by default)
- **Frontend:** Custom HTML/CSS/JavaScript

## Project Structure

```
mini-rag-indecimal/
  app.py
  rag_pipeline.py
  requirements.txt
  .env.example
  docs/                 # put provided assignment docs here
  templates/index.html
  static/style.css
  static/app.js
```

## Setup

1. Create and activate virtual environment:
   - Windows PowerShell:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

2. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```

3. Configure environment:
   - Copy `.env.example` to `.env`
   - Add your `OPENROUTER_API_KEY`

4. Add assignment documents:
   - Put all provided files (PDF/TXT/MD) inside `docs/`

5. Run server:
   ```powershell
   uvicorn app:app --reload
   ```

6. Open in browser:
   - `http://127.0.0.1:8000`

## How Grounding Is Enforced

- Retrieval gets top-k chunks using cosine similarity over local vectors.
- The prompt explicitly instructs the LLM to answer **only from retrieved chunks**.
- If information is missing, the model is instructed to return:
  - `I don't have enough information in the provided documents.`
- UI always displays:
  - Retrieved document chunks (source, chunk id, score, text)
  - Final generated answer

## Suggested Evaluation Questions (8-15)

Create question sets from your provided docs, for example:
- What factors affect construction project delays?
- What is the refund policy timeline?
- Which quality checks are mandatory before handover?
- What are client responsibilities in procurement?
- What are penalties for delayed vendor delivery?
- What safety compliance standards are listed?
- What documentation is needed for milestone approval?
- What happens when change requests are raised late?

Track:
- Relevance of retrieved chunks
- Hallucination presence
- Completeness and clarity of answers

## Deployment Options

- Render / Railway / Hugging Face Spaces (Docker)
- Ensure `OPENROUTER_API_KEY` is set as environment variable in deployment

## Notes

- If `OPENROUTER_API_KEY` is missing, retrieval still runs but answer generation is disabled with a clear message.
- Bonus extension: upgrade to sentence-transformers + FAISS on a machine with stable torch support.
