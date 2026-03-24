# Indecimal ML Intern Assignment - Answer Key

Candidate: `<YOUR_NAME>`  
Institute: NIT Bhopal  
Date: 24 March 2026

## 1) Objective Coverage

Implemented a Mini RAG pipeline for a construction marketplace assistant that:
- Retrieves relevant information from provided internal documents
- Generates responses grounded strictly in retrieved content
- Clearly displays retrieved context and final answer for transparency

## 2) Document Chunking and Vectorization

- Documents are loaded from local `docs/` folder (`.pdf`, `.txt`, `.md`)
- Text is chunked into overlapping segments:
  - Chunk size: ~700 characters
  - Overlap: ~120 characters
- Vector representations are generated using:
  - TF-IDF vectorization (`scikit-learn`)

Rationale:
- This approach is lightweight, stable on Windows, and suitable for short internal document retrieval.

## 3) Vector Indexing and Retrieval

- Local vector matrix is built from TF-IDF features
- Retrieval uses cosine similarity scoring
- For each user query, top-k relevant chunks are retrieved semantically
- No managed vector DB is used (as requested)

## 4) Grounded Answer Generation

- LLM used via OpenRouter (free model by default):
  - `meta-llama/llama-3.1-8b-instruct:free`
- LLM prompt enforces strict grounding:
  - Answer only from retrieved chunks
  - If insufficient context, return:
    - `I don't have enough information in the provided documents.`
  - Avoid unsupported claims/hallucinations

## 5) Transparency and Explainability

The chatbot UI displays:
- Retrieved chunks with:
  - source file
  - chunk id
  - retrieval score
  - chunk text
- Final generated answer separately

This makes answer provenance easy to inspect.

## 6) Deliverables Included

- Custom frontend chatbot interface (HTML/CSS/JS)
- Python implementation (FastAPI backend + RAG pipeline)
- README with setup and implementation details
- Local run instructions

## 7) How to Run

1. Install dependencies: `pip install -r requirements.txt`
2. Add `OPENROUTER_API_KEY` in `.env`
3. Place assignment documents in `docs/`
4. Run: `uvicorn app:app --reload`
5. Open: `http://127.0.0.1:8000`

## 8) Optional Quality Analysis Template

Prepared test-question workflow in README (8-15 questions recommended) to evaluate:
- Retrieval relevance
- Hallucinations/unsupported claims
- Completeness and clarity

## 9) Notes

- Implementation is modular and easy to extend with:
  - local LLM via Ollama (bonus)
  - automated eval scripts
  - reranking for improved retrieval quality
