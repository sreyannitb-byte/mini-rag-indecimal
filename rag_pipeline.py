import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import requests
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer


SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


@dataclass
class RetrievedChunk:
    source: str
    chunk_id: int
    text: str
    score: float


class MiniRAG:
    def __init__(
        self,
        docs_dir: str = "docs",
        embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        llm_model: str = "meta-llama/llama-3.1-8b-instruct:free",
    ) -> None:
        self.docs_dir = Path(docs_dir)
        self.embed_model_name = embed_model_name
        self.llm_model = llm_model
        # CPU-safe fallback embedding/retrieval for Windows machines where torch DLL may fail.
        self.vectorizer: TfidfVectorizer | None = None
        self.doc_vectors = None
        self.chunks: List[Dict] = []

    def build_index(self) -> int:
        documents = self._load_documents()
        self.chunks = self._chunk_documents(documents)
        if not self.chunks:
            self.vectorizer = None
            self.doc_vectors = None
            return 0

        texts = [c["text"] for c in self.chunks]
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=20000)
        self.doc_vectors = self.vectorizer.fit_transform(texts)
        return len(self.chunks)

    def answer(self, query: str, top_k: int = 4) -> Tuple[List[RetrievedChunk], str]:
        if self.vectorizer is None or self.doc_vectors is None or not self.chunks:
            raise ValueError("Index is empty. Add docs and rebuild index first.")

        retrieved = self.retrieve(query=query, top_k=top_k)
        answer = self._generate_grounded_answer(query=query, retrieved_chunks=retrieved)
        return retrieved, answer

    def retrieve(self, query: str, top_k: int = 4) -> List[RetrievedChunk]:
        if self.vectorizer is None or self.doc_vectors is None:
            raise ValueError("Index is empty. Build index first.")

        query_vec = self.vectorizer.transform([query])
        # Cosine similarity for sparse TF-IDF vectors.
        scores = (self.doc_vectors @ query_vec.T).toarray().ravel()
        if scores.size == 0:
            return []
        top_indices = np.argsort(scores)[::-1][:top_k]

        retrieved: List[RetrievedChunk] = []
        for idx in top_indices:
            score = float(scores[idx])
            if idx < 0:
                continue
            chunk = self.chunks[idx]
            retrieved.append(
                RetrievedChunk(
                    source=chunk["source"],
                    chunk_id=chunk["chunk_id"],
                    text=chunk["text"],
                    score=float(score),
                )
            )
        return retrieved

    def _load_documents(self) -> List[Tuple[str, str]]:
        if not self.docs_dir.exists():
            return []

        docs: List[Tuple[str, str]] = []
        for path in sorted(self.docs_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            text = self._read_file(path)
            if text.strip():
                docs.append((str(path.relative_to(self.docs_dir)), text))
        return docs

    def _read_file(self, path: Path) -> str:
        if path.suffix.lower() in {".txt", ".md"}:
            return path.read_text(encoding="utf-8", errors="ignore")
        if path.suffix.lower() == ".pdf":
            reader = PdfReader(str(path))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        return ""

    def _chunk_documents(self, documents: List[Tuple[str, str]], chunk_size: int = 700, overlap: int = 120) -> List[Dict]:
        chunks: List[Dict] = []
        for source, text in documents:
            clean_text = " ".join(text.split())
            if not clean_text:
                continue
            start = 0
            chunk_id = 0
            while start < len(clean_text):
                end = min(start + chunk_size, len(clean_text))
                snippet = clean_text[start:end].strip()
                if snippet:
                    chunks.append(
                        {
                            "source": source,
                            "chunk_id": chunk_id,
                            "text": snippet,
                        }
                    )
                    chunk_id += 1
                if end == len(clean_text):
                    break
                start = max(end - overlap, start + 1)
        return chunks

    def _generate_grounded_answer(self, query: str, retrieved_chunks: List[RetrievedChunk]) -> str:
        openrouter_api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        if not openrouter_api_key:
            return (
                "OPENROUTER_API_KEY is not set. Retrieval worked, but answer generation is disabled.\n"
                "Please set the key in your environment or .env file."
            )

        context_blocks = []
        for c in retrieved_chunks:
            context_blocks.append(
                f"[SOURCE: {c.source} | CHUNK: {c.chunk_id} | SCORE: {c.score:.4f}]\n{c.text}"
            )
        context_text = "\n\n".join(context_blocks)

        system_prompt = (
            "You are a strict RAG assistant. Answer only using the provided context.\n"
            "Rules:\n"
            "1) If context is insufficient, say exactly: 'I don't have enough information in the provided documents.'\n"
            "2) Do not use prior/world knowledge.\n"
            "3) Keep answer concise and factual.\n"
            "4) When possible, mention source names used."
        )
        user_prompt = f"Question:\n{query}\n\nContext:\n{context_text}"

        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {openrouter_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.llm_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.1,
            },
            timeout=45,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()
