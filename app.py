import os
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
import requests

from rag_pipeline import MiniRAG


load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "docs"

app = FastAPI(title="Mini RAG - Indecimal Assignment")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

rag = MiniRAG(
    docs_dir=str(DOCS_DIR),
    embed_model_name=os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
    llm_model=os.getenv("LLM_MODEL", "meta-llama/llama-3.1-8b-instruct:free"),
)


class QueryPayload(BaseModel):
    query: str = Field(..., min_length=3, description="User question")
    top_k: int = Field(default=4, ge=1, le=10)


@app.on_event("startup")
def startup_event() -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    rag.build_index()


@app.get("/", response_class=HTMLResponse)
def home(request: Request) -> Any:
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"docs_dir": str(DOCS_DIR)},
    )


@app.get("/api/status")
def status() -> Dict[str, Any]:
    indexed_chunks = len(rag.chunks)
    indexed_sources = len({c["source"] for c in rag.chunks})
    return {
        "indexed_chunks": indexed_chunks,
        "indexed_sources": indexed_sources,
        "docs_dir": str(DOCS_DIR),
    }


@app.post("/api/reindex")
def reindex() -> Dict[str, Any]:
    count = rag.build_index()
    return {"indexed_chunks": count}


@app.post("/api/ask")
def ask(payload: QueryPayload) -> JSONResponse:
    if not rag.chunks:
        raise HTTPException(
            status_code=400,
            detail="No documents indexed. Add files to docs/ and click Reindex.",
        )
    try:
        retrieved, answer = rag.answer(query=payload.query, top_k=payload.top_k)
    except requests.exceptions.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"LLM request failed: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to answer query: {exc}") from exc

    return JSONResponse(
        {
            "query": payload.query,
            "retrieved_context": [
                {
                    "source": r.source,
                    "chunk_id": r.chunk_id,
                    "score": round(r.score, 4),
                    "text": r.text,
                }
                for r in retrieved
            ],
            "answer": answer,
        }
    )
