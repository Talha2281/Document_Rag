from pathlib import Path
import logging
import time
import os
from dotenv import load_dotenv
load_dotenv(override=True)

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import get_settings
from app.rag import ask, get_vectorstore, index_pdf
from app.schemas import QueryRequest, QueryResponse, EvaluationRequest
from app.guardrails_runtime import validate_query_safety

settings = get_settings()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("documind")

# FastAPI instance
app = FastAPI(
    title="DocuMind API",
    description="Self-correcting enterprise RAG agent powered by LangChain, LangGraph and Gemini.",
    version="1.0.0",
)

# Frontend Static Files setup
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

if FRONTEND_DIR.exists():
    app.mount(
        "/ui",
        StaticFiles(directory=str(FRONTEND_DIR), html=True),
        name="ui",
    )
else:
    logger.warning("Frontend directory not found at: %s", FRONTEND_DIR)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {
        "name": "DocuMind",
        "status": "running",
        "docs": "/docs",
        "ui": "/ui",
        "message": "Upload PDFs at POST /documents/upload, then ask questions at POST /query.",
    }

@app.get("/health")
def health():
    return {"status": "ok", "gemini_key_configured": bool(settings.google_api_key)}

@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    filename = Path(file.filename or "upload.pdf").name
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(content) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File exceeds {settings.max_upload_mb} MB limit.")
    try:
        result = index_pdf(filename, content)
        return {"status": "indexed", **result}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Document indexing failed")
        raise HTTPException(status_code=500, detail=f"Indexing failed: {type(exc).__name__}. Check server logs.") from exc

@app.post("/query", response_model=QueryResponse)
async def query(payload: QueryRequest):
    try:
        started = time.perf_counter()

        # 1. Guardrail check call
        safety_check = await validate_query_safety(payload.question)
        
        if not safety_check.get("allowed", True):
            return {
                "answer": safety_check.get("blocked_reason"),
                "sources": [],
                "retries": 0,
                "steps": ["guardrails_blocked"],
                "abstained": True,
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            }

        # 2. Agar safe hai to RAG chale
        result = ask(payload.question)
        result["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
        return result
    except Exception as exc:
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail=str(exc))
    
@app.post("/evaluate")
def evaluate(payload: EvaluationRequest):
    """Lightweight evaluation endpoint with extracted contexts support."""
    results = []
    keyword_hits = 0
    source_hits = 0
    for case in payload.cases:
        try:
            result = ask(case.question)
            answer_lower = result["answer"].lower()
            matched = [kw for kw in case.expected_keywords if kw.lower() in answer_lower]
            keyword_score = len(matched) / len(case.expected_keywords) if case.expected_keywords else None
            source_hit = None
            if case.expected_source:
                source_hit = any(
                    case.expected_source.lower() in s["source"].lower()
                    for s in result["sources"]
                )
                source_hits += int(source_hit)
            if case.expected_keywords:
                keyword_hits += int(bool(matched))
            
            contexts = [
                doc.page_content if hasattr(doc, "page_content") else str(doc)
                for doc in result.get("documents", [])
            ]

            results.append({
                "question": case.question,
                "answer": result.get("answer", ""),
                "contexts": contexts,
                "abstained": result["abstained"],
                "keyword_coverage": keyword_score,
                "matched_keywords": matched,
                "expected_source_found": source_hit,
                "retries": result["retries"],
                "steps": result["steps"],
            })
        except Exception as exc:
            results.append({"question": case.question, "error": type(exc).__name__})
    n = len(results)
    scored_keywords = sum(c.expected_keywords != [] for c in payload.cases)
    scored_sources = sum(c.expected_source is not None for c in payload.cases)
    return {
        "cases": results,
        "summary": {
            "total_cases": n,
            "keyword_case_pass_rate": round(keyword_hits / scored_keywords, 3) if scored_keywords else None,
            "expected_source_hit_rate": round(source_hits / scored_sources, 3) if scored_sources else None,
            "note": "Keyword coverage and source hit rate are simple diagnostic metrics, not semantic correctness or a substitute for human review/RAGAS.",
        },
    }

@app.get("/graph")
def graph_info():
    return {
        "nodes": ["retrieve", "grade", "rewrite", "generate"],
        "flow": "START -> retrieve -> grade -> (generate OR rewrite -> retrieve) -> END",
        "max_retries": settings.max_retries,
    }