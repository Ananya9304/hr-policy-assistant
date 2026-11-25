
import hashlib
import logging

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from app import config
from app.embeddings import get_embedder
from app.ingest import IngestError, chunk_document
from app.models import AskRequest, AskResponse, DocumentInfo
from app.rag import RAGService
from app.retrieval import Retriever
from app.store import Store

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

app = FastAPI(title="HR Policy Assistant", version="1.0")
store = Store(config.DB_PATH)
rag = RAGService(Retriever(store))


def require_admin(x_admin_token: str | None = Header(default=None)):
    """Hardcoded admin flag (allowed by the brief). Real SSO/RBAC is out of scope."""
    if x_admin_token != config.ADMIN_TOKEN:
        raise HTTPException(status_code=403, detail="Admin token required to manage documents.")


def index_file(filename: str, data: bytes) -> tuple[DocumentInfo, bool]:
    document, _meta, chunks = chunk_document(filename, data)
    vectors = get_embedder().embed([c.embed_text() for c in chunks])
    return store.upsert_document(document, filename, hashlib.sha256(data).hexdigest(), chunks, vectors)


@app.post("/documents", dependencies=[Depends(require_admin)])
async def upload_document(file: UploadFile = File(...)):
    data = await file.read(config.MAX_UPLOAD_BYTES + 1)
    if len(data) > config.MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large.")
    filename = (file.filename or "").rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    try:
        
        info, changed = await run_in_threadpool(index_file, filename, data)
    except IngestError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"document": info, "changed": changed,
            "message": ("Indexed." if changed else "Identical file already indexed; nothing changed.")}


@app.get("/documents", response_model=list[DocumentInfo])
def list_documents():
    return store.list_documents()


@app.delete("/documents/{doc_id}", dependencies=[Depends(require_admin)])
def delete_document(doc_id: str):
    if not store.delete_document(doc_id):
        raise HTTPException(status_code=404, detail="Document not found.")
    return {"deleted": doc_id}


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    return await run_in_threadpool(rag.ask, req.question)


@app.get("/health")
def health():
    return {"status": "ok", "documents": len(store.list_documents()), "llm_model": config.LLM_MODEL,
            "embedding_backend": config.EMBEDDING_BACKEND}
