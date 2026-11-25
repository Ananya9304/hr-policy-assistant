
import hashlib
import os
import pathlib

os.environ["EMBEDDING_BACKEND"] = "hash"

import pytest  
from fastapi.testclient import TestClient 

from app import config  
from app.embeddings import get_embedder  
from app.ingest import IngestError, chunk_document 
from app.rag import RAGService, answer_numbers_supported, quote_is_supported  
from app.retrieval import Retriever  
from app.store import Store 

POLICIES = pathlib.Path(__file__).resolve().parent.parent / "sample_policies"


class FakeLLM:
    """Returns a canned JSON reply, so we can test what the pipeline does with good and bad model output."""

    def __init__(self, reply):
        self.reply, self.calls = reply, 0

    def complete_json(self, system, user):
        self.calls += 1
        return self.reply


@pytest.fixture
def store():
    s = Store(":memory:")
    for p in POLICIES.glob("*.md"):
        data = p.read_bytes()
        doc, _, chunks = chunk_document(p.name, data)
        s.upsert_document(doc, p.name, hashlib.sha256(data).hexdigest(), chunks,
                          get_embedder().embed([c.embed_text() for c in chunks]))
    return s



def test_sections_and_metadata():
    doc, meta, chunks = chunk_document("leave-policy.md", (POLICIES / "leave-policy.md").read_bytes())
    assert doc == "Leave Policy" and meta["version"] == "1.0"
    cf = [c for c in chunks if c.section.startswith("4.1")]
    assert len(cf) == 1 and "8 days" in cf[0].text
    assert cf[0].section_path == "4. Carry-forward and encashment > 4.1 Casual leave carry-forward"


def test_table_rows_keep_headers():
    _, _, chunks = chunk_document("benefits-policy.md", (POLICIES / "benefits-policy.md").read_bytes())
    standard = [c for c in chunks if c.kind == "table_row" and "Health tier: Standard" in c.text]
    assert len(standard) == 1
    assert "Dental implants: Not covered" in standard[0].text


def test_plain_text_headings():
    data = b"Travel Policy\n\n1. Purpose\nThis covers travel.\n\n1.1 Per diem\nPer diem is 2000 per day.\n"
    _, _, chunks = chunk_document("travel.txt", data)
    assert any(c.section == "1.1 Per diem" and "2000" in c.text for c in chunks)


@pytest.mark.parametrize("name,data", [("x.md", b""), ("x.docx", b"hello"), ("x.pdf", b"not a pdf"),
                                       ("x.md", b"# Title only\n")])
def test_unreadable_files_raise_clean_error(name, data):
    with pytest.raises(IngestError):
        chunk_document(name, data)



def test_quote_verification():
    src = "A maximum of 8 days of unused casual leave may be carried forward to the next calendar year."
    assert quote_is_supported("A maximum of 8 days of unused casual leave may be carried forward", src)
    assert not quote_is_supported("A maximum of 10 days of unused casual leave may be carried forward", src)
    assert not quote_is_supported("Employees may expense home gyms", src)


def test_numbers_in_answer_must_be_in_sources():
    src = ["Health tier: Plus | Dental implants: Covered up to ₹40,000"]
    assert answer_numbers_supported("Plus covers implants up to ₹40,000.", src)
    assert not answer_numbers_supported("Plus covers implants up to ₹50,000.", src)
    assert answer_numbers_supported("Not reimbursable (see section 4).", ["4. Wellness\nGym equipment is not reimbursable."])
    assert answer_numbers_supported("For a 5 day leave, apply 7 days ahead.", ["at least 7 days in advance"], "5 day leave?")



def test_answer_with_verified_citation(store):
    llm = FakeLLM({"answerable": True, "answer": "You can carry forward up to 8 days of casual leave.",
                   "citations": [{"source_id": "S1", "quote": "PLACEHOLDER"}]})
    rag = RAGService(Retriever(store), llm)
    hits = rag.retriever.search("casual leave carry forward limit")
    s = next(i for i, h in enumerate(hits) if h.chunk.section.startswith("4.1"))
    llm.reply["citations"] = [{"source_id": f"S{s + 1}", "quote": "A maximum of 8 days of unused casual leave may be carried forward"}]
    res = rag.ask("casual leave carry forward limit")
    assert res.status == "answered"
    assert res.citations[0].document == "Leave Policy" and res.citations[0].section.startswith("4.1")


def test_fabricated_quote_is_refused(store):
    llm = FakeLLM({"answerable": True, "answer": "Yes, up to 20 days.",
                   "citations": [{"source_id": "S1", "quote": "Employees may carry forward 20 days."}]})
    res = RAGService(Retriever(store), llm).ask("casual leave carry forward limit")
    assert res.status == "refused" and res.reason == "unverified_citations"


def test_model_declines(store):
    llm = FakeLLM({"answerable": False, "answer": "", "citations": []})
    res = RAGService(Retriever(store), llm).ask("maternity leave weeks")
    assert res.status == "refused" and res.reason == "model_declined"


def test_low_score_refuses_without_calling_llm(store, monkeypatch):
    monkeypatch.setattr(config, "MIN_DENSE_SCORE", 0.99)
    llm = FakeLLM({})
    res = RAGService(Retriever(store), llm).ask("what is the capital of France")
    assert res.reason == "low_retrieval_score" and llm.calls == 0


def test_empty_query_and_bad_json(store):
    rag = RAGService(Retriever(store), FakeLLM({"nonsense": 1, "answerable": "maybe"}))
    assert rag.ask("   ").reason == "empty_query"
    assert rag.ask("casual leave carry forward").status in ("refused", "error")


def test_api_upload_ask_and_errors(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", str(tmp_path / "t.db"))
    import importlib
    import app.main as main
    importlib.reload(main)
    main.rag.llm = FakeLLM({"answerable": False, "answer": "", "citations": []})
    client = TestClient(main.app)

    data = (POLICIES / "leave-policy.md").read_bytes()
    assert client.post("/documents", files={"file": ("leave-policy.md", data)}).status_code == 403
    hdr = {"X-Admin-Token": config.ADMIN_TOKEN}
    r = client.post("/documents", headers=hdr, files={"file": ("leave-policy.md", data)})
    assert r.status_code == 200 and r.json()["document"]["version"] == 1
    r = client.post("/documents", headers=hdr, files={"file": ("leave-policy.md", data + b"\nExtra line.\n")})
    assert r.json()["document"]["version"] == 2                      # re-upload replaces, bumps version
    assert len(client.get("/documents").json()) == 1
    assert client.post("/documents", headers=hdr, files={"file": ("bad.pdf", b"%PDF-broken")}).status_code == 400

    r = client.post("/ask", json={"question": ""})
    assert r.status_code == 200 and r.json()["reason"] == "empty_query"
    r = client.post("/ask", json={"question": "maternity leave?"})
    assert set(r.json()) >= {"status", "answer", "citations", "reason"}
