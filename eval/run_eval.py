
import argparse
import hashlib
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import config  
from app.embeddings import get_embedder  
from app.ingest import chunk_document  
from app.rag import RAGService  
from app.retrieval import Retriever 
from app.store import Store 


def build_store(policy_dir: pathlib.Path) -> Store:
    store = Store(":memory:")
    for path in sorted(policy_dir.iterdir()):
        if path.suffix.lower() in {".md", ".txt", ".pdf"}:
            data = path.read_bytes()
            doc, _, chunks = chunk_document(path.name, data)
            vecs = get_embedder().embed([c.embed_text() for c in chunks])
            store.upsert_document(doc, path.name, hashlib.sha256(data).hexdigest(), chunks, vecs)
    return store


def section_matches(case, doc, section_path) -> bool:
    return doc == case["doc"] and case["section"].lower() in section_path.lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieval-only", action="store_true")
    ap.add_argument("--policies", default=str(ROOT / "sample_policies"))
    ap.add_argument("--sleep", type=float, default=8.0,
                    help="seconds between LLM calls; free tiers limit tokens per minute (Groq: 8k TPM)")
    args = ap.parse_args()

    cases = json.loads((ROOT / "eval" / "questions.json").read_text())
    store = build_store(pathlib.Path(args.policies))
    retriever = Retriever(store)
    passed = 0

    for case in cases:
        hits = retriever.search(case["question"])
        top = max((h.dense_score for h in hits), default=0.0)
        rank = next((i + 1 for i, h in enumerate(hits)
                     if "doc" in case and section_matches(case, h.chunk.document, h.chunk.section_path)), None)

        extra = ""
        if args.retrieval_only:
            if case["expect"] == "refused":
                ok = True     # cannot judge refusals without the LLM; we only print the score
                detail = f"top_score={top:.2f} (gate={config.MIN_DENSE_SCORE})"
            else:
                ok = rank is not None
                detail = f"expected section at rank {rank}" if ok else "expected section NOT in top-k"
        else:
            res = RAGService(retriever).ask(case["question"])
            time.sleep(args.sleep)
            ans = res.answer.lower()
            if case["expect"] == "refused":
                ok = res.status == "refused"
            else:
                cited_ok = any(section_matches(case, c.document, c.section_path) for c in res.citations)
                contains_ok = all(s.lower() in ans for s in case.get("answer_contains", []))
                excludes_ok = not any(s.lower() in ans for s in case.get("answer_excludes", []))
                answered_ok = res.status == "answered" and cited_ok and contains_ok and excludes_ok
                ok = answered_ok or (case["expect"] == "answered_or_refused" and res.status == "refused")
                if not ok and res.status == "answered":
                    why = [n for n, good in (("wrong/missing citation", cited_ok), ("missing expected text", contains_ok),
                                             ("contains forbidden text", excludes_ok)) if not good]
                    cites = "; ".join(f"{c.document} > {c.section}" for c in res.citations)
                    extra = f"\n      why: {', '.join(why)} | cited: {cites}\n      answer: {res.answer!r}"
            detail = f"status={res.status.value} reason={res.reason and res.reason.value} top={res.top_score} :: {res.answer[:90]}"

        passed += ok
        print(f"{'PASS' if ok else 'FAIL'}  {case['id']:<28} {detail}{extra}")

    print(f"\n{passed}/{len(cases)} passed")
    sys.exit(0 if passed == len(cases) else 1)


if __name__ == "__main__":
    main()
