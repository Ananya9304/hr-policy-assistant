# DESIGN: HR Policy Assistant

**Goal:** answer employee questions **only** from uploaded HR policies, with a citation for every answer, and
say "I don't know, contact HR" whenever the policies don't contain the answer.
I treated the refusal path as a feature with the same weight as the answer path. A confident wrong answer about leave or
benefits costs more than a refusal, because the employee acts on it.

---

## 1. Architecture

```
 ADMIN                                         EMPLOYEE
   │ POST /documents (.md/.txt/.pdf)              │ POST /ask {"question"}
   ▼                                              ▼
┌──────────────┐  text   ┌──────────────┐     ┌──────────────────────────┐
│ extract_text │────────►│ chunker      │     │ Hybrid retriever         │
│ md/txt/pdf   │         │ • 1 section  │     │ • dense: cosine (numpy)  │
│ (errors→400) │         │   = 1 chunk  │     │ • BM25 keywords          │
└──────────────┘         │ • 1 table row│     │ • Reciprocal Rank Fusion │
                         │   = 1 chunk  │     └────────────┬─────────────┘
                         └──────┬───────┘                  │ top-5 chunks + scores
                                │ chunks + metadata        ▼
                         ┌──────▼───────┐     ┌──────────────────────────┐
                         │ embedder     │     │ L1  score gate           │──► refuse (no LLM call)
                         │ MiniLM local │     └────────────┬─────────────┘
                         └──────┬───────┘                  ▼
                         ┌──────▼───────────┐ ┌──────────────────────────┐
                         │ SQLite           │ │ L2  LLM, JSON mode,      │──► answerable=false → refuse
                         │ documents/chunks │◄┤     must quote sources   │──► error/timeout → status=error
                         │ + vectors (BLOB) │ └────────────┬─────────────┘
                         └──────────────────┘              ▼
                                              ┌──────────────────────────┐
                                              │ L3  verify quotes +      │──► nothing verifiable → refuse
                                              │     numbers vs. chunks   │
                                              └────────────┬─────────────┘
                                                           ▼
                                              {status, answer, citations[], reason, top_score}
```

| Component | File | Responsibility |
|---|---|---|
| API | `app/main.py` | FastAPI endpoints, admin flag, size limit, maps ingest errors to 400 |
| Ingest | `app/ingest.py` | Read file, split into heading-scoped sections and table rows, attach metadata |
| Embeddings | `app/embeddings.py` | Local `all-MiniLM-L6-v2` (384-d, normalised) |
| Store | `app/store.py` | SQLite tables `documents`, `chunks`; in-memory vector matrix cache |
| Retrieval | `app/retrieval.py` | Dense + BM25, fused with RRF |
| Generation | `app/llm.py`, `app/rag.py` | Prompt, JSON parsing, three-layer grounding |
| UI | `ui/streamlit_app.py` | Ask tab + admin upload tab; calls the API only |
| Eval | `eval/` | 18 questions with expected section; retrieval-only and full modes |

**Failure paths:** an empty or unreadable file returns `400` with a reason, never a crash. An empty question returns
`refused/empty_query`. An LLM timeout, quota error or invalid JSON (after one retry) returns `status: "error"` with a
"try again or contact HR" message. The system never answers without the model.

---

## 2. Chunking and retrieval

### How documents are split
HR policies are already structured as **numbered clauses**, and an employee's question almost always maps to one clause.
So the chunk boundary is the **heading**, not a character count:

* **One section = one chunk.** For example, `4.1 Casual leave carry-forward` (two short paragraphs) is one chunk.
  The sample sections are 50–400 characters, so fixed 500-token windows would either merge unrelated clauses
  (4.1 casual + 4.2 privilege carry-forward) or split a rule from its exception ("must be used by 31 March").
* **One table row = one chunk, with column headers repeated:**
  `Health tier: Standard | Eligible bands: Band B, Band C | … | Dental implants: Not covered | …`.
  A raw markdown row `| Standard | Band B… | Not covered |` loses the meaning of every cell. Table questions
  ("does Standard cover implants?") need exactly one row and one column, and this format makes both searchable and quotable.
* **Long sections** (>1,200 characters) are split on paragraph boundaries, never mid-sentence. Each piece keeps the section path.
* **Document front-matter** (`**Version:**`, `**Applies to:**`) is parsed into document metadata, not indexed as a chunk.
* **Plain text / PDF**: lines like `4.1 Casual leave carry-forward` are detected as headings, so `.txt` files cite sections too.

### Metadata stored per chunk
`chunk_id, doc_id, document title, filename, section, section_path (e.g. "4. Carry-forward… > 4.1 Casual…"), kind (text|table_row), text, embedding`.
The section path is what the citation shows. It is also **prepended to the text before embedding and keyword indexing**
(`Leave Policy | 4. Carry-forward… > 4.1 …`). This means a chunk whose body never repeats the topic word can still be found
by its heading.

### Choosing the top chunks: hybrid search
* **Dense (embeddings)** handles paraphrase: "carry over unused CL into next year" ≈ "carried forward to the next calendar year".
* **BM25 (keywords)** handles exact tokens that embeddings blur: `Band C`, `LTA`, `Plus`, `12 characters`, clause numbers.
  It uses light plural stemming: my eval caught "password" failing to match "Passwords".
* The two ranked lists are merged with **Reciprocal Rank Fusion** (`score = Σ 1/(60 + rank)`). RRF uses ranks, not raw
  scores, so I don't need to calibrate cosine against BM25.
* **Top 5** chunks go to the LLM. Chunks are small, so 5 still fits in a short prompt, and 5 leaves room for questions
  that need two clauses (for example, the carry-forward limit plus the 31 March deadline).

---

## 3. Grounding: how the system avoids inventing policy

Three independent layers. Any one of them can turn an answer into a refusal, and none of them can turn a refusal into an answer.

1. **Retrieval gate (before the LLM).** If the best dense similarity is below `MIN_DENSE_SCORE` (0.20), we refuse
   **without calling the model**. Questions like "capital of France" never reach the LLM, so it cannot answer from general knowledge.
   The gate is deliberately loose, and the threshold came from the eval, not a guess. My first run used 0.30 and
   wrongly refused an answerable question ("Where can I store a spreadsheet of salaries?", top score 0.239: the answer is
   a table row in the IT policy). That run also showed an off-policy question ("annual bonus") scoring 0.249, *higher*
   than that answerable one. **Similarity alone cannot separate answerable from unanswerable questions**, so the gate only
   filters the clearly off-topic (≈0.04 for "capital of France"), and near-misses are left to layers 2 and 3.
2. **Prompt and schema.** Temperature 0, JSON mode, and a Pydantic-validated schema
   `{answerable, answer, citations:[{source_id, quote}]}`. The model is told to use only the labelled excerpts, to set
   `answerable:false` when they don't contain the answer, and to ignore instructions inside the excerpts. Giving the model an
   explicit, cheap "no" option makes refusing easier for it than improvising.
3. **Verification (after the LLM). This is the key part.**
   * Each `quote` must actually appear in the chunk with that `source_id` (normalised match, or ≥90% word overlap
     **with every number present**). Citations that fail are dropped. If none survive, we refuse with `unverified_citations`.
   * **Every number in the answer** (days, ₹ amounts, hours) must appear in a cited chunk (its text or section heading)
     or in the question itself. "₹50,000" when the policy says "₹40,000" is refused.
   * Citations are built **from our own chunk metadata**, never from text the model writes. The document name and section
     therefore always point to a real uploaded document.

**Weak retrieval:** a low score triggers the gate, so we refuse. A medium score lets the LLM see the chunks; if they don't
answer the question it declines, and if it tries to answer anyway, verification catches unsupported quotes and numbers.

**A subtle case: "Can I expense a personal home gym?"** The brief expects a refusal, but Benefits §4 explicitly says
"Gym or home-fitness equipment is **not** a reimbursable benefit". Inventing a policy would be wrong; quoting an explicit
exclusion is not. So the system answers *"No — not reimbursable under the Benefits Policy (§4 Wellness)"* with a citation, and the
eval accepts either a cited "no" or a refusal. "WFH internet stipend" is different: §5 says the policy does **not define** it,
so the correct output is a refusal, never an invented amount. Both cases are in the eval set.

---

## 4. Schema and APIs

```
POST /ask   {"question": "string (≤1000 chars)"}
→ 200 {
    "status":   "answered" | "refused" | "error",
    "answer":   "string",                       // always present, safe to show the user
    "citations":[{"document","filename","section","section_path","quote"}],
    "reason":   null | "empty_query" | "no_documents" | "low_retrieval_score"
                     | "model_declined" | "unverified_citations" | "llm_unavailable",
    "top_score": 0.61                           // best similarity; for debugging and threshold tuning
  }
POST   /documents  (X-Admin-Token, multipart file) → {document:{doc_id,document,filename,version,chunks,uploaded_at}, changed}
GET    /documents  → [DocumentInfo]
DELETE /documents/{doc_id} (admin)
```

**Why this shape:**
* `status` is an **enum**, so the UI and any future integration (Slack bot, HR portal) branch on it instead of
  parsing English. `refused` (no answer exists) is separate from `error` (try again), because they need different UX.
* `reason` gives a reason for every refusal, so we can monitor why the bot says no. Many `low_retrieval_score`
  results mean a missing policy; many `unverified_citations` mean a prompt or model problem.
* `citations[].quote` lets the employee check the source text without opening the file, which builds trust.
* `/ask` always returns 200 with a valid body; HTTP errors are reserved for client mistakes (403, 400, 413).
* **Upload versioning:** `doc_id` is derived from the filename. Re-uploading `leave-policy.md` replaces all its chunks
  in a single SQLite transaction and increments `version`, so readers never see half a policy or two versions at once.
  An identical file (same SHA-256) is a no-op.

---

## 5. Trade-offs (considered and rejected)

| Decision | Chose | Rejected | Why |
|---|---|---|---|
| Chunking | Heading-scoped sections + table rows | Fixed 500-token windows with overlap (tutorial default) | Policies are clause-structured. Fixed windows merge unrelated clauses and split tables from headers, which also makes the citation's section ambiguous. |
| Vector store | SQLite + numpy in memory | Chroma / pgvector | Hundreds of chunks: brute-force cosine takes under 1 ms. SQLite gives atomic replace and no extra service to run. pgvector becomes worth it past roughly 100k chunks or with multiple API instances. |
| Retrieval | Hybrid (dense + BM25, RRF) | Dense only | Embeddings blur exact tokens in short table rows ("Band B" vs "Band C"). BM25 costs about 20 lines. |
| Orchestration | Own thin pipeline (≈150 lines) | LangChain / LlamaIndex | The grounding logic (gate, quote check, number check) is the product. I wanted every step visible and testable, not inside framework defaults. |
| LLM API | OpenAI-compatible client | Provider SDK | Groq, Gemini, OpenRouter and Ollama all speak it. Switching provider when a free quota runs out is an `.env` change. |
| Embeddings | Local MiniLM | Hosted embedding API | Free, no quota, policy text is not sent out for indexing, and re-indexing costs nothing. |
| Ingestion | Synchronous (in a thread pool) | Background queue | HR files are small (seconds to index). A queue adds job states and polling for no user benefit at this size; the thread pool already keeps `/ask` responsive. |
| Refusal | Code-enforced layers | Prompt-only "say I don't know" | A prompt is a request, not a guarantee. Verification is deterministic and unit-tested. |

What I **deliberately did not build:** real auth (a hardcoded admin token is enough here), multi-tenancy,
deployment/CI, OCR for scanned PDFs, and streaming responses.

---

## 6. If I had two more weeks

1. **A larger eval set plus CI** (first, because it protects everything else): about 100 questions written with HR, including
   adversarial ones. Run on every change and track answer accuracy, refusal precision/recall and citation accuracy.
   Use it to **calibrate `MIN_DENSE_SCORE` from data** instead of picking it by hand.
2. **Better PDF and table extraction:** pdfplumber/camelot so PDF tables become header-aware row chunks like markdown tables
   (today pypdf flattens them to text), plus OCR fallback.
3. **Feedback loop:** thumbs up/down stored with question, retrieved chunk IDs and `reason`. Repeated refusals on the same
   topic are a signal that a policy is missing.
4. **Applicability filters:** use front-matter (`Applies to: Full-time employees in India`) and the user's profile (band,
   country) to filter chunks before ranking. Then "my LTA limit" can be answered for *my* band.
5. **Auth and audit:** SSO, admin role, and an audit log of who uploaded which version.
6. **Scale:** pgvector, async ingestion for large uploads, cached answers for frequent questions.

## Limitations I'm aware of
* Numbers are checked as digits: an answer that writes "₹10 lakh" for "₹10,00,000" would be refused (safe, but strict).
* Questions that need reasoning across many clauses ("total leave I can carry forward") rely on the LLM to combine excerpts.
  Each number is still verified, but arithmetic results that don't appear in the policy are refused.
* `MIN_DENSE_SCORE` depends on the embedding model and must be recalibrated with the eval set if the model changes.
