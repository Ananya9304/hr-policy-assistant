# HR Policy Assistant (RAG)

**Name:** Ananya Ghosh
**College:** Kalyani Government Engineering College
**Email:** ghoshanu984@gmail.com
**Phone:** 9064003516
**Demo video:** [add link]

---

A small RAG service that answers employee questions about HR policies **only from the uploaded policy files**.
Every answer comes back as **JSON with citations** (document + section). If the policies don't contain the answer,
it **refuses** and tells the employee to contact HR. It doesn't guess.

📄 **The design write-up is in [DESIGN.md](DESIGN.md).**

```
Upload:  .md/.txt/.pdf → split by heading + table rows → local embeddings → SQLite
Ask:     question → meaning + keyword search → score check → AI (JSON + quotes) → verify quotes & numbers → answer or refusal
```

**Eval result: 18/18 passing** (12 answers with correct citations, 4 refusals, gym case, prompt-injection case).

---

## How to run it (about 10 minutes)

You need **Python 3.10+** and a free **Groq API key** from https://console.groq.com/keys.

### 1. Set up

**Windows (PowerShell):**
```powershell
git clone https://github.com/<your-username>/hr-policy-assistant.git
cd hr-policy-assistant
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

**Mac / Linux:**
```bash
git clone https://github.com/<your-username>/hr-policy-assistant.git
cd hr-policy-assistant
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Open `.env` and paste your Groq key into `LLM_API_KEY=`.

> If PowerShell blocks `activate`, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once.
> Keep the project **outside OneDrive**: OneDrive locks files while `venv` is being created.

### 2. Start the API (terminal 1)
```
uvicorn app.main:app --reload
```
API docs: http://localhost:8000/docs

### 3. Start the chat UI (terminal 2)
```
.venv\Scripts\activate
streamlit run ui/streamlit_app.py
```
Opens at http://localhost:8501.

### 4. Use it
1. **Admin tab:** enter the admin token from `.env` (default `change-me`) and upload the 3 files from `sample_policies/`.
   The first upload takes 1–2 minutes because the embedding model (~90 MB) downloads once.
2. **Ask tab:** try:
   - *What is the casual leave carry-forward limit?* → 8 days, cites Leave Policy 4.1
   - *Does the Standard health tier cover dental implants?* → Not covered, cites the Benefits Policy table
   - *How many weeks of maternity leave do I get?* → refuses and points to HR

The first question after starting the server can take ~30 seconds while models load; after that, answers take a few seconds.

---

## Models used

| Part | Model | Where it runs | Cost |
|---|---|---|---|
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | Locally on CPU | Free |
| LLM | `openai/gpt-oss-120b` on **Groq** (OpenAI-compatible API) | Groq cloud | Free tier |

To switch to **Gemini** or **Ollama** (fully local), change `LLM_BASE_URL`, `LLM_API_KEY` and `LLM_MODEL` in `.env`
(examples are in `.env.example`). Groq model names change over time; if you get a "model does not exist" error, run:
```
python -c "from app.llm import LLMClient; print([m.id for m in LLMClient().client.models.list()])"
```
and put one of the listed models into `LLM_MODEL`.

**Privacy note:** the question and the matching policy excerpts are sent to the LLM provider. Use Ollama if policy text
must not leave the machine.

## Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `LLM_BASE_URL` | Groq URL | Any OpenAI-compatible endpoint |
| `LLM_API_KEY` | – | Your API key (**never commit `.env`**) |
| `LLM_MODEL` | `openai/gpt-oss-120b` | Chat model |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformers model |
| `EMBEDDING_BACKEND` | `sentence-transformers` | `hash` = offline test mode only |
| `TOP_K` | `5` | Number of chunks sent to the AI |
| `MIN_DENSE_SCORE` | `0.20` | Below this similarity the bot refuses without calling the AI (tuned with the eval) |
| `DB_PATH` | `data/index.db` | SQLite database file |
| `ADMIN_TOKEN` | `admin` | Admin password for upload/delete |
| `HR_CONTACT` | `HR at hr@example.com` | Shown in refusal messages |

## API

| Method | Endpoint | Who | Description |
|---|---|---|---|
| POST | `/documents` | admin (`X-Admin-Token` header) | Upload a policy file |
| GET | `/documents` | anyone | List uploaded policies |
| DELETE | `/documents/{doc_id}` | admin | Delete a policy |
| POST | `/ask` | anyone | `{"question": "..."}` → JSON answer |
| GET | `/health` | anyone | Server status |

Example answer:
```json
{
  "status": "answered",
  "answer": "The casual leave carry-forward limit is a maximum of 8 days.",
  "citations": [{
    "document": "Leave Policy",
    "filename": "leave-policy.md",
    "section": "4.1 Casual leave carry-forward",
    "section_path": "4. Carry-forward and encashment > 4.1 Casual leave carry-forward",
    "quote": "A maximum of 8 days of unused casual leave may be carried forward to the next calendar year"
  }],
  "reason": null,
  "top_score": 0.75
}
```

Example refusal:
```json
{"status": "refused", "answer": "I don't have enough information in the uploaded HR policies to answer that. Please contact HR at hr@example.com.", "citations": [], "reason": "model_declined", "top_score": 0.411}
```

## Tests and evaluation

```
pytest -q                                 # 15 unit tests, run offline (fake AI + simple embedder)
python -m eval.run_eval --retrieval-only  # checks the right section is found, no API key needed
python -m eval.run_eval                   # full check with the AI (~3 min, pauses for the free-tier rate limit)
```

## Project structure

```
app/
  config.py       settings from .env
  models.py       request/response shapes (Pydantic)
  ingest.py       read files, split by heading, table rows as chunks
  embeddings.py   local embeddings
  store.py        SQLite storage, versioned re-upload
  retrieval.py    meaning search + keyword search, merged with RRF
  llm.py          prompt and AI call
  rag.py          the pipeline: score check → AI → verify quotes/numbers → answer or refuse
  main.py         FastAPI endpoints
ui/streamlit_app.py   chat UI
eval/                 18 test questions + runner
tests/                unit tests
sample_policies/      the 3 sample policies
```