
import os

import pandas as pd
import requests
import streamlit as st

API = os.getenv("API_URL", "http://localhost:8000")

REASONS = {
    "empty_query": "The question was empty.",
    "no_documents": "No policies have been uploaded yet.",
    "low_retrieval_score": "No policy section was close enough to the question, so the AI was not even asked.",
    "model_declined": "The AI read the closest policy sections and found they don't answer this question.",
    "unverified_citations": "The AI's answer could not be matched to the exact policy text, so it was blocked.",
    "llm_unavailable": "The AI service did not respond (network or rate limit). Please try again.",
}
EXAMPLES = [
    ("📅", "What is the casual leave carry-forward limit?"),
    ("🦷", "Does the Standard health tier cover dental implants?"),
    ("👶", "How many weeks of maternity leave do I get?"),
    ("🏋️", "Can I expense a personal home gym?"),
]

st.set_page_config(page_title="HR Policy Assistant", page_icon="🗂️", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&family=JetBrains+Mono:wght@400;600&display=swap');
html, body, [class*="css"] {font-family: 'Inter', sans-serif;}
.block-container {padding-top: 5rem; max-width: 1100px;}
.hero {padding: 22px 26px; border-radius: 14px; margin-bottom: 18px;
       background: linear-gradient(135deg, rgba(99,102,241,.22), rgba(16,185,129,.10));
       border: 1px solid rgba(129,140,248,.35);}
.hero h1 {margin: 0; font-size: 1.9rem; font-weight: 800; letter-spacing: -.5px;}
.hero p {margin: 6px 0 0; opacity: .8;}
.hero .stack {margin-top: 10px; font-family: 'JetBrains Mono', monospace; font-size: .75rem; opacity: .75;}
.mono {font-family: 'JetBrains Mono', monospace;}
.badge {display:inline-block; padding:3px 11px; border-radius:999px; font-size:.75rem; font-weight:600;
        font-family:'JetBrains Mono', monospace; margin-bottom:8px; letter-spacing:.3px}
.ok  {background:rgba(16,185,129,.15); color:#34d399; border:1px solid rgba(52,211,153,.4)}
.no  {background:rgba(245,158,11,.13); color:#fbbf24; border:1px solid rgba(251,191,36,.4)}
.err {background:rgba(239,68,68,.13); color:#f87171; border:1px solid rgba(248,113,113,.4)}
.cite {border-left:3px solid #818cf8; padding:8px 14px; margin:8px 0; border-radius:6px;
       background:rgba(129,140,248,.08)}
.cite .src {font-family:'JetBrains Mono', monospace; font-size:.75rem; color:#a5b4fc}
.cite .q {font-size:.9rem; opacity:.9; margin-top:3px}
.pipe {display:flex; align-items:center; gap:6px; flex-wrap:wrap; margin:8px 0 12px;
       font-family:'JetBrains Mono', monospace; font-size:.74rem}
.step {padding:6px 11px; border-radius:8px; border:1px solid rgba(255,255,255,.12); background:rgba(255,255,255,.03)}
.step.pass {border-color:rgba(52,211,153,.55); color:#6ee7b7}
.step.fail {border-color:rgba(248,113,113,.6); color:#fca5a5}
.step.skip {opacity:.35}
.arrow {opacity:.4}
.why {font-size:.85rem; opacity:.75; margin-top:4px}
.doc {padding:9px 12px; border-radius:8px; margin-bottom:8px; background:rgba(255,255,255,.03);
      border:1px solid rgba(255,255,255,.07)}
.doc b {font-size:.9rem} .doc small {font-family:'JetBrains Mono', monospace; opacity:.6; font-size:.72rem}
</style>
""", unsafe_allow_html=True)



def ask_api(question: str) -> dict:
    try:
        r = requests.post(f"{API}/ask", json={"question": question}, timeout=120)
        r.raise_for_status()
        return r.json()
    except requests.RequestException:
        return {"status": "error", "answer": f"Could not reach the API at {API}.", "citations": [],
                "reason": "llm_unavailable"}


def step(label: str, state: str, note: str = "") -> str:
    icon = {"pass": "✓", "fail": "✕", "skip": "·"}[state]
    return f'<span class="step {state}">{icon} {label}{(" · " + note) if note else ""}</span>'


def render_pipeline(t: dict, status: str):
    """Four boxes: Retrieve → Gate → LLM → Verify, each green (passed), red (stopped here) or grey (not reached)."""
    n = len(t.get("retrieved", []))
    top = max((c["dense_score"] for c in t.get("retrieved", [])), default=0.0)
    gate = t.get("gate_passed")
    steps = [step("RETRIEVE", "pass" if n else "fail", f"{n} chunks")]
    steps.append(step("GATE", "skip" if gate is None else ("pass" if gate else "fail"),
                      f"{top:.2f} ≥ {t['gate_threshold']:.2f}" if gate else f"{top:.2f} < {t['gate_threshold']:.2f}"
                      if gate is False else ""))
    if not t.get("llm_called"):
        steps.append(step("LLM", "skip"))
    elif status == "error":
        steps.append(step("LLM", "fail", "unavailable"))
    else:
        steps.append(step("LLM", "pass" if t.get("llm_answerable") else "fail",
                          "answerable" if t.get("llm_answerable") else "declined"))
    if t.get("numbers_verified") is None:
        steps.append(step("VERIFY", "skip"))
    else:
        ok = status == "answered"
        steps.append(step("VERIFY", "pass" if ok else "fail",
                          f"{t['citations_verified']}/{t['citations_proposed']} quotes"
                          + ("" if t["numbers_verified"] else " · numbers ✕")))
    st.markdown('<div class="pipe">' + '<span class="arrow">→</span>'.join(steps) + "</div>", unsafe_allow_html=True)


def render_trace_details(t: dict):
    c1, c2, c3, c4 = st.columns(4)
    top = max((c["dense_score"] for c in t.get("retrieved", [])), default=0.0)
    c1.metric("Top similarity", f"{top:.3f}", f"gate {t['gate_threshold']:.2f}", delta_color="off")
    c2.metric("Quotes verified", f"{t.get('citations_verified', 0)}/{t.get('citations_proposed', 0)}")
    c3.metric("Retrieval", f"{t.get('timings_ms', {}).get('retrieval', 0):.0f} ms")
    c4.metric("Total", f"{t.get('timings_ms', {}).get('total', 0):.0f} ms")
    if t.get("retrieved"):
        df = pd.DataFrame(t["retrieved"]).rename(columns={
            "document": "Document", "section": "Section", "kind": "Type",
            "dense_score": "Similarity", "fused_score": "RRF", "cited": "Cited"})
        df.insert(0, "#", [f"S{i + 1}" for i in range(len(df))])
        st.dataframe(df, hide_index=True, use_container_width=True, column_config={
            "Similarity": st.column_config.ProgressColumn("Similarity", min_value=0, max_value=1, format="%.3f"),
            "RRF": st.column_config.NumberColumn("RRF", format="%.4f"),
            "Cited": st.column_config.CheckboxColumn("Cited"),
        })


def render_reply(data: dict):
    status = data.get("status")
    badge = {"answered": ('ok', '● ANSWERED · grounded in policy'),
             "refused": ('no', '● REFUSED · not in the policies')}.get(status, ('err', '● ERROR'))
    st.markdown(f'<span class="badge {badge[0]}">{badge[1]}</span>', unsafe_allow_html=True)
    if data.get("trace"):
        render_pipeline(data["trace"], status)

    st.markdown(data.get("answer", ""))

    for c in data.get("citations", []):
        st.markdown(f'<div class="cite"><div class="src">{c["document"]} › {c["section_path"]}</div>'
                    f'<div class="q">“{c["quote"]}”</div></div>', unsafe_allow_html=True)

    if data.get("reason"):
        st.markdown(f'<div class="why">Why: {REASONS.get(data["reason"], data["reason"])}</div>',
                    unsafe_allow_html=True)

    if data.get("trace"):
        with st.expander("🔬 Pipeline trace: retrieved chunks, scores, timings"):
            render_trace_details(data["trace"])
    with st.expander("{ } Raw JSON response"):
        st.json(data)



with st.sidebar:
    st.markdown("### 🗂️ Policy library")
    try:
        requests.get(f"{API}/health", timeout=5).raise_for_status()
        docs = requests.get(f"{API}/documents", timeout=10).json()
        st.markdown(f'<span class="badge ok">● API ONLINE</span> <span class="badge ok">{len(docs)} POLICIES</span>',
                    unsafe_allow_html=True)
    except requests.RequestException:
        docs = []
        st.markdown('<span class="badge err">● API OFFLINE</span>', unsafe_allow_html=True)
        st.code("uvicorn app.main:app --reload", language="bash")

    for d in docs:
        st.markdown(f'<div class="doc"><b>{d["document"]}</b><br>'
                    f'<small>{d["filename"]} · v{d["version"]} · {d["chunks"]} chunks</small></div>',
                    unsafe_allow_html=True)

    with st.expander("🔐 Admin: upload / delete"):
        token = st.text_input("Admin token", type="password")
        files = st.file_uploader("Policy files", type=["md", "txt", "pdf"], accept_multiple_files=True)
        if st.button("Upload & index", use_container_width=True, disabled=not files, type="primary"):
            for f in files:
                with st.spinner(f"Chunking + embedding {f.name}..."):
                    r = requests.post(f"{API}/documents", headers={"X-Admin-Token": token},
                                      files={"file": (f.name, f.getvalue())}, timeout=300)
                if r.ok:
                    st.success(f"{f.name}: {r.json()['message']}")
                else:
                    st.error(f"{f.name}: {r.json().get('detail', r.text)}")
            st.rerun()
        if docs:
            target = st.selectbox("Delete a policy", [d["filename"] for d in docs])
            if st.button("Delete", use_container_width=True):
                doc = next(d for d in docs if d["filename"] == target)
                r = requests.delete(f"{API}/documents/{doc['doc_id']}", headers={"X-Admin-Token": token}, timeout=10)
                if r.ok:
                    st.rerun()
                st.error(r.json().get("detail", r.text))

    if st.button("🧹 Clear chat", use_container_width=True):
        st.session_state.history = []
        st.rerun()

    st.markdown('<div class="mono" style="font-size:.7rem;opacity:.5;margin-top:14px">'
                'retrieval: MiniLM + BM25 → RRF<br>llm: Groq · JSON mode<br>guards: gate → declare → verify</div>',
                unsafe_allow_html=True)


st.markdown("""
<div class="hero">
  <h1>HR Policy Assistant</h1>
  <p>Answers only from your uploaded HR policies, with the exact sentence as proof. If the policies don't say, it says so.</p>
  <div class="stack">FastAPI · hybrid retrieval (vectors + BM25) · grounded JSON · quote & number verification</div>
</div>
""", unsafe_allow_html=True)

if "history" not in st.session_state:
    st.session_state.history = []

question = st.chat_input("Ask an HR policy question...")

if not st.session_state.history and not question:
    st.markdown("**Try one of these**")
    cols = st.columns(4)
    for i, (icon, ex) in enumerate(EXAMPLES):
        if cols[i].button(f"{icon}  {ex}", use_container_width=True, key=f"ex{i}"):
            st.session_state.pending = ex
            st.rerun()

question = question or st.session_state.pop("pending", None)

for turn in st.session_state.history:
    with st.chat_message("user", avatar="🧑‍💼"):
        st.write(turn["question"])
    with st.chat_message("assistant", avatar="🗂️"):
        render_reply(turn["reply"])

if question:
    with st.chat_message("user", avatar="🧑‍💼"):
        st.write(question)
    with st.chat_message("assistant", avatar="🗂️"):
        with st.spinner("Retrieving → checking → asking the model → verifying quotes..."):
            reply = ask_api(question)
        render_reply(reply)
    st.session_state.history.append({"question": question, "reply": reply})