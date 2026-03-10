import streamlit as st
import os
import tempfile
import math
import re
from pypdf import PdfReader
from bytez import Bytez

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocChat",
    page_icon="✦",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── API key from secrets ───────────────────────────────────────────────────────
def get_api_key() -> str | None:
    try:
        return st.secrets["BYTEZ_API_KEY"]
    except Exception:
        return None

MODELS = [
    "openai/gpt-4o",
    "openai/gpt-4o-mini",
    "anthropic/claude-3-5-sonnet-20241022",
    "anthropic/claude-3-haiku-20240307",
    "meta-llama/Llama-3.1-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
]

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500;9..40,600&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, .stApp {
    background: #0c0c0f !important;
    font-family: 'DM Sans', sans-serif;
    color: #e8e6e1;
}

#MainMenu, footer, header { visibility: hidden; }

.block-container {
    padding: 0 16px 120px 16px !important;
    max-width: 780px !important;
}

/* ── Top navbar ── */
.navbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 18px 0 14px 0;
    border-bottom: 1px solid #1a1a24;
    margin-bottom: 20px;
    position: sticky;
    top: 0;
    background: #0c0c0f;
    z-index: 100;
}
.nav-brand {
    display: flex;
    align-items: center;
    gap: 10px;
}
.nav-logo {
    width: 34px; height: 34px;
    background: linear-gradient(135deg, #5b5bd6, #9898d8);
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1rem; color: white; font-weight: 700;
    flex-shrink: 0;
}
.nav-title {
    font-family: 'Instrument Serif', serif;
    font-size: 1.3rem;
    color: #e8e6e1;
    letter-spacing: -0.02em;
}
.nav-doc {
    background: #16161e;
    border: 1px solid #22222e;
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 0.72rem;
    color: #7878c8;
    max-width: 180px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

/* ── Upload card ── */
.upload-card {
    background: #13131a;
    border: 1.5px dashed #2a2a3e;
    border-radius: 18px;
    padding: 28px 20px;
    margin-bottom: 20px;
    transition: border-color 0.2s;
}
.upload-card:hover { border-color: #5b5bd6; }
.upload-label {
    font-size: 0.7rem;
    font-weight: 600;
    color: #5b5b72;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 10px;
}

/* ── Settings row ── */
.settings-card {
    background: #10101a;
    border: 1px solid #1e1e2c;
    border-radius: 14px;
    padding: 16px;
    margin-bottom: 20px;
}
.settings-title {
    font-size: 0.68rem;
    font-weight: 600;
    color: #4a4a62;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 12px;
}

/* ── Stat pills ── */
.stat-row {
    display: flex;
    gap: 8px;
    margin: 12px 0;
}
.stat-box {
    flex: 1;
    background: #16161e;
    border: 1px solid #22222e;
    border-radius: 12px;
    padding: 10px;
    text-align: center;
}
.stat-num {
    font-family: 'Instrument Serif', serif;
    font-size: 1.3rem;
    color: #9898d8;
}
.stat-lbl {
    font-size: 0.62rem;
    color: #44445a;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* ── Chat messages ── */
.msg-row {
    display: flex;
    gap: 10px;
    margin-bottom: 16px;
    animation: fadeUp 0.25s ease;
}
.msg-row.user-row { flex-direction: row-reverse; }

@keyframes fadeUp {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}

.avatar {
    width: 30px; height: 30px;
    border-radius: 50%;
    flex-shrink: 0;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.7rem; font-weight: 700;
    margin-top: 2px;
}
.avatar.ai-av  { background: linear-gradient(135deg,#5b5bd6,#9898d8); color:#fff; }
.avatar.usr-av { background: #1e1e2c; border:1px solid #2a2a3e; color:#7878c8; }

.bubble {
    max-width: min(520px, 78vw);
    padding: 12px 16px;
    font-size: 0.91rem;
    line-height: 1.65;
    word-break: break-word;
}
.bubble.ai-bub {
    background: #14141e;
    border: 1px solid #20202e;
    color: #d8d5d0;
    border-radius: 4px 16px 16px 16px;
}
.bubble.usr-bub {
    background: linear-gradient(135deg, #5b5bd6, #7575e0);
    color: #fff;
    border-radius: 16px 4px 16px 16px;
}
.src-row {
    display: flex; flex-wrap: wrap; gap: 5px;
    margin-top: 10px; padding-top: 10px;
    border-top: 1px solid #252535;
}
.src-pill {
    background: #1c1c2a; border: 1px solid #2c2c40;
    border-radius: 20px; padding: 2px 9px;
    font-size: 0.68rem; color: #7070b8; font-weight: 500;
}

/* ── Empty state ── */
.empty-wrap {
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    padding: 48px 20px; gap: 10px; text-align: center;
}
.empty-icon { font-size: 2.6rem; opacity: 0.3; }
.empty-title {
    font-family: 'Instrument Serif', serif;
    font-size: 1.4rem; color: #35354a;
}
.empty-sub { font-size: 0.82rem; color: #2e2e40; }
.chips-wrap {
    display: flex; flex-wrap: wrap;
    gap: 8px; justify-content: center; margin-top: 14px;
}
.chip {
    background: #14141e; border: 1px solid #22222e;
    border-radius: 20px; padding: 7px 15px;
    font-size: 0.78rem; color: #5a5a72;
}

/* ── Fixed input bar at bottom ── */
.input-bar-wrap {
    position: fixed;
    bottom: 0; left: 50%;
    transform: translateX(-50%);
    width: 100%;
    max-width: 780px;
    padding: 12px 16px 18px 16px;
    background: linear-gradient(to top, #0c0c0f 80%, transparent);
    z-index: 200;
}

/* Override Streamlit input inside fixed bar */
.input-bar-wrap .stTextInput input {
    background: #14141e !important;
    border: 1.5px solid #24243a !important;
    border-radius: 14px !important;
    color: #e8e6e1 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.92rem !important;
    padding: 13px 18px !important;
}
.input-bar-wrap .stTextInput input:focus {
    border-color: #5b5bd6 !important;
    box-shadow: 0 0 0 3px rgba(91,91,214,0.15) !important;
    outline: none !important;
}
.input-bar-wrap .stButton button {
    background: linear-gradient(135deg, #5b5bd6, #7575e0) !important;
    border: none !important;
    border-radius: 14px !important;
    color: #fff !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.92rem !important;
    font-weight: 600 !important;
    height: 50px !important;
    width: 100% !important;
}

/* Global input/select overrides */
.stTextInput input, .stSelectbox > div > div {
    background: #14141e !important;
    border: 1px solid #22222e !important;
    border-radius: 10px !important;
    color: #e8e6e1 !important;
    font-family: 'DM Sans', sans-serif !important;
}
.stTextInput input:focus {
    border-color: #5b5bd6 !important;
    box-shadow: 0 0 0 3px rgba(91,91,214,0.12) !important;
}
label {
    color: #5a5a72 !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}
.stButton button {
    background: #16161e !important;
    border: 1px solid #24243a !important;
    border-radius: 10px !important;
    color: #8888c8 !important;
    font-family: 'DM Sans', sans-serif !important;
    transition: all 0.15s !important;
}
.stButton button:hover {
    border-color: #5b5bd6 !important;
    color: #c0c0f0 !important;
}

/* Key status banner */
.key-ok {
    background: #0d1f0d; border: 1px solid #1e3a1e;
    border-radius: 10px; padding: 8px 14px;
    font-size: 0.78rem; color: #4caf50; margin-bottom: 16px;
}
.key-warn {
    background: #1c1010; border: 1px solid #3a1e1e;
    border-radius: 10px; padding: 8px 14px;
    font-size: 0.78rem; color: #e57373; margin-bottom: 8px;
}

/* Divider */
.div-line {
    border: none; border-top: 1px solid #1a1a24; margin: 18px 0;
}

/* Slider thumb color */
[data-testid="stSlider"] [data-testid="stThumbValue"] {
    color: #9898d8 !important;
}

/* Collapse sidebar toggle on mobile */
[data-testid="collapsedControl"] { display: none !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 3px; }
::-webkit-scrollbar-thumb { background: #22222e; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ── PDF & Retrieval helpers ───────────────────────────────────────────────────
def extract_text(file_bytes: bytes) -> list[dict]:
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(file_bytes)
        path = tmp.name
    try:
        reader = PdfReader(path)
        pages = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                pages.append({"page": i + 1, "text": text.strip()})
        return pages
    finally:
        os.unlink(path)

def chunk_pages(pages, size=400, overlap=60):
    chunks = []
    for p in pages:
        words = p["text"].split()
        start = 0
        while start < len(words):
            end = min(start + size, len(words))
            chunks.append({"text": " ".join(words[start:end]), "page": p["page"], "id": len(chunks)})
            if end == len(words): break
            start += size - overlap
    return chunks

def build_index(chunks):
    vocab = {}
    for c in chunks:
        for w in c["text"].lower().split():
            if w not in vocab: vocab[w] = len(vocab)
    N = len(chunks)
    df = [0] * len(vocab)
    for c in chunks:
        for w in set(c["text"].lower().split()):
            if w in vocab: df[vocab[w]] += 1
    idf = [math.log((N + 1) / (d + 1)) + 1 for d in df]
    def vec(text):
        words = text.lower().split()
        tf = {}
        for w in words: tf[w] = tf.get(w, 0) + 1
        n = len(words) or 1
        v = [0.0] * len(vocab)
        for w, cnt in tf.items():
            if w in vocab: v[vocab[w]] = (cnt / n) * idf[vocab[w]]
        return v
    vecs = [vec(c["text"]) for c in chunks]
    return vocab, idf, vecs, vec

def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    return dot / ((sum(x**2 for x in a)**0.5) * (sum(x**2 for x in b)**0.5) + 1e-9)

def retrieve(query, chunks, vecs, vec_fn, k=4):
    qv = vec_fn(query)
    scored = sorted(enumerate(vecs), key=lambda x: cosine(qv, x[1]), reverse=True)
    return [chunks[i] for i, _ in scored[:k]]

def clean_output(raw) -> str:
    if isinstance(raw, str):
        raw = raw.strip()
        m = re.search(r"'content':\s*'(.*?)'(?:\s*\}|$)", raw, re.DOTALL)
        if m: return m.group(1).replace("\\n", "\n")
        m = re.search(r'"content":\s*"(.*?)"(?:\s*\}|$)', raw, re.DOTALL)
        if m: return m.group(1).replace("\\n", "\n")
        return raw
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                if "message" in item and "content" in item["message"]:
                    return item["message"]["content"]
                if "generated_text" in item:
                    txt = item["generated_text"]
                    if "assistant" in txt.lower():
                        parts = re.split(r'(?i)assistant\s*[:\n]', txt)
                        if len(parts) > 1:
                            return parts[-1].strip()
                    return txt.strip()
        return str(raw)
    if isinstance(raw, dict):
        return raw.get("content", raw.get("generated_text", str(raw)))
    return str(raw)

def ask(question, context_chunks, history, api_key, model_id):
    context = "\n\n".join(f"[Page {c['page']}]: {c['text']}" for c in context_chunks)
    system = (
        "You are a precise document assistant. Answer ONLY from the provided document context. "
        "Be concise and direct. Cite page numbers inline like (p.2). "
        "If unsure, say so. Never fabricate.\n\nDOCUMENT CONTEXT:\n" + context
    )
    messages = [{"role": "system", "content": system}]
    for m in history[-6:]:
        messages.append({"role": m["role"], "content": m["content"]})
    messages.append({"role": "user", "content": question})
    sdk = Bytez(api_key)
    model = sdk.model(model_id)
    result = model.run(messages)
    if result.error:
        raise RuntimeError(result.error)
    return clean_output(result.output)

# ── Session state ─────────────────────────────────────────────────────────────
for k, v in [("chunks",[]),("vecs",[]),("vec_fn",None),("history",[]),
              ("pdf_name",None),("last_src",[]),("page_count",0),
              ("show_settings", False)]:
    if k not in st.session_state: st.session_state[k] = v

# ── Resolve API key ───────────────────────────────────────────────────────────
secret_key = get_api_key()

# ── NAVBAR ────────────────────────────────────────────────────────────────────
doc_badge = (
    f"<span class='nav-doc'>📄 {st.session_state.pdf_name[:28]}…"
    if st.session_state.pdf_name and len(st.session_state.pdf_name) > 28
    else f"<span class='nav-doc'>📄 {st.session_state.pdf_name}</span>"
    if st.session_state.pdf_name else ""
)
st.markdown(
    f"<div class='navbar'>"
    f"  <div class='nav-brand'>"
    f"    <div class='nav-logo'>✦</div>"
    f"    <span class='nav-title'>DocChat</span>"
    f"  </div>"
    f"  {doc_badge}"
    f"</div>",
    unsafe_allow_html=True,
)

# ── API KEY SECTION ───────────────────────────────────────────────────────────
if secret_key:
    api_key = secret_key
    st.markdown(
        "<div class='key-ok'>🔒 API key loaded from Secrets</div>",
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        "<div class='key-warn'>⚠️ No secret found. Enter your Bytez API key below.</div>",
        unsafe_allow_html=True,
    )
    api_key = st.text_input(
        "Bytez API Key", type="password",
        placeholder="Paste your Bytez API key…",
    )
    with st.expander("💡 How to add as a permanent Secret"):
        st.markdown("""
**Streamlit Community Cloud** → App Settings → Secrets:
```toml
BYTEZ_API_KEY = "your-key-here"
        
