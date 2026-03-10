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
    layout="wide",
    initial_sidebar_state="expanded",
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
@import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Reset & Base ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, .stApp {
    background: #0c0c0f !important;
    font-family: 'DM Sans', sans-serif;
    color: #e8e6e1;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding: 0 !important;
    max-width: 100% !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #111116 !important;
    border-right: 1px solid #1e1e26 !important;
    padding: 0 !important;
}
[data-testid="stSidebar"] > div:first-child {
    padding: 28px 20px 20px 20px !important;
}

/* Sidebar labels */
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label {
    color: #6b6b7e !important;
    font-size: 0.7rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* Sidebar inputs */
[data-testid="stSidebar"] .stTextInput input,
[data-testid="stSidebar"] .stSelectbox > div > div {
    background: #18181f !important;
    border: 1px solid #2a2a36 !important;
    border-radius: 10px !important;
    color: #e8e6e1 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.88rem !important;
}
[data-testid="stSidebar"] .stTextInput input:focus {
    border-color: #5b5bd6 !important;
    box-shadow: 0 0 0 3px rgba(91,91,214,0.15) !important;
}

/* Slider */
[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] {
    margin-top: 6px !important;
}
[data-testid="stSidebar"] .stSlider [data-testid="stThumbValue"] {
    color: #9898d8 !important;
    font-size: 0.75rem !important;
}

/* File uploader */
[data-testid="stSidebar"] [data-testid="stFileUploader"] {
    background: #18181f !important;
    border: 1.5px dashed #2a2a36 !important;
    border-radius: 12px !important;
    padding: 16px !important;
    transition: border-color 0.2s;
}
[data-testid="stSidebar"] [data-testid="stFileUploader"]:hover {
    border-color: #5b5bd6 !important;
}

/* Sidebar button */
[data-testid="stSidebar"] .stButton button {
    background: #18181f !important;
    border: 1px solid #2a2a36 !important;
    border-radius: 10px !important;
    color: #9898d8 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    width: 100% !important;
    padding: 8px 0 !important;
    transition: all 0.15s !important;
}
[data-testid="stSidebar"] .stButton button:hover {
    background: #22222e !important;
    border-color: #5b5bd6 !important;
    color: #c4c4f0 !important;
}

/* ── Main area ── */
.main-wrap {
    display: flex;
    flex-direction: column;
    height: 100vh;
    padding: 0;
    overflow: hidden;
}

/* Header */
.app-header {
    padding: 22px 36px 18px 36px;
    border-bottom: 1px solid #1a1a22;
    display: flex;
    align-items: center;
    gap: 14px;
    flex-shrink: 0;
    background: #0c0c0f;
}
.app-logo {
    width: 36px;
    height: 36px;
    background: linear-gradient(135deg, #5b5bd6, #9898d8);
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1rem;
    color: white;
    flex-shrink: 0;
}
.app-title {
    font-family: 'Instrument Serif', serif;
    font-size: 1.45rem;
    color: #e8e6e1;
    letter-spacing: -0.02em;
}
.app-doc-badge {
    margin-left: auto;
    background: #18181f;
    border: 1px solid #2a2a36;
    border-radius: 20px;
    padding: 5px 14px;
    font-size: 0.75rem;
    color: #9898d8;
    font-weight: 500;
    max-width: 260px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

/* Chat container */
.chat-scroll {
    flex: 1;
    overflow-y: auto;
    padding: 28px 36px;
    display: flex;
    flex-direction: column;
    gap: 20px;
    scrollbar-width: thin;
    scrollbar-color: #2a2a36 transparent;
}
.chat-scroll::-webkit-scrollbar { width: 4px; }
.chat-scroll::-webkit-scrollbar-thumb { background: #2a2a36; border-radius: 4px; }

/* Message rows */
.msg-row {
    display: flex;
    gap: 12px;
    animation: fadeUp 0.3s ease;
}
.msg-row.user { flex-direction: row-reverse; }

@keyframes fadeUp {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* Avatars */
.avatar {
    width: 32px;
    height: 32px;
    border-radius: 50%;
    flex-shrink: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.78rem;
    font-weight: 700;
    margin-top: 2px;
}
.avatar.ai  { background: linear-gradient(135deg,#5b5bd6,#9898d8); color:#fff; }
.avatar.usr { background: #22222e; border: 1px solid #2a2a36; color: #9898d8; }

/* Bubbles */
.bubble {
    max-width: min(560px, 72vw);
    padding: 13px 18px;
    border-radius: 16px;
    font-size: 0.92rem;
    line-height: 1.65;
    word-break: break-word;
}
.bubble.ai {
    background: #16161e;
    border: 1px solid #22222e;
    color: #dddad4;
    border-radius: 4px 16px 16px 16px;
}
.bubble.user {
    background: linear-gradient(135deg, #5b5bd6, #7b7be8);
    color: #ffffff;
    border-radius: 16px 4px 16px 16px;
}

/* Source pills inside AI bubble */
.source-row {
    display: flex;
    flex-wrap: wrap;
    gap: 5px;
    margin-top: 10px;
    padding-top: 10px;
    border-top: 1px solid #2a2a36;
}
.src-pill {
    background: #1e1e2a;
    border: 1px solid #2e2e40;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.7rem;
    color: #7878c8;
    font-weight: 500;
}

/* Empty states */
.empty-wrap {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 12px;
    padding: 40px;
}
.empty-icon {
    font-size: 2.8rem;
    opacity: 0.25;
}
.empty-title {
    font-family: 'Instrument Serif', serif;
    font-size: 1.5rem;
    color: #3a3a4a;
    text-align: center;
}
.empty-sub {
    font-size: 0.85rem;
    color: #2e2e3e;
    text-align: center;
}

/* Suggestion chips */
.chips-wrap {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    justify-content: center;
    margin-top: 16px;
}
.chip {
    background: #16161e;
    border: 1px solid #22222e;
    border-radius: 20px;
    padding: 7px 16px;
    font-size: 0.8rem;
    color: #6b6b7e;
    cursor: pointer;
    transition: all 0.15s;
}
.chip:hover { border-color: #5b5bd6; color: #9898d8; }

/* Input bar */
.input-bar {
    padding: 16px 36px 20px 36px;
    border-top: 1px solid #1a1a22;
    background: #0c0c0f;
    flex-shrink: 0;
}
.input-bar .stTextInput input {
    background: #14141c !important;
    border: 1.5px solid #22222e !important;
    border-radius: 14px !important;
    color: #e8e6e1 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.92rem !important;
    padding: 12px 18px !important;
    transition: border-color 0.2s, box-shadow 0.2s;
}
.input-bar .stTextInput input:focus {
    border-color: #5b5bd6 !important;
    box-shadow: 0 0 0 3px rgba(91,91,214,0.12) !important;
}
.input-bar .stButton button {
    background: linear-gradient(135deg,#5b5bd6,#7b7be8) !important;
    border: none !important;
    border-radius: 14px !important;
    color: #fff !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.92rem !important;
    font-weight: 600 !important;
    height: 46px !important;
    width: 100% !important;
    transition: opacity 0.15s !important;
}
.input-bar .stButton button:hover { opacity: 0.88 !important; }

/* Dividers in sidebar */
.side-divider {
    border: none;
    border-top: 1px solid #1e1e26;
    margin: 18px 0;
}

/* Stat row */
.stat-row {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    margin-top: 8px;
}
.stat-box {
    flex: 1;
    min-width: 70px;
    background: #18181f;
    border: 1px solid #22222e;
    border-radius: 10px;
    padding: 8px 10px;
    text-align: center;
}
.stat-num {
    font-size: 1.1rem;
    font-weight: 700;
    color: #9898d8;
    font-family: 'Instrument Serif', serif;
}
.stat-lbl {
    font-size: 0.65rem;
    color: #4a4a5e;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin-top: 2px;
}

/* Responsive: mobile */
@media (max-width: 768px) {
    .app-header { padding: 14px 16px; }
    .app-title { font-size: 1.15rem; }
    .app-doc-badge { display: none; }
    .chat-scroll { padding: 16px; gap: 14px; }
    .bubble { max-width: 88vw; font-size: 0.88rem; }
    .input-bar { padding: 12px 14px 16px 14px; }
    .avatar { width: 26px; height: 26px; font-size: 0.65rem; }
}
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
    """Extract clean string from any Bytez output shape."""
    if isinstance(raw, str):
        # Strip dict-like wrappers if model leaked them
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
                    # Strip system/user prompt leakage
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

def make_suggestions(pdf_name: str) -> list[str]:
    return [
        "What is this document about?",
        "Summarize the key points",
        "What are the main conclusions?",
        "List the important terms",
    ]

# ── Session state ─────────────────────────────────────────────────────────────
for k, v in [("chunks",[]),("vecs",[]),("vec_fn",None),("history",[]),
              ("pdf_name",None),("last_src",[]),("page_count",0)]:
    if k not in st.session_state: st.session_state[k] = v

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<div style='font-family:Instrument Serif,serif;font-size:1.3rem;"
        "color:#e8e6e1;margin-bottom:4px'>✦ DocChat</div>"
        "<div style='font-size:0.72rem;color:#3a3a4e;margin-bottom:20px'>"
        "PDF Retrieval-Augmented Generation</div>",
        unsafe_allow_html=True,
    )

    # API key
    secret_key = get_api_key()
    if secret_key:
        api_key = secret_key
        st.markdown(
            "<div style='background:#0f1f0f;border:1px solid #1e3a1e;border-radius:8px;"
            "padding:7px 12px;font-size:0.75rem;color:#4caf50;margin-bottom:14px'>"
            "🔒 API key loaded from Secrets</div>",
            unsafe_allow_html=True,
        )
    else:
        api_key = st.text_input("Bytez API Key", type="password", placeholder="paste key…")
        with st.expander("How to add as Secret"):
            st.markdown("""
**Streamlit Cloud** → Settings → Secrets:
```toml
BYTEZ_API_KEY = "your-key"
```
**HF Spaces** → Settings → Variables & Secrets
            """)

    st.markdown("<hr class='side-divider'>", unsafe_allow_html=True)

    model_id = st.selectbox("Model", MODELS, index=0, label_visibility="visible")
    c1, c2 = st.columns(2)
    with c1: chunk_size = st.slider("Chunk size", 200, 800, 400, 50)
    with c2: top_k = st.slider("Top-k", 2, 8, 4)
    overlap = st.slider("Overlap", 0, 150, 60, 10)

    st.markdown("<hr class='side-divider'>", unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size:0.7rem;color:#6b6b7e;font-weight:600;"
        "letter-spacing:0.1em;text-transform:uppercase;margin-bottom:10px'>"
        "Upload PDF</div>",
        unsafe_allow_html=True,
    )
    uploaded = st.file_uploader("", type="pdf", label_visibility="collapsed")

    if uploaded:
        is_new = uploaded.name != st.session_state.pdf_name
        if is_new or st.button("↺ Re-index"):
            with st.spinner("Indexing…"):
                try:
                    pages = extract_text(uploaded.read())
                    chunks = chunk_pages(pages, chunk_size, overlap)
                    vocab, idf, vecs, vec_fn = build_index(chunks)
                    st.session_state.update({
                        "chunks": chunks, "vecs": vecs, "vec_fn": vec_fn,
                        "pdf_name": uploaded.name, "history": [],
                        "last_src": [], "page_count": len(pages),
                    })
                    st.success(f"✓ Ready")
                except Exception as e:
                    st.error(str(e))

    if st.session_state.chunks:
        st.markdown(
            f"<div class='stat-row'>"
            f"<div class='stat-box'><div class='stat-num'>{st.session_state.page_count}</div>"
            f"<div class='stat-lbl'>Pages</div></div>"
            f"<div class='stat-box'><div class='stat-num'>{len(st.session_state.chunks)}</div>"
            f"<div class='stat-lbl'>Chunks</div></div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    if st.session_state.history:
        st.markdown("<hr class='side-divider'>", unsafe_allow_html=True)
        if st.button("🗑 Clear chat"):
            st.session_state.history = []
            st.session_state.last_src = []
            st.rerun()

# ── Main layout ───────────────────────────────────────────────────────────────
header_col, _ = st.columns([1, 0])
with header_col:
    doc_badge = (
        f"<span class='app-doc-badge'>📄 {st.session_state.pdf_name}</span>"
        if st.session_state.pdf_name else ""
    )
    st.markdown(
        f"<div class='app-header'>"
        f"<div class='app-logo'>✦</div>"
        f"<span class='app-title'>DocChat</span>"
        f"{doc_badge}"
        f"</div>",
        unsafe_allow_html=True,
    )

# Chat area
if not api_key:
    st.markdown(
        "<div class='empty-wrap'>"
        "<div class='empty-icon'>🔑</div>"
        "<div class='empty-title'>Add your Bytez API key</div>"
        "<div class='empty-sub'>Paste it in the sidebar or add it as a Secret</div>"
        "</div>",
        unsafe_allow_html=True,
    )
elif not st.session_state.chunks:
    st.markdown(
        "<div class='empty-wrap'>"
        "<div class='empty-icon'>📄</div>"
        "<div class='empty-title'>Upload a PDF to begin</div>"
        "<div class='empty-sub'>Drag & drop your document in the sidebar</div>"
        "</div>",
        unsafe_allow_html=True,
    )
else:
    # Render messages
    if not st.session_state.history:
        suggestions = make_suggestions(st.session_state.pdf_name)
        chips_html = "".join(f"<div class='chip'>{s}</div>" for s in suggestions)
        st.markdown(
            "<div class='empty-wrap'>"
            "<div class='empty-icon'>💬</div>"
            f"<div class='empty-title'>Ask about {st.session_state.pdf_name[:40]}</div>"
            "<div class='empty-sub'>Try one of these or type your own question</div>"
            f"<div class='chips-wrap'>{chips_html}</div>"
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        chat_html = ""
        for i, msg in enumerate(st.session_state.history):
            is_user = msg["role"] == "user"
            avatar = (
                "<div class='avatar usr'>You</div>"
                if is_user else
                "<div class='avatar ai'>✦</div>"
            )
            bubble_class = "user" if is_user else "ai"
            row_class = "user" if is_user else "ai"
            content = msg["content"].replace("\n", "<br>")

            sources_html = ""
            if not is_user and i == len(st.session_state.history) - 1 and st.session_state.last_src:
                pages = sorted(set(c["page"] for c in st.session_state.last_src))
                pills = "".join(f"<span class='src-pill'>Page {p}</span>" for p in pages)
                sources_html = f"<div class='source-row'>{pills}</div>"

            chat_html += (
                f"<div class='msg-row {row_class}'>"
                f"{avatar}"
                f"<div class='bubble {bubble_class}'>{content}{sources_html}</div>"
                f"</div>"
            )

        st.markdown(
            f"<div class='chat-scroll'>{chat_html}</div>",
            unsafe_allow_html=True,
        )

    # Input bar
    st.markdown("<div class='input-bar'>", unsafe_allow_html=True)
    col_inp, col_btn = st.columns([6, 1])
    with col_inp:
        user_input = st.text_input(
            "", placeholder="Ask anything about your document…",
            label_visibility="collapsed", key="inp"
        )
    with col_btn:
        send = st.button("Send", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if send and user_input.strip():
        q = user_input.strip()
        st.session_state.history.append({"role": "user", "content": q})
        with st.spinner(""):
            try:
                srcs = retrieve(q, st.session_state.chunks, st.session_state.vecs,
                                st.session_state.vec_fn, top_k)
                ans = ask(q, srcs, st.session_state.history[:-1], api_key, model_id)
                st.session_state.history.append({"role": "assistant", "content": ans})
                st.session_state.last_src = srcs
            except Exception as e:
                st.session_state.history.append({"role": "assistant", "content": f"Error: {e}"})
                st.session_state.last_src = []
        st.rerun()
