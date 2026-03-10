import streamlit as st
import os
import tempfile
import math
import re
from pypdf import PdfReader
from bytez import Bytez

st.set_page_config(
    page_title="DocChat",
    page_icon="✦",
    layout="centered",
    initial_sidebar_state="collapsed",
)

def get_api_key():
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

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Instrument+Serif&family=DM+Sans:wght@300;400;500;600&display=swap');

*, *::before, *::after { box-sizing: border-box; }
html, body, .stApp { background: #0c0c0f !important; font-family: 'DM Sans', sans-serif; color: #e8e6e1; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 16px 140px 16px !important; max-width: 780px !important; }

.navbar {
    display: flex; align-items: center; justify-content: space-between;
    padding: 18px 0 14px 0; border-bottom: 1px solid #1a1a24;
    margin-bottom: 20px; position: sticky; top: 0;
    background: #0c0c0f; z-index: 100;
}
.nav-brand { display: flex; align-items: center; gap: 10px; }
.nav-logo {
    width: 34px; height: 34px;
    background: linear-gradient(135deg, #5b5bd6, #9898d8);
    border-radius: 10px; display: flex; align-items: center;
    justify-content: center; font-size: 1rem; color: white; font-weight: 700;
}
.nav-title { font-family: 'Instrument Serif', serif; font-size: 1.3rem; color: #e8e6e1; }
.nav-doc {
    background: #16161e; border: 1px solid #22222e; border-radius: 20px;
    padding: 4px 12px; font-size: 0.72rem; color: #7878c8;
    max-width: 180px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}

.stat-row { display: flex; gap: 8px; margin: 12px 0; }
.stat-box { flex: 1; background: #16161e; border: 1px solid #22222e; border-radius: 12px; padding: 10px; text-align: center; }
.stat-num { font-family: 'Instrument Serif', serif; font-size: 1.3rem; color: #9898d8; }
.stat-lbl { font-size: 0.62rem; color: #44445a; text-transform: uppercase; letter-spacing: 0.08em; }

.msg-row { display: flex; gap: 10px; margin-bottom: 16px; animation: fadeUp 0.25s ease; }
.msg-row.user-row { flex-direction: row-reverse; }
@keyframes fadeUp { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }

.avatar { width: 30px; height: 30px; border-radius: 50%; flex-shrink: 0; display: flex; align-items: center; justify-content: center; font-size: 0.7rem; font-weight: 700; margin-top: 2px; }
.avatar.ai-av { background: linear-gradient(135deg,#5b5bd6,#9898d8); color:#fff; }
.avatar.usr-av { background: #1e1e2c; border:1px solid #2a2a3e; color:#7878c8; }

.bubble { max-width: min(520px, 78vw); padding: 12px 16px; font-size: 0.91rem; line-height: 1.65; word-break: break-word; }
.bubble.ai-bub { background: #14141e; border: 1px solid #20202e; color: #d8d5d0; border-radius: 4px 16px 16px 16px; }
.bubble.usr-bub { background: linear-gradient(135deg, #5b5bd6, #7575e0); color: #fff; border-radius: 16px 4px 16px 16px; }

.src-row { display: flex; flex-wrap: wrap; gap: 5px; margin-top: 10px; padding-top: 10px; border-top: 1px solid #252535; }
.src-pill { background: #1c1c2a; border: 1px solid #2c2c40; border-radius: 20px; padding: 2px 9px; font-size: 0.68rem; color: #7070b8; font-weight: 500; }

.empty-wrap { display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 48px 20px; gap: 10px; text-align: center; }
.empty-icon { font-size: 2.6rem; opacity: 0.3; }
.empty-title { font-family: 'Instrument Serif', serif; font-size: 1.4rem; color: #35354a; }
.empty-sub { font-size: 0.82rem; color: #2e2e40; }
.chips-wrap { display: flex; flex-wrap: wrap; gap: 8px; justify-content: center; margin-top: 14px; }
.chip { background: #14141e; border: 1px solid #22222e; border-radius: 20px; padding: 7px 15px; font-size: 0.78rem; color: #5a5a72; }

.input-bar-wrap {
    position: fixed; bottom: 0; left: 50%; transform: translateX(-50%);
    width: 100%; max-width: 780px; padding: 12px 16px 20px 16px;
    background: linear-gradient(to top, #0c0c0f 80%, transparent); z-index: 200;
}

.stTextInput input { background: #14141e !important; border: 1.5px solid #24243a !important; border-radius: 14px !important; color: #e8e6e1 !important; font-family: 'DM Sans', sans-serif !important; font-size: 0.92rem !important; padding: 13px 18px !important; }
.stTextInput input:focus { border-color: #5b5bd6 !important; box-shadow: 0 0 0 3px rgba(91,91,214,0.15) !important; }
.stSelectbox > div > div { background: #14141e !important; border: 1px solid #22222e !important; border-radius: 10px !important; color: #e8e6e1 !important; }

label { color: #5a5a72 !important; font-size: 0.72rem !important; font-weight: 600 !important; text-transform: uppercase !important; letter-spacing: 0.08em !important; }
.stButton button { background: #16161e !important; border: 1px solid #24243a !important; border-radius: 10px !important; color: #8888c8 !important; font-family: 'DM Sans', sans-serif !important; transition: all 0.15s !important; }
.stButton button:hover { border-color: #5b5bd6 !important; color: #c0c0f0 !important; }

.key-ok { background: #0d1f0d; border: 1px solid #1e3a1e; border-radius: 10px; padding: 8px 14px; font-size: 0.78rem; color: #4caf50; margin-bottom: 16px; }
.key-warn { background: #1c1010; border: 1px solid #3a1e1e; border-radius: 10px; padding: 8px 14px; font-size: 0.78rem; color: #e57373; margin-bottom: 8px; }
.div-line { border: none; border-top: 1px solid #1a1a24; margin: 18px 0; }
[data-testid="collapsedControl"] { display: none !important; }
::-webkit-scrollbar { width: 3px; }
::-webkit-scrollbar-thumb { background: #22222e; border-radius: 3px; }
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)


def extract_text(file_bytes):
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
            if end == len(words):
                break
            start += size - overlap
    return chunks


def build_index(chunks):
    vocab = {}
    for c in chunks:
        for w in c["text"].lower().split():
            if w not in vocab:
                vocab[w] = len(vocab)
    N = len(chunks)
    df = [0] * len(vocab)
    for c in chunks:
        for w in set(c["text"].lower().split()):
            if w in vocab:
                df[vocab[w]] += 1
    idf = [math.log((N + 1) / (d + 1)) + 1 for d in df]

    def vec(text):
        words = text.lower().split()
        tf = {}
        for w in words:
            tf[w] = tf.get(w, 0) + 1
        n = len(words) or 1
        v = [0.0] * len(vocab)
        for w, cnt in tf.items():
            if w in vocab:
                v[vocab[w]] = (cnt / n) * idf[vocab[w]]
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


def clean_output(raw):
    if isinstance(raw, str):
        raw = raw.strip()
        m = re.search(r"'content':\s*'(.*?)'(?:\s*\}|$)", raw, re.DOTALL)
        if m:
            return m.group(1).replace("\\n", "\n")
        m = re.search(r'"content":\s*"(.*?)"(?:\s*\}|$)', raw, re.DOTALL)
        if m:
            return m.group(1).replace("\\n", "\n")
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


# Session state
for k, v in [("chunks", []), ("vecs", []), ("vec_fn", None), ("history", []),
              ("pdf_name", None), ("last_src", []), ("page_count", 0)]:
    if k not in st.session_state:
        st.session_state[k] = v

# Resolve API key
secret_key = get_api_key()

# Navbar
doc_badge = ""
if st.session_state.pdf_name:
    name = st.session_state.pdf_name
    display = (name[:26] + "…") if len(name) > 26 else name
    doc_badge = f"<span class='nav-doc'>📄 {display}</span>"

st.markdown(
    f"<div class='navbar'>"
    f"<div class='nav-brand'>"
    f"<div class='nav-logo'>✦</div>"
    f"<span class='nav-title'>DocChat</span>"
    f"</div>"
    f"{doc_badge}"
    f"</div>",
    unsafe_allow_html=True,
)

# API key
if secret_key:
    api_key = secret_key
    st.markdown("<div class='key-ok'>🔒 API key loaded from Secrets</div>", unsafe_allow_html=True)
else:
    st.markdown("<div class='key-warn'>⚠️ No secret found. Enter your Bytez API key below.</div>", unsafe_allow_html=True)
    api_key = st.text_input("Bytez API Key", type="password", placeholder="Paste your Bytez API key…")
    with st.expander("💡 How to add as a permanent Secret"):
        st.markdown("""
**Streamlit Community Cloud** → App Settings → Secrets:
```toml
BYTEZ_API_KEY = "your-key-here"
Hugging Face Spaces → Settings → Variables and Secrets → add BYTEZ_API_KEY
""")
st.markdown("", unsafe_allow_html=True)
Upload
st.markdown(
"📂 Upload your PDF",
unsafe_allow_html=True,
)
uploaded = st.file_uploader("Choose a PDF", type="pdf", label_visibility="collapsed")
Settings
with st.expander("⚙️ Settings", expanded=False):
col1, col2 = st.columns(2)
with col1:
model_id = st.selectbox("Model", MODELS, index=0)
chunk_size = st.slider("Chunk size", 200, 800, 400, 50)
with col2:
top_k = st.slider("Top-k chunks", 2, 8, 4)
overlap = st.slider("Overlap", 0, 150, 60, 10)
Index PDF
if uploaded:
is_new = uploaded.name != st.session_state.pdf_name
col_a, col_b = st.columns([3, 1])
with col_b:
reindex = st.button("↺ Re-index", use_container_width=True)
if is_new or reindex:
with st.spinner("Extracting & indexing…"):
try:
pages = extract_text(uploaded.read())
chunks = chunk_pages(pages, chunk_size, overlap)
vocab, idf, vecs, vec_fn = build_index(chunks)
st.session_state.update({
"chunks": chunks, "vecs": vecs, "vec_fn": vec_fn,
"pdf_name": uploaded.name, "history": [],
"last_src": [], "page_count": len(pages),
})
st.rerun()
except Exception as e:
st.error(str(e))
Stats
if st.session_state.chunks:
st.markdown(
f""
f"{st.session_state.page_count}Pages"
f"{len(st.session_state.chunks)}Chunks"
f"{top_k}Top-k"
f"",
unsafe_allow_html=True,
)
if st.session_state.history:
if st.button("🗑 Clear chat"):
st.session_state.history = []
st.session_state.last_src = []
st.rerun()
st.markdown("", unsafe_allow_html=True)
Chat area
if not api_key:
st.markdown(
"🔑"
"Enter your API key above"
"Paste your Bytez key to get started",
unsafe_allow_html=True,
)
elif not st.session_state.chunks:
st.markdown(
"📄"
"Upload a PDF above"
"Your document will be indexed automatically",
unsafe_allow_html=True,
)
else:
if not st.session_state.history:
chips = ["What is this about?", "Summarize key points", "What are the conclusions?", "List important terms"]
chips_html = "".join(f"{c}" for c in chips)
st.markdown(
"💬"
"Ask about your document"
"Try a suggestion or type your own"
f"{chips_html}",
unsafe_allow_html=True,
)
else:
chat_html = ""
for i, msg in enumerate(st.session_state.history):
is_user = msg["role"] == "user"
avatar_cls = "usr-av" if is_user else "ai-av"
avatar_txt = "You" if is_user else "✦"
bubble_cls = "usr-bub" if is_user else "ai-bub"
row_cls = "user-row" if is_user else ""
content = msg["content"].replace("\n", "")
sources_html = ""
if not is_user and i == len(st.session_state.history) - 1 and st.session_state.last_src:
pages_used = sorted(set(c["page"] for c in st.session_state.last_src))
pills = "".join(f"p.{p}" for p in pages_used)
sources_html = f"{pills}"
chat_html += (
f""
f"{avatar_txt}"
f"{content}{sources_html}"
f""
)
st.markdown(chat_html, unsafe_allow_html=True)
# Input bar
st.markdown("<div class='input-bar-wrap'>", unsafe_allow_html=True)
col_inp, col_btn = st.columns([5, 1])
with col_inp:
    user_input = st.text_input(
        "msg", placeholder="Ask anything about your document…",
        label_visibility="collapsed", key="inp",
    )
with col_btn:
    send = st.button("↑", use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

if send and user_input.strip():
    q = user_input.strip()
    st.session_state.history.append({"role": "user", "content": q})
    with st.spinner("Thinking…"):
        try:
            srcs = retrieve(q, st.session_state.chunks, st.session_state.vecs, st.session_state.vec_fn, top_k)
            ans = ask(q, srcs, st.session_state.history[:-1], api_key, model_id)
            st.session_state.history.append({"role": "assistant", "content": ans})
            st.session_state.last_src = srcs
        except Exception as e:
            st.session_state.history.append({"role": "assistant", "content": f"⚠️ {e}"})
            st.session_state.last_src = []
    st.rerun()
