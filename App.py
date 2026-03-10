import streamlit as st
import os
import tempfile
from pypdf import PdfReader
import math
from bytez import Bytez

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PDF RAG Chat",
    page_icon="📄",
    layout="wide",
)

# ── Load API key: st.secrets first, fallback to sidebar input ─────────────────
def get_api_key_from_secrets() -> str | None:
    try:
        return st.secrets["BYTEZ_API_KEY"]
    except Exception:
        return None

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .stApp { background-color: #0f1117; }

    [data-testid="stSidebar"] {
        background-color: #1a1d27;
        border-right: 1px solid #2e3147;
    }
    .user-bubble {
        background: linear-gradient(135deg, #4f46e5, #7c3aed);
        color: white;
        padding: 12px 18px;
        border-radius: 18px 18px 4px 18px;
        margin: 8px 0 8px 60px;
        font-size: 0.95rem;
        line-height: 1.5;
    }
    .assistant-bubble {
        background-color: #1e2133;
        color: #e2e8f0;
        padding: 12px 18px;
        border-radius: 18px 18px 18px 4px;
        margin: 8px 60px 8px 0;
        font-size: 0.95rem;
        line-height: 1.6;
        border: 1px solid #2e3147;
    }
    .source-badge {
        display: inline-block;
        background-color: #2e3147;
        color: #a5b4fc;
        font-size: 0.72rem;
        padding: 2px 8px;
        border-radius: 10px;
        margin: 4px 4px 0 0;
    }
    .chunk-box {
        background-color: #151826;
        border-left: 3px solid #4f46e5;
        padding: 10px 14px;
        margin: 6px 0;
        border-radius: 0 8px 8px 0;
        font-size: 0.82rem;
        color: #94a3b8;
    }
    .stat-pill {
        display: inline-block;
        background-color: #1e2133;
        border: 1px solid #2e3147;
        color: #a5b4fc;
        font-size: 0.78rem;
        padding: 3px 10px;
        border-radius: 12px;
        margin: 2px;
    }
    .empty-state {
        text-align: center;
        padding: 60px 20px;
        color: #4a5568;
    }
    .secret-banner {
        background-color: #1a2e1a;
        border: 1px solid #2d5a2d;
        border-radius: 8px;
        padding: 8px 14px;
        color: #6fcf6f;
        font-size: 0.8rem;
        margin-bottom: 8px;
    }
    .secret-banner-warn {
        background-color: #2e1a1a;
        border: 1px solid #5a2d2d;
        border-radius: 8px;
        padding: 8px 14px;
        color: #cf6f6f;
        font-size: 0.8rem;
        margin-bottom: 8px;
    }
    h1 { color: #e2e8f0 !important; }
    label { color: #a5b4fc !important; }
</style>
""", unsafe_allow_html=True)

# ── Available models via Bytez ─────────────────────────────────────────────────
MODELS = [
    "openai/gpt-4o",
    "openai/gpt-4o-mini",
    "anthropic/claude-3-5-sonnet-20241022",
    "anthropic/claude-3-haiku-20240307",
    "meta-llama/Llama-3.1-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
]

# ── PDF helpers ───────────────────────────────────────────────────────────────

def extract_text_from_pdf(file_bytes: bytes) -> list[dict]:
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        reader = PdfReader(tmp_path)
        pages = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                pages.append({"page": i + 1, "text": text.strip()})
        return pages
    finally:
        os.unlink(tmp_path)


def chunk_pages(pages: list[dict], chunk_size: int = 500, overlap: int = 50) -> list[dict]:
    chunks = []
    for page in pages:
        words = page["text"].split()
        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunks.append({
                "text": " ".join(words[start:end]),
                "page": page["page"],
                "chunk_id": len(chunks),
            })
            if end == len(words):
                break
            start += chunk_size - overlap
    return chunks


# ── TF-IDF retrieval ──────────────────────────────────────────────────────────

def build_vocab(chunks: list[dict]) -> dict[str, int]:
    vocab: dict[str, int] = {}
    for c in chunks:
        for w in c["text"].lower().split():
            if w not in vocab:
                vocab[w] = len(vocab)
    return vocab


def build_idf(chunks: list[dict], vocab: dict[str, int]) -> list[float]:
    N = len(chunks)
    df = [0] * len(vocab)
    for c in chunks:
        seen = set(c["text"].lower().split())
        for w in seen:
            if w in vocab:
                df[vocab[w]] += 1
    return [math.log((N + 1) / (d + 1)) + 1 for d in df]


def tfidf_vector(text: str, vocab: dict[str, int], idf: list[float]) -> list[float]:
    words = text.lower().split()
    tf: dict[str, float] = {}
    for w in words:
        tf[w] = tf.get(w, 0) + 1
    n = len(words) or 1
    vec = [0.0] * len(vocab)
    for w, count in tf.items():
        if w in vocab:
            vec[vocab[w]] = (count / n) * idf[vocab[w]]
    return vec


def embed_chunks(chunks: list[dict]) -> tuple[dict, list[float], list[list[float]]]:
    vocab = build_vocab(chunks)
    idf = build_idf(chunks, vocab)
    embeddings = [tfidf_vector(c["text"], vocab, idf) for c in chunks]
    return vocab, idf, embeddings


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = sum(x ** 2 for x in a) ** 0.5
    mag_b = sum(x ** 2 for x in b) ** 0.5
    return dot / (mag_a * mag_b + 1e-9)


def retrieve_chunks(
    query: str,
    chunks: list[dict],
    vocab: dict,
    idf: list[float],
    embeddings: list[list[float]],
    top_k: int = 4,
) -> list[dict]:
    q_vec = tfidf_vector(query, vocab, idf)
    scored = [(cosine_similarity(q_vec, emb), chunk) for emb, chunk in zip(embeddings, chunks)]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [chunk for _, chunk in scored[:top_k]]


# ── LLM via Bytez ─────────────────────────────────────────────────────────────

def ask_llm(
    question: str,
    context_chunks: list[dict],
    history: list[dict],
    api_key: str,
    model_id: str,
) -> str:
    context_text = "\n\n---\n\n".join(
        f"[Page {c['page']}] {c['text']}" for c in context_chunks
    )
    system_content = (
        "You are a helpful assistant that answers questions strictly based on the provided PDF context. "
        "If the answer is not in the context, say so clearly. "
        "Always cite the page number(s) you used.\n\n"
        f"CONTEXT:\n{context_text}"
    )
    messages = [{"role": "system", "content": system_content}]
    for msg in history[-6:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": question})

    sdk = Bytez(api_key)
    model = sdk.model(model_id)
    result = model.run(messages)

    if result.error:
        raise RuntimeError(result.error)

    output = result.output
    if isinstance(output, list):
        for item in output:
            if isinstance(item, dict):
                if "message" in item and "content" in item["message"]:
                    return item["message"]["content"]
                if "generated_text" in item:
                    return item["generated_text"]
        return str(output)
    return str(output)


# ── Session state ─────────────────────────────────────────────────────────────
for key, default in [
    ("chunks", []),
    ("embeddings", []),
    ("vocab", {}),
    ("idf", []),
    ("chat_history", []),
    ("pdf_name", None),
    ("last_sources", []),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Resolve API key (secrets > sidebar input) ─────────────────────────────────
secret_api_key = get_api_key_from_secrets()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    # Show key source status
    if secret_api_key:
        st.markdown(
            '<div class="secret-banner">🔒 API key loaded from Secrets</div>',
            unsafe_allow_html=True,
        )
        api_key = secret_api_key
    else:
        st.markdown(
            '<div class="secret-banner-warn">⚠️ No secret found — enter key manually</div>',
            unsafe_allow_html=True,
        )
        api_key = st.text_input(
            "Bytez API Key",
            type="password",
            placeholder="Paste your Bytez key here…",
            help="To avoid entering this every time, add BYTEZ_API_KEY to your deployment secrets.",
        )

    model_id = st.selectbox("Model", MODELS, index=0)
    top_k = st.slider("Chunks to retrieve (top-k)", 2, 8, 4)
    chunk_size = st.slider("Chunk size (words)", 200, 1000, 500, step=50)
    overlap = st.slider("Chunk overlap (words)", 0, 200, 50, step=10)

    st.markdown("---")
    st.markdown("## 📂 Upload PDF")

    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file:
        new_pdf = uploaded_file.name != st.session_state.pdf_name
        if new_pdf or st.button("🔄 Re-index PDF"):
            with st.spinner("Extracting & indexing PDF…"):
                try:
                    pages = extract_text_from_pdf(uploaded_file.read())
                    chunks = chunk_pages(pages, chunk_size=chunk_size, overlap=overlap)
                    vocab, idf, embeddings = embed_chunks(chunks)
                    st.session_state.chunks = chunks
                    st.session_state.embeddings = embeddings
                    st.session_state.vocab = vocab
                    st.session_state.idf = idf
                    st.session_state.pdf_name = uploaded_file.name
                    st.session_state.chat_history = []
                    st.session_state.last_sources = []
                    st.success(f"✅ Indexed {len(chunks)} chunks from {len(pages)} pages")
                except Exception as e:
                    st.error(f"Error: {e}")

    if st.session_state.chunks:
        st.markdown("---")
        st.markdown("### 📊 Index Stats")
        st.markdown(
            f'<span class="stat-pill">📄 {st.session_state.pdf_name}</span>'
            f'<span class="stat-pill">🧩 {len(st.session_state.chunks)} chunks</span>',
            unsafe_allow_html=True,
        )

    if st.session_state.chat_history:
        if st.button("🗑️ Clear chat"):
            st.session_state.chat_history = []
            st.session_state.last_sources = []
            st.rerun()

    # ── How to add secrets (collapsible) ──────────────────────────────────────
    with st.expander("📖 How to add your API key as a Secret"):
        st.markdown("""
**Streamlit Community Cloud**
1. Go to your app dashboard
2. Click ⋮ → **Settings → Secrets**
3. Add:
```toml
BYTEZ_API_KEY = "your-key-here"
```

**Hugging Face Spaces**
1. Go to your Space → **Settings**
2. Scroll to **Variables and Secrets**
3. Add secret name `BYTEZ_API_KEY`

**Local `.streamlit/secrets.toml`**
```toml
BYTEZ_API_KEY = "your-key-here"
```
        """)

# ── Main area ─────────────────────────────────────────────────────────────────
st.markdown(
    "# 📄 PDF RAG Chat &nbsp;<small style='font-size:0.5em;color:#6366f1'>powered by Bytez</small>",
    unsafe_allow_html=True,
)

if not api_key:
    st.info("👈 Add your Bytez API key — either as a Secret (recommended) or in the sidebar.")
elif not st.session_state.chunks:
    st.markdown(
        '<div class="empty-state">'
        '<div style="font-size:3rem">📂</div>'
        '<div style="font-size:1.1rem;margin-top:12px">Upload a PDF in the sidebar to start chatting</div>'
        '</div>',
        unsafe_allow_html=True,
    )
else:
    with st.container():
        if not st.session_state.chat_history:
            st.markdown(
                '<div class="empty-state">'
                '<div style="font-size:2.5rem">💬</div>'
                f'<div style="font-size:1rem;margin-top:8px">Ask anything about <b>{st.session_state.pdf_name}</b></div>'
                '</div>',
                unsafe_allow_html=True,
            )
        else:
            for i, msg in enumerate(st.session_state.chat_history):
                if msg["role"] == "user":
                    st.markdown(f'<div class="user-bubble">{msg["content"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="assistant-bubble">{msg["content"]}</div>', unsafe_allow_html=True)
                    if i == len(st.session_state.chat_history) - 1 and st.session_state.last_sources:
                        with st.expander("📎 Retrieved chunks", expanded=False):
                            for chunk in st.session_state.last_sources:
                                st.markdown(
                                    f'<div class="chunk-box">'
                                    f'<span class="source-badge">Page {chunk["page"]}</span><br>'
                                    f'{chunk["text"][:300]}{"…" if len(chunk["text"]) > 300 else ""}'
                                    f'</div>',
                                    unsafe_allow_html=True,
                                )

    st.markdown("---")
    col1, col2 = st.columns([5, 1])
    with col1:
        user_input = st.text_input(
            "Ask a question",
            placeholder="e.g. What are the main conclusions of this document?",
            label_visibility="collapsed",
            key="user_input",
        )
    with col2:
        send = st.button("Send ➤", use_container_width=True)

    if send and user_input.strip():
        question = user_input.strip()
        st.session_state.chat_history.append({"role": "user", "content": question})

        with st.spinner("Thinking…"):
            try:
                sources = retrieve_chunks(
                    question,
                    st.session_state.chunks,
                    st.session_state.vocab,
                    st.session_state.idf,
                    st.session_state.embeddings,
                    top_k=top_k,
                )
                answer = ask_llm(
                    question,
                    sources,
                    st.session_state.chat_history[:-1],
                    api_key,
                    model_id,
                )
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
                st.session_state.last_sources = sources
            except Exception as e:
                st.session_state.chat_history.append({"role": "assistant", "content": f"⚠️ Error: {e}"})
                st.session_state.last_sources = []

        st.rerun()
