import streamlit as st
import os
import tempfile
import math
import json
from pypdf import PdfReader
from docx import Document
from bytez import Bytez


st.set_page_config(
    page_title="DocChat",
    page_icon="📄",
    layout="centered"
)


# ---------------- API KEY ----------------
def get_api_key():
    try:
        return st.secrets["BYTEZ_API_KEY"]
    except Exception:
        return None


# ---------------- MODELS ----------------
MODELS = [
    "openai/gpt-4o",
    "openai/gpt-4o-mini",
    "anthropic/claude-3-5-sonnet-20241022",
    "anthropic/claude-3-haiku-20240307",
    "meta-llama/Llama-3.1-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
]


# ---------------- CLEAN OUTPUT ----------------
def clean_output(raw):

    if isinstance(raw, str):

        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict) and "content" in parsed:
                return parsed["content"]
        except Exception:
            pass

        if "'content':" in raw:
            try:
                return raw.split("'content':")[1].split("'}")[0].strip(" '")
            except Exception:
                pass

        return raw

    if isinstance(raw, dict):

        if "content" in raw:
            return raw["content"]

        if "generated_text" in raw:
            return raw["generated_text"]

    if isinstance(raw, list):

        for item in raw:

            if isinstance(item, dict):

                if "message" in item:
                    return item["message"]["content"]

                if "generated_text" in item:
                    return item["generated_text"]

    return str(raw)


# ---------------- PDF EXTRACTION ----------------
def extract_pdf(file_bytes):

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        path = tmp.name

    pages = []

    try:

        reader = PdfReader(path)

        for i, page in enumerate(reader.pages):

            text = page.extract_text() or ""

            if text.strip():

                pages.append({
                    "page": i + 1,
                    "text": text.strip()
                })

        return pages

    finally:
        os.unlink(path)


# ---------------- DOCX EXTRACTION ----------------
def extract_docx(file_bytes):

    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
        tmp.write(file_bytes)
        path = tmp.name

    pages = []

    try:

        doc = Document(path)

        text = "\n".join(
            p.text for p in doc.paragraphs if p.text.strip()
        )

        pages.append({
            "page": 1,
            "text": text
        })

        return pages

    finally:
        os.unlink(path)


# ---------------- CHUNKING ----------------
def chunk_pages(pages, size=400, overlap=60):

    chunks = []

    for p in pages:

        words = p["text"].split()

        start = 0

        while start < len(words):

            end = min(start + size, len(words))

            chunks.append({
                "text": " ".join(words[start:end]),
                "page": p["page"],
                "id": len(chunks)
            })

            if end == len(words):
                break

            start += size - overlap

    return chunks


# ---------------- VECTOR INDEX ----------------
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

    return vocab, vecs, vec


# ---------------- COSINE ----------------
def cosine(a, b):

    dot = sum(x*y for x, y in zip(a, b))

    norm_a = sum(x**2 for x in a) ** 0.5
    norm_b = sum(x**2 for x in b) ** 0.5

    return dot / (norm_a * norm_b + 1e-9)


# ---------------- RETRIEVAL ----------------
def retrieve(query, chunks, vecs, vec_fn, k=4):

    qv = vec_fn(query)

    scored = sorted(
        enumerate(vecs),
        key=lambda x: cosine(qv, x[1]),
        reverse=True
    )

    return [chunks[i] for i, _ in scored[:k]]


# ---------------- MODEL CALL ----------------
def ask(question, context_chunks, history, api_key, model_id):

    context = "\n\n".join(
        f"(Page {c['page']}) {c['text']}"
        for c in context_chunks
    )

    system_prompt = f"""
You are a document analysis assistant.

Follow these rules strictly:

1. Answer ONLY using the provided document context.
2. If the answer is not in the document say:
   "The document does not contain that information."
3. Be concise and accurate.
4. When possible cite page numbers like (Page 2).
5. Use bullet points for lists.

DOCUMENT CONTEXT:
{context}
"""

    messages = [{"role": "system", "content": system_prompt}]

    for m in history[-6:]:
        messages.append(m)

    messages.append({"role": "user", "content": question})

    try:

        sdk = Bytez(api_key)

        model = sdk.model(model_id)

        result = model.run(messages)

        if result.error:
            return f"Model error: {result.error}"

        return clean_output(result.output)

    except Exception as e:

        return f"API error: {str(e)}"


# ---------------- SESSION ----------------
if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "vecs" not in st.session_state:
    st.session_state.vecs = []

if "vec_fn" not in st.session_state:
    st.session_state.vec_fn = None

if "history" not in st.session_state:
    st.session_state.history = []

if "file_name" not in st.session_state:
    st.session_state.file_name = None


# ---------------- UI ----------------
st.title("📄 DocChat")


secret_key = get_api_key()

if secret_key:
    api_key = secret_key
else:
    api_key = st.text_input("Bytez API Key", type="password")


uploaded = st.file_uploader(
    "Upload PDF or DOCX",
    type=["pdf", "docx"]
)


with st.expander("Settings"):

    model_id = st.selectbox("Model", MODELS)

    chunk_size = st.slider("Chunk Size", 200, 800, 400)

    overlap = st.slider("Overlap", 0, 150, 60)

    top_k = st.slider("Top Chunks", 2, 8, 4)


# ---------------- INDEX ----------------
if uploaded:

    is_new = uploaded.name != st.session_state.file_name

    if is_new:

        with st.spinner("Indexing document..."):

            file_bytes = uploaded.read()

            if uploaded.name.endswith(".pdf"):
                pages = extract_pdf(file_bytes)
            else:
                pages = extract_docx(file_bytes)

            chunks = chunk_pages(pages, chunk_size, overlap)

            vocab, vecs, vec_fn = build_index(chunks)

            st.session_state.chunks = chunks
            st.session_state.vecs = vecs
            st.session_state.vec_fn = vec_fn
            st.session_state.file_name = uploaded.name
            st.session_state.history = []

        st.success("Document indexed")


# ---------------- CHAT ----------------
if st.session_state.chunks:

    for msg in st.session_state.history:

        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input("Ask about the document")

    if prompt:

        st.session_state.history.append({
            "role": "user",
            "content": prompt
        })

        with st.spinner("Thinking..."):

            srcs = retrieve(
                prompt,
                st.session_state.chunks,
                st.session_state.vecs,
                st.session_state.vec_fn,
                top_k
            )

            answer = ask(
                prompt,
                srcs,
                st.session_state.history[:-1],
                api_key,
                model_id
            )

        st.session_state.history.append({
            "role": "assistant",
            "content": answer
        })

        st.rerun()
