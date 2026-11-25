import re
import time
import streamlit as st
from google import genai
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

client = genai.Client()   # reads your key from GEMINI_API_KEY
MODELS = ["gemini-3.1-flash-lite", "gemini-3.5-flash-lite", "gemini-3.8-flash"]

STOP_WORDS = {
    "the", "and", "how", "what", "does", "do", "is", "are", "an", "of", "in",
    "to", "on", "with", "work", "works", "about", "explain", "tell", "me",
    "can", "you", "use", "used", "why", "when", "which", "give", "define",
}


@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")


def load_pdf(file):
    reader = PdfReader(file)
    return [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]


def chunk_text(pages, size=800, overlap=150):
    chunks = []
    for page_no, text in pages:
        start = 0
        while start < len(text):
            piece = text[start:start + size].strip()
            is_contents = piece.count(". .") >= 5   # skip table-of-contents lines
            too_short = len(piece) < 200            # skip titles, footers, tiny leftovers
            if piece and not is_contents and not too_short:
                chunks.append({"page": page_no, "text": piece})
            start += size - overlap
    return chunks


def stem(word):
    return word[:-1] if len(word) > 3 and word.endswith("s") else word


def tokens(text):
    return {stem(w) for w in re.findall(r"[a-z0-9]+", text.lower())}


def keywords(query):
    words = re.findall(r"[a-z0-9]+", query.lower())
    return [stem(w) for w in words if w not in STOP_WORDS and len(w) > 1]


def build_index(chunks, embedder):
    embeddings = embedder.encode([c["text"] for c in chunks], normalize_embeddings=True)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(np.array(embeddings, dtype="float32"))
    return index


def search(query, embedder, index, chunks, chunk_tokens, k=6):
    # 1. meaning similarity for every chunk
    q = embedder.encode([query], normalize_embeddings=True)
    scores, ids = index.search(np.array(q, dtype="float32"), len(chunks))
    sim = np.zeros(len(chunks))
    sim[ids[0]] = scores[0]
    # 2. keyword bonus: how many of the question's keywords appear in the chunk
    kws = keywords(query)
    final = []
    for i in range(len(chunks)):
        kw_score = sum(1 for w in kws if w in chunk_tokens[i]) / len(kws) if kws else 0.0
        final.append(sim[i] + 0.3 * kw_score)
    top = np.argsort(final)[::-1][:k]
    return [(chunks[i], float(final[i])) for i in top]


def ask_llm(question, retrieved):
    context = "\n\n".join(f"[Page {c['page']}]\n{c['text']}" for c, _ in retrieved)
    prompt = f"""You are a helpful assistant. Answer the question using ONLY the context below.
The context comes from a PDF, so code and spacing may look messy. Read it carefully.
If the document uses a different word for the same idea (for example "array" for "list"), still answer from it.
If the answer is not in the context, reply exactly: "I could not find this in the document."
Give a clear explanation in 3 to 5 sentences, with a short example if the context has one.
Mention the page numbers you used.

Context:
{context}

Question: {question}
Answer:"""
    for model_name in MODELS:
        for attempt in range(2):
            try:
                response = client.models.generate_content(model=model_name, contents=prompt)
                return response.text
            except Exception as e:
                print(f"[{model_name}] attempt {attempt + 1} failed: {str(e)[:80]}")
                time.sleep(2)
    return "The AI service is busy right now. Please try again in a minute."


st.set_page_config(page_title="PDF Q&A Chatbot", page_icon="📄")
st.title("📄 PDF Q&A Chatbot (RAG)")
st.caption("Upload a PDF, ask a question, and get an answer with page citations.")

embedder = load_embedder()
uploaded = st.file_uploader("Upload a text-based PDF", type="pdf")

if uploaded:
    key = f"{uploaded.name}-{uploaded.size}-v3"
    if st.session_state.get("doc_key") != key:
        with st.spinner("Reading and indexing the PDF..."):
            pages = load_pdf(uploaded)
            chunks = chunk_text(pages)
            if not chunks:
                st.error("No text found. This may be a scanned PDF. Try a text-based one.")
                st.stop()
            index = build_index(chunks, embedder)
            chunk_tokens = [tokens(c["text"]) for c in chunks]
            st.session_state.update(
                doc_key=key, chunks=chunks, index=index,
                chunk_tokens=chunk_tokens, n_pages=len(pages),
            )

    st.success(f"Ready: {st.session_state.n_pages} pages, {len(st.session_state.chunks)} chunks indexed.")
    question = st.text_input("Ask a question about this document")

    if question:
        with st.spinner("Thinking..."):
            results = search(
                question, embedder, st.session_state.index,
                st.session_state.chunks, st.session_state.chunk_tokens,
            )
            answer = ask_llm(question, results)
        st.subheader("Answer")
        st.write(answer)
        pages_used = sorted({c["page"] for c, _ in results})
        st.subheader("Sources")
        st.write("Pages searched: " + ", ".join(str(p) for p in pages_used))
        for chunk, score in results:
            with st.expander(f"Page {chunk['page']} (relevance {score:.2f})"):
                st.write(chunk["text"])