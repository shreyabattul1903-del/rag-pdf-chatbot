import time
from google import genai
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

client = genai.Client()   # reads your key from GEMINI_API_KEY
MODELS = ["gemini-3.5-flash", "gemini-2.5-flash", "gemini-3.7-flash"]

# 1. Read the PDF page by page
def load_pdf(path):
    reader = PdfReader(path)
    return [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]

# 2. Cut the text into small overlapping pieces (chunks)
def chunk_text(pages, size=800, overlap=150):
    chunks = []
    for page_no, text in pages:
        start = 0
        while start < len(text):
            piece = text[start:start + size].strip()
            if piece:
                chunks.append({"page": page_no, "text": piece})
            start += size - overlap
    return chunks

pages = load_pdf("sample.pdf")
chunks = chunk_text(pages)
print(f"Loaded {len(pages)} pages -> {len(chunks)} chunks")

# 3. Turn every chunk into an embedding
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode([c["text"] for c in chunks], normalize_embeddings=True)

# 4. Store the embeddings in a FAISS index so we can search them
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(np.array(embeddings, dtype="float32"))

# 5. Search: find the chunks closest in meaning to a question
def search(query, k=4):
    q = model.encode([query], normalize_embeddings=True)
    scores, ids = index.search(np.array(q, dtype="float32"), k)
    return [(chunks[i], float(s)) for i, s in zip(ids[0], scores[0])]

# 6. Ask the LLM to answer using only the retrieved chunks
def ask_llm(question, retrieved):
    context = "\n\n".join(f"[Page {c['page']}]\n{c['text']}" for c, _ in retrieved)
    prompt = f"""You are a helpful assistant. Answer the question using ONLY the context below.
If the answer is not in the context, reply exactly: "I could not find this in the document."
Mention the page numbers you used.

Context:
{context}

Question: {question}
Answer:"""

    for model_name in MODELS:
        for attempt in range(3):
            try:
                response = client.models.generate_content(model=model_name, contents=prompt)
                return response.text
            except Exception as e:
                print(f"[{model_name}] attempt {attempt + 1} failed: {str(e)[:80]}")
                time.sleep(2 * (attempt + 1))
    return "The AI service is busy right now. Please try again in a minute."

while True:
    question = input("\nAsk a question (or type 'quit'): ")
    if question.lower() == "quit":
        break
    results = search(question)
    print("\nAnswer:\n", ask_llm(question, results))
    print("\nSources: pages", sorted({c["page"] for c, _ in results}))