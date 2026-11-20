import os
import joblib
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.file_parser import extract_text_from_file
from src.preprocess import clean_text

app = FastAPI(title="SemantiCheck 3D Studio")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load NLP Models once at startup
sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
tfidf_vectorizer = (
    joblib.load("models/tfidf_vectorizer.pkl")
    if os.path.exists("models/tfidf_vectorizer.pkl")
    else None
)


class UploadWrapper:
    """Adapts FastAPI UploadFile to our existing extract_text_from_file helper."""
    def __init__(self, filename: str, content: bytes):
        self.name = filename
        self._content = content
        self._offset = 0

    def read(self):
        return self._content

    def seek(self, pos: int):
        self._offset = pos


def chunk_text(text: str, chunk_size: int = 350) -> list[str]:
    words = text.split()
    if len(words) <= chunk_size:
        return [text]
    step = chunk_size - 50
    return [" ".join(words[i : i + chunk_size]) for i in range(0, len(words), step)]


def compute_semantic_similarity(text1: str, text2: str) -> float:
    chunks1 = chunk_text(text1)
    chunks2 = chunk_text(text2)
    emb1 = sbert_model.encode(chunks1, normalize_embeddings=True)
    emb2 = sbert_model.encode(chunks2, normalize_embeddings=True)
    sim_matrix = cosine_similarity(emb1, emb2)
    score = 0.5 * (np.max(sim_matrix, axis=1).mean() + np.max(sim_matrix, axis=0).mean())
    return float(max(0.0, min(1.0, score)))


def compute_tfidf_similarity(text1: str, text2: str) -> float:
    c1, c2 = clean_text(text1), clean_text(text2)
    if not c1 or not c2:
        return 0.0
    if tfidf_vectorizer is not None:
        v1, v2 = tfidf_vectorizer.transform([c1]), tfidf_vectorizer.transform([c2])
    else:
        temp_vec = TfidfVectorizer().fit([c1, c2])
        v1, v2 = temp_vec.transform([c1]), temp_vec.transform([c2])
    return float(max(0.0, min(1.0, cosine_similarity(v1, v2)[0][0])))


@app.get("/")
def serve_home():
    return FileResponse("static/index.html")


@app.post("/api/compare")
async def compare_texts(
    text1: str = Form(""),
    text2: str = Form(""),
    engine: str = Form("sbert"),
    file1: UploadFile | None = File(None),
    file2: UploadFile | None = File(None),
):
    final_t1 = text1.strip()
    final_t2 = text2.strip()

    if file1 and file1.filename:
        raw1 = await file1.read()
        final_t1 = extract_text_from_file(UploadWrapper(file1.filename, raw1))

    if file2 and file2.filename:
        raw2 = await file2.read()
        final_t2 = extract_text_from_file(UploadWrapper(file2.filename, raw2))

    if not final_t1 or not final_t2:
        raise HTTPException(status_code=400, detail="Please enter text or upload a file in both slots.")

    sbert_score = compute_semantic_similarity(final_t1, final_t2)
    tfidf_score = compute_tfidf_similarity(final_t1, final_t2)

    tokens_1 = set(clean_text(final_t1).split())
    tokens_2 = set(clean_text(final_t2).split())
    shared_keywords = sorted(tokens_1.intersection(tokens_2))
    jaccard_score = len(shared_keywords) / max(1, len(tokens_1.union(tokens_2)))

    active_score = sbert_score if engine == "sbert" else tfidf_score

    return {
        "active_score": active_score,
        "sbert_score": sbert_score,
        "tfidf_score": tfidf_score,
        "jaccard_score": jaccard_score,
        "words1": len(final_t1.split()),
        "words2": len(final_t2.split()),
        "shared_keywords": shared_keywords[:16],
    }