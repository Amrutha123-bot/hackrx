import faiss
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

def build_faiss_index(chunks):
    embeddings = model.encode(chunks)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    return index, embeddings

def get_similar_chunks(chunks, index, embeddings, query, top_k=5):
    query_embedding = model.encode([query])
    D, I = index.search(query_embedding, top_k)
    return [chunks[i] for i in I[0]]
