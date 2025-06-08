from sentence_transformers import SentenceTransformer
from pymilvus import Collection, connections, utility

def embed_and_insert(documents):
    connections.connect(host="localhost", port="19530")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(documents).tolist()
    
    collection = Collection("esewa_docs")
    collection.insert([documents, embeddings])
