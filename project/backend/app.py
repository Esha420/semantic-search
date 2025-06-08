from fastapi import FastAPI, Query
from pydantic import BaseModel
from pymilvus import connections, Collection
from sentence_transformers import SentenceTransformer

app = FastAPI()
model = SentenceTransformer("all-MiniLM-L6-v2")
connections.connect(host="localhost", port="19530")
collection = Collection("esewa_docs")
collection.load()

@app.get("/search")
def search(query: str):
    embedding = model.encode(query).tolist()
    results = collection.search([embedding], anns_field="embedding",
                                param={"metric_type": "IP", "params": {"nprobe": 10}},
                                limit=5, output_fields=["text"])
    return [{"text": hit.entity.get("text"), "score": hit.score} for hit in results[0]]
