from pymilvus import Collection, connections
from sentence_transformers import SentenceTransformer
from utils.config import MILVUS_HOST, MILVUS_PORT

# Connect
connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
collection = Collection("esewa_docs")
collection.load()

model = SentenceTransformer("all-MiniLM-L6-v2")

def search_documents(query: str, top_k: int = 5):
    embedding = model.encode(query).tolist()
    search_params = {"metric_type": "IP", "params": {"nprobe": 10}}

    results = collection.search(
        data=[embedding],
        anns_field="embedding",
        param=search_params,
        limit=top_k,
        output_fields=["metadata"]
    )
    return [hit.entity.get("metadata") for hit in results[0]]
