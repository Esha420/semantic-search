import json
from kafka import KafkaConsumer
from pymilvus import Collection, connections
from sentence_transformers import SentenceTransformer
from utils.config import MILVUS_HOST, MILVUS_PORT

# Connect to Milvus
connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
collection = Collection("esewa_docs")
collection.load()

model = SentenceTransformer("all-MiniLM-L6-v2")

consumer = KafkaConsumer(
    "esewa_docs",
    bootstrap_servers=["localhost:9092"],
    value_deserializer=lambda x: json.loads(x.decode("utf-8")),
    group_id="esewa_consumer"
)

print("✅ Listening to Kafka...")

for message in consumer:
    data = message.value
    doc_type = data.get("type")
    if doc_type == "transaction":
        text = f"{data['description']} {data.get('merchant_name', '')}"
    elif doc_type == "saved_payment":
        text = f"{data['description']} {data.get('payee_name', '')}"
    elif doc_type == "service":
        text = data.get("content", "")
    elif doc_type == "product":
        text = f"{data['description']} {data.get('product_name', '')}"
    else:
        continue

    embedding = model.encode(text).tolist()
    try:
        collection.insert([
            [data["id"]],
            [embedding],
            [data]
        ])
        print(f"✅ Inserted: {data['id']}")
    except Exception as e:
        print(f"⚠️ Failed to insert {data['id']}: {e}")
