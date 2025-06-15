import json
from kafka import KafkaProducer

with open("../data/esewa_dataset.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

producer = KafkaProducer(
    bootstrap_servers=["localhost:9092"],
    value_serializer=lambda x: json.dumps(x).encode("utf-8")
)

for category in dataset:
    for item in dataset[category]:
        producer.send("esewa_docs", value=item)
        print(f"✅ Sent to Kafka: {item['id']}")

producer.flush()
producer.close()