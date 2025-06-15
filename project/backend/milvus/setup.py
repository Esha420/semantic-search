from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection
from utils.config import MILVUS_HOST, MILVUS_PORT

connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)

fields = [
    FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=64, is_primary=True, auto_id=False),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),
    FieldSchema(name="metadata", dtype=DataType.JSON)
]

schema = CollectionSchema(fields, description="eSewa documents collection")
collection = Collection(name="esewa_docs", schema=schema)

index_params = {
    "index_type": "IVF_FLAT",
    "metric_type": "IP",
    "params": {"nlist": 128}
}
collection.create_index(field_name="embedding", index_params=index_params)
collection.load()
print("✅ Milvus collection ready.")
