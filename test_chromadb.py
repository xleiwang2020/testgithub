import chromadb
from sentence_transformers import SentenceTransformer


client = chromadb.Client()
collection = client.create_collection("Collection")

# 添加数据到集合
collection.add(
    documents=["Hello world", "Chroma is great!", "Python programming"],
    metadatas=[{"source": "doc1"}, {"source": "doc2"}, {"source": "doc3"}],
    ids=["doc1", "doc2", "doc3"]  # 添加 ids 参数
)

sentences = ["Hello Amy"]
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embeddings = model.encode(sentences)

results = collection.query(query_embeddings=[embeddings], n_results=2)
print(results)
