# chromadb/search_versions.py

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

embedding_fn = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
client = chromadb.Client()
collection = client.get_or_create_collection(
    name="book_versions",
    embedding_function=embedding_fn
)

def search_versions(query, top_k=2):
    results = collection.query(
        query_texts=[query],
        n_results=top_k
    )

    print(f"\n Query: {query}")
    for i in range(len(results["documents"][0])):
        doc = results['documents'][0][i]
        metadata = results['metadatas'][0][i]
        print(f"\n Result {i+1}:")
        print(f"Role: {metadata['role']}, Author: {metadata['author']}, Timestamp: {metadata['timestamp']}")
        print(f"Excerpt: {doc[:300]}...")

if __name__ == "__main__":
    sample_query = "tropical island scene with early morning sunrise"
    search_versions(sample_query)
