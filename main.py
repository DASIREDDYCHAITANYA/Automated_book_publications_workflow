# main.py
import os
import json
from utils.rl_search import RLSearchAgent
import torch
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from ppo_trainer import PolicyNetwork
import torch

# Load PPO model
model_path = "utils/ppo_model.pt"
model = PolicyNetwork()
model.load_state_dict(torch.load(model_path, weights_only=False))  # ✅ FIXED HERE
model.eval()


# Feedback file to align IDs
with open("/Users/chaitanya/Automated_book_pub_workflow/utils/feedback_rewards.json", "r") as f:
    feedback_data = json.load(f)

# Setup ChromaDB
embedding_fn = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
client = chromadb.Client()
collection = client.get_or_create_collection(name="book_versions", embedding_function=embedding_fn)

# Run semantic search
query = "tropical island sunrise"
results = collection.query(query_texts=[query], n_results=4)

# Rerank using PPO model
version_scores = []
for i in range(len(results['ids'][0])):
    version_id = results['ids'][0][i]
    doc = results['documents'][0][i]
    meta = results['metadatas'][0][i]

    if version_id not in feedback_data:
        continue

    index_tensor = torch.tensor([[i]], dtype=torch.float32)
    predicted_score = model(index_tensor).item()

    version_scores.append((predicted_score, version_id, doc, meta))

# Sort descending by PPO-predicted reward
version_scores.sort(reverse=True)

# Show reranked results
for rank, (score, version_id, doc, meta) in enumerate(version_scores, 1):
    print(f"\n🔁 Rank {rank} — ID: {version_id} — PPO Score: {score:.2f}")
    print(f"Role: {meta['role']}, Author: {meta['author']}")
    print(f"Excerpt: {doc[:300]}...")
