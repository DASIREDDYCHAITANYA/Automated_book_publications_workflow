# chromadb/version_tracker.py
# chromadb/version_tracker.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import datetime
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from utils.rl_search import RLSearchAgent
# Initialize ChromaDB client and collection
embedding_fn = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
client = chromadb.Client()
collection = client.get_or_create_collection(
    name="book_versions",
    embedding_function=embedding_fn
)

# Define metadata for each version
version_map = [
    {
        "file": "/Users/chaitanya/Automated_book_pub_workflow/data/raw/chapter_1.png",
        "version_id": "chapter1_v1",
        "role": "scraper",
        "author": "Playwright",
        "timestamp": "2025-06-21T10:00:00Z"
    },
    {
        "file": "/Users/chaitanya/Automated_book_pub_workflow/data/spun_chapter_1.txt",
        "version_id": "chapter1_v2",
        "role": "writer",
        "author": "AI Writer",
        "timestamp": "2025-06-21T10:30:00Z"
    },
    {
        "file": "/Users/chaitanya/Automated_book_pub_workflow/data/reviewed_chapter_1.txt",
        "version_id": "chapter1_v3",
        "role": "reviewer",
        "author": "LLM Reviewer",
        "timestamp": "2025-06-21T11:00:00Z"
    },
    {
        "file": "/Users/chaitanya/Automated_book_pub_workflow/data/Final_chapter.txt",
        "version_id": "chapter1_v4",
        "role": "editor",
        "author": "Human",
        "timestamp": "2025-06-21T11:30:00Z"
    }
]
def update_rewards_to_chromadb():
    agent = RLSearchAgent()
    agent.load_rewards()

    for version_id, reward in agent.rewards.items():
        try:
            collection.update(
                ids=[version_id],
                metadatas=[{"reward": reward}]
            )
            print(f"[↑] Updated reward for {version_id}: {reward}")
        except Exception as e:
            print(f"[!] Could not update reward for {version_id}: {e}")

def store_versions():
    for entry in version_map:
        file_path = os.path.join(".", entry["file"])
        if not os.path.exists(file_path):
            print(f"[!] Missing file: {file_path}")
            continue

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        collection.add(
            documents=[content],
            ids=[entry["version_id"]],
            metadatas=[{
                "role": entry["role"],
                "author": entry["author"],
                "chapter": 1,
                "timestamp": entry["timestamp"]
            }]
        )
        print(f"[+] Stored: {entry['version_id']}")

if __name__ == "__main__":
    store_versions()
    update_rewards_to_chromadb()
