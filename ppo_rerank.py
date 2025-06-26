import torch
import torch.nn as nn

class PPOModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=16):
        super(PPOModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        return self.model(x)

# Rebuild the model and load weights
model = PPOModel()
state_dict = torch.load("/Users/chaitanya/Automated_book_pub_workflow/ppo.pt", map_location=torch.device('cpu'))
model.load_state_dict(state_dict)
model.eval()  
feedback_data = {
    "chapter1_v1": 0.5,
    "chapter1_v2": 1.0,
    "chapter1_v3": 2.5,
    "chapter1_v4": 4.0
}

version_map = [
    {
        "file": "/Users/chaitanya/Automated_book_pub_workflow/data/raw/chapter_1.txt",
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

# Prepare data for model inference
results = {
    "ids": [[]],
    "documents": [[]],
    "metadatas": [[]]
}

for version in version_map:
    version_id = version["version_id"]
    file_path = version["file"]

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        content = f"[Error reading file: {e}]"

    results["ids"][0].append(version_id)
    results["documents"][0].append(content)
    results["metadatas"][0].append({
        "role": version["role"],
        "author": version["author"],
        "timestamp": version["timestamp"]
    })

# Rerank using PPO model
version_scores = []
for i in range(len(results['ids'][0])):
    version_id = results['ids'][0][i]
    doc = results['documents'][0][i]
    meta = results['metadatas'][0][i]

    if version_id not in feedback_data:
        continue

    feedback_score = torch.tensor([[feedback_data[version_id]]], dtype=torch.float32)
    predicted_score = model(feedback_score).item()

    version_scores.append({
        "version_id": version_id,
        "score": predicted_score,
        "author": meta["author"],
        "role": meta["role"],
        "timestamp": meta["timestamp"]
    })

# Sort and print ranked results
ranked_versions = sorted(version_scores, key=lambda x: x["score"], reverse=True)

print("🔢 Ranked Chapter Versions:")
for i, v in enumerate(ranked_versions, 1):
    print(f"{i}. {v['version_id']} ({v['role']} by {v['author']}) - Score: {v['score']:.3f}")
    # Save ranked results to a plain text file
with open("ranked_versions.txt", "w", encoding="utf-8") as f:
    f.write("🔢 Ranked Chapter Versions:\n")
    for i, v in enumerate(ranked_versions, 1):
        line = f"{i}. {v['version_id']} ({v['role']} by {v['author']}) - Score: {v['score']:.3f}\n"
        f.write(line)

print("\n✅ Ranked results saved to 'ranked_versions.txt'")
