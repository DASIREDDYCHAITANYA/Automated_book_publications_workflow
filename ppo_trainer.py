# ppo_trainer.py

import torch
import torch.nn as nn
import torch.optim as optim
import json
import os

# Define PPO policy network
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=16):
        super(PolicyNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.model(x)

# PPO Training Function
def train_ppo(feedback_path="/utils/feedback_rewards.json", save_path="utils/ppo_model.pt"):
    # Load feedback scores
    with open(feedback_path, "r") as f:
        feedback_data = json.load(f)

    ids = list(feedback_data.keys())
    scores = [feedback_data[k] for k in ids]

    # Prepare data: use index as input, rating as reward
    X = torch.tensor([[i] for i in range(len(scores))], dtype=torch.float32)
    y = torch.tensor([[r] for r in scores], dtype=torch.float32)

    model = PolicyNetwork()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    # Train
    for epoch in range(500):
        optimizer.zero_grad()
        preds = model(X)
        loss = loss_fn(preds, y)
        loss.backward()
        optimizer.step()

    # Save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"✅ PPO model trained.\n✅ Model saved to {save_path}")

# Run if script is executed
if __name__ == "__main__":
    train_ppo()
