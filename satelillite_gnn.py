import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.data import DataLoader, Data
from torch_geometric.utils import from_networkx
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import time
import json
import os


# Core Graph Neural Network Design
class BellmanFordGNN(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(BellmanFordGNN, self).__init__(aggr="min")  # Min aggregation aligns with dynamic programming
        self.message_linear = nn.Linear(in_channels + 1, out_channels)  # Process node and edge features
        self.update_linear = nn.Linear(in_channels + out_channels, out_channels)

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        return self.message_linear(torch.cat([x_j, edge_attr.unsqueeze(-1)], dim=-1))

    def update(self, aggr_out, x):
        return self.update_linear(torch.cat([x, aggr_out], dim=-1))


# Encode-Process-Decode Architecture
class EncodeProcessDecode(nn.Module):
    def __init__(self, node_in_dim, edge_in_dim, hidden_dim):
        super(EncodeProcessDecode, self).__init__()
        self.encoder = nn.Linear(node_in_dim, hidden_dim)
        self.processor = BellmanFordGNN(hidden_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, 1)  # Predict quality score or optimality

    def forward(self, data):
        x = F.relu(self.encoder(data.x))
        x = self.processor(x, data.edge_index, data.edge_attr)
        return torch.sigmoid(self.decoder(x))


# Actor-Critic for Reinforcement Learning
class ActorCritic(nn.Module):
    def __init__(self, hidden_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        policy = F.softmax(self.actor(x), dim=-1)
        value = self.critic(x)
        return policy, value


# Data Loader for JSON
def parse_graph_data(json_data):
    graphs = []
    for graph_data in json_data["graphs"]:
        nodes = graph_data["nodes"]
        edges = graph_data["edges"]

        node_features = torch.tensor([list(node["features"].values()) for node in nodes], dtype=torch.float)
        edge_indices = torch.tensor([[edge["source"] - 1, edge["target"] - 1] for edge in edges], dtype=torch.long).T
        edge_features = torch.tensor([list(edge["features"].values()) for edge in edges], dtype=torch.float)

        graph = Data(x=node_features, edge_index=edge_indices, edge_attr=edge_features)
        graphs.append(graph)

    return graphs


# Training Function for Supervised Learning
def train_supervised(model, loader, optimizer, criterion, epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        y_true, y_pred = [], []

        for data in loader:
            optimizer.zero_grad()
            out = model(data).squeeze()
            loss = criterion(out, data.y.float())
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            y_true.extend(data.y.cpu().numpy())
            y_pred.extend((out > 0.5).cpu().numpy())

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}, Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")


# Reinforcement Learning Training
def train_rl(actor_critic, env, optimizer, epochs, gamma=0.99):
    for epoch in range(epochs):
        state = env.reset()
        log_probs, rewards, values = [], [], []

        while not env.done():
            policy, value = actor_critic(state)
            action = torch.multinomial(policy, 1).item()
            next_state, reward = env.step(action)

            log_probs.append(torch.log(policy[action]))
            rewards.append(reward)
            values.append(value)

            state = next_state

        returns = compute_returns(rewards, gamma)
        policy_loss = -sum(log_prob * (R - value) for log_prob, R, value in zip(log_probs, returns, values))
        value_loss = F.mse_loss(torch.tensor(values), torch.tensor(returns))

        loss = policy_loss + value_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch + 1}, Policy Loss: {policy_loss.item():.4f}, Value Loss: {value_loss.item():.4f}")


# Utility Functions
def compute_returns(rewards, gamma):
    returns = []
    discounted_sum = 0
    for reward in reversed(rewards):
        discounted_sum = reward + gamma * discounted_sum
        returns.insert(0, discounted_sum)
    return returns


def save_model(model, path="gnn_model.pth"):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


def load_model(path, model_class, **kwargs):
    model = model_class(**kwargs)
    model.load_state_dict(torch.load(path))
    model.eval()
    print(f"Model loaded from {path}")
    return model


def run_inference(model, input_data):
    model.eval()
    with torch.no_grad():
        prediction = model(input_data).squeeze()
        print("Inference Result:", prediction)
        return prediction


# Example Usage
if __name__ == "__main__":
    # Load data
    with open("example_data.json", "r") as f:
        json_data = json.load(f)
    graphs = parse_graph_data(json_data)
    loader = DataLoader(graphs, batch_size=2, shuffle=True)

    # Define model, optimizer, and loss
    model = EncodeProcessDecode(node_in_dim=6, edge_in_dim=4, hidden_dim=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()

    # Supervised training
    train_supervised(model, loader, optimizer, criterion, epochs=10)

    # Save and load model
    save_model(model, path="gnn_model.pth")
    model = load_model("gnn_model.pth", EncodeProcessDecode, node_in_dim=6, edge_in_dim=4, hidden_dim=16)

    # Inference
    for graph in graphs:
        run_inference(model, graph)
