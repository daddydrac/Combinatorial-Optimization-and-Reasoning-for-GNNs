import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import json
import os

# Environment variables for troubleshooting
os.environ['MKLDNN_VERBOSE'] = '0'
os.environ['MKLDNN_DISABLE'] = '1'

# Core Bellman-Ford GNN
class BellmanFordGNN(MessagePassing):
    def __init__(self, node_feature_dim, edge_feature_dim, hidden_dim):
        super(BellmanFordGNN, self).__init__(aggr="min")
        self.message_linear = nn.Linear(node_feature_dim + edge_feature_dim, hidden_dim)
        self.update_linear = nn.Linear(node_feature_dim + hidden_dim, hidden_dim)

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        return self.message_linear(torch.cat([x_j, edge_attr], dim=-1))

    def update(self, aggr_out, x):
        return self.update_linear(torch.cat([x, aggr_out], dim=-1))


# Encode-Process-Decode Architecture
class EncodeProcessDecode(nn.Module):
    def __init__(self, node_in_dim, edge_in_dim, hidden_dim, decoder_in_features):
        super(EncodeProcessDecode, self).__init__()
        self.encoder = nn.Linear(node_in_dim, hidden_dim)
        self.processor = BellmanFordGNN(
            node_feature_dim=hidden_dim,
            edge_feature_dim=edge_in_dim,
            hidden_dim=hidden_dim,
        )
        self.decoder_in_features = decoder_in_features
        self.decoder = nn.Linear(decoder_in_features, 1, bias=True)

    def forward(self, data):
        x = F.relu(self.encoder(data.x.float()))
        print(f"Shape after encoder: {x.shape}, dtype: {x.dtype}")

        x = self.processor(x, data.edge_index, data.edge_attr.float())
        print(f"Shape after processor: {x.shape}, dtype: {x.dtype}")

        # Adjust input to match decoder expectations
        if x.size(1) < self.decoder.in_features:
            padding = torch.zeros(
                x.size(0), self.decoder.in_features - x.size(1),
                device=x.device, dtype=torch.float32
            )
            x = torch.cat([x, padding], dim=1)
        elif x.size(1) > self.decoder.in_features:
            x = x[:, :self.decoder.in_features]

        print(f"Shape before decoder: {x.shape}, Expected: {self.decoder.in_features}, dtype: {x.dtype}")
        x = self.decoder(x.float())  # Ensure float32
        return torch.sigmoid(x)




# Actor-Critic Framework
class ActorCritic(nn.Module):
    def __init__(self, hidden_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        policy = F.softmax(self.actor(x), dim=-1)
        value = self.critic(x)
        return policy, value


# Supervised Training Function
# Supervised Training Function
def train_supervised(model, loader, optimizer, criterion, epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        y_true, y_pred = [], []

        for data in loader:
            # Ensure tensors are float32
            data.x = data.x.float()
            data.edge_attr = data.edge_attr.float()
            data.y = data.y.float()

            optimizer.zero_grad()
            data = data.to(next(model.parameters()).device)
            
            # Forward pass
            out = model(data).squeeze(-1)  # Node-level predictions
            
            # Aggregate node-level predictions to graph-level
            out = out.mean()  # Take the mean of node-level predictions for the entire graph
            
            # Compute loss
            loss = criterion(out.unsqueeze(0), data.y)  # Match dimensions with graph-level label
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            y_true.append(data.y.item())  # Graph-level label
            y_pred.append(out.item() > 0.5)  # Binary prediction at graph-level

        # Metrics for graph-level classification
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}, Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")


# Reinforcement Learning Training Function
def train_rl(actor_critic, env, optimizer, epochs, gamma=0.99):
    actor_critic.train()
    for epoch in range(epochs):
        state = env.reset()
        log_probs, rewards, values = [], [], []

        while not env.done():
            state = state.to(next(actor_critic.parameters()).device)
            policy, value = actor_critic(state)
            action = torch.multinomial(policy, 1).item()
            next_state, reward = env.step(action)

            log_probs.append(torch.log(policy.squeeze(0)[action]))
            rewards.append(reward)
            values.append(value)

            state = next_state

        returns = compute_returns(rewards, gamma)
        returns = torch.tensor(returns, dtype=torch.float32).to(next(actor_critic.parameters()).device)

        values = torch.cat(values).squeeze()
        log_probs = torch.stack(log_probs)

        advantage = returns - values.detach()
        policy_loss = -(log_probs * advantage).mean()
        value_loss = F.mse_loss(values, returns)

        optimizer.zero_grad()
        (policy_loss + value_loss).backward()
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


def flatten_features(features):
    return [v for value in features.values() for v in (value if isinstance(value, list) else [value])]


def parse_graph_data(json_data):
    graphs = []
    for graph_data in json_data.get("graphs", []):
        try:
            nodes = graph_data["nodes"]
            edges = graph_data["edges"]
            label = graph_data.get("label")  # Extract the label

            node_features = torch.tensor(
                [flatten_features(node["features"]) for node in nodes],
                dtype=torch.float32  # Force float32
            )

            edge_features = torch.tensor(
                [flatten_features(edge["features"]) for edge in edges],
                dtype=torch.float32  # Force float32
            )

            edge_indices = torch.tensor(
                [[edge["source"] - 1, edge["target"] - 1] for edge in edges],
                dtype=torch.long  # Long type for indices
            ).t().contiguous()

            y = torch.tensor([label], dtype=torch.float32)  # Force float32

            graph = Data(
                x=node_features,
                edge_index=edge_indices,
                edge_attr=edge_features,
                y=y
            )
            graphs.append(graph)

        except KeyError as e:
            print(f"Error parsing graph: Missing key {e}")
        except Exception as e:
            print(f"Unexpected error parsing graph: {e}")

    return graphs




def create_dataloader(json_data, batch_size=1):
    graphs = []
    for item in json_data:
        parsed_graphs = parse_graph_data(item)
        if parsed_graphs:
            graphs.extend(parsed_graphs)

    if not graphs:
        raise ValueError("No valid graphs found in the dataset")

    print("First parsed graph:", graphs[0])
    return DataLoader(graphs, batch_size=batch_size, shuffle=True)


# Placeholder Environment
class RLEnv:
    def __init__(self, hidden_dim):
        self.hidden_dim = hidden_dim
        self.done_flag = False

    def reset(self):
        self.done_flag = False
        return torch.rand(1, self.hidden_dim)

    def step(self, action):
        self.done_flag = np.random.rand() < 0.1
        next_state = torch.rand(1, self.hidden_dim)
        reward = np.random.rand()
        return next_state, reward

    def done(self):
        return self.done_flag


# Save and Load Model
def save_model(model, optimizer, epoch, path="model_checkpoint.pth"):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
    }
    torch.save(checkpoint, path)
    print(f"Model and optimizer state saved to {path}")


# Main Execution
if __name__ == "__main__":
    with open("generated_samples.json") as f:
        json_data = json.load(f)

    loader = create_dataloader(json_data, batch_size=1)

    # Check the structure and dimensions of the first graph in the DataLoader
    for data in loader:
        print(f"Graph data: x shape: {data.x.shape}, edge_attr shape: {data.edge_attr.shape}")
        break

    # Ensure the first graph is on the correct device (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    first_graph = next(iter(loader)).to(device)

    # Extract node and edge input dimensions from the first graph
    node_in_dim = first_graph.x.size(1)
    edge_in_dim = first_graph.edge_attr.size(1)

    # Initialize the EncodeProcessDecode model
    model = EncodeProcessDecode(
        node_in_dim=node_in_dim,
        edge_in_dim=edge_in_dim,
        hidden_dim=16,
        decoder_in_features=16  # Ensure this matches the output dim of the processor
    ).to(device)


    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()

    print("Starting supervised training...")
    train_supervised(model, loader, optimizer, criterion, epochs=50)

    save_model(model, optimizer, epoch=50, path="gnn_model.pth")

    print("Starting reinforcement learning...")
    actor_critic = ActorCritic(hidden_dim=16, action_dim=4).to(device)
    rl_optimizer = torch.optim.Adam(actor_critic.parameters(), lr=0.001)
    env = RLEnv(hidden_dim=16)
    train_rl(actor_critic, env, rl_optimizer, epochs=5)

    save_model(actor_critic, rl_optimizer, epoch=5, path="actor_critic_model.pth")
