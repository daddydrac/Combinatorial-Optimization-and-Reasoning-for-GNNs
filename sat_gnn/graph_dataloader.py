import json
import torch
from torch_geometric.data import Data, DataLoader

def preprocess_graph(graph, node_in_dim, edge_in_dim):
    """
    Preprocesses a graph to ensure the node and edge features match the required dimensions.

    Args:
        graph (Data): PyTorch Geometric Data object.
        node_in_dim (int): Desired dimensionality of node features.
        edge_in_dim (int): Desired dimensionality of edge features.

    Returns:
        Data: Preprocessed PyTorch Geometric Data object.
    """
    # Adjust node features
    if graph.x.size(1) < node_in_dim:
        padding = torch.zeros(graph.x.size(0), node_in_dim - graph.x.size(1))
        graph.x = torch.cat([graph.x, padding], dim=1)
    elif graph.x.size(1) > node_in_dim:
        graph.x = graph.x[:, :node_in_dim]

    # Adjust edge features
    if graph.edge_attr.size(1) < edge_in_dim:
        padding = torch.zeros(graph.edge_attr.size(0), edge_in_dim - graph.edge_attr.size(1))
        graph.edge_attr = torch.cat([graph.edge_attr, padding], dim=1)
    elif graph.edge_attr.size(1) > edge_in_dim:
        graph.edge_attr = graph.edge_attr[:, :edge_in_dim]

    return graph

def parse_graph(json_data):
    """
    Parses a single graph from the provided JSON data.
    
    Args:
        json_data (dict): A dictionary containing nodes and edges.

    Returns:
        Data: Parsed graph data in PyTorch Geometric format, with labels included if applicable.
    """
    nodes = json_data["nodes"]
    edges = json_data["edges"]

    # Node features and labels
    node_features = []
    node_labels = []
    node_mapping = {}

    for i, node in enumerate(nodes):
        features = [
            *node["features"]["position"],
            *node["features"]["velocity"],
            node["features"]["remaining_fuel"],
            node["features"]["load"],
            node["features"]["energy"]
        ]
        node_features.append(features)
        
        # 'status' is used as a label for classification
        node_labels.append(1 if node["features"]["status"] >= 1 else 0)

        node_mapping[node["id"]] = i

    # Edge features and connections
    edge_index = []
    edge_features = []

    for edge in edges:
        source = node_mapping[edge["source"]]
        target = node_mapping[edge["target"]]
        features = [
            edge["features"]["distance"],
            edge["features"]["signal_strength"],
            edge["features"]["bandwidth"],
            edge["features"]["latency"]
        ]
        edge_index.append([source, target])
        edge_features.append(features)

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_features = torch.tensor(edge_features, dtype=torch.float)
    node_features = torch.tensor(node_features, dtype=torch.float)
    node_labels = torch.tensor(node_labels, dtype=torch.float)

    # Create PyTorch Geometric Data object
    graph = Data(x=node_features, edge_index=edge_index, edge_attr=edge_features, y=node_labels)
    return graph

def create_dataloader(json_data, batch_size=1, node_in_dim=16, edge_in_dim=4):
    """
    Creates a DataLoader from a JSON dataset and preprocesses graphs to match the required dimensions.

    Args:
        json_data (list): The dataset loaded from JSON.
        batch_size (int): Number of graphs per batch.
        node_in_dim (int): Desired dimensionality of node features.
        edge_in_dim (int): Desired dimensionality of edge features.

    Returns:
        DataLoader: PyTorch Geometric DataLoader for the dataset.
    """
    graphs = []
    for item in json_data:
        for graph_data in item["graphs"]:
            graph = parse_graph(graph_data)
            # Preprocess graph to match required dimensions
            graph = preprocess_graph(graph, node_in_dim, edge_in_dim)
            graphs.append(graph)

    return DataLoader(graphs, batch_size=batch_size, shuffle=True)