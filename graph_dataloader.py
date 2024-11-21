from torch_geometric.data import Data, DataLoader
import torch

def parse_graph(json_data):
    """
    Parses a single graph from the provided JSON data.
    
    Args:
        json_data (dict): A dictionary containing nodes and edges.

    Returns:
        torch_geometric.data.Data: Parsed graph data in PyTorch Geometric format.
    """
    nodes = json_data["nodes"]
    edges = json_data["edges"]

    # Node features and labels
    node_features = []
    node_labels = []  # If you have labels, replace this with actual values
    node_mapping = {}

    for i, node in enumerate(nodes):
        features = [
            *node["features"]["position"],
            *node["features"]["velocity"],
            node["features"]["remaining_fuel"],
            node["features"]["status"],
            node["features"]["load"],
            node["features"]["energy"]
        ]
        node_features.append(features)
        node_labels.append(1 if node["features"]["status"] == 1 else 0)  # Example label logic
        node_mapping[node["id"]] = i  # Map node ID to index

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
    node_labels = torch.tensor(node_labels, dtype=torch.long)

    # Create PyTorch Geometric Data object
    graph = Data(x=node_features, edge_index=edge_index, edge_attr=edge_features, y=node_labels)
    return graph


def create_dataloader(json_dataset, batch_size=1):
    """
    Creates a DataLoader from a JSON dataset.

    Args:
        json_dataset (dict): The dataset loaded from JSON.
        batch_size (int): Number of graphs per batch.

    Returns:
        DataLoader: PyTorch Geometric DataLoader for the dataset.
    """
    graphs = []
    for graph_data in json_dataset["graphs"]:
        graph = parse_graph(graph_data)
        graphs.append(graph)
    
    return DataLoader(graphs, batch_size=batch_size, shuffle=True)



# Use DataLoader
# dataloader = create_dataloader(json.loads(example_json), batch_size=1)
