import random
import copy
import json

def write_json_to_file(data, filename):
    """
    Write JSON data to a file on disk.

    Args:
        data (dict or list): The JSON-serializable data to write.
        filename (str): The name of the file to write to.

    Returns:
        None
    """
    try:
        with open(filename, 'w') as json_file:
            json.dump(data, json_file, indent=2)
        print(f"JSON data successfully written to {filename}")
    except Exception as e:
        print(f"Error writing JSON to file: {e}")

def generate_graph_samples_with_labels(baseline, N, variation=0.2):
    """
    Generate N graph samples with variations within a specified range and assign labels.

    Args:
        baseline (dict): The baseline graph structure with nodes and edges.
        N (int): Number of graph samples to generate.
        variation (float): Percentage variation from the baseline (default is 0.2 for 20%).

    Returns:
        list: A list of graph samples with variations and labels.
    """
    def apply_variation(value, variation):
        if isinstance(value, (int, float)):
            delta = value * variation
            return value + random.uniform(-delta, delta)
        elif isinstance(value, list):
            return [apply_variation(v, variation) for v in value]
        return value  # Return as is for unsupported types

    def compute_label(graph):
        """
        Compute a label for the graph based on its features.
        This function assigns a label based on the sum of energy levels of nodes.

        Args:
            graph (dict): The graph structure.

        Returns:
            int: The computed label for classification or a float for regression.
        """
        total_energy = sum(node["features"]["energy"] for node in graph["nodes"])
        if total_energy > 2500:
            return 1  # Example classification: high-energy graph
        else:
            return 0  # Example classification: low-energy graph

    samples = []
    for _ in range(N):
        sample = copy.deepcopy(baseline)
        try:
            for node in sample["graphs"][0]["nodes"]:
                for key, value in node["features"].items():
                    node["features"][key] = apply_variation(value, variation)

            for edge in sample["graphs"][0]["edges"]:
                for key, value in edge["features"].items():
                    edge["features"][key] = apply_variation(value, variation)

            # Assign a label to the graph
            sample["graphs"][0]["label"] = compute_label(sample["graphs"][0])

            samples.append(sample)
        except KeyError as e:
            print(f"Error in graph generation: Missing key {e}")
            continue

    return samples


# Baseline graph data
baseline_graph = {
    "graphs": [
        {
            "nodes": [
                {"id": 1, "features": {"position": [100.0, 200.0, 300.0], "velocity": [1.0, 0.0, -1.0], "remaining_fuel": 80.0, "status": 1.0, "load": 50.0, "energy": 1000.0}},
                {"id": 2, "features": {"position": [200.0, 300.0, 400.0], "velocity": [0.0, 1.0, -1.0], "remaining_fuel": 70.0, "status": 1.0, "load": 30.0, "energy": 900.0}},
                {"id": 3, "features": {"position": [300.0, 400.0, 500.0], "velocity": [-1.0, 0.0, 1.0], "remaining_fuel": 60.0, "status": 1.0, "load": 40.0, "energy": 800.0}}
            ],
            "edges": [
                {"source": 1, "target": 2, "features": {"distance": 150.0, "signal_strength": -50.0, "bandwidth": 100.0, "latency": 20.0}},
                {"source": 2, "target": 3, "features": {"distance": 170.0, "signal_strength": -55.0, "bandwidth": 80.0, "latency": 25.0}},
                {"source": 3, "target": 1, "features": {"distance": 200.0, "signal_strength": -60.0, "bandwidth": 90.0, "latency": 30.0}}
            ]
        }
    ]
}

# Generate 1000 graph samples with labels
generated_samples = generate_graph_samples_with_labels(baseline_graph, N=10)

# Save the data to a JSON file
write_json_to_file(generated_samples, "generated_samples.json")
