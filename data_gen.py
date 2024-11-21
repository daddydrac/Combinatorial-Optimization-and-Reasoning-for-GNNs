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
      
def generate_graph_samples(baseline, N, variation=0.2):
    """
    Generate N graph samples with variations within a specified range.

    Args:
        baseline (dict): The baseline graph structure with nodes and edges.
        N (int): Number of graph samples to generate.
        variation (float): Percentage variation from the baseline (default is 0.2 for 20%).

    Returns:
        list: A list of graph samples with variations.
    """
    def apply_variation(value, variation):
        if isinstance(value, (int, float)):
            delta = value * variation
            return value + random.uniform(-delta, delta)
        elif isinstance(value, list):
            return [apply_variation(v, variation) for v in value]
        return value  # Return as is for unsupported types
    
    samples = []
    for _ in range(N):
        sample = copy.deepcopy(baseline)
        for node in sample["graphs"][0]["nodes"]:
            for key, value in node["features"].items():
                node["features"][key] = apply_variation(value, variation)
        
        for edge in sample["graphs"][0]["edges"]:
            for key, value in edge["features"].items():
                edge["features"][key] = apply_variation(value, variation)
        
        samples.append(sample)
    
    return samples

# Baseline graph data
baseline_graph = {
  "graphs": [
    {
      "nodes": [
        {"id": 1, "features": {"position": [100, 200, 300], "velocity": [1, 0, -1], "remaining_fuel": 80, "status": 1, "load": 50, "energy": 1000}},
        {"id": 2, "features": {"position": [200, 300, 400], "velocity": [0, 1, -1], "remaining_fuel": 70, "status": 1, "load": 30, "energy": 900}},
        {"id": 3, "features": {"position": [300, 400, 500], "velocity": [-1, 0, 1], "remaining_fuel": 60, "status": 1, "load": 40, "energy": 800}}
      ],
      "edges": [
        {"source": 1, "target": 2, "features": {"distance": 150.0, "signal_strength": -50, "bandwidth": 100, "latency": 20}},
        {"source": 2, "target": 3, "features": {"distance": 170.0, "signal_strength": -55, "bandwidth": 80, "latency": 25}},
        {"source": 3, "target": 1, "features": {"distance": 200.0, "signal_strength": -60, "bandwidth": 90, "latency": 30}}
      ]
    }
  ]
}

# Generate 10000 graph samples
generated_samples = generate_graph_samples(baseline_graph, N=10000)

# Print the first sample
write_json_to_file(generated_samples, "generated_samples.json")

