# Combinatorial Optimization and Reasoning for GNNs
Using Graph Neural Networks (GNNs) for combinatorial optimization (CO) offers specific advantages for problems where traditional methods struggle due to complexity, data dependency, or the need for real-time solutions. Only works and tested on Linux.

 - Codebase for Medium article [here](https://medium.com/@joehoeller/optimizing-satellite-constellations-with-graph-neural-networks-6ce87d50a29f)

 - Notebook has explaination of math used with equations [here](https://github.com/daddydrac/Combinatorial-Optimization-and-Reasoning-for-GNNs/blob/main/math_explained.ipynb)

### How to use

```docker build -f CPU.Dockerfile -t satgnn .``` 

(swap CPU. for GPU. if on NVIDIA GPUs)

#### GPU
```docker run -it -d --name satgnn --gpus all -v ${PWD}:/sat_gnn satgnn```

#### CPU
```docker run -it -d --name satgnn -v ${PWD}:/sat_gnn satgnn```

#### Exec into container and run the code

```docker exec -it satgnn /bin/bash```

### Train model 

```python model.py```

-------------

### Framing the Data Science Problem and Solution 

<strong>Problem:</strong> Managing mega-constellations (e.g., Starlink, OneWeb) involves optimizing communication paths, maintaining orbits, and minimizing fuel usage for adjustments. Why GNNs? Graph Representation: Satellites form a communication network, where nodes represent satellites and edges represent communication links. 

<strong>GNN Role:</strong> Optimize routing paths for satellite-to-satellite communication to minimize latency. Predict failures or disruptions in the network by aggregating historical telemetry and sensor data. 

<strong>Benefit over Baseline:</strong> Traditional heuristics may struggle to handle the dynamic and large-scale nature of constellations, while GNNs can generalize and adapt to evolving network conditions.

### Explaination of 
Supervised Learning Labels: For supervised training, you need ground-truth labels:

- Routing Paths: Optimal paths between specific satellite pairs.

```{"source": 1, "target": 3, "path": [1, 2, 3]}```

- Predicted Failures: Binary labels indicating whether a link or satellite is likely to fail in the next time step.

```{"node_id": 2, "failure": 0}, {"edge": [1, 3], "failure": 1}```

- Fuel-Efficient Adjustments: Recommendations for satellite repositioning.

```{"node_id": 1, "delta_position": [10, -10, 5]}```

Generated data model

```
{
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

```
