import random

# Parameters
num_nodes = 300
num_edges = 5000

# First, create a spanning tree to ensure connectivity (999 edges)
edges = []
nodes = list(range(num_nodes))
connected = [0]  # Start with node 0
unconnected = list(range(1, num_nodes))

# Build spanning tree
while unconnected:
    u = random.choice(connected)
    v = unconnected.pop(random.randint(0, len(unconnected) - 1))
    w = random.randint(1, 100)  # Random weight
    edges.append((u, v, w))
    connected.append(v)

# Add remaining edges randomly (1500 - 999 = 501 edges)
edge_set = set((min(u, v), max(u, v)) for u, v, _ in edges)
while len(edges) < num_edges:
    u = random.randint(0, num_nodes - 1)
    v = random.randint(0, num_nodes - 1)
    if u != v:
        edge_tuple = (min(u, v), max(u, v))
        if edge_tuple not in edge_set:
            w = random.randint(1, 100)
            edges.append((u, v, w))
            edge_set.add(edge_tuple)

# Write all edges to file
with open('../K-Machines/graph.txt', 'w') as f:
    for u, v, w in edges:
        f.write(f"{u} {v} {w}\n")

print(f"Successfully wrote {len(edges)} edges to K-Machines/graph.txt")