#include <iostream>
#include <vector>
#include <algorithm>
#include <fstream>
#include <numeric>

// --- 1. Edge Structure ---
// Represents an edge with its two vertices (u, v) and weight (w)
struct Edge {
    int u, v, weight;
    // Overload the less than operator for sorting based on weight
    bool operator<(const Edge& other) const {
        return weight < other.weight;
    }
};

// --- 2. Disjoint Set Union (DSU) Structure ---
// Used to efficiently manage sets of vertices and detect cycles
struct DSU {
    std::vector<int> parent;
    
    // Initialize DSU for 'n' vertices
    DSU(int n) {
        parent.resize(n + 1); // +1 for 1-based indexing if needed, but safer to size appropriately
        // Initialize each vertex as its own parent (its own set)
        std::iota(parent.begin(), parent.end(), 0); // Fills parent[i] with i
    }
    
    // Find the representative (root) of the set containing 'i'
    // Uses path compression for efficiency
    int find(int i) {
        if (parent[i] == i)
            return i;
        // Path compression: set parent[i] directly to the root
        return parent[i] = find(parent[i]);
    }
    
    // Union by rank or size is generally better, but simple union works fine for this
    // Joins the sets containing 'u' and 'v'
    // Returns true if a union was performed, false if they were already in the same set (a cycle)
    bool unite(int u, int v) {
        int root_u = find(u);
        int root_v = find(v);
        
        // If they are not in the same set, join them (no cycle detected)
        if (root_u != root_v) {
            parent[root_u] = root_v;
            return true; // Successfully united
        }
        return false; // Already connected, adding this edge creates a cycle
    }
};

// --- 3. Kruskal's Algorithm Function ---
void kruskal(std::vector<Edge>& edges, int num_vertices) {
    // 
    
    // Step 1: Sort all edges by weight in non-decreasing order
    std::sort(edges.begin(), edges.end());
    
    // Initialize the DSU structure for all vertices
    DSU dsu(num_vertices);
    
    std::vector<Edge> mst_edges;
    int mst_weight = 0;
    
    std::cout << "\n--- MST Edges Selected ---\n";
    
    // Step 2: Iterate through the sorted edges
    for (const auto& edge : edges) {
        // Check if adding the current edge (u, v) connects two different components
        if (dsu.unite(edge.u, edge.v)) {
            // If true, it means no cycle is formed, so include the edge in the MST
            mst_edges.push_back(edge);
            mst_weight += edge.weight;
            
            std::cout << "Edge (" << edge.u << " - " << edge.v << ") with weight " << edge.weight << " added.\n";
            
            // Optimization: If we have V-1 edges, we have found the MST
            if (mst_edges.size() == num_vertices - 1) {
                break;
            }
        } else {
            // std::cout << "Edge (" << edge.u << " - " << edge.v << ") with weight " << edge.weight << " skipped (forms a cycle).\n";
        }
    }
    
    // Output the final results
    std::cout << "\n--------------------------\n";
    if (mst_edges.size() == num_vertices - 1 && num_vertices > 1) {
        std::cout << "Minimum Spanning Tree (MST) Found:\n";
        std::cout << "Total MST Weight: **" << mst_weight << "**\n";
    } else if (num_vertices <= 1) {
        std::cout << "The graph has only " << num_vertices << " vertex. MST weight is 0.\n";
    }
    else {
        std::cout << "The graph is not connected or the input was incomplete. Failed to find a spanning tree.\n";
    }
}

// --- 4. Main Function and Input Handling ---
int main() {
    // Read the graph from the specified file
    std::ifstream inputFile("graph.txt");
    
    if (!inputFile.is_open()) {
        std::cerr << "Error: Could not open graph.txt\n";
        // Prompt for file creation/format
        std::cerr << "Please ensure 'graph.txt' exists in the same directory and is formatted as 'u v w' per line.\n";
        return 1;
    }
    
    std::vector<Edge> edges;
    int u, v, w;
    int max_vertex_id = 0;
    
    // Read edges from the file
    while (inputFile >> u >> v >> w) {
        edges.push_back({u, v, w});
        // Track the largest vertex ID to determine the total number of vertices (N)
        max_vertex_id = std::max({max_vertex_id, u, v});
    }
    
    inputFile.close();
    
    // The number of vertices (N) is max_vertex_id, assuming vertices are labeled 1 to N
    int num_vertices = max_vertex_id;
    
    if (edges.empty()) {
        std::cout << "The graph file is empty. Exiting.\n";
        return 0;
    }
    
    std::cout << "Graph Loaded:\n";
    std::cout << "Total Edges: " << edges.size() << "\n";
    std::cout << "Inferred Vertices: " << num_vertices << "\n";
    
    // Run Kruskal's algorithm
    kruskal(edges, num_vertices);
    
    return 0;
}