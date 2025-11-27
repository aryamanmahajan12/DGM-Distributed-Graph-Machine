#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <limits.h>

#define MAX_NODES 6
#define INF INT_MAX

// -----------------------------------------------------
// 1. Data Structures
// -----------------------------------------------------

// Edge structure
typedef struct {
    int u, v;     // Vertices of the edge
    int weight;   // Edge weight
} Edge;

// MPI Datatype for the Edge structure
MPI_Datatype MPI_EDGE;

// -----------------------------------------------------
// 2. Union-Find Utility (Centralized for simplicity)
// -----------------------------------------------------

int parent[MAX_NODES]; // Array to store parent pointers

// Initializes the Union-Find structure
void UF_init() {
    for (int i = 0; i < MAX_NODES; i++) {
        parent[i] = i;
    }
}

// Finds the root/representative of the set (with path compression)
int UF_find(int i) {
    if (parent[i] == i)
        return i;
    return parent[i] = UF_find(parent[i]);
}

// Unites two sets (by rank/size for better performance, but simplified here)
// Returns 1 if a merge happened, 0 if they were already in the same set
int UF_union(int i, int j) {
    int root_i = UF_find(i);
    int root_j = UF_find(j);
    if (root_i != root_j) {
        parent[root_i] = root_j; // Simple union (root_i points to root_j)
        return 1;
    }
    return 0;
}

// -----------------------------------------------------
// 3. MPI Edge Datatype Creation
// -----------------------------------------------------

void create_mpi_edge_type() {
    int block_lengths[3] = {1, 1, 1}; // u, v, weight
    MPI_Aint displacements[3];
    MPI_Datatype typelist[3] = {MPI_INT, MPI_INT, MPI_INT};

    // Get the memory addresses of the members of the struct
    MPI_Aint start_address;
    Edge temp_edge;
    MPI_Get_address(&temp_edge, &start_address);
    MPI_Get_address(&temp_edge.u, &displacements[0]);
    MPI_Get_address(&temp_edge.v, &displacements[1]);
    MPI_Get_address(&temp_edge.weight, &displacements[2]);

    // Make displacements relative to the start_address
    displacements[0] = MPI_Aint_diff(displacements[0], start_address);
    displacements[1] = MPI_Aint_diff(displacements[1], start_address);
    displacements[2] = MPI_Aint_diff(displacements[2], start_address);

    // Create the new type
    MPI_Type_create_struct(3, block_lengths, displacements, typelist, &MPI_EDGE);
    MPI_Type_commit(&MPI_EDGE);
}

// -----------------------------------------------------
// 4. Test Data and Distribution
// -----------------------------------------------------

// Test Graph (6 vertices, 9 edges, MST weight 15)
const Edge GLOBAL_EDGE_LIST[] = {
    {0, 1, 1}, {0, 2, 9}, {0, 3, 4},
    {1, 2, 5}, {1, 3, 7}, {1, 4, 3},
    {2, 5, 8}, {3, 4, 10}, {4, 5, 2}
};
const int NUM_GLOBAL_EDGES = 9;

// Function to partition and send the graph data (Edge Partitioning)
void distribute_graph(int rank, int size, Edge** local_edges_ptr, int* num_local_edges_ptr) {
    int *sendcounts = NULL;
    int *displs = NULL;

    if (rank == 0) {
        // Calculate counts and displacements for block distribution
        sendcounts = (int*)malloc(size * sizeof(int));
        displs = (int*)malloc(size * sizeof(int));
        int total_edges = NUM_GLOBAL_EDGES;
        int sum = 0;

        for (int i = 0; i < size; i++) {
            sendcounts[i] = total_edges / size + (i < total_edges % size ? 1 : 0);
            displs[i] = sum;
            sum += sendcounts[i];
        }
    }

    // Process 0 broadcasts the count for its own use and for all others to know
    int local_count;
    if (rank == 0) {
        local_count = sendcounts[rank];
    }
    MPI_Scatter(sendcounts, 1, MPI_INT, &local_count, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    *num_local_edges_ptr = local_count;
    *local_edges_ptr = (Edge*)malloc(local_count * sizeof(Edge));
    
    // Scatter the edge data (uses the custom MPI_EDGE type)
    MPI_Scatterv((void*)GLOBAL_EDGE_LIST, sendcounts, displs, MPI_EDGE,
                 *local_edges_ptr, local_count, MPI_EDGE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        free(sendcounts);
        free(displs);
    }
}

// -----------------------------------------------------
// 5. Distributed Borůvka Logic
// -----------------------------------------------------

// Finds the best MWOE for the fragments visible to this process
// Uses the globally updated parent[] array
Edge find_local_best_mwoe(Edge* local_edges, int num_local_edges) {
    Edge local_mwoe = {-1, -1, INF};
    
    for (int i = 0; i < num_local_edges; i++) {
        Edge current_edge = local_edges[i];
        
        // Find roots of the current components
        int root_u = UF_find(current_edge.u);
        int root_v = UF_find(current_edge.v);
        
        // Check outgoing condition: must connect different fragments
        if (root_u != root_v) {
            if (current_edge.weight < local_mwoe.weight) {
                local_mwoe = current_edge;
            }
        }
    }
    return local_mwoe;
}


int run_distributed_mst(int rank, int size) {
    Edge* local_edges = NULL;
    int num_local_edges = 0;
    int total_mst_weight = 0;
    int edges_in_mst = 0;
    
    // Create custom MPI datatype and partition graph
    create_mpi_edge_type();
    distribute_graph(rank, size, &local_edges, &num_local_edges);
    UF_init(); // Initialize Union-Find structure

    Edge mwoe_to_add; // The edge agreed upon globally in each round

    if (rank == 0) {
         printf("Starting Distributed Boruvka (Processes: %d)...\n", size);
    }

    // Boruvka's main loop: runs at most log(V) times (or until V-1 edges are found)
    while (edges_in_mst < MAX_NODES - 1) {
        
        // --- 1. Local Computation: Find the best MWOE *visible* to this process ---
        Edge local_mwoe = find_local_best_mwoe(local_edges, num_local_edges);

        // --- 2. Global Agreement: Find the overall minimum weight MWOE ---
        
        // We use two allreduces to find the actual MWOE:
        // A. Find the global minimum weight
        int global_min_weight;
        MPI_Allreduce(&local_mwoe.weight, &global_min_weight, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        
        if (global_min_weight == INF) {
             // Graph is disconnected or MST is complete
             break;
        }

        // B. Identify and Broadcast the actual edge that holds the global minimum weight.
        // The process that found the edge with weight == global_min_weight must broadcast it.
        // A simple way is to identify the root process (e.g., lowest rank) that owns it.
        int root_proc = 0; 
        
        // A simplified way to find the actual MWOE edge by checking who found it locally:
        if (local_mwoe.weight == global_min_weight) {
             mwoe_to_add = local_mwoe;
        } 
        
        // Broadcast the final agreed-upon MWOE edge to all processes
        // Note: If multiple processes find the same minimum weight, one must be chosen consistently.
        // Here, the edge found by rank 0 (or the default rank if no one found it) is used.
        MPI_Bcast(&mwoe_to_add, 1, MPI_EDGE, root_proc, MPI_COMM_WORLD); 
        
        // --- 3. Fragment Merging: Update the component IDs (Union-Find) ---
        // All processes now perform the merge operation using the agreed-upon MWOE.
        if (UF_union(mwoe_to_add.u, mwoe_to_add.v)) {
            // Only update total weight if a new merge successfully occurred
            // This condition is handled by the root process for the total weight count.
            if (rank == 0) {
                total_mst_weight += mwoe_to_add.weight;
                edges_in_mst++;
                printf("[Rank 0] Added edge (%d, %d, %d). MST Edges: %d\n", 
                       mwoe_to_add.u, mwoe_to_add.v, mwoe_to_add.weight, edges_in_mst);
            }
        }
        
        // Synchronize the edge count for the loop termination condition
        MPI_Bcast(&edges_in_mst, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }
    
    free(local_edges);
    MPI_Type_free(&MPI_EDGE);

    return total_mst_weight;
}

// -----------------------------------------------------
// 6. Main Function and Testing
// -----------------------------------------------------

int main(int argc, char* argv[]) {
    int rank, size;
    int final_mst_weight = 0;
    const int EXPECTED_MST_WEIGHT = 15;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
        if (rank == 0) {
            printf("--- Test requires at least 2 MPI processes to demonstrate distribution. ---\n");
            printf("Usage: mpiexec -n <num_procs> ./executable\n");
        }
        MPI_Finalize();
        return 0;
    }

    // Run the distributed MST algorithm
    final_mst_weight = run_distributed_mst(rank, size);

    // --- Verification (Only on Root Process) ---
    if (rank == 0) {
        printf("\n=== Distributed Boruvka MST Results ===\n");
        printf("Graph Size: %d Vertices, %d Edges\n", MAX_NODES, NUM_GLOBAL_EDGES);
        printf("MPI Processes Used: %d\n", size);
        printf("--------------------------------------\n");
        printf("Final Calculated MST Weight: %d\n", final_mst_weight);
        printf("Expected MST Weight: %d\n", EXPECTED_MST_WEIGHT);
        
        if (final_mst_weight == EXPECTED_MST_WEIGHT) {
            printf("✅ TEST PASSED: Calculated weight matches expected MST weight.\n");
        } else {
            printf("❌ TEST FAILED: Calculated weight does not match expected MST weight.\n");
        }
        printf("--------------------------------------\n");
    }

    MPI_Finalize();
    
    return 0;
}