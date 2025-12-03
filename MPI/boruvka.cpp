// boruvka_mpi_fixed.cpp
#include <mpi.h>
#include <bits/stdc++.h>
using namespace std;

struct EdgeInput { int u,v; double w; };

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 2) {
        if (rank == 0) cerr << "Usage: mpirun -n <N> ./boruvka_mpi_fixed graph.txt\n";
        MPI_Finalize();
        return 1;
    }

    string fname = argv[1];

    // root reads file and broadcasts edges to everyone
    vector<int> us, vs;
    vector<double> ws;
    int numEdges = 0;
    int maxVertex = -1;

    if (rank == 0) {
        ifstream fin(fname);
        if (!fin.is_open()) {
            cerr << "Cannot open " << fname << "\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        int a,b; double w;
        while (fin >> a >> b >> w) {
            us.push_back(a);
            vs.push_back(b);
            ws.push_back(w);
            maxVertex = max(maxVertex, max(a,b));
        }
        fin.close();
        numEdges = (int)us.size();
    }

    // broadcast numEdges and maxVertex
    MPI_Bcast(&numEdges, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&maxVertex, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // ensure number of processes covers vertices
    if (size < maxVertex + 1) {
        if (rank == 0) cerr << "Error: need at least " << (maxVertex + 1) << " processes.\n";
        MPI_Finalize();
        return 1;
    }

    // allocate arrays on non-root ranks
    if (rank != 0) {
        us.resize(numEdges);
        vs.resize(numEdges);
        ws.resize(numEdges);
    }

    // broadcast arrays
    if (numEdges > 0) {
        MPI_Bcast(us.data(), numEdges, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(vs.data(), numEdges, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(ws.data(), numEdges, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    // Build adjacency list local to this process (edges incident to this vertex)
    vector<pair<int,double>> adj; // (neighbor, weight)
    for (int i = 0; i < numEdges; ++i) {
        if (us[i] == rank) adj.emplace_back(vs[i], ws[i]);
        else if (vs[i] == rank) adj.emplace_back(us[i], ws[i]);
    }

    // initial component id = vertex id
    vector<int> comp_id(size);
    for (int i = 0; i < size; ++i) comp_id[i] = i;

    // global component array to be shared each round
    vector<int> global_comp(size);

    // MST edges collected at root
    vector<EdgeInput> mst_edges;

    // Main Boruvka loop
    while (true) {
        int my_comp = comp_id[rank];

        // Use MPI_Allgather so every process gets the full component array
        MPI_Allgather(&my_comp, 1, MPI_INT, global_comp.data(), 1, MPI_INT, MPI_COMM_WORLD);

        // Each process finds its local best outgoing edge
        double local_best_w = numeric_limits<double>::infinity();
        int local_best_nb = -1;
        for (auto &e : adj) {
            int nb = e.first;
            double w = e.second;
            if (global_comp[nb] != global_comp[rank]) {
                if (w < local_best_w || (w == local_best_w && nb < local_best_nb)) {
                    local_best_w = w;
                    local_best_nb = nb;
                }
            }
        }

        // Prepare buffers for root to gather
        int send_nb = (local_best_nb == -1 ? -1 : local_best_nb);
        double send_w = (local_best_nb == -1 ? numeric_limits<double>::infinity() : local_best_w);

        // All ranks must provide recv buffers for gather on root; allocate on all ranks for simplicity
        vector<int> all_nb(size);
        vector<double> all_w(size);

        MPI_Gather(&send_nb, 1, MPI_INT, all_nb.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Gather(&send_w, 1, MPI_DOUBLE, all_w.data(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // Root: for each component, pick minimal outgoing edge among collected candidates
        vector<int> new_comp(size, -1);
        if (rank == 0) {
            // Initialize Union-Find over vertices (for merging chosen edges)
            vector<int> parent(size);
            vector<int> rnk(size, 0);
            for (int i = 0; i < size; ++i) parent[i] = i;
            function<int(int)> findp = [&](int x){ return parent[x]==x ? x : parent[x]=findp(parent[x]); };
            auto unite = [&](int a, int b)->bool {
                a = findp(a); b = findp(b);
                if (a == b) return false;
                if (rnk[a] < rnk[b]) swap(a,b);
                parent[b] = a;
                if (rnk[a] == rnk[b]) ++rnk[a];
                return true;
            };

            // Map current component id -> vertices in it
            unordered_map<int, vector<int>> comp_to_vertices;
            for (int vtx = 0; vtx < size; ++vtx) {
                int c = global_comp[vtx];
                comp_to_vertices[c].push_back(vtx);
            }

            struct Chosen { int u,v; double w; bool valid; };
            vector<Chosen> chosen_list;
            chosen_list.reserve(comp_to_vertices.size());

            for (auto &kv : comp_to_vertices) {
                int comp = kv.first;
                double bestw = numeric_limits<double>::infinity();
                int bestu = -1, bestv = -1;
                for (int u_v : kv.second) {
                    int nb = all_nb[u_v];
                    double w = all_w[u_v];
                    if (nb == -1) continue;
                    if (global_comp[nb] == comp) continue; // not outgoing
                    if (w < bestw || (w == bestw && (bestu == -1 || u_v < bestu))) {
                        bestw = w;
                        bestu = u_v;
                        bestv = nb;
                    }
                }
                if (bestu != -1) chosen_list.push_back({bestu, bestv, bestw, true});
            }

            // Sort chosen edges (deterministic)
            sort(chosen_list.begin(), chosen_list.end(), [](const Chosen &A, const Chosen &B){
                if (A.w != B.w) return A.w < B.w;
                if (A.u != B.u) return A.u < B.u;
                return A.v < B.v;
            });

            // Merge using union-find and add to MST if they connect different components
            for (auto &ch : chosen_list) {
                if (unite(ch.u, ch.v)) {
                    mst_edges.push_back({ch.u, ch.v, ch.w});
                }
            }

            // Compute new component id for every vertex as findp(vertex)
            for (int vtx = 0; vtx < size; ++vtx) {
                new_comp[vtx] = findp(vtx);
            }

            // Relabel to compact ids
            unordered_map<int,int> comp_rename;
            int nextid = 0;
            for (int vtx = 0; vtx < size; ++vtx) {
                int rootc = new_comp[vtx];
                if (!comp_rename.count(rootc)) comp_rename[rootc] = nextid++;
                new_comp[vtx] = comp_rename[rootc];
            }
        }

        // Broadcast new_comp to all processes
        MPI_Bcast(new_comp.data(), size, MPI_INT, 0, MPI_COMM_WORLD);

        // Update local comp_id for next round
        for (int i = 0; i < size; ++i) comp_id[i] = new_comp[i];

        // Root checks whether single component remains
        int unique_comps = 0;
        if (rank == 0) {
            unordered_set<int> s;
            for (int i = 0; i < size; ++i) s.insert(new_comp[i]);
            unique_comps = (int)s.size();
        }
        MPI_Bcast(&unique_comps, 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (unique_comps <= 1) break;
    } // end while

    // root prints MST edges and total weight
    if (rank == 0) {
        double total = 0.0;
        cout << "Boruvka MST edges:\n";
        for (auto &e : mst_edges) {
            cout << e.u << " - " << e.v << " (w=" << e.w << ")\n";
            total += e.w;
        }
        cout << "Total MST weight: " << total << "\n";
    }

    MPI_Finalize();
    return 0;
}
