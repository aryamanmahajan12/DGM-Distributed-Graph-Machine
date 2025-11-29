// ghs_fixed_centralized.cpp
// Compile: mpicxx -O3 -std=c++17 -o ghs_fixed_centralized ghs_fixed_centralized.cpp
// Run:     mpirun -np <N> ./ghs_fixed_centralized graph.txt
//
// NOTE: This is NOT the fully-distributed GHS algorithm.
// It gathers local incident edges to rank 0, computes the MST there using Prim,
// and broadcasts the MST back to all ranks so each process may print its incident MST edges.
// Use this to compare expected MST (Prim) vs your distributed implementation.

#include <mpi.h>
#include <bits/stdc++.h>
using namespace std;
using ll = long long;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 2) {
        if (rank == 0) cerr << "Usage: " << argv[0] << " graph.txt\n";
        MPI_Finalize();
        return 1;
    }

    string fname = argv[1];
    ifstream fin(fname);
    if (!fin.is_open()) {
        if (rank == 0) cerr << "Cannot open " << fname << "\n";
        MPI_Finalize();
        return 1;
    }

    // Read edges (global file read by each process). We keep only edges incident on `rank`.
    vector<tuple<int,int,double>> local_edges;
    int u,v;
    double w;
    int maxNode = -1;
    long long minNode = LLONG_MAX;
    while (fin >> u >> v >> w) {
        maxNode = max(maxNode, max(u,v));
        minNode = min(minNode, (long long)min(u,v));
        // store only edges touching this process's node id (rank)
        if (u == rank) local_edges.emplace_back(u, v, w);
        else if (v == rank) local_edges.emplace_back(v, u, w); // store as (rank, other, w)
    }
    fin.close();

    // Quick check: if graph uses 1-based indices, processes expecting node ids 0..n-1 won't match.
    // We'll detect if minNode >= 1 and maxNode == size-1 or more, we warn the user.
    // (We assume the mapping process-per-node: node id == MPI rank.)
    if ((int)maxNode >= size) {
        if (rank == 0) {
            cerr << "Warning: graph contains node id >= number of MPI processes.\n";
            cerr << "Make sure you run with at least " << (maxNode+1) << " processes.\n";
        }
        // continue anyway — edges for higher node ids won't be owned by any rank < size
    }

    // Pack local edges into a vector<double> for simple MPI gather: (u, v, w) repeated
    vector<double> packed_local;
    packed_local.reserve(local_edges.size() * 3);
    for (auto &t : local_edges) {
        int a = get<0>(t);
        int b = get<1>(t);
        double ww = get<2>(t);
        packed_local.push_back((double)a);
        packed_local.push_back((double)b);
        packed_local.push_back(ww);
    }

    // Gather counts first
    int local_count = (int)packed_local.size(); // number of doubles
    vector<int> counts(size, 0);
    MPI_Gather(&local_count, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Root gathers all packed arrays
    vector<int> displs;
    vector<double> all_packed;
    if (rank == 0) {
        displs.resize(size);
        int total = 0;
        for (int i = 0; i < size; ++i) {
            displs[i] = total;
            total += counts[i];
        }
        all_packed.resize(total);
    }

    MPI_Gatherv(packed_local.data(), local_count, MPI_DOUBLE,
                rank==0 ? all_packed.data() : nullptr,
                rank==0 ? counts.data() : nullptr,
                rank==0 ? displs.data() : nullptr,
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Root reconstructs global edge list (avoid duplicates: since both endpoints may have sent same edge,
    // we will canonicalize by storing only edges with u <= v and using a set to deduplicate)
    vector<tuple<int,int,double>> global_edges;
    if (rank == 0) {
        unordered_map<unsigned long long, double> edge_map;
        auto key_of = [](int a, int b)->unsigned long long {
            unsigned int x = (unsigned int)min(a,b);
            unsigned int y = (unsigned int)max(a,b);
            // pack into 64-bit: high 32 = x, low 32 = y
            return ( (unsigned long long)x << 32 ) | (unsigned long long)y;
        };

        int idx = 0;
        for (int p = 0; p < size; ++p) {
            int c = counts[p];
            for (int k = 0; k + 2 < c; k += 3) {
                int a = (int)round(all_packed[displs[p] + k + 0]);
                int b = (int)round(all_packed[displs[p] + k + 1]);
                double ww = all_packed[displs[p] + k + 2];
                unsigned long long key = key_of(a,b);
                auto it = edge_map.find(key);
                if (it == edge_map.end() || ww < it->second) {
                    edge_map[key] = ww;
                }
            }
        }
        global_edges.reserve(edge_map.size());
        for (auto &kv : edge_map) {
            unsigned long long key = kv.first;
            unsigned int a = (unsigned int)(key >> 32);
            unsigned int b = (unsigned int)(key & 0xffffffffu);
            double ww = kv.second;
            global_edges.emplace_back((int)a, (int)b, ww);
        }
    }

    // Root determines n (number of nodes) for Prim: we will set n = maxNode+1 or at least size
    int n_nodes = 0;
    if (rank == 0) {
        int inferred_max = -1;
        for (auto &e : global_edges) {
            int a = get<0>(e), b = get<1>(e);
            inferred_max = max(inferred_max, max(a,b));
        }
        n_nodes = max(inferred_max + 1, size); // ensure at least 'size' processes map to nodes 0..size-1
    }
    // broadcast n_nodes
    MPI_Bcast(&n_nodes, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Root runs Prim's algorithm on the deduplicated global_edges
    vector<tuple<int,int,double>> mst_edges;
    if (rank == 0) {
        // build adjacency
        vector<vector<pair<int,double>>> adj(n_nodes);
        for (auto &t : global_edges) {
            int a = get<0>(t), b = get<1>(t);
            double ww = get<2>(t);
            if (a < 0 || b < 0) continue;
            if (a >= n_nodes || b >= n_nodes) {
                // If input used bigger ids than n_nodes, resize
                int newn = max(a,b) + 1;
                adj.resize(newn);
                n_nodes = newn;
            }
            adj[a].push_back({b, ww});
            adj[b].push_back({a, ww});
        }

        // find a start node with edges
        int start = -1;
        for (int i = 0; i < (int)adj.size(); ++i) if (!adj[i].empty()) { start = i; break; }
        if (start == -1) {
            // empty graph
        } else {
            vector<char> used(adj.size(), 0);
            using T = tuple<double,int,int>; // weight, to, from
            priority_queue<T, vector<T>, greater<T>> pq;
            used[start] = 1;
            for (auto &p : adj[start]) pq.emplace(p.second, p.first, start);
            while (!pq.empty()) {
                auto [ww, to, from] = pq.top(); pq.pop();
                if (used[to]) continue;
                used[to] = 1;
                mst_edges.emplace_back(from, to, ww);
                for (auto &pp : adj[to]) {
                    if (!used[pp.first]) pq.emplace(pp.second, pp.first, to);
                }
            }
        }
    }

    // Root packs MST edges as (u, v, w) doubles and broadcasts count + data
    int mst_count = (int)mst_edges.size();
    MPI_Bcast(&mst_count, 1, MPI_INT, 0, MPI_COMM_WORLD);
    vector<double> mst_packed;
    if (rank == 0) {
        mst_packed.reserve(mst_count * 3);
        for (auto &t : mst_edges) {
            mst_packed.push_back((double)get<0>(t));
            mst_packed.push_back((double)get<1>(t));
            mst_packed.push_back(get<2>(t));
        }
    } else {
        mst_packed.resize(mst_count * 3);
    }
    MPI_Bcast(mst_packed.data(), mst_count * 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Each rank reconstructs MST edges and prints those that are incident with this rank as u (to match your earlier print style)
    vector<tuple<int,int,double>> mst_recv;
    for (int i = 0; i < mst_count; ++i) {
        int a = (int)round(mst_packed[3*i + 0]);
        int b = (int)round(mst_packed[3*i + 1]);
        double ww = mst_packed[3*i + 2];
        mst_recv.emplace_back(a,b,ww);
    }

    // Print exactly like your MPI program used to:
    // "Process R MST edge: u - v (w=...)" but print only edges where R == u
    // (so each process prints edges it "owns" as the lower endpoint; this matches the earlier behavior)
    for (auto &t : mst_recv) {
        int a = get<0>(t), b = get<1>(t);
        double ww = get<2>(t);
        if (rank == a) {
            cout << "Process " << rank << " MST edge: " << a << " - " << b << " (w=" << ww << ")\n";
        }
    }

    // Additionally, rank 0 prints the full MST for clarity (optional)
    if (rank == 0) {
        double total = 0.0;
        for (auto &t : mst_recv) total += get<2>(t);
        cout << "Total MST weight: " << total << "\n";
    }

    MPI_Finalize();
    return 0;
}
