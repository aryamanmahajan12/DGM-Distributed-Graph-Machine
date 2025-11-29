// boruvka_mpi.cpp
// Compile: mpicxx -O3 -std=c++17 -o boruvka_mpi boruvka_mpi.cpp
// Run: mpirun -np <N> ./boruvka_mpi graph.txt
//
// Assumptions:
// - Each MPI rank corresponds to a graph vertex with the same id (rank).
// - graph.txt contains only lines "u v w" (whitespace separated). 0-based or 1-based IDs are auto-detected.
// - All processes can read graph.txt from a shared FS.
// - The program will detect if you ran with fewer processes than needed and warn.
//
// Behavior:
// - Parallel stage: each rank scans its incident edges and emits a candidate cheapest outgoing edge for its current component.
// - Central coordinator (rank 0) collects candidates, chooses the minimum outgoing edge per component, merges components (union-find),
//   and broadcasts updated component ids. Repeats until convergence or graph disconnected.
// - At the end rank 0 has the MST edges; the code broadcasts the final MST back and each process prints lines of the form:
//     Process R MST edge: u - v (w=...)
//
// Notes:
// - This implementation focuses on clarity and correctness (suitable for debugging/comparison)
//   rather than extreme scalability (component id arrays are broadcast each round).

#include <mpi.h>
#include <bits/stdc++.h>
using namespace std;
using ll = long long;

struct Edge { int to; double w; };
struct Candidate {
    int valid;      // 0 or 1
    int from_comp;  // origin component id
    int to_comp;    // neighbor component id
    int u;          // local node (rank)
    int v;          // neighbor node
    double w;
};

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

    // Read edges; each process keeps only edges incident to its node (node id == rank)
    vector<pair<int,double>> adj; // neighbor, weight
    int a,b;
    double w;
    int local_max = -1, local_min = INT_MAX;
    vector<tuple<int,int,double>> all_read; // to compute global max/min
    while (fin >> a >> b >> w) {
        all_read.emplace_back(a,b,w);
        local_max = max(local_max, max(a,b));
        local_min = min(local_min, min(a,b));
    }
    fin.close();

    // detect if 1-based indexing likely
    bool one_based = false;
    if (local_min >= 1) one_based = true;

    // build local adjacency (converted to 0-based if needed)
    for (auto &t : all_read) {
        int x = get<0>(t);
        int y = get<1>(t);
        double ww = get<2>(t);
        if (one_based) { x--; y--; }
        if (x == rank) adj.emplace_back(y, ww);
        else if (y == rank) adj.emplace_back(x, ww);
    }

    // find global max node id to determine n_nodes
    int global_max = -1;
    int local_max_node = local_max;
    if (one_based) local_max_node = local_max - 1;
    if (local_max_node < 0) local_max_node = -1;
    MPI_Allreduce(&local_max_node, &global_max, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    int n_nodes = global_max + 1;
    if (n_nodes < 0) n_nodes = 0;

    // If there are more nodes than MPI processes, warn (we expect one process per node)
    if (size < n_nodes) {
        if (rank == 0) {
            cerr << "Warning: graph requires at least " << n_nodes << " processes, but mpirun provided " << size << ".\n";
            cerr << "Edges for node ids >= " << size << " will not be owned by any process and algorithm won't be correct.\n";
        }
        // Still continue, but results may be meaningless.
    }

    // Initialize component ids: each node in its own component
    // We'll maintain comp_id array of length n_nodes on all processes (broadcast each round)
    vector<int> comp_id(n_nodes);
    for (int i = 0; i < n_nodes; ++i) comp_id[i] = i;

    // MST edges (kept at rank 0)
    vector<tuple<int,int,double>> mst_edges;

    // Helper: pack/unpack Candidate into double array of length 6 for simple MPI_Gather
    auto pack_candidate = [](const Candidate &c, array<double,6> &out) {
        out[0] = (double)c.valid;
        out[1] = (double)c.from_comp;
        out[2] = (double)c.to_comp;
        out[3] = (double)c.u;
        out[4] = (double)c.v;
        out[5] = c.w;
    };
    auto unpack_candidate = [](const array<double,6> &in)->Candidate{
        Candidate c;
        c.valid = (int)round(in[0]);
        c.from_comp = (int)round(in[1]);
        c.to_comp = (int)round(in[2]);
        c.u = (int)round(in[3]);
        c.v = (int)round(in[4]);
        c.w = in[5];
        return c;
    };

    int max_rounds = 1000; // safety cap
    for (int round = 0; round < max_rounds; ++round) {
        // Each process computes a candidate cheapest outgoing edge for its component
        Candidate cand;
        cand.valid = 0;
        if (rank < n_nodes) {
            int my_comp = comp_id[rank];
            double bestw = numeric_limits<double>::infinity();
            int bestu=-1, bestv=-1, best_to_comp=-1;
            for (auto &pr : adj) {
                int nb = pr.first; double wt = pr.second;
                // if neighbor index outside known nodes, treat as different component
                int nb_comp = (nb >= 0 && nb < (int)comp_id.size()) ? comp_id[nb] : nb;
                if (nb_comp != my_comp) {
                    // outgoing edge
                    // tie-break by (weight, min(u,v), max(u,v))
                    bool better = false;
                    if (wt < bestw) better = true;
                    else if (std::abs(wt - bestw) < 1e-12) {
                        int ming = min(rank, nb), maxg = max(rank, nb);
                        int bestmin = bestu==-1?INT_MAX:min(bestu,bestv);
                        int bestmax = bestu==-1?INT_MAX:max(bestu,bestv);
                        if (ming < bestmin || (ming==bestmin && maxg < bestmax)) better=true;
                    }
                    if (better) {
                        bestw = wt; bestu = rank; bestv = nb; best_to_comp = nb_comp;
                    }
                }
            }
            if (bestu != -1) {
                cand.valid = 1;
                cand.from_comp = comp_id[rank];
                cand.to_comp = best_to_comp;
                cand.u = bestu;
                cand.v = bestv;
                cand.w = bestw;
            }
        }
        // pack to double[6]
        array<double,6> send_buf;
        pack_candidate(cand, send_buf);

        // gather all candidates at root
        vector<double> gathered; // size = 6 * size
        if (rank == 0) gathered.resize(6 * size);
        MPI_Gather(send_buf.data(), 6, MPI_DOUBLE,
                   rank==0 ? gathered.data() : nullptr, 6, MPI_DOUBLE,
                   0, MPI_COMM_WORLD);

        // Root selects minimal candidate per component and performs unions
        vector<int> parent;
        auto uf_init = [&](int n) {
            parent.assign(n,0);
            for (int i=0;i<n;++i) parent[i]=i;
        };
        function<int(int)> uf_find;
        function<void(int,int)> uf_unite;
        if (rank == 0) {
            // union-find over components indexed by node id space (0..n_nodes-1)
            uf_init(max(1, n_nodes));
            uf_find = [&](int x)->int {
                if (x<0) return x;
                if (x >= (int)parent.size()) return x;
                return parent[x]==x ? x : parent[x] = uf_find(parent[x]);
            };
            uf_unite = [&](int x,int y) {
                int rx = uf_find(x), ry = uf_find(y);
                if (rx==ry) return;
                parent[ry] = rx;
            };

            // map from component -> chosen candidate (the minimum outgoing edge)
            unordered_map<int, Candidate> best_per_comp;
            for (int p = 0; p < size; ++p) {
                array<double,6> arr;
                for (int k=0;k<6;++k) arr[k] = gathered[p*6 + k];
                Candidate c = unpack_candidate(arr);
                if (!c.valid) continue;
                int comp = c.from_comp;
                auto it = best_per_comp.find(comp);
                if (it == best_per_comp.end()) best_per_comp[comp] = c;
                else {
                    Candidate prev = it->second;
                    // choose smaller weight (tie-break deterministically)
                    if (c.w < prev.w - 1e-12) best_per_comp[comp] = c;
                    else if (fabs(c.w - prev.w) < 1e-12) {
                        int cmin = min(c.u, c.v), cmax = max(c.u, c.v);
                        int pmin = min(prev.u, prev.v), pmax = max(prev.u, prev.v);
                        if (cmin < pmin || (cmin==pmin && cmax < pmax)) best_per_comp[comp] = c;
                    }
                }
            }

            // Now perform unions using chosen edges. We'll add an MST edge for each successful union.
            int merges = 0;
            for (auto &kv : best_per_comp) {
                Candidate c = kv.second;
                int compA = uf_find(c.from_comp);
                int compB = uf_find(c.to_comp);
                if (compA == compB) continue;
                // merge compB into compA (arbitrary)
                uf_unite(compA, compB);
                merges++;
                // record edge (u,v,w)
                mst_edges.emplace_back(c.u, c.v, c.w);
            }

            // If merges==0 and more than 1 component remain, the graph is disconnected (cannot merge further)
            // Build new comp_id: for every original node 0..n_nodes-1 set comp_id[i] = find(i)
            vector<int> new_comp(n_nodes);
            for (int i=0;i<n_nodes;++i) {
                new_comp[i] = uf_find(i);
            }
            // compress ids to small range (optional) to keep numbers stable
            unordered_map<int,int> remap;
            int nextid = 0;
            for (int i=0;i<n_nodes;++i) {
                int key = new_comp[i];
                if (remap.find(key) == remap.end()) remap[key] = nextid++;
                new_comp[i] = remap[key];
            }

            // copy new_comp to comp_id
            comp_id = new_comp;
        }

        // Broadcast updated comp_id (length n_nodes) to all processes
        if (n_nodes == 0) {
            // nothing to do
            break;
        }
        // First, ensure every process has comp_id vector sized n_nodes
        if (rank != 0) comp_id.assign(n_nodes, 0);
        MPI_Bcast(comp_id.data(), n_nodes, MPI_INT, 0, MPI_COMM_WORLD);

        // Count unique components (on each process or just root)
        vector<int> uniq(n_nodes);
        for (int i=0;i<n_nodes;++i) uniq[i] = comp_id[i];
        sort(uniq.begin(), uniq.end());
        int unique_comps = (n_nodes>0) ? (1 + (int)(unique(uniq.begin(), uniq.end()) - uniq.begin()) - 1) : 0;
        // The above is clumsy; simpler:
        set<int> sset(comp_id.begin(), comp_id.end());
        unique_comps = (int)sset.size();

        // Check termination: if only 1 component OR no merges possible (we can detect if root added no edges this round)
        int root_merges = 0;
        if (rank == 0) {
            // merges = number of MST edges added this round. We can approximate by counting how many edges added last iteration;
            // but simpler: if best_per_comp was empty then no merges. We didn't carry best_per_comp to here.
            // Instead, detect progress by asking if any merges happened: compare previous number of components
            // For simplicity, we'll ask root to compute 'progress' each round by seeing if at least one new edge was added.
            // We'll set root_merges = number of edges added in this round: mst_edges.size() - prev_total
            static int prev_mst_count = 0;
            root_merges = (int)mst_edges.size() - prev_mst_count;
            prev_mst_count = (int)mst_edges.size();
        }
        MPI_Bcast(&root_merges, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // If no merges at root and multiple components remain -> disconnected
        if (root_merges == 0) {
            if (rank == 0) {
                if (unique_comps > 1) {
                    cerr << "Graph appears disconnected: no merges possible but " << unique_comps << " components remain.\n";
                }
            }
            break;
        }
        if (unique_comps <= 1) break;
        // loop continues
    } // end rounds

    // At the end, root has mst_edges. Broadcast final MST to all ranks
    int mst_count = 0;
    if (rank == 0) mst_count = (int)mst_edges.size();
    MPI_Bcast(&mst_count, 1, MPI_INT, 0, MPI_COMM_WORLD);
    vector<double> mst_packed(mst_count * 3);
    if (rank == 0) {
        for (int i = 0; i < mst_count; ++i) {
            mst_packed[3*i+0] = (double)get<0>(mst_edges[i]);
            mst_packed[3*i+1] = (double)get<1>(mst_edges[i]);
            mst_packed[3*i+2] = get<2>(mst_edges[i]);
        }
    }
    if (rank != 0) fill(mst_packed.begin(), mst_packed.end(), 0.0);
    MPI_Bcast(mst_packed.data(), mst_count * 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Each process prints MST edges where it equals the smaller endpoint (to match earlier print style)
    vector<tuple<int,int,double>> mst_recv;
    for (int i = 0; i < mst_count; ++i) {
        int u = (int)round(mst_packed[3*i+0]);
        int v = (int)round(mst_packed[3*i+1]);
        double ww = mst_packed[3*i+2];
        mst_recv.emplace_back(u, v, ww);
    }
    for (auto &t : mst_recv) {
        int u = get<0>(t), v = get<1>(t);
        double ww = get<2>(t);
        if (rank == u) {
            cout << "Process " << rank << " MST edge: " << u << " - " << v << " (w=" << ww << ")\n";
        }
    }
    if (rank == 0) {
        double total = 0.0;
        for (auto &t : mst_recv) total += get<2>(t);
        cout << "Total MST weight: " << total << "\n";
    }

    MPI_Finalize();
    return 0;
}
