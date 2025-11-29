// boruvka_kmachines_fixed_output_with_total.cpp
// Fixed K-machine Boruvka MST using MPI with final MST output formatting.
// Changes from original:
//  - Final MST edges are sorted deterministically and printed in a clear "u v w" format
//  - Added MPI_Barrier before final printing to avoid interleaved output
//  - Root prints the canonical MST once; non-roots remain silent
//  - Fixed precision for weights
//  - Added printing of the total weight (sum of weights) of the final MST

#include <mpi.h>
#include <bits/stdc++.h>
using namespace std;

struct Edge { int to; double w; };
struct Cand { int frag; int u; int v; double w; };

struct DSU {
    int n;
    vector<int> p, r;
    DSU(int n_=0){ init(n_); }
    void init(int n_){ n = n_; p.resize(n); r.assign(n,0); for(int i=0;i<n;++i) p[i]=i; }
    int find(int x){ return p[x]==x ? x : (p[x]=find(p[x])); }
    bool unite(int a,int b){
        a = find(a); b = find(b);
        if(a==b) return false;
        if(r[a] < r[b]) swap(a,b);
        p[b] = a;
        if(r[a]==r[b]) r[a]++;
        return true;
    }
};

int main(int argc,char** argv){
    MPI_Init(&argc,&argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);

    if(argc < 2){
        if(rank==0) cerr<<"Usage: mpirun -n K ./boruvka_kmachines_fixed graph.txt\n";
        MPI_Finalize();
        return 1;
    }
    string fname = argv[1];

    // Read graph (everyone)
    ifstream fin(fname);
    if(!fin.is_open()){
        if(rank==0) cerr<<"Cannot open "<<fname<<"\n";
        MPI_Finalize();
        return 1;
    }
    vector<tuple<int,int,double>> edges_all;
    int a,b; double w;
    int maxVertex=-1;
    while(fin >> a >> b >> w){
        edges_all.emplace_back(a,b,w);
        maxVertex = max(maxVertex, max(a,b));
    }
    fin.close();
    int n = maxVertex + 1;
    if(n <= 0){
        if(rank==0) cerr<<"Empty graph\n";
        MPI_Finalize();
        return 1;
    }

    // Partition nodes across machines: contiguous blocks
    int per = (n + size - 1) / size;
    int start = rank * per;
    int end = min(n, start + per);
    int local_n = max(0, end - start);

    // local adjacency for owned vertices
    vector<vector<Edge>> adj_local(local_n);
    for(auto &t : edges_all){
        int u,vv; double ww; tie(u,vv,ww) = t;
        if(u >= start && u < end) adj_local[u - start].push_back({vv, ww});
        if(vv >= start && vv < end) adj_local[vv - start].push_back({u, ww});
    }
    for(auto &lst : adj_local) sort(lst.begin(), lst.end(), [](const Edge& A,const Edge& B){
        if (A.w != B.w) return A.w < B.w;
        return A.to < B.to;
    });

    // parent[]: representative (fragment id) for each vertex; every machine keeps full copy
    vector<int> parent(n);
    for(int i=0;i<n;++i) parent[i] = i;

    // MST accumulator at root: store canonical edges (u < v)
    set<pair<pair<int,int>, double>> mst_set; // ((u,v), w)

    if(rank==0) cerr<<"Starting Boruvka on "<<size<<" machines; n="<<n<<", per="<<per<<"\n";

    bool done = false;
    int phase = 0;
    while(!done){
        phase++;
        // Step 1: local candidate MOE per fragment we see
        unordered_map<int, Cand> local_best;
        for(int i=0;i<local_n;++i){
            int v_global = start + i;
            int frag_v = parent[v_global];
            for(const Edge &e: adj_local[i]){
                int u_global = e.to;
                int frag_u = parent[u_global];
                if(frag_u != frag_v){
                    // candidate for frag_v
                    auto it = local_best.find(frag_v);
                    if(it==local_best.end() || e.w < it->second.w ||
                       (e.w == it->second.w && make_pair(v_global,u_global) < make_pair(it->second.u, it->second.v))){
                        local_best[frag_v] = Cand{frag_v, v_global, u_global, e.w};
                    }
                    break; // amortized
                }
            }
        }

        // Pack local candidates
        int local_count = (int)local_best.size();
        vector<int> ints; ints.reserve(local_count*3);
        vector<double> dws; dws.reserve(local_count);
        for(auto &kv : local_best){
            const Cand &c = kv.second;
            ints.push_back(c.frag);
            ints.push_back(c.u);
            ints.push_back(c.v);
            dws.push_back(c.w);
        }

        // Gather at root
        if(rank == 0){
            vector<Cand> all_cands;
            // include root's own
            for(int i=0;i<local_count;++i){
                all_cands.push_back(Cand{ints[3*i], ints[3*i+1], ints[3*i+2], dws[i]});
            }
            for(int src=1; src<size; ++src){
                MPI_Status st;
                int recv_count;
                MPI_Recv(&recv_count, 1, MPI_INT, src, 10, MPI_COMM_WORLD, &st);
                if(recv_count == 0) continue;
                vector<int> rints(3*recv_count);
                vector<double> rdws(recv_count);
                MPI_Recv(rints.data(), 3*recv_count, MPI_INT, src, 11, MPI_COMM_WORLD, &st);
                MPI_Recv(rdws.data(), recv_count, MPI_DOUBLE, src, 12, MPI_COMM_WORLD, &st);
                for(int i=0;i<recv_count;++i){
                    all_cands.push_back(Cand{rints[3*i], rints[3*i+1], rints[3*i+2], rdws[i]});
                }
            }

            // Pick global best per fragment (based on fragment id in parent[] snapshot)
            unordered_map<int, Cand> best_per_frag;
            for(const auto &c : all_cands){
                auto it = best_per_frag.find(c.frag);
                if(it==best_per_frag.end() || c.w < it->second.w ||
                   (c.w == it->second.w && make_pair(c.u,c.v) < make_pair(it->second.u, it->second.v))){
                    best_per_frag[c.frag] = c;
                }
            }

            // Important fix: union fragments by fragment-representatives (parent[u], parent[v])
            // Create DSU initialized from current parent[] snapshot
            DSU dsu(n);
            for(int i=0;i<n;++i) dsu.p[i] = parent[i];
            for(int i=0;i<n;++i) dsu.p[i] = dsu.find(dsu.p[i]);

            vector<tuple<int,int,double>> added_this_phase;

            // To avoid ordering issues that could merge multiple best edges for same fragment
            // we process best_per_frag in deterministic order (by fragment id)
            vector<int> frag_keys;
            frag_keys.reserve(best_per_frag.size());
            for(auto &kv : best_per_frag) frag_keys.push_back(kv.first);
            sort(frag_keys.begin(), frag_keys.end());

            for(int frag : frag_keys){
                const Cand &c = best_per_frag[frag];
                int fu = parent[c.u]; // fragment rep for u
                int fv = parent[c.v]; // fragment rep for v
                fu = dsu.find(fu);
                fv = dsu.find(fv);
                if(fu != fv){
                    bool merged = dsu.unite(fu, fv);
                    if(merged){
                        // canonicalize edge (min,max)
                        int uu = c.u, vv = c.v;
                        if(uu > vv) swap(uu, vv);
                        added_this_phase.emplace_back(uu, vv, c.w);
                        // Also add to mst_set (deduplicated global set)
                        mst_set.insert({{min(uu,vv), max(uu,vv)}, c.w});
                    }
                }
            }

            // update parent[] from DSU
            for(int i=0;i<n;++i) parent[i] = dsu.find(i);

            // component count
            unordered_set<int> comps;
            for(int i=0;i<n;++i) comps.insert(parent[i]);
            int num_comps = (int)comps.size();
            int done_flag = (num_comps <= 1) ? 1 : 0;

            if(rank==0){
                cerr<<"Phase "<<phase<<": components="<<num_comps<<", edges_added_this_phase="<<added_this_phase.size()<<"\n";
            }

            // broadcast done_flag & parent[]
            MPI_Bcast(&done_flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Bcast(parent.data(), n, MPI_INT, 0, MPI_COMM_WORLD);
            done = (done_flag == 1);
        } else {
            // non-root sends and waits for broadcast
            MPI_Send(&local_count, 1, MPI_INT, 0, 10, MPI_COMM_WORLD);
            if(local_count > 0){
                MPI_Send(ints.data(), 3*local_count, MPI_INT, 0, 11, MPI_COMM_WORLD);
                MPI_Send(dws.data(), local_count, MPI_DOUBLE, 0, 12, MPI_COMM_WORLD);
            }
            int done_flag;
            MPI_Bcast(&done_flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
            parent.assign(n,0);
            MPI_Bcast(parent.data(), n, MPI_INT, 0, MPI_COMM_WORLD);
            done = (done_flag == 1);
        }
    } // end phases

    // Root composes final MST vector (deduplicated) from mst_set and broadcasts
    vector<pair<int,int>> edges_out;
    vector<double> weights_out;
    if(rank == 0){
        edges_out.reserve(mst_set.size());
        weights_out.reserve(mst_set.size());
        for(auto &it : mst_set){
            edges_out.push_back(it.first);
            weights_out.push_back(it.second);
        }
        // sort deterministically by (u,v)
        vector<int> idx(edges_out.size());
        iota(idx.begin(), idx.end(), 0);
        sort(idx.begin(), idx.end(), [&](int A,int B){
            if(edges_out[A].first != edges_out[B].first) return edges_out[A].first < edges_out[B].first;
            return edges_out[A].second < edges_out[B].second;
        });
        vector<pair<int,int>> edges_sorted;
        vector<double> weights_sorted;
        edges_sorted.reserve(edges_out.size());
        weights_sorted.reserve(weights_out.size());
        for(int id : idx){ edges_sorted.push_back(edges_out[id]); weights_sorted.push_back(weights_out[id]); }

        int mcount = (int)edges_sorted.size();
        MPI_Bcast(&mcount, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if(mcount > 0){
            vector<int> ints(2*mcount);
            for(int i=0;i<mcount;++i){ ints[2*i] = edges_sorted[i].first; ints[2*i+1] = edges_sorted[i].second; }
            MPI_Bcast(ints.data(), 2*mcount, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Bcast(weights_sorted.data(), mcount, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        }

        // print final MST in a simple deterministic format: one edge per line "u v w"
        cout.setf(std::ios::fixed); cout<<setprecision(6);

        // Barrier to ensure other ranks have finished any pending I/O (avoids interleaving)
        MPI_Barrier(MPI_COMM_WORLD);

        cout<<"MST edges : "<<endl;

        for(size_t i=0;i<edges_sorted.size();++i){
            cout<<edges_sorted[i].first<<" "<<edges_sorted[i].second<<" "<<weights_sorted[i]<<endl;
        }

        // compute and print total weight of MST
        double total_weight = 0.0;
        for(double val : weights_sorted) total_weight += val;
        cout<<"Total MST weight: "<<total_weight<<endl;
        cout.flush();
    } else {
        // Non-root: receive the canonical MST but stay silent (no printing)
        int mcount;
        MPI_Bcast(&mcount, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if(mcount > 0){
            vector<int> ints(2*mcount);
            vector<double> dws(mcount);
            MPI_Bcast(ints.data(), 2*mcount, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Bcast(dws.data(), mcount, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            // intentionally do nothing: only root prints final MST
        }
    }

    // Ensure all prints complete before finalizing
    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Finalize();
    return 0;
}
