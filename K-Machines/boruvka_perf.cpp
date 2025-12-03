// boruvka_kmachines_fixed_output_with_total_and_logging.cpp
// Boruvka MST (K-machine MPI) with timing, per-phase metrics and CSV logging.
// Edits added:
//  - MPI_Wtime timers: total wall time and per-phase local times
//  - per-phase max (across ranks) recorded at root
//  - edges added per phase recorded at root
//  - final CSV logging (append) with JSON-encoded per-phase arrays
//  - CSV header automatically written if file doesn't exist

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

// Helper: check if file exists
static bool file_exists(const string &fname){
    ifstream f(fname);
    return f.good();
}

// Helper: check if file is empty (or does not exist)
static bool file_is_empty(const string &fname){
    ifstream f(fname);
    if(!f.good()) return true;
    // peek for first non-whitespace
    char c;
    while(f.get(c)){
        if(!isspace((unsigned char)c)) return false;
    }
    return true;
}

int main(int argc,char** argv){
    MPI_Init(&argc,&argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);

    if(argc < 2){
        if(rank==0) cerr<<"Usage: mpirun -n K ./boruvka graph.txt [run_id]\n";
        MPI_Finalize();
        return 1;
    }
    string fname = argv[1];
    string run_id = "";
    if(argc >= 3) run_id = argv[2];
    if(run_id.empty()){
        // fallback run id: timestamp
        time_t t = time(NULL);
        run_id = to_string((long long)t);
    }

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
    int m = (int)edges_all.size();
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

    if(rank==0) cerr<<"Starting Boruvka on "<<size<<" machines; n="<<n<<", m="<<m<<", per="<<per<<"\n";

    bool done = false;
    int phase = 0;

    // timing and per-phase arrays (collected at root)
    vector<double> per_phase_max_time; // per-phase maximum time across ranks
    vector<int> edges_added_per_phase; // edges added each phase at root

    // start total timer
    double t_start = MPI_Wtime();

    while(!done){
        phase++;

        double phase_start = MPI_Wtime();

        // Step 1: local candidate MOE per fragment we see
        unordered_map<int, Cand> local_best;
        for(int i=0;i<local_n;++i){
            int v_global = start + i;
            int frag_v = parent[v_global];
            for(const Edge &e: adj_local[i]){
                int u_global = e.to;
                int frag_u = parent[u_global];
                if(frag_u != frag_v){
                    auto it = local_best.find(frag_v);
                    if(it==local_best.end() || e.w < it->second.w ||
                       (e.w == it->second.w && make_pair(v_global,u_global) < make_pair(it->second.u, it->second.v))){
                        local_best[frag_v] = Cand{frag_v, v_global, u_global, e.w};
                    }
                    break;
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

            unordered_map<int, Cand> best_per_frag;
            for(const auto &c : all_cands){
                auto it = best_per_frag.find(c.frag);
                if(it==best_per_frag.end() || c.w < it->second.w ||
                   (c.w == it->second.w && make_pair(c.u,c.v) < make_pair(it->second.u, it->second.v))){
                    best_per_frag[c.frag] = c;
                }
            }

            // Create DSU initialized from current parent[]
            DSU dsu(n);
            for(int i=0;i<n;++i) dsu.p[i] = parent[i];
            for(int i=0;i<n;++i) dsu.p[i] = dsu.find(dsu.p[i]);

            vector<tuple<int,int,double>> added_this_phase;

            vector<int> frag_keys;
            for(auto &kv : best_per_frag) frag_keys.push_back(kv.first);
            sort(frag_keys.begin(), frag_keys.end());

            for(int frag : frag_keys){
                const Cand &c = best_per_frag[frag];
                int fu = parent[c.u];
                int fv = parent[c.v];
                fu = dsu.find(fu);
                fv = dsu.find(fv);
                if(fu != fv){
                    bool merged = dsu.unite(fu, fv);
                    if(merged){
                        int uu = c.u, vv = c.v;
                        if(uu > vv) swap(uu, vv);
                        added_this_phase.emplace_back(uu, vv, c.w);
                        mst_set.insert({{min(uu,vv), max(uu,vv)}, c.w});
                    }
                }
            }

            for(int i=0;i<n;++i) parent[i] = dsu.find(i);

            unordered_set<int> comps;
            for(int i=0;i<n;++i) comps.insert(parent[i]);
            int num_comps = comps.size();
            int done_flag = (num_comps <= 1);

            MPI_Bcast(&done_flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Bcast(parent.data(), n, MPI_INT, 0, MPI_COMM_WORLD);
            done = done_flag;

            edges_added_per_phase.push_back(added_this_phase.size());

        } else {
            MPI_Send(&local_count, 1, MPI_INT, 0, 10, MPI_COMM_WORLD);
            if(local_count > 0){
                MPI_Send(ints.data(), 3*local_count, MPI_INT, 0, 11, MPI_COMM_WORLD);
                MPI_Send(dws.data(), local_count, MPI_DOUBLE, 0, 12, MPI_COMM_WORLD);
            }
            int done_flag;
            MPI_Bcast(&done_flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
            parent.assign(n,0);
            MPI_Bcast(parent.data(), n, MPI_INT, 0, MPI_COMM_WORLD);
            done = done_flag;
        }

        double phase_end = MPI_Wtime();
        double local_phase_time = phase_end - phase_start;

        // compute max local phase time across ranks for this phase
        double max_phase_time = 0.0;
        MPI_Reduce(&local_phase_time, &max_phase_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if(rank == 0){
            per_phase_max_time.push_back(max_phase_time);
        }

    } // end phases

    double t_end = MPI_Wtime();
    double total_wall_time = t_end - t_start;
    int phases_count = phase;

    // Root composes final MST vector
    vector<pair<int,int>> edges_out;
    vector<double> weights_out;
    if(rank == 0){
        edges_out.reserve(mst_set.size());
        weights_out.reserve(mst_set.size());
        for(auto &it : mst_set){
            edges_out.push_back(it.first);
            weights_out.push_back(it.second);
        }

        vector<int> idx(edges_out.size());
        iota(idx.begin(), idx.end(), 0);
        sort(idx.begin(), idx.end(), [&](int A,int B){
            if(edges_out[A].first != edges_out[B].first) return edges_out[A].first < edges_out[B].first;
            return edges_out[A].second < edges_out[B].second;
        });

        vector<pair<int,int>> edges_sorted;
        vector<double> weights_sorted;
        for(int id : idx){
            edges_sorted.push_back(edges_out[id]);
            weights_sorted.push_back(weights_out[id]);
        }

        int mcount = edges_sorted.size();
        MPI_Bcast(&mcount, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if(mcount > 0){
            vector<int> ints(2*mcount);
            for(int i=0;i<mcount;++i){
                ints[2*i] = edges_sorted[i].first;
                ints[2*i+1] = edges_sorted[i].second;
            }
            MPI_Bcast(ints.data(), 2*mcount, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Bcast(weights_sorted.data(), mcount, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        }

        cout.setf(std::ios::fixed); cout<<setprecision(6);

        MPI_Barrier(MPI_COMM_WORLD);

        cout<<"MST edges : "<<endl;
        for(size_t i=0;i<edges_sorted.size();++i){
            cout<<edges_sorted[i].first<<" "<<edges_sorted[i].second<<" "<<weights_sorted[i]<<endl;
        }

        double total_weight = 0.0;
        for(double val : weights_sorted) total_weight += val;
        cout<<"Total MST weight: "<<total_weight<<"\n";
        cout.flush();

        // CSV logging
        string csv_file = "results.csv";
        bool need_header = !file_exists(csv_file) || file_is_empty(csv_file);

        auto make_json_array_double = [&](const vector<double>& arr)->string{
            string s = "[";
            for(size_t i=0;i<arr.size();++i){
                if(i) s += ", ";
                s += to_string(arr[i]);
            }
            s += "]";
            return s;
        };
        auto make_json_array_int = [&](const vector<int>& arr)->string{
            string s = "[";
            for(size_t i=0;i<arr.size();++i){
                if(i) s += ", ";
                s += to_string(arr[i]);
            }
            s += "]";
            return s;
        };

        string per_phase_max_json = make_json_array_double(per_phase_max_time);
        string edges_added_json = make_json_array_int(edges_added_per_phase);

        ofstream out(csv_file, ios::app);
        if(!out.good()){
            cerr<<"Error writing "<<csv_file<<"\n";
        } else {
            if(need_header){
                out<<"run_id,n,m,K,total_time,phases_count,edges_in_mst,per_phase_max_time_json,edges_added_per_phase_json,notes\n";
            }

            out<<run_id<<","<<n<<","<<m<<","<<size<<","
               <<fixed<<setprecision(6)<<total_wall_time<<","<<phases_count<<","
               <<mst_set.size()<<","
               <<"\""<<per_phase_max_json<<"\","
               <<"\""<<edges_added_json<<"\","
               <<"\"graph:"<<fname<<"\""
               <<"\n";

            cerr<<"Logged run_id="<<run_id<<" to "<<csv_file<<"\n";
        }

    } else {
        int mcount;
        MPI_Bcast(&mcount, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if(mcount > 0){
            vector<int> ints(2*mcount);
            vector<double> dws(mcount);
            MPI_Bcast(ints.data(), 2*mcount, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Bcast(dws.data(), mcount, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Finalize();
    return 0;
}
