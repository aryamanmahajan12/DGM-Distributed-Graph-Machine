// boruvka_zmq_with_logging.cpp
// Boruvka K-machine MST implemented with ZeroMQ, now with timing and CSV logging.
// Usage: ./boruvka_zmq graph.txt <id> <K> [run_id]
// Root is id == 0
//
// Build: g++ -std=c++17 -O2 -lzmq -pthread -o boruvka_zmq boruvka_zmq_with_logging.cpp

#include <zmq.hpp>
#include <bits/stdc++.h>
#include <chrono>
#include <iomanip>
using namespace std;
using clk = std::chrono::steady_clock;
using dur = std::chrono::duration<double>;

struct Edge { int to; double w; };
struct Cand { int frag; int u; int v; double w; };

// join ints to single string "a,b,c,..."
static string join_ints(const vector<int>& a, char sep=',') {
    string s;
    for(size_t i=0;i<a.size();++i){
        if(i) s.push_back(sep);
        s += to_string(a[i]);
    }
    return s;
}
static vector<int> split_to_ints(const string &s, char sep=',') {
    vector<int> out;
    string cur;
    for(char c: s){
        if(c==sep){ if(!cur.empty()) { out.push_back(stoi(cur)); cur.clear(); } }
        else cur.push_back(c);
    }
    if(!cur.empty()) out.push_back(stoi(cur));
    return out;
}

// parse worker msg format: "count;phase_time;frag,u,v,w|frag,u,v,w|..."
static vector<Cand> parse_worker_msg(const string &msg, double &out_phase_time) {
    vector<Cand> out;
    out_phase_time = 0.0;
    // find first ';' and second ';'
    size_t p1 = msg.find(';');
    if(p1==string::npos) return out;
    size_t p2 = msg.find(';', p1+1);
    if(p2==string::npos) return out;
    int cnt = 0;
    try { cnt = stoi(msg.substr(0,p1)); }
    catch(...) { return out; }
    string phase_str = msg.substr(p1+1, p2-(p1+1));
    try { out_phase_time = stod(phase_str); } catch(...) { out_phase_time = 0.0; }
    string rest = (p2+1 < msg.size()) ? msg.substr(p2+1) : "";
    if(cnt==0) return out;
    string token;
    for(size_t i=0;i<rest.size();++i){
        char c = rest[i];
        if(c=='|'){
            if(!token.empty()){
                vector<string> parts;
                string t;
                for(char ch: token){
                    if(ch==','){ parts.push_back(t); t.clear(); }
                    else t.push_back(ch);
                }
                if(!t.empty()) parts.push_back(t);
                if(parts.size()==4){
                    Cand cnd;
                    cnd.frag = stoi(parts[0]);
                    cnd.u = stoi(parts[1]);
                    cnd.v = stoi(parts[2]);
                    cnd.w = stod(parts[3]);
                    out.push_back(cnd);
                }
            }
            token.clear();
        } else token.push_back(c);
    }
    if(!token.empty()){
        vector<string> parts;
        string t;
        for(char ch: token){
            if(ch==','){ parts.push_back(t); t.clear(); }
            else t.push_back(ch);
        }
        if(!t.empty()) parts.push_back(t);
        if(parts.size()==4){
            Cand cnd;
            cnd.frag = stoi(parts[0]);
            cnd.u = stoi(parts[1]);
            cnd.v = stoi(parts[2]);
            cnd.w = stod(parts[3]);
            out.push_back(cnd);
        }
    }
    return out;
}

// create worker message: "count;phase_time;frag,u,v,w|frag,u,v,w|..."
static string make_worker_msg_with_time(const vector<Cand>& list, double phase_time) {
    ostringstream ss;
    ss.setf(std::ios::fixed); ss<<setprecision(6);
    if(list.empty()){
        ss << "0;" << phase_time << ";";
        return ss.str();
    }
    ss << (int)list.size() << ";" << phase_time << ";";
    for(size_t i=0;i<list.size();++i){
        if(i) ss << "|";
        ss << list[i].frag << "," << list[i].u << "," << list[i].v << "," << list[i].w;
    }
    return ss.str();
}

// file exists / empty helpers
static bool file_exists(const string &fname){
    ifstream f(fname);
    return f.good();
}
static bool file_is_empty(const string &fname){
    ifstream f(fname);
    if(!f.good()) return true;
    char c;
    while(f.get(c)){ if(!isspace((unsigned char)c)) return false; }
    return true;
}

int main(int argc,char** argv){
    if(argc < 4){
        cerr<<"Usage: "<<argv[0]<<" graph.txt <id> <K> [run_id]\n";
        return 1;
    }
    string fname = argv[1];
    int id = stoi(argv[2]);
    int K = stoi(argv[3]);
    string run_id = "";
    if(argc >= 5) run_id = argv[4];
    if(run_id.empty()){
        time_t t = time(NULL);
        run_id = to_string((long long)t);
    }
    if(id < 0 || id >= K){ cerr<<"id must be in [0,K-1]\n"; return 1; }

    // Read graph
    ifstream fin(fname);
    if(!fin.is_open()){ cerr<<"Cannot open "<<fname<<"\n"; return 1; }
    vector<tuple<int,int,double>> edges_all;
    int a,b; double w;
    int maxv = -1;
    while(fin >> a >> b >> w){
        edges_all.emplace_back(a,b,w);
        maxv = max(maxv, max(a,b));
    }
    fin.close();
    int n = maxv + 1;
    int m = (int)edges_all.size();
    if(n <= 0){ cerr<<"Empty graph\n"; return 1; }

    // Partition vertices across K machines; contiguous blocks
    int per = (n + K - 1) / K;
    int start = id * per;
    int end = min(n, start + per);
    int local_n = max(0, end - start);

    vector<vector<Edge>> adj_local(local_n);
    for(auto &t : edges_all){
        int u0,v0; double ww; tie(u0,v0,ww) = t;
        if(u0 >= start && u0 < end) adj_local[u0-start].push_back({v0, ww});
        if(v0 >= start && v0 < end) adj_local[v0-start].push_back({u0, ww});
    }
    for(auto &lst: adj_local){
        sort(lst.begin(), lst.end(), [](const Edge &A,const Edge &B){
            if(A.w != B.w) return A.w < B.w;
            return A.to < B.to;
        });
    }

    // parent[] snapshot per process
    vector<int> parent(n);
    for(int i=0;i<n;++i) parent[i] = i;

    // ZeroMQ context & sockets
    zmq::context_t ctx(1);
    const string pull_addr = "tcp://127.0.0.1:5557";
    const string pub_addr  = "tcp://127.0.0.1:5558";

    zmq::socket_t pull_socket(ctx, zmq::socket_type::pull);
    zmq::socket_t pub_socket(ctx, zmq::socket_type::pub);
    zmq::socket_t push_socket(ctx, zmq::socket_type::push);
    zmq::socket_t sub_socket(ctx, zmq::socket_type::sub);

    if(id == 0){
        pull_socket.bind(pull_addr);
        pub_socket.bind(pub_addr);
        this_thread::sleep_for(chrono::milliseconds(150));
        cerr<<"[root] listening at "<<pull_addr<<" and publishing at "<<pub_addr<<"\n";
    } else {
        push_socket.connect(pull_addr);
        sub_socket.connect(pub_addr);
        sub_socket.set(zmq::sockopt::subscribe, "");
        this_thread::sleep_for(chrono::milliseconds(200));
        cerr<<"[worker "<<id<<"] connected to root\n";
    }

    auto recv_string = [&](zmq::socket_t &sock)->string{
        zmq::message_t msg;
        sock.recv(msg, zmq::recv_flags::none);
        return string(static_cast<char*>(msg.data()), msg.size());
    };
    auto send_string = [&](zmq::socket_t &sock, const string &s){
        zmq::message_t msg(s.size());
        memcpy(msg.data(), s.data(), s.size());
        sock.send(msg, zmq::send_flags::none);
    };

    // Root's MST set (deduplicated by canonical (min,max))
    set<pair<pair<int,int>, double>> mst_set;

    // Logging containers (root)
    vector<double> per_phase_max_time;   // per-phase max time across ranks
    vector<int> edges_added_per_phase;  // edges added each phase (root view)

    bool done = false;
    int phase = 0;

    // Start total timer (root)
    auto overall_start = clk::now();

    while(!done){
        phase++;

        // local phase timer start
        auto phase_local_start = clk::now();

        // Step 1: each machine finds local candidates (one per fragment it sees)
        unordered_map<int, Cand> local_best;
        for(int i=0;i<local_n;++i){
            int v_global = start + i;
            int frag_v = parent[v_global];
            for(const Edge &e : adj_local[i]){
                int u_global = e.to;
                int frag_u = parent[u_global];
                if(frag_u != frag_v){
                    auto it = local_best.find(frag_v);
                    if(it==local_best.end() || e.w < it->second.w ||
                       (e.w == it->second.w &&
                        make_pair(v_global,u_global) < make_pair(it->second.u, it->second.v))){
                        local_best[frag_v] = Cand{frag_v, v_global, u_global, e.w};
                    }
                    break;
                }
            }
        }

        // pack to message
        vector<Cand> local_list;
        local_list.reserve(local_best.size());
        for(auto &kv : local_best) local_list.push_back(kv.second);

        // local phase timer end (before sending)
        auto phase_local_end = clk::now();
        double local_phase_time = dur(phase_local_end - phase_local_start).count();

        string msg_out = make_worker_msg_with_time(local_list, local_phase_time);

        if(id == 0){
            // root: include its own message (we already have local_list and local_phase_time)
            // Collect all messages from other K-1 workers (each message includes its phase time)
            double root_phase_time = local_phase_time;
            vector<Cand> all_cands = local_list;

            // Also compute per-phase times: start by taking root's own
            double max_phase_time_this_round = root_phase_time;

            for(int src=1; src < K; ++src){
                string rec = recv_string(pull_socket);
                double worker_phase_time = 0.0;
                auto rec_list = parse_worker_msg(rec, worker_phase_time);
                // update max phase time
                if(worker_phase_time > max_phase_time_this_round) max_phase_time_this_round = worker_phase_time;
                for(auto &c : rec_list) all_cands.push_back(c);
            }

            // Pick best per fragment
            unordered_map<int, Cand> best_per_frag;
            for(const auto &c: all_cands){
                auto it = best_per_frag.find(c.frag);
                if(it==best_per_frag.end() || c.w < it->second.w ||
                   (c.w == it->second.w && make_pair(c.u,c.v) < make_pair(it->second.u, it->second.v))){
                    best_per_frag[c.frag] = c;
                }
            }

            // create DSU from parent[]
            vector<int> dsu_p = parent;
            function<int(int)> dsu_find = [&](int x)->int{ return dsu_p[x]==x ? x : dsu_p[x]=dsu_find(dsu_p[x]); };
            auto dsu_unite = [&](int a, int b)->bool{
                a = dsu_find(a); b = dsu_find(b);
                if(a==b) return false;
                dsu_p[b] = a;
                return true;
            };

            vector<tuple<int,int,double>> added_this_phase;
            vector<int> frag_keys;
            frag_keys.reserve(best_per_frag.size());
            for(auto &kv : best_per_frag) frag_keys.push_back(kv.first);
            sort(frag_keys.begin(), frag_keys.end());
            for(int frag : frag_keys){
                const Cand &c = best_per_frag[frag];
                int fu = dsu_find(parent[c.u]);
                int fv = dsu_find(parent[c.v]);
                if(fu != fv){
                    bool merged = dsu_unite(fu, fv);
                    if(merged){
                        int uu = c.u, vv = c.v;
                        if(uu > vv) swap(uu, vv);
                        added_this_phase.emplace_back(uu, vv, c.w);
                        mst_set.insert({{uu,vv}, c.w});
                    }
                }
            }

            // update parent[] from dsu_p
            for(int i=0;i<n;++i) parent[i] = dsu_find(dsu_p[i]);

            // compute components & done flag
            unordered_set<int> comps;
            for(int i=0;i<n;++i) comps.insert(parent[i]);
            int num_comps = (int)comps.size();
            int done_flag = (num_comps <= 1) ? 1 : 0;
            cerr<<"[root] Phase "<<phase<<": components="<<num_comps<<", added="<<added_this_phase.size()<<"\n";

            // record per-phase metrics
            per_phase_max_time.push_back(max_phase_time_this_round);
            edges_added_per_phase.push_back((int)added_this_phase.size());

            // Broadcast done flag and parent[] to all workers
            // message format: "flag;n;parent0,parent1,..."
            string pstr = join_ints(parent, ',');
            string bmsg = to_string(done_flag) + ";" + to_string(n) + ";" + pstr;
            send_string(pub_socket, bmsg);

            if(done_flag) {
                done = true;
                // Build final MST message and broadcast "MST;count;u,v,w|u,v,w|..."
                string mstmsg;
                if(mst_set.empty()) mstmsg = string("MST;0;");
                else {
                    vector<string> pieces;
                    pieces.reserve(mst_set.size());
                    for(auto &it : mst_set){
                        int aa = it.first.first, bb = it.first.second; double ww = it.second;
                        ostringstream ss; ss.setf(std::ios::fixed); ss<<setprecision(6);
                        ss<<aa<<","<<bb<<","<<ww;
                        pieces.push_back(ss.str());
                    }
                    string body;
                    for(size_t i=0;i<pieces.size();++i){ if(i) body.push_back('|'); body += pieces[i]; }
                    mstmsg = string("MST;") + to_string((int)pieces.size()) + ";" + body;
                }
                send_string(pub_socket, mstmsg);

                // compute total wall time and log CSV
                auto overall_end = clk::now();
                double total_wall_time = dur(overall_end - overall_start).count();

                // Print MST and total weight
                double total_weight = 0.0;
                ostringstream out_ss;
                out_ss.setf(std::ios::fixed); out_ss<<setprecision(6);
                out_ss << "MST edges (root):\n";
                for(auto &it : mst_set){
                    out_ss<<it.first.first<<" - "<<it.first.second<<" (w="<<it.second<<")\n";
                    total_weight += it.second;
                }
                out_ss << "Total MST weight: " << total_weight << "\n";
                cout << out_ss.str();

                // CSV logging: run_id,n,m,K,total_time,phases_count,edges_in_mst,per_phase_max_time_json,edges_added_per_phase_json,notes
                string csv_file = "results_zmq.csv";
                bool need_header = !file_exists(csv_file) || file_is_empty(csv_file);

                auto make_json_array_double = [&](const vector<double>& arr)->string{
                    ostringstream s; s.setf(std::ios::fixed); s<<setprecision(6);
                    s << "[";
                    for(size_t i=0;i<arr.size();++i){ if(i) s<<", "; s<<arr[i]; }
                    s << "]";
                    return s.str();
                };
                auto make_json_array_int = [&](const vector<int>& arr)->string{
                    ostringstream s;
                    s << "[";
                    for(size_t i=0;i<arr.size();++i){ if(i) s<<", "; s<<arr[i]; }
                    s << "]";
                    return s.str();
                };

                string per_phase_json = make_json_array_double(per_phase_max_time);
                string edges_added_json = make_json_array_int(edges_added_per_phase);

                ofstream outf(csv_file, ios::app);
                if(!outf.good()){
                    cerr<<"Error: cannot open "<<csv_file<<" for appending\n";
                } else {
                    if(need_header){
                        outf<<"run_id,n,m,K,total_time,phases_count,edges_in_mst,per_phase_max_time_json,edges_added_per_phase_json,notes\n";
                    }
                    outf << run_id << "," << n << "," << m << "," << K << ",";
                    outf.setf(std::ios::fixed); outf<<setprecision(6);
                    outf << total_wall_time << "," << phase << "," << (int)mst_set.size() << ",";
                    outf << "\"" << per_phase_json << "\"" << "," << "\"" << edges_added_json << "\"" << ",";
                    outf << "\"graph:" << fname << "\"\n";
                    outf.close();
                    cerr<<"[root] appended results to "<<csv_file<<" (run_id="<<run_id<<")\n";
                }
            }

        } else {
            // worker: send local message to root, then wait for broadcast
            send_string(push_socket, msg_out);

            // blocking receive: either parent broadcast or possibly MST directly in rare race
            string rec = recv_string(sub_socket);

            // If we directly receive an MST message (rare), parse and exit
            if(rec.rfind("MST;", 0) == 0){
                // parse MST and print total weight
                size_t p = rec.find(';');
                size_t q = rec.find(';', p+1);
                int mcount = 0; string body;
                if(p!=string::npos && q!=string::npos){
                    mcount = stoi(rec.substr(p+1, q-(p+1)));
                    body = rec.substr(q+1);
                }
                if(mcount==0){
                    cout<<"Process "<<id<<" received MST empty\n";
                } else {
                    cout<<"Process "<<id<<" MST edges (received):\n";
                    string token; double total_weight = 0.0;
                    for(char ch : body){
                        if(ch=='|'){
                            if(!token.empty()){
                                vector<string> parts; string t;
                                for(char c: token){ if(c==','){ parts.push_back(t); t.clear(); } else t.push_back(c); }
                                if(!t.empty()) parts.push_back(t);
                                if(parts.size()==3){
                                    cout<<parts[0]<<" - "<<parts[1]<<" (w="<<parts[2]<<")\n";
                                    try{ total_weight += stod(parts[2]); } catch(...) {}
                                }
                            }
                            token.clear();
                        } else token.push_back(ch);
                    }
                    if(!token.empty()){
                        vector<string> parts; string t;
                        for(char c: token){ if(c==','){ parts.push_back(t); t.clear(); } else t.push_back(c); }
                        if(!t.empty()) parts.push_back(t);
                        if(parts.size()==3){ cout<<parts[0]<<" - "<<parts[1]<<" (w="<<parts[2]<<")\n"; try{ total_weight += stod(parts[2]); } catch(...) {} }
                    }
                    cout.setf(std::ios::fixed); cout<<setprecision(6);
                    cout<<"Total MST weight: "<<total_weight<<"\n";
                }
                done = true;
                break;
            }

            // Otherwise parse parent broadcast: "flag;n;parent0,parent1,..."
            size_t p1 = rec.find(';');
            size_t p2 = rec.find(';', p1+1);
            if(p1==string::npos || p2==string::npos){
                cerr<<"[worker "<<id<<"] malformed broadcast\n";
            } else {
                int done_flag = stoi(rec.substr(0,p1));
                int nn = stoi(rec.substr(p1+1, p2-(p1+1)));
                string plist = rec.substr(p2+1);
                auto parr = split_to_ints(plist, ',');
                if((int)parr.size() == nn) parent = parr;
                else {
                    // shouldn't happen
                }
                if(done_flag){
                    // now receive final MST message
                    string mstrec = recv_string(sub_socket);
                    if(mstrec.rfind("MST;",0)==0){
                        size_t p = mstrec.find(';'); size_t q = mstrec.find(';', p+1);
                        int mcount = 0; string body;
                        if(p!=string::npos && q!=string::npos){
                            mcount = stoi(mstrec.substr(p+1, q-(p+1)));
                            body = mstrec.substr(q+1);
                        }
                        if(mcount==0){
                            cout<<"Process "<<id<<" received MST empty\n";
                        } else {
                            cout<<"Process "<<id<<" MST edges (received):\n";
                            string token; double total_weight = 0.0;
                            for(char ch : body){
                                if(ch=='|'){
                                    if(!token.empty()){
                                        vector<string> parts; string t;
                                        for(char c: token){ if(c==','){ parts.push_back(t); t.clear(); } else t.push_back(c); }
                                        if(!t.empty()) parts.push_back(t);
                                        if(parts.size()==3){
                                            cout<<parts[0]<<" - "<<parts[1]<<" (w="<<parts[2]<<")\n";
                                            try{ total_weight += stod(parts[2]); } catch(...) {}
                                        }
                                    }
                                    token.clear();
                                } else token.push_back(ch);
                            }
                            if(!token.empty()){
                                vector<string> parts; string t;
                                for(char c: token){ if(c==','){ parts.push_back(t); t.clear(); } else t.push_back(c); }
                                if(!t.empty()) parts.push_back(t);
                                if(parts.size()==3){ cout<<parts[0]<<" - "<<parts[1]<<" (w="<<parts[2]<<")\n"; try{ total_weight += stod(parts[2]); } catch(...) {} }
                            }
                            cout.setf(std::ios::fixed); cout<<setprecision(6);
                            cout<<"Total MST weight: "<<total_weight<<"\n";
                        }
                    }
                    done = true;
                    break;
                }
            }
        } // end worker branch

        // small sleep to avoid busy loop
        this_thread::sleep_for(chrono::milliseconds(20));
    } // end while

    // Final root print (in case not already printed)
    if(id == 0){
        if(!mst_set.empty()){
            double total_weight = 0.0;
            cout.setf(std::ios::fixed); cout<<setprecision(6);
            cout<<"MST edges (root summary):\n";
            for(auto &it : mst_set){
                cout<<it.first.first<<" - "<<it.first.second<<" (w="<<it.second<<")\n";
                total_weight += it.second;
            }
            cout<<"Total MST weight: "<<total_weight<<"\n";
        } else {
            cout<<"MST edges (root): empty\nTotal MST weight: 0.000000\n";
        }
    }

    // cleanup
    if(id==0){ pull_socket.close(); pub_socket.close(); }
    else { push_socket.close(); sub_socket.close(); }
    ctx.close();
    return 0;
}
