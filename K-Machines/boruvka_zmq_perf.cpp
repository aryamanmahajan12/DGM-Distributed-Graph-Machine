// boruvka_zmq.cpp
// Boruvka K-machine MST implemented with ZeroMQ
// Usage: ./boruvka_zmq graph.txt <id> <K>
// Root is id == 0
//
// Build: g++ -std=c++17 -O2 -lzmq -pthread -o boruvka_zmq boruvka_zmq.cpp

#include <zmq.hpp>
#include <bits/stdc++.h>
#include <iomanip>
using namespace std;

struct Edge { int to; double w; };
struct Cand { int frag; int u; int v; double w; };

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

// parse "count;frag,u,v,w|frag,u,v,w|..."
static vector<Cand> parse_worker_msg(const string &msg) {
    vector<Cand> out;
    size_t p = msg.find(';');
    if(p==string::npos) return out;
    int cnt = stoi(msg.substr(0,p));
    string rest = (p+1 < msg.size()) ? msg.substr(p+1) : "";
    if(cnt==0) return out;
    string token;
    for(size_t i=0;i<rest.size();++i){
        char c = rest[i];
        if(c=='|'){
            if(!token.empty()){
                // token like frag,u,v,w
                vector<string> parts;
                string t;
                for(char ch: token) {
                    if(ch==',') { parts.push_back(t); t.clear(); }
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
        for(char ch: token) {
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

// create worker message from list of Cand
static string make_worker_msg(const vector<Cand>& list) {
    if(list.empty()) return string("0;"); 
    string s = to_string((int)list.size()) + ";";
    for(size_t i=0;i<list.size();++i){
        if(i) s.push_back('|');
        s += to_string(list[i].frag) + "," + to_string(list[i].u) + "," + to_string(list[i].v) + "," + to_string(list[i].w);
    }
    return s;
}

int main(int argc,char** argv){
    if(argc < 4){
        cerr<<"Usage: "<<argv[0]<<" graph.txt <id> <K>\n";
        return 1;
    }
    string fname = argv[1];
    int id = stoi(argv[2]);
    int K = stoi(argv[3]);
    if(id < 0 || id >= K){ cerr<<"id must be in [0,K-1]\n"; return 1; }

    // Read graph
    ifstream fin(fname);
    if(!fin.is_open()){ cerr<<"Cannot open "<<fname<<"\n"; return 1; }
    vector<tuple<int,int,double>> edges_all;
    int u,v; double w;
    int maxv = -1;
    while(fin >> u >> v >> w){
        edges_all.emplace_back(u,v,w);
        maxv = max(maxv, max(u,v));
    }
    fin.close();
    int n = maxv + 1;
    if(n <= 0){ cerr<<"Empty graph\n"; return 1; }

    // Partition vertices across K machines; contiguous blocks
    int per = (n + K - 1) / K;
    int start = id * per;
    int end = min(n, start + per);
    int local_n = max(0, end - start);

    vector<vector<Edge>> adj_local(local_n);
    for(auto &t : edges_all){
        int a,b; double ww; tie(a,b,ww) = t;
        if(a >= start && a < end) adj_local[a-start].push_back({b, ww});
        if(b >= start && b < end) adj_local[b-start].push_back({a, ww});
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

    // ZeroMQ context
    zmq::context_t ctx(1);

    // Root (id==0) binds PULL and PUB. Workers connect PUSH and SUB.
    const string pull_addr = "tcp://127.0.0.1:5557";
    const string pub_addr  = "tcp://127.0.0.1:5558";

    zmq::socket_t pull_socket(ctx, zmq::socket_type::pull);
    zmq::socket_t pub_socket(ctx, zmq::socket_type::pub);
    zmq::socket_t push_socket(ctx, zmq::socket_type::push);
    zmq::socket_t sub_socket(ctx, zmq::socket_type::sub);

    if(id == 0){
        pull_socket.bind(pull_addr);
        pub_socket.bind(pub_addr);
        // Allow sockets to be established
        this_thread::sleep_for(chrono::milliseconds(150));
        cerr<<"[root] listening at "<<pull_addr<<" and publishing at "<<pub_addr<<"\n";
    } else {
        // worker: connect push and sub
        push_socket.connect(pull_addr);
        sub_socket.connect(pub_addr);
        // subscribe to everything
        sub_socket.set(zmq::sockopt::subscribe, "");
        // small sleep to give root time to bind
        this_thread::sleep_for(chrono::milliseconds(200));
        cerr<<"[worker "<<id<<"] connected to root\n";
    }

    // helper to recv a string from a zmq socket (blocking)
    auto recv_string = [&](zmq::socket_t &sock)->string{
        zmq::message_t msg;
        sock.recv(msg, zmq::recv_flags::none);
        return string(static_cast<char*>(msg.data()), msg.size());
    };

    // helper to send string
    auto send_string = [&](zmq::socket_t &sock, const string &s){
        zmq::message_t msg(s.size());
        memcpy(msg.data(), s.data(), s.size());
        sock.send(msg, zmq::send_flags::none);
    };

    // Root's MST set (deduplicated by canonical (min,max))
    set<pair<pair<int,int>, double>> mst_set;

    bool done = false;
    int phase = 0;

    while(!done){
        phase++;
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
                       (e.w == it->second.w && make_pair(v_global,u_global) < make_pair(it->second.u, it->second.v))){
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
        string msg_out = make_worker_msg(local_list);

        if(id == 0){
            // root: include its own candidates directly and then receive K-1 messages
            vector<Cand> all_cands = local_list;
            for(int src=1; src < K; ++src){
                string rec = recv_string(pull_socket);
                auto rec_list = parse_worker_msg(rec);
                for(auto &c : rec_list) all_cands.push_back(c);
            }

            // pick best per fragment
            unordered_map<int, Cand> best_per_frag;
            for(const auto &c: all_cands){
                auto it = best_per_frag.find(c.frag);
                if(it==best_per_frag.end() || c.w < it->second.w ||
                   (c.w == it->second.w && make_pair(c.u,c.v) < make_pair(it->second.u, it->second.v))){
                    best_per_frag[c.frag] = c;
                }
            }

            // create DSU from parent[]
            vector<int> dsu_p = parent; // simple DSU with path compression via lambda
            function<int(int)> dsu_find = [&](int x)->int{ return dsu_p[x]==x ? x : dsu_p[x]=dsu_find(dsu_p[x]); };
            auto dsu_unite = [&](int a, int b)->bool{
                a = dsu_find(a); b = dsu_find(b);
                if(a==b) return false;
                dsu_p[b] = a;
                return true;
            };

            vector<tuple<int,int,double>> added_this_phase;
            // process fragment keys deterministic order
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

            unordered_set<int> comps;
            for(int i=0;i<n;++i) comps.insert(parent[i]);
            int num_comps = (int)comps.size();
            int done_flag = (num_comps <= 1) ? 1 : 0;
            cerr<<"[root] Phase "<<phase<<": components="<<num_comps<<", added="<<added_this_phase.size()<<"\n";

            // Broadcast done flag and parent[] to all workers
            // message format: "flag;n;parent0,parent1,..."
            string pstr = join_ints(parent, ',');
            string bmsg = to_string(done_flag) + ";" + to_string(n) + ";" + pstr;
            send_string(pub_socket, bmsg);

            if(done_flag) {
                done = true;
                // send final MST as "MST;count;u,v,w|u,v,w|..."
                string mstmsg;
                if(mst_set.empty()) mstmsg = string("MST;0;");
                else {
                    vector<string> pieces;
                    for(auto &it : mst_set){
                        int a = it.first.first, b = it.first.second; double ww = it.second;
                        pieces.push_back(to_string(a)+","+to_string(b)+","+to_string(ww));
                    }
                    string body;
                    for(size_t i=0;i<pieces.size();++i){ if(i) body.push_back('|'); body += pieces[i]; }
                    mstmsg = string("MST;") + to_string((int)pieces.size()) + ";" + body;
                }
                // broadcast final MST
                send_string(pub_socket, mstmsg);

                // Also print total weight at root
                double total_weight = 0.0;
                for(auto &it : mst_set) total_weight += it.second;
                cout<<"MST edges (root):\n";
                cout.setf(std::ios::fixed); cout<<setprecision(6);
                for(auto &it : mst_set){
                    cout<<it.first.first<<" - "<<it.first.second<<" (w="<<it.second<<")\n";
                }
                cout<<"Total MST weight: "<<total_weight<<"\n";
            }
        } else {
            // worker: send local message to root, then wait for broadcast
            send_string(push_socket, msg_out);

            // wait for broadcast (blocking)
            string rec = recv_string(sub_socket);

            // If we directly receive an MST message (rare), handle it here
            if(rec.rfind("MST;", 0) == 0){
                // parse MST and print total weight
                size_t p = rec.find(';');
                size_t q = rec.find(';', p+1);
                int mcount = 0;
                string body;
                if(p!=string::npos && q!=string::npos){
                    mcount = stoi(rec.substr(p+1, q-(p+1)));
                    body = rec.substr(q+1);
                }
                if(mcount==0){
                    cout<<"Process "<<id<<" received MST empty\n";
                } else {
                    cout<<"Process "<<id<<" MST edges (received):\n";
                    string token;
                    double total_weight = 0.0;
                    for(char ch : body){
                        if(ch=='|'){
                            if(!token.empty()){
                                vector<string> parts;
                                string t;
                                for(char c: token){
                                    if(c==','){ parts.push_back(t); t.clear(); }
                                    else t.push_back(c);
                                }
                                if(!t.empty()) parts.push_back(t);
                                if(parts.size()==3){
                                    cout<<parts[0]<<" - "<<parts[1]<<" (w="<<parts[2]<<")\n";
                                    try { total_weight += stod(parts[2]); } catch(...) {}
                                }
                            }
                            token.clear();
                        } else token.push_back(ch);
                    }
                    if(!token.empty()){
                        vector<string> parts;
                        string t;
                        for(char c: token){
                            if(c==','){ parts.push_back(t); t.clear(); }
                            else t.push_back(c);
                        }
                        if(!t.empty()) parts.push_back(t);
                        if(parts.size()==3){
                            cout<<parts[0]<<" - "<<parts[1]<<" (w="<<parts[2]<<")\n";
                            try { total_weight += stod(parts[2]); } catch(...) {}
                        }
                    }
                    cout.setf(std::ios::fixed); cout<<setprecision(6);
                    cout<<"Total MST weight: "<<total_weight<<"\n";
                }
                done = true;
                // done - we can break or continue to cleanup
                break;
            }

            // We might receive both broadcasts sequentially: first done+parent, possibly then MST.
            // Parse received message: check if it starts with "MST;" or "flag;"
            if(rec.rfind("MST;", 0) == 0){
                // handled above
            } else {
                // normal parent update
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
                    if(done_flag) {
                        done = true;
                        // after receiving done we expect final MST message next - block and receive it
                        string mstrec = recv_string(sub_socket);
                        if(mstrec.rfind("MST;",0)==0){
                            // parse MST and print
                            size_t p = mstrec.find(';');
                            size_t q = mstrec.find(';', p+1);
                            int mcount = 0;
                            string body;
                            if(p!=string::npos && q!=string::npos){
                                mcount = stoi(mstrec.substr(p+1, q-(p+1)));
                                body = mstrec.substr(q+1);
                            }
                            if(mcount==0){
                                cout<<"Process "<<id<<" received MST empty\n";
                            } else {
                                cout<<"Process "<<id<<" MST edges (received):\n";
                                string token;
                                double total_weight = 0.0;
                                for(char ch : body){
                                    if(ch=='|'){
                                        if(!token.empty()){
                                            // token u,v,w
                                            vector<string> parts;
                                            string t;
                                            for(char c: token){
                                                if(c==','){ parts.push_back(t); t.clear(); }
                                                else t.push_back(c);
                                            }
                                            if(!t.empty()) parts.push_back(t);
                                            if(parts.size()==3){
                                                cout<<parts[0]<<" - "<<parts[1]<<" (w="<<parts[2]<<")\n";
                                                try { total_weight += stod(parts[2]); } catch(...) {}
                                            }
                                        }
                                        token.clear();
                                    }
                                    else token.push_back(ch);
                                }
                                if(!token.empty()){
                                    vector<string> parts;
                                    string t;
                                    for(char c: token){
                                        if(c==','){ parts.push_back(t); t.clear(); }
                                        else t.push_back(c);
                                    }
                                    if(!t.empty()) parts.push_back(t);
                                    if(parts.size()==3) {
                                        cout<<parts[0]<<" - "<<parts[1]<<" (w="<<parts[2]<<")\n";
                                        try { total_weight += stod(parts[2]); } catch(...) {}
                                    }
                                }
                                cout.setf(std::ios::fixed); cout<<setprecision(6);
                                cout<<"Total MST weight: "<<total_weight<<"\n";
                            }
                        }
                    }
                }
            }
        }

        // small sleep to avoid busy loop (and give sockets time)
        this_thread::sleep_for(chrono::milliseconds(50));
    } // end while

    if(id == 0){
        // print MST (in case we didn't already)
        if(!mst_set.empty()){
            double total_weight = 0.0;
            for(auto &it : mst_set) total_weight += it.second;
            // If we already printed inside the done_flag branch, this will repeat; that's OK.
            cout<<"MST edges (root):\n";
            cout.setf(std::ios::fixed); cout<<setprecision(6);
            for(auto &it : mst_set){
                cout<<it.first.first<<" - "<<it.first.second<<" (w="<<it.second<<")\n";
            }
            cout<<"Total MST weight: "<<total_weight<<"\n";
        } else {
            cout<<"MST edges (root): empty\n";
            cout<<"Total MST weight: 0.000000\n";
        }
    }

    // clean up sockets
    if(id==0){ pull_socket.close(); pub_socket.close(); }
    else { push_socket.close(); sub_socket.close(); }
    ctx.close();

    return 0;
}
