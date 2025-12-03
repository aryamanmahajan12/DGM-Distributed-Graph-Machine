// prim_edges_only.cpp
// Compile: g++ -O3 -std=c++17 -o prim_edges_only prim_edges_only.cpp
// Run:     ./prim_edges_only graph.txt
#include <bits/stdc++.h>
using namespace std;
using ll = long long;

int main(int argc, char** argv) {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    
    // Start total time measurement
    auto start_total = chrono::high_resolution_clock::now();
    
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " graph.txt\n";
        return 1;
    }
    string fname = argv[1];
    
    // Time file reading
    auto start_read = chrono::high_resolution_clock::now();
    ifstream fin(fname);
    if (!fin) {
        cerr << "Error: cannot open file '" << fname << "'\n";
        return 1;
    }
    vector<tuple<long long,long long,long long>> edges;
    long long u,v,w;
    long long maxnode = -1, minnode = LLONG_MAX;
    string line;
    while (std::getline(fin, line)) {
        // trim leading spaces
        size_t pos = line.find_first_not_of(" \t\r\n");
        if (pos == string::npos) continue;
        if (line[pos] == '#') continue;
        stringstream ss(line);
        if (!(ss >> u >> v >> w)) continue; // ignore malformed lines
        // record
        edges.emplace_back(u, v, w);
        maxnode = max(maxnode, max(u,v));
        minnode = min(minnode, min(u,v));
    }
    fin.close();
    auto end_read = chrono::high_resolution_clock::now();
    
    if (edges.empty()) {
        cerr << "No valid edges found in file.\n";
        return 1;
    }
    
    // Time graph construction
    auto start_construct = chrono::high_resolution_clock::now();
    // Determine indexing: if minnode >= 1 => assume 1-based, else 0-based
    bool one_based = (minnode >= 1);
    // Build adjacency list (convert to 0-based if needed)
    int n = (int)(maxnode + 1);
    if (one_based) n = (int)(maxnode + 1); // nodes numbered 1..maxnode -> after convert will be 0..maxnode-1
    vector<vector<pair<int,ll>>> adj;
    if (one_based) {
        adj.assign((size_t)maxnode, {});
    } else {
        adj.assign((size_t)n, {});
    }
    for (auto &t : edges) {
        long long uu = std::get<0>(t);
        long long vv = std::get<1>(t);
        long long ww = std::get<2>(t);
        if (one_based) { uu--; vv--; }
        if (uu < 0 || vv < 0) continue; // skip invalid after conversion
        // resize if necessary
        int need = (int)max(uu, vv) + 1;
        if ((int)adj.size() < need) adj.resize(need);
        if (uu == vv) continue; // ignore self-loop
        adj[(size_t)uu].emplace_back((int)vv, (ll)ww);
        adj[(size_t)vv].emplace_back((int)uu, (ll)ww);
    }
    int NN = (int)adj.size();
    // find a start node that has edges
    int start = -1;
    for (int i = 0; i < NN; ++i) if (!adj[i].empty()) { start = i; break; }
    if (start == -1) {
        cerr << "Graph has no edges (all nodes isolated).\n";
        return 1;
    }
    auto end_construct = chrono::high_resolution_clock::now();
    
    // Time Prim's algorithm
    auto start_prim = chrono::high_resolution_clock::now();
    // Prim's algorithm using min-heap of (weight, to, from)
    vector<char> used(NN, 0);
    using T = tuple<ll,int,int>;
    priority_queue<T, vector<T>, greater<T>> pq;
    used[start] = 1;
    for (auto &pr : adj[start]) {
        pq.emplace(pr.second, pr.first, start);
    }
    vector<tuple<int,int,ll>> mst;
    while (!pq.empty()) {
        auto [ww, to, from] = pq.top(); pq.pop();
        if (used[to]) continue;
        used[to] = 1;
        mst.emplace_back(from, to, ww);
        for (auto &pr : adj[to]) {
            int nxt = pr.first;
            ll wt = pr.second;
            if (!used[nxt]) pq.emplace(wt, nxt, to);
        }
    }
    auto end_prim = chrono::high_resolution_clock::now();
    
    // Count non-isolated vertices in graph
    int nonIsolated = 0;
    for (int i = 0; i < NN; ++i) if (!adj[i].empty()) ++nonIsolated;
    if ((int)mst.size() != max(0, nonIsolated - 1)) {
        cerr << "Warning: graph may be disconnected. MST spans connected component containing node " << start << ".\n";
    }
    
    ll total = 0;
    for (auto &tup : mst) {
        int a,b; ll ww;
        tie(a,b,ww) = tup;
        cout << "MST edge: " << a << " - " << b << " (w=" << ww << ")\n";
        total += ww;
    }
    cout << "Total MST weight: " << total << "\n";
    
    // End total time measurement
    auto end_total = chrono::high_resolution_clock::now();
    
    // Calculate durations
    auto read_time = chrono::duration_cast<chrono::milliseconds>(end_read - start_read).count();
    auto construct_time = chrono::duration_cast<chrono::milliseconds>(end_construct - start_construct).count();
    auto prim_time = chrono::duration_cast<chrono::milliseconds>(end_prim - start_prim).count();
    auto total_time = chrono::duration_cast<chrono::milliseconds>(end_total - start_total).count();
    
    // Print timing information
    cout << "\n=== Timing Information ===\n";
    cout << "File reading time:      " << read_time << " ms\n";
    cout << "Graph construction time: " << construct_time << " ms\n";
    cout << "Prim's algorithm time:  " << prim_time << " ms\n";
    cout << "Total execution time:   " << total_time << " ms\n";
    
    // Log results to CSV file
    int n_vertices = nonIsolated;
    int m_edges = (int)edges.size();
    
    // Generate run_id based on timestamp
    auto now = chrono::system_clock::now();
    auto timestamp = chrono::duration_cast<chrono::milliseconds>(now.time_since_epoch()).count();
    string run_id = to_string(timestamp);
    
    // Check if file exists to determine if we need to write header
    bool file_exists = false;
    ifstream check_file("results-single-machine.csv");
    if (check_file.good()) {
        file_exists = true;
    }
    check_file.close();
    
    // Append to CSV file
    ofstream csv_file("results-single-machine.csv", ios::app);
    if (csv_file.is_open()) {
        // Write header if file is new
        if (!file_exists) {
            csv_file << "run_id,n,m,read_time_ms,construct_time_ms,prim_time_ms,total_time_ms\n";
        }
        // Write data
        csv_file << run_id << "," << n_vertices << "," << m_edges << "," 
                 << read_time << "," << construct_time << "," << prim_time << "," << total_time << "\n";
        csv_file.close();
        cout << "\nResults logged to results-single-machine.csv\n";
    } else {
        cerr << "Warning: Could not open results-single-machine.csv for writing\n";
    }
    
    return 0;
}