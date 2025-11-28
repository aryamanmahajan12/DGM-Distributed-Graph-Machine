#include <mpi.h>
#include <bits/stdc++.h>
using namespace std;

enum MsgType {
    MSG_CONNECT = 1,
    MSG_INITIATE = 2,
    MSG_REPORT = 3,
    MSG_TEST = 4,
    MSG_ACCEPT = 5,
    MSG_REJECT = 6,
    MSG_WAKEUP = 7,
    MSG_CHANGEROOT = 8,
    MSG_DONE = 9,
    MSG_DEBUG = 20
};

struct Edge {
    int nb;
    double w;
    int state;
    Edge(int n=0,double ww=0): nb(n), w(ww), state(0) {}
};

int rank_,size_;
vector<Edge> adj;

void send_int_msg(int dest,int tag,const vector<int>& data) {
    MPI_Send((void*)data.data(),data.size(), MPI_INT, dest, tag, MPI_COMM_WORLD);
}

void send_double_msg(int dest,int tag,const vector<double>& data) {
    MPI_Send((void*)data.data(),data.size(), MPI_DOUBLE, dest, tag, MPI_COMM_WORLD);
}

bool probe_and_recv_any(int &src, int &tag, vector<int> &idata, vector<double> &ddata) {
    int flag;
    MPI_Status st;
    MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, &st);
    if(!flag)return false;
    src = st.MPI_SOURCE;
    tag = st.MPI_TAG;

    if(tag == MSG_CONNECT || tag == MSG_TEST || tag == MSG_INITIATE || tag ==MSG_REPORT) {
        int count;
        MPI_Get_count(&st, MPI_DOUBLE, &count);
        ddata.resize(count);
        MPI_Recv(ddata.data(),count,MPI_DOUBLE,src,tag,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        idata.clear();
    } 
    else {
        int count;
        MPI_Get_count(&st, MPI_INT, &count);
        idata.resize(count);
        MPI_Recv(idata.data(),count, MPI_INT, src, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        ddata.clear();
    }

    return true;
}

int state_found = 0; // 0 = SLEEPING, 1 = FIND, 2 = FOUND
int level_frag = 0;
int fragment_id = -1;
int in_branch = -1;
int best_neighbor = -1;
double best_weight = numeric_limits<double>::infinity();
int test_edge = -1;

int find_edge_index(int nb) {
    for(int i=0;i<(int)adj.size();++i) if(adj[i].nb==nb)return i;
    return -1;
}

void debug_print(const string &s)
{
    // debug statements
    cout<<s<<endl;
}

//choose min weight incident edge, mark as branch and send CONNECT to it
void wakeup() {
    if(adj.empty()) return;
    int idx = 0;

    for(int i=1;i<(int)adj.size();++i)
    {
        if(adj[i].w < adj[idx].w) idx = i;
    }

    adj[idx].state=1; //branch

    in_branch = adj[idx].nb;
    level_frag=0;
    fragment_id=rank_;
    state_found=1;

    vector<double> msg = {(double)MSG_CONNECT, (double)level_frag};
    send_double_msg(adj[idx].nb, MSG_CONNECT, msg);
    debug_print("wakeup: sent CONNECT to " + to_string(adj[idx].nb));
}

void send_initiate(int to, int level, int fid, int st)
{
    vector<double> msg = {(double)MSG_INITIATE, (double)level, (double)fid, (double)st};
    send_double_msg(to, MSG_INITIATE, msg);
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
    MPI_Comm_size(MPI_COMM_WORLD, &size_);

    if(argc < 2) {
        if(rank_ == 0)
        {
            cerr << "Usage: mpirun -n <N> ./ghs graph.txt\n";
        }
        MPI_Finalize();
        return 1;
    }

    string fname = argv[1];
    ifstream fin(fname);
    if(!fin.is_open()){
        if(rank_==0)cerr<<"Can not open "<<fname<<endl;
        MPI_Finalize();
        return 1;
    }
    vector<tuple<int,int,double>>edges;
    int u,v;
    double w;
    int maxVertex = -1;
    while(fin>>u>>v>>w){
        edges.emplace_back(u,v,w);
        maxVertex=max(maxVertex, max(u,v));
    }
    fin.close();

    if(size_ < maxVertex+1)
    {
        if(rank_==0)cerr<<"Error : need at least "<<(maxVertex+1)<<" processes for this graph.\n";
        MPI_Finalize();
        return 1;
    }

    for(auto &e:edges)
    {
        int a,b; double ww;
        tie(a,b,ww)=e;
        if(a==rank_)adj.emplace_back(b,ww);
        else if(b==rank_)adj.emplace_back(a,ww);
    }

    sort(adj.begin(), adj.end(), [](const Edge &A, const Edge &B){ return A.w < B.w; });

    state_found = 0;
    level_frag = -1;
    fragment_id = rank_;
    in_branch = -1;
    best_neighbor = -1;
    best_weight = numeric_limits<double>::infinity();
    test_edge = -1;

    // global wake up

    if(rank_==0)
    {
        for(int r=0;r<size_;++r)
        {
            if(r==0)continue;
            vector<int> m = {MSG_WAKEUP};
            send_int_msg(r,MSG_WAKEUP, m);
        }
    }

    wakeup();

    bool terminated = false;
    int max_rounds=10*size_;

    for(int round=0;round<max_rounds && !terminated; ++round)
    {
        while(true)
        {
            int src,tag;
            vector<int> idata;
            vector<double> ddata;

            if(!probe_and_recv_any(src,tag,idata,ddata)) break;

            if(tag==MSG_WAKEUP){
                if(state_found==0){
                    wakeup();
                }
            }

            else if (tag == MSG_CONNECT)
            {
                int recv_level = (int) ddata[1];

                int ei = find_edge_index(src);

                if(ei<0)
                {
                    continue;
                }
                if(recv_level < level_frag)
                {
                    adj[ei].state = 1;
                    send_initiate(src,level_frag,fragment_id,2);
                } else {
                    adj[ei].state=1;
                    level_frag=max(level_frag,recv_level) + 1;
                    fragment_id = min(fragment_id,src);

                    for(auto &e: adj) {
                        if(e.state == 1 && e.nb !=src) {
                            send_initiate(e.nb,level_frag,fragment_id,2);
                        }
                    }

                    send_initiate(src, level_frag, fragment_id, 2);
                }
            } 
            else if(tag == MSG_INITIATE)
            {
                int lvl = (int)ddata[1];
                int fid = (int)ddata[2];
                int st = (int)ddata[3];
                level_frag=lvl;
                fragment_id=fid;
                state_found=st;
                in_branch=src;

                best_neighbor = -1;
                best_weight = numeric_limits<double>::infinity();
                test_edge = -1;

                for(auto &e: adj)
                {
                    if(e.state == 1 && e.nb != src){
                        send_initiate(e.nb, level_frag,fragment_id, state_found);
                    }
                }
            }
            else if(tag == MSG_TEST)
            {
                int lvl = (int)ddata[1];
                int fid = (int)ddata[2];
                int ei = find_edge_index(src);
                if(ei<0)continue;
                if(lvl> level_frag || fid != fragment_id) {
                    vector<int> m = {MSG_ACCEPT};
                    send_int_msg(src,MSG_ACCEPT, m);
                } 
                else
                {
                    vector<int> m = {MSG_REJECT};
                    send_int_msg(src,MSG_REJECT,m);
                }
            }
            else if(tag==MSG_ACCEPT)
            {
                int ei = find_edge_index(src);
                if(ei>=0)
                {
                    if(adj[ei].w < best_weight) {
                        best_weight = adj[ei].w;
                        best_neighbor=src;
                    }
                }
            }
            else if(tag==MSG_REJECT)
            {
                int ei = find_edge_index(src);
                if(ei>=0)adj[ei].state=2;
            }
            else if(tag==MSG_REPORT)
            {
                double b = ddata[1];
                if(b<best_weight)
                {
                    best_weight=b;
                }
            }
            else if(tag == MSG_CHANGEROOT)
            {
                int ei = find_edge_index(src);
                if(ei > 0 && adj[ei].state != 1) 
                {
                    adj[ei].state=1;
                    vector<double>msg={(double)MSG_CONNECT,(double)level_frag};
                    send_double_msg(src, MSG_CONNECT, msg);
                }
            }
        }

        if(state_found==1)
        {
            int candidate=-1;
            for(auto&e : adj) 
            {
                if(e.state==0) 
                {
                    candidate = e.nb;
                    break;
                }
            }
            if(candidate !=-1)
            {
                vector<double> msg = {(double)MSG_TEST, (double)level_frag, (double)fragment_id};
                send_double_msg(candidate, MSG_TEST, msg);
            }
            else
            {
                if (in_branch != -1)
                {
                    vector<double> msg = {(double)MSG_REPORT, numeric_limits<double>::infinity()};
                    send_double_msg(in_branch, MSG_REPORT, msg);
                    state_found = 2;
                }
                else
                {
                    state_found = 2;
                }
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);

        int local_done = 0;
        bool any_basic = false;

        for(auto &e:adj) if(e.state ==0)any_basic = true;
        if(state_found == 2 && !any_basic) local_done = 1;

        int global_done = 0;

        MPI_Allreduce(&local_done, &global_done, 1,MPI_INT,MPI_MIN,MPI_COMM_WORLD);
        if(global_done == 1)
        {
            terminated = true;
            for(int r=0;r<size_;++r)
            {
                if(r!=rank_)
                {
                    vector<int> msg = {MSG_DONE};
                    send_int_msg(r,MSG_DONE,msg);
                }
            }
        }

    }

    vector<tuple<int,int,double>> tree;

    for(auto &e:adj)
    {
        if(e.state == 1 && rank_ < e.nb) {
            cout << "Process " << rank_ << " MST edge: " << rank_ << " - " << e.nb << " (w=" << e.w << ")\n";
            tree.push_back(make_tuple(rank_,e.nb,e.w));
        }
    }


    MPI_Finalize();
    return 0;
}