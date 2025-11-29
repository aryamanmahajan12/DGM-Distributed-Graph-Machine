# DGM-Distributed-Graph-Machine
The repository contains the implementation of Graph algorithms, such as the MST finding algorithm (GHS). The main frameworks used are MPI and K-Machine

## Minumum Spanning Tree - GHS Algoorithm - MPI Framework based


### Requirements : OpenMPI , Standard Template Library (STL)

Instructions for intalling Open MPI on Linux :

sudo apt install openmpi-bin openmpi-common libopenmpi-dev 


### Commands to run the script :

Commands to run : 

mpicxx -std=c++17 -O2 -o {object filename} ghs.cpp

mpirun -n {number of nodes in input graph} ./{object filename} {input grapph filename}.txt

Example : 

mpicxx -std=c++17 -O2 -o ghs ghs.cpp

mpirun -n 5 ./ghs graph.txt


## Minumum Spanning Tree - Boruvka Algoorithm - using ZeroMQ


### Requirements : ZeroMQ

Installation instructions for Linux : 

sudo apt-get install -y libzmq3-dev


## Command to compile and build : 

g++ -std=c++17 -O2 -pthread boruvka_zmq.cpp -o boruvka_zmq $(pkg-config --cflags --libs libzmq)


## Command to run :

Root terminal : 

./boruvka_zmq graph.txt 0 K


Terminals 2..K-1 (workers) :

./boruvka_zmq graph.txt 1 4

./boruvka_zmq graph.txt 2 4

./boruvka_zmq graph.txt 3 4




## commands for cluster :

salloc --nodes=1 --ntasks=10 --time=00:30:00
