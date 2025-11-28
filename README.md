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