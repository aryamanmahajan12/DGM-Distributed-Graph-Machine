#!/bin/bash

# Check if K is provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <K>"
    echo "Example: $0 4"
    exit 1
fi

K=$1

# Validate K is a number and greater than 1
if ! [[ "$K" =~ ^[0-9]+$ ]] || [ "$K" -lt 2 ]; then
    echo "Error: K must be a number greater than 1"
    exit 1
fi

echo "Starting Boruvka ZeroMQ with K=$K processes..."

# Start the root process (process 0) in the background
echo "Starting root process (ID 0)..."
./boruvka_zmq_perf ../Input/graph_s.txt 0 $K &
ROOT_PID=$!

# Give root process time to initialize
sleep 5

# Start worker processes (1 to K-1)
for ((i=1; i<K; i++)); do
    echo "Starting worker process (ID $i)..."
    ./boruvka_zmq_perf ../Input/graph_s.txt $i $K &
done

echo "All $K processes started!"
echo "Root process PID: $ROOT_PID"
echo "Press Ctrl+C to terminate all processes"

# Wait for root process to complete
wait $ROOT_PID

# Kill any remaining worker processes
echo "Root process completed. Cleaning up workers..."
pkill -P $$ boruvka_zmq

echo "Done!"