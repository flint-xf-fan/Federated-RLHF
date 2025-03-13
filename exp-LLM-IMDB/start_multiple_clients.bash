#!/bin/bash

# Read num_clients from config.py using Python
NUM_CLIENTS=$(python -c 'import config; print(config.NUM_CLIENTS)')

# Loop to start multiple clients
for (( i=0; i<$NUM_CLIENTS; i++ ))
do
    echo "Starting client $i"
    python start_client.py $i &
done

# Wait for all background processes to finish
wait

echo "All clients have finished"