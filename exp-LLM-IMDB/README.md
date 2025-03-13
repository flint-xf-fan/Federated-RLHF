# FedRLHF
## Task: IMDB
pass
## Description 
pass
## TODO

## Set up
```
conda env create -f environment.yml
conda activate fedrlhf
```

## Running Experiments
first, ensure `num_clients` are set to the same number in both `main.py` and `run_multiple_clients.sh`

then, start the FedRLHF server by running

```
python server.py
```

in another terminal, run the bash script to simulate multiple clients connecting to the server via gRPC protocol (check and verify the port number is not in use; default is 8080):


```
bash run_multiple_clients.sh
```