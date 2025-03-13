# start_client.py

import flwr as fl
from client import FedRLHFClient
import sys
from config import NUM_CLIENTS, NUM_ROUNDS, LAMBDA_LMs

def main(client_id: int, num_clients: int, num_rounds: int, lambda_lm: float) -> None:
    # Initialize your client with the new arguments
    client = FedRLHFClient(client_id, num_clients, num_rounds, lambda_lm)

    # Start Flower client
    fl.client.start_client(
        server_address="127.0.0.1:8080",
        client=client,
        grpc_max_message_length=1024 * 1024 * 1024  # 1 GB
    )

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python start_client.py <client_id>")
        sys.exit(1)
    client_id = int(sys.argv[1])
    client_lambda_lm = LAMBDA_LMs[client_id]
    main(client_id, NUM_CLIENTS, NUM_ROUNDS, client_lambda_lm)
