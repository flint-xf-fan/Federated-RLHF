# server.py

import os
import flwr as fl
from flwr.server.strategy import FedAvg
from typing import List, Tuple, Dict, Optional
from flwr.common import Parameters, FitRes, Status, Code
from flwr.server.client_proxy import ClientProxy
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import json
import seaborn as sns
sns.set_theme(style="darkgrid")

from config import NUM_CLIENTS, NUM_ROUNDS, FLOWER_SIM_MODE, LAMBDA_LMs

# Custom FedAvg strategy with overridden aggregate_fit
class CustomFedAvg(FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client_metrics = defaultdict(list)
        self.global_metrics = defaultdict(list)

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, float]]:
        # Use the superclass method to aggregate parameters
        aggregated_params, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_params is not None:
            # Process client-specific metrics
            total_steps = 0
            total_samples = 0
            rewards = []
            losses = []
            for _, fit_res in results:
                client_id = fit_res.metrics["client_id"]
                rewards.append(fit_res.metrics["avg_reward"])
                losses.append(fit_res.metrics["avg_loss"])
                total_steps += fit_res.metrics["total_steps"]
                total_samples += fit_res.metrics["total_samples"]

                self.client_metrics[client_id].append({
                    "round": server_round,
                    "avg_reward": fit_res.metrics["avg_reward"],
                    "avg_loss": fit_res.metrics["avg_loss"],
                    "total_steps": fit_res.metrics["total_steps"],
                    "total_samples": fit_res.metrics["total_samples"],
                })

            # Calculate simple averages
            avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
            avg_loss = sum(losses) / len(losses) if losses else 0.0

            # Save metrics
            self.global_metrics["rounds"].append(server_round)
            self.global_metrics["avg_reward"].append(avg_reward)
            self.global_metrics["avg_loss"].append(avg_loss)
            self.global_metrics["total_steps"].append(total_steps)
            self.global_metrics["total_samples"].append(total_samples)

            print(f"Round {server_round} - Global Metrics:")
            print(f"  Average Reward: {avg_reward:.4f}")
            print(f"  Average Loss: {avg_loss:.4f}")
            print(f"  Total Steps: {total_steps}")
            print(f"  Total Samples: {total_samples}")

            # Generate and save the visualization
            self.visualize_metrics()

        return aggregated_params, aggregated_metrics

    def visualize_metrics(self):
        rounds = range(1, len(self.global_metrics["avg_reward"]) + 1)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))  # Increased figure height

        # Rewards subplot
        for client_id, metrics in self.client_metrics.items():
            client_rounds = [m['round'] for m in metrics]
            rewards = [m['avg_reward'] for m in metrics]
            ax1.plot(client_rounds, rewards, marker='o', label=f'Client {client_id} Reward')
        ax1.plot(rounds, self.global_metrics["avg_reward"], marker='s', linewidth=2, linestyle='--', label='Global Avg Reward')
        # ax1.set_title("Per-Client and Global Average Rewards")
        ax1.set_xlabel("Round",fontsize=16)
        ax1.set_ylabel("Reward",fontsize=16)

        # Losses subplot
        for client_id, metrics in self.client_metrics.items():
            client_rounds = [m['round'] for m in metrics]
            losses = [m['avg_loss'] for m in metrics]
            ax2.plot(client_rounds, losses, marker='o', label=f'Client {client_id} Loss')
        ax2.plot(rounds, self.global_metrics["avg_loss"], marker='s', linewidth=2, linestyle='--', label='Global Avg Loss')
        # ax2.set_title("Per-Client and Global Average Losses")
        ax2.set_xlabel("Round",fontsize=16)
        ax2.set_ylabel("Loss",fontsize=16)

        # Adjust layout and add a single legend
        plt.tight_layout()
        handles, labels = ax1.get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(0.52, -0.05), fontsize=16, handlelength=2, columnspacing=1)

        # Adjust subplot spacing to make room for the legend
        plt.subplots_adjust(bottom=0.1)

        os.makedirs("training_logs", exist_ok=True)
        plt.savefig("training_logs/global_performance.pdf",dpi=300, bbox_inches='tight')
        plt.close()
        print("Visualization saved as 'training_logs/global_performance.png'")

        # Save metrics to a single JSON file for combined plotting
        metrics_data = {
            "rounds": self.global_metrics["rounds"],
            "avg_rewards": self.global_metrics["avg_reward"],
            "avg_losses": self.global_metrics["avg_loss"],
            "total_steps": self.global_metrics["total_steps"],
            "total_samples": self.global_metrics["total_samples"],
        }
        os.makedirs("metrics", exist_ok=True)
        with open(f"metrics/metrics_federated_k{NUM_CLIENTS}.json", "w") as f:
            json.dump(metrics_data, f)
        print(f"Federated metrics saved as 'metrics/metrics_federated_k{NUM_CLIENTS}.json'")

    def plot_performance_vs_samples(self):
        """Plot global average reward vs total samples."""
        plt.figure(figsize=(10, 6))
        total_samples = self.global_metrics["total_samples"]
        avg_rewards = self.global_metrics["avg_reward"]
        plt.plot(total_samples, avg_rewards, marker='s', linewidth=2, linestyle='--', label=f'FedRLHF K={NUM_CLIENTS}')
        plt.xlabel("Total Samples")
        plt.ylabel("Average Reward")
        plt.title("Performance vs. Number of Samples")
        plt.legend()
        plt.grid(True)
        os.makedirs("training_logs", exist_ok=True)
        plt.savefig(f"training_logs/performance_vs_samples_k{NUM_CLIENTS}.png")
        plt.close()
        print(f"Performance vs. Samples plot saved as 'training_logs/performance_vs_samples_k{NUM_CLIENTS}.png'")

def start_federated_server():
    print("Starting federated server")
    strategy = CustomFedAvg(
        fraction_fit=1.0,
        min_fit_clients=NUM_CLIENTS,
        min_evaluate_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
        on_fit_config_fn=lambda rnd: {"round": rnd}  # Pass the current round number
    )

    # Start the server
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
        grpc_max_message_length=1024 * 1024 * 1024  # 1 GB
    )

def start_federated_simulation():
    print("Starting federated simulation")

    # Import the client class
    from client import FedRLHFClient

    # Define the client function
    def client_fn(cid: str):
        # Convert client ID to integer
        client_id = int(cid)
        client_lambda_lm = LAMBDA_LMs[client_id]
        return FedRLHFClient(client_id=client_id, num_clients=NUM_CLIENTS, num_rounds=NUM_ROUNDS, lambda_lm=client_lambda_lm)

    # Create the strategy
    strategy = CustomFedAvg(
        fraction_fit=1.0,
        min_fit_clients=NUM_CLIENTS,
        min_evaluate_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
        on_fit_config_fn=lambda rnd: {"round": rnd}  # Pass the current round number
    )

    # Start the simulation
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
        client_resources={"num_cpus": 1},
    )

if __name__ == "__main__":
    if FLOWER_SIM_MODE == "server":
        start_federated_server()
    elif FLOWER_SIM_MODE == "simulation":
        start_federated_simulation()
    else:
        raise ValueError(f"Invalid server mode: {FLOWER_SIM_MODE}. Please set FLOWER_SIM_MODE to 'server' or 'simulation'.")
