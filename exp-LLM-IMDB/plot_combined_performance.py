import matplotlib.pyplot as plt
import json
import os
import numpy as np
from config import NUM_CLIENTS
import seaborn as sns

def load_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def interpolate_data(x, y, new_x):
    return np.interp(new_x, x, y)

def plot_combined_performance():
    sns.set_theme(style="darkgrid")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Load centralized data
    cent_data = load_json('metrics/metrics_centralized.json')
    cent_samples = np.array(cent_data['total_samples'])
    cent_rewards = np.array(cent_data['avg_rewards'])
    cent_losses = np.array(cent_data['avg_losses'])

    # Load and aggregate client data
    client_data = [load_json(f'metrics/metrics_client_{i}.json') for i in range(NUM_CLIENTS)]
    fed_samples = np.array(client_data[0]['total_samples'])  # Assuming all clients have same sample points
    fed_rewards = np.array([data['avg_rewards'] for data in client_data])
    fed_losses = np.array([data['losses'] for data in client_data])

    # Calculate mean and std of federated rewards and losses
    fed_mean_rewards = np.mean(fed_rewards, axis=0)
    fed_std_rewards = np.std(fed_rewards, axis=0)
    fed_mean_losses = np.mean(fed_losses, axis=0)
    fed_std_losses = np.std(fed_losses, axis=0)

    # Interpolate centralized data to match federated sample points
    cent_rewards_interp = interpolate_data(cent_samples, cent_rewards, fed_samples)
    cent_losses_interp = interpolate_data(cent_samples, cent_losses, fed_samples)

    # Plotting Rewards
    ax1.plot(fed_samples, cent_rewards_interp, label='Centralized RLHF', color='blue')
    ax1.plot(fed_samples, fed_mean_rewards, label=f'FedRLHF (K={NUM_CLIENTS})', color='red')
    ax1.fill_between(fed_samples, fed_mean_rewards - fed_std_rewards, fed_mean_rewards + fed_std_rewards, alpha=0.3, color='red')
    ax1.set_ylabel('Average Reward',fontsize=16)
    # ax1.set_title('Rewards Comparison: Centralized vs Federated',fontsize=14)
    ax1.legend(fontsize=16)
    ax1.grid(True)

    # Plotting Losses
    ax2.plot(fed_samples, cent_losses_interp, label='Centralized RLHF', color='blue')
    ax2.plot(fed_samples, fed_mean_losses, label=f'FedRLHF (K={NUM_CLIENTS})', color='red')
    ax2.fill_between(fed_samples, fed_mean_losses - fed_std_losses, fed_mean_losses + fed_std_losses, alpha=0.3, color='red')
    ax2.set_xlabel('Total Samples',fontsize=16)
    ax2.set_ylabel('Average Loss',fontsize=16)
    # ax2.set_title('Losses Comparison: Centralized vs Federated',fontsize=14)
    ax2.legend(fontsize=16)
    ax2.grid(True)

    # Adjust x-axis to start from 0
    ax1.set_xlim(0, max(fed_samples))
    ax2.set_xlim(0, max(fed_samples))

    plt.tight_layout()

    # Save the plot
    os.makedirs("training_logs", exist_ok=True)
    plt.savefig(f"training_logs/performance_comparison_k{NUM_CLIENTS}.pdf", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Combined performance plot saved as 'training_logs/performance_comparison_k{NUM_CLIENTS}.pdf'")

    # Additional analysis
    print("\nPerformance Analysis:")
    print(f"Centralized final reward: {cent_rewards[-1]:.4f}")
    print(f"Federated final mean reward: {fed_mean_rewards[-1]:.4f}")
    print(f"Federated final reward std: {fed_std_rewards[-1]:.4f}")
    print(f"Centralized final loss: {cent_losses[-1]:.4f}")
    print(f"Federated final mean loss: {fed_mean_losses[-1]:.4f}")
    print(f"Federated final loss std: {fed_std_losses[-1]:.4f}")

if __name__ == "__main__":
    plot_combined_performance()