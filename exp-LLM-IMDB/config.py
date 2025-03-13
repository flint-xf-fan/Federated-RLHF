# config.py

# Global configuration variables

# Number of federated learning rounds
NUM_ROUNDS = 5  # Adjust as needed

# Number of clients participating in the federated learning
NUM_CLIENTS = 5  # Adjust as needed

# LAMBDA_LM = 0.5

# Generate LAMBDA_LMs dynamically
LAMBDA_LMs = {i: 0.1 + (0.8 * i / (NUM_CLIENTS - 1)) for i in range(NUM_CLIENTS)}

DATASET_DIVISION = 1  # Adjust as needed

# Flower server mode -- "server" or "simulation"
FLOWER_SIM_MODE = "server"
# FLOWER_SIM_MODE = "simulation"

SEED = 42

# Config for the trainer
TRAINER_CONFIG = {
    "BATCH_SIZE": 16,
    "MINI_BATCH_SIZE": 16,
    "LEARNING_RATE": 1e-5,
    "num_epochs": 5,
    "num_warmup_steps": 100,
    "num_training_steps": 1000,
}

VERBOSE = False
