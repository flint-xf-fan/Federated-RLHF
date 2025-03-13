# client.py

import flwr as fl
from flwr.common import (
    FitIns,
    FitRes,
    Parameters,
    GetParametersIns,
    GetParametersRes,
    EvaluateIns,
    EvaluateRes,
    NDArrays,
    Status,
    Code,
)
from typing import Dict, List
from flwr.common.parameter import ndarrays_to_parameters, parameters_to_ndarrays

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, pipeline
from trl import PPOTrainer, AutoModelForCausalLMWithValueHead, PPOConfig
import os
from collections import OrderedDict
import logging
import csv
import json

from config import TRAINER_CONFIG, DATASET_DIVISION, SEED, VERBOSE

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# Define PPO configuration
ppo_config = PPOConfig(
    model_name="gpt2",
    batch_size=TRAINER_CONFIG['BATCH_SIZE'],          # Adjust based on GPU memory
    mini_batch_size=TRAINER_CONFIG['MINI_BATCH_SIZE'],     # Adjust based on GPU memory
    learning_rate=TRAINER_CONFIG['LEARNING_RATE'],
    gradient_accumulation_steps=1,
    seed=SEED,
    query_dataset="imdb",
    dataset_num_proc=4,
)

class FedRLHFClient(fl.client.Client):
    def __init__(self, client_id: int, num_clients: int, num_rounds: int, lambda_lm: float):
        super().__init__()
        logger.info(f"Initializing FedRLHFClient {client_id} with lambda_lm: {lambda_lm}")
        self.client_id = client_id
        self.num_clients = num_clients
        self.num_rounds = num_rounds  # Total number of federation rounds
        # Initialize metrics lists
        self.rewards_over_samples = []
        self.losses_over_samples = []
        # Initialize step counter
        self.step = 0
        self.total_samples = 0  # Cumulative samples processed

        self.lambda_lm = lambda_lm

        # Initialize components
        self.model = None
        self.tokenizer = None
        self.train_dataset = None
        self.eval_dataset = None
        self.round_datasets = []  # List to hold data partitions for each round
        self.trainer = None
        self.sentiment_pipe = None
        self.stats_log = {"rewards": [], "losses": []}
        self.client_metrics = {
            "avg_rewards": [],
            "avg_losses": [],
            "num_examples": 0,
            "total_samples": [],
        }

        # Clear previous evaluation logs
        self.clear_evaluation_logs()

        self.evaluation_samples = None  # List to hold evaluation samples for actual response

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        self.verbose = VERBOSE  # Set to True for detailed logging

        self.initialize_components()

    def initialize_components(self):
        logger.info("Initializing components")
        try:
            # Initialize model and tokenizer
            self.model = AutoModelForCausalLMWithValueHead.from_pretrained(ppo_config.model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(ppo_config.model_name, padding_side='left', clean_up_tokenization_spaces=True)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.tokenizer.eos_token_id
            self.model.config.padding_side = 'left'

            # Build dataset
            self.build_dataset(ppo_config.query_dataset, ppo_config.dataset_num_proc)
            logger.info(f"Training dataset size: {len(self.train_dataset)}")
            logger.info(f"Evaluation dataset size: {len(self.eval_dataset)}")

            # Partition the training data into num_rounds parts
            self.partition_data()

            # Move model to device
            self.model = self.model.to(self.device)
            logger.info(f"Model moved to {self.device}")

            # Initialize sentiment analysis pipeline
            self.sentiment_pipe = pipeline("sentiment-analysis", model="lvwerra/distilbert-imdb", device=0 if torch.cuda.is_available() else -1)
            logger.info("Sentiment analysis pipeline initialized")

            # Sample evaluation data once for consistent evaluation
            self.sample_evaluation_data(num_samples=30)
        except Exception as e:
            logger.info(f"Error in initialize_components: {str(e)}")
            raise

    def build_dataset(self, query_dataset, dataset_num_proc, input_min_text_length=2, input_max_text_length=8):
        logger.info("Building dataset")
        full_ds = load_dataset(query_dataset, split="train")
        full_ds = full_ds.rename_columns({"text": "review"})
        full_ds = full_ds.filter(lambda x: len(x["review"]) > 200, num_proc=dataset_num_proc)

        # Limit dataset size for testing
        max_total_samples = len(full_ds) // DATASET_DIVISION
        full_ds = full_ds.select(range(min(len(full_ds), max_total_samples)))

        # Partition dataset among clients
        total_size = len(full_ds)
        partition_size = total_size // self.num_clients

        start_idx = self.client_id * partition_size
        end_idx = start_idx + partition_size if self.client_id < self.num_clients - 1 else total_size

        ds = full_ds.select(range(start_idx, end_idx))

        def tokenize(sample):
            encoding = self.tokenizer(
                sample["review"],
                truncation=True,
                max_length=input_max_text_length,
                return_tensors="pt",
            )
            sample["input_ids"] = encoding["input_ids"][0]
            sample["attention_mask"] = encoding["attention_mask"][0]
            sample["query"] = self.tokenizer.decode(
                sample["input_ids"],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            return sample

        ds = ds.map(tokenize, num_proc=dataset_num_proc)
        ds.set_format(type="torch", columns=["input_ids", "attention_mask", "query"])
        logger.info(f"Dataset built successfully for client {self.client_id} with {len(ds)} samples")

        # Split into train and eval datasets
        split_ds = ds.train_test_split(test_size=0.2, seed=ppo_config.seed)
        self.train_dataset = split_ds['train']
        self.eval_dataset = split_ds['test']

    def partition_data(self):
        # Partition the training data into num_rounds parts
        num_samples = len(self.train_dataset)
        samples_per_round = num_samples // self.num_rounds
        self.round_datasets = []

        for i in range(self.num_rounds):
            start_idx = i * samples_per_round
            end_idx = start_idx + samples_per_round if i < self.num_rounds - 1 else num_samples
            round_dataset = self.train_dataset.select(range(start_idx, end_idx))
            self.round_datasets.append(round_dataset)
            logger.info(f"Client {self.client_id} - Round {i+1}: {len(round_dataset)} samples")

    def collator(self, data):
        return {
            "input_ids": torch.nn.utils.rnn.pad_sequence([d["input_ids"] for d in data], batch_first=True, padding_value=self.tokenizer.pad_token_id),
            "attention_mask": torch.nn.utils.rnn.pad_sequence([torch.ones_like(d["input_ids"]) for d in data], batch_first=True, padding_value=0),
            "query": [d["query"] if "query" in d else self.tokenizer.decode(d["input_ids"]) for d in data]
        }

    def compute_rewards(self, queries, responses):
        texts = [q + r for q, r in zip(queries, responses)]
        max_length = 512

        # Sentiment analysis reward
        pipe_outputs = self.sentiment_pipe(texts, truncation=True, max_length=max_length)
        sentiment_rewards = []
        for i, output in enumerate(pipe_outputs):
            if isinstance(output, dict) and "label" in output and "score" in output:
                reward = torch.tensor(output["score"] if output["label"] == "POSITIVE" else 1 - output["score"], device=self.device)
            else:
                logger.info(f"Unexpected output format: {output}")
                reward = torch.tensor(0.5, device=self.device)  # Default reward if no valid sentiment data
            sentiment_rewards.append(reward)

            # Verbose logging for first few samples
            if self.verbose and i < 5:
                logger.info(f"Sample {i}: Query: {queries[i][:50]}... Response: {responses[i][:50]}... Sentiment Reward: {reward.item():.4f}")

        # Convert list of sentiment rewards to tensor
        sentiment_rewards = torch.stack(sentiment_rewards)

        # Calculate intrinsic reward (Negative log probability)
        with torch.no_grad():
            # Tokenize responses and move to the device
            lm_inputs = self.tokenizer(responses, return_tensors="pt", padding=True, truncation=True).to(self.device)

            # Pass the inputs through the model (assuming model outputs a tuple)
            lm_outputs = self.model(**lm_inputs)
            lm_logits = lm_outputs[0]

            # Get the actual tokens in the responses (i.e., target tokens for calculating probabilities)
            input_ids = lm_inputs["input_ids"]

            # Shift input_ids to get the correct next-token targets
            shift_labels = input_ids[..., 1:].contiguous()

            # Get the corresponding logits for the shifted tokens
            shift_logits = lm_logits[..., :-1, :].contiguous()

            # Compute log probabilities for the actual tokens (shift_labels)
            log_probs = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction='none'
            )

            # Reshape log_probs back to [batch_size, seq_len-1]
            log_probs = log_probs.view(shift_labels.size())

            # Compute the intrinsic reward as the negative log probability (mean over tokens in the sequence)
            intrinsic_rewards = -log_probs.mean(dim=1)

        # Define the theoretical minimum and maximum intrinsic rewards
        V = self.tokenizer.vocab_size  # Vocabulary size, e.g., 50,000
        min_possible = -torch.log(torch.tensor(V, dtype=torch.float32, device=self.device))
        max_possible = torch.tensor(0.0, device=self.device)

        # Normalize intrinsic rewards to [0, 1]
        intrinsic_rewards_norm = (intrinsic_rewards - min_possible) / (max_possible - min_possible)
        intrinsic_rewards_norm = torch.clamp(intrinsic_rewards_norm, min=0.0, max=1.0)

        # Compute combined rewards
        combined_rewards = self.lambda_lm * sentiment_rewards + (1 - self.lambda_lm) * intrinsic_rewards_norm

        # Ensure rewards are lists of tensors
        combined_rewards = [r for r in combined_rewards]
        sentiment_rewards = [r for r in sentiment_rewards]
        intrinsic_rewards = [r for r in intrinsic_rewards_norm]

        # Log combined reward for debugging
        if self.verbose:
            logger.info(f"Combined Reward for Batch: {torch.stack(combined_rewards).mean().item():.4f}")

        return combined_rewards, sentiment_rewards, intrinsic_rewards_norm

    def plot_metrics(self):
        if not self.rewards_over_samples or not self.losses_over_samples:
            logger.info(f"No metrics to plot for client {self.client_id}.")
            return

        samples_rewards, avg_rewards = zip(*self.rewards_over_samples)
        samples_losses, losses = zip(*self.losses_over_samples)

        plt.figure(figsize=(12, 6))

        # Rewards subplot
        plt.subplot(2, 1, 1)
        plt.plot(samples_rewards, avg_rewards, marker='o', label='Average Reward')
        plt.title(f"Client {self.client_id} - Average Reward over Samples")
        plt.xlabel("Total Samples")
        plt.ylabel("Average Reward")
        plt.legend()

        # Losses subplot
        plt.subplot(2, 1, 2)
        plt.plot(samples_losses, losses, marker='o', label='Loss')
        plt.title(f"Client {self.client_id} - Loss over Samples")
        plt.xlabel("Total Samples")
        plt.ylabel("Loss")
        plt.legend()

        plt.tight_layout()
        os.makedirs("training_logs", exist_ok=True)
        plt.savefig(f"training_logs/ppo_training_client_{self.client_id}.png")
        plt.close()

        logger.info(f"Training metrics plotted and saved for client {self.client_id}.")

        # Save metrics to a JSON file for individual client analysis
        metrics_data = {
            "total_samples": [s for s, _ in self.rewards_over_samples],
            "avg_rewards": [r for _, r in self.rewards_over_samples],
            "losses": [l for _, l in self.losses_over_samples],
            "client_id": self.client_id,
        }
        os.makedirs("metrics", exist_ok=True)
        with open(f"metrics/metrics_client_{self.client_id}.json", "w") as f:
            json.dump(metrics_data, f)
        logger.info(f"Metrics saved for client {self.client_id}.")

    def fit(self, ins: FitIns) -> FitRes:
        self.set_parameters(ins.parameters)

        # Retrieve the current round number from FitIns.config
        round_num = int(ins.config.get("round", 1))
        logger.info(f"Client {self.client_id} - Starting training for round {round_num}")

        # Select the data partition for the current round
        if round_num <= len(self.round_datasets):
            current_dataset = self.round_datasets[round_num - 1]
        else:
            current_dataset = self.round_datasets[-1]

        # Initialize PPO Trainer with the current round's dataset
        self.trainer = PPOTrainer(
            config=ppo_config,
            model=self.model,
            ref_model=None,
            tokenizer=self.tokenizer,
            dataset=current_dataset,
            data_collator=self.collator
        )

        generation_kwargs = {
            "min_length": -1,
            "top_k": 0.0,
            "top_p": 1.0,
            "do_sample": True,
            "pad_token_id": self.tokenizer.eos_token_id,
            "max_new_tokens": 32,
        }

        total_rewards = []
        total_losses = []
        num_examples = 0  # Number of samples processed in this round

        for batch_idx, batch in enumerate(tqdm(self.trainer.dataloader, desc=f"Client {self.client_id} - Round {round_num} Training")):
            query_tensors = batch["input_ids"].to(self.device)
            query_tensor_list = [tensor for tensor in query_tensors]

            response_tensors = self.trainer.generate(
                query_tensor_list,
                return_prompt=False,
                **generation_kwargs
            )

            decoded_queries = self.tokenizer.batch_decode(query_tensors, clean_up_tokenization_spaces=True)
            decoded_responses = self.tokenizer.batch_decode(response_tensors, clean_up_tokenization_spaces=True)

            combined_rewards, _, _ = self.compute_rewards(decoded_queries, decoded_responses)
            rewards = combined_rewards
            avg_reward = sum([r.item() for r in rewards]) / len(rewards)
            total_rewards.extend([r.item() for r in rewards])

            stats = self.trainer.step(query_tensor_list, response_tensors, rewards)

            loss = stats.get("ppo/loss/total", 0.0)
            total_losses.append(loss)

            # Increment the total sample count
            batch_size = len(query_tensors)
            self.total_samples += batch_size
            num_examples += batch_size

            # Collect metrics
            self.rewards_over_samples.append((self.total_samples, avg_reward))
            self.losses_over_samples.append((self.total_samples, loss))

            if self.verbose:
                logger.info(f"Batch {batch_idx + 1}:")
                logger.info(f"  Average reward for the batch: {avg_reward:.4f}")
                logger.info(f"  Loss: {loss:.4f}")

            # Increment the total step count
            self.step += 1

        logger.info(f"Training completed for client {self.client_id} - Round {round_num}")

        # Save metrics to file after each round
        self.plot_metrics()

        # Save the model after training
        self.save_model(round_num)

        # Sample and evaluate on a few samples from eval dataset
        self.sample_and_evaluate(round_num)

        # Update total number of examples processed
        self.client_metrics["num_examples"] += num_examples
        self.client_metrics["total_samples"].append(self.total_samples)

        # Calculate average reward and loss for this round
        avg_reward = sum(total_rewards) / len(total_rewards) if total_rewards else 0.0
        avg_loss = sum(total_losses) / len(total_losses) if total_losses else 0.0

        self.client_metrics["avg_rewards"].append(avg_reward)
        self.client_metrics["avg_losses"].append(avg_loss)

        # Prepare metrics to send to the server
        metrics = {
            "client_id": self.client_id,
            "avg_reward": avg_reward,
            "avg_loss": avg_loss,
            "num_examples": num_examples,
            "total_steps": self.step,
            "total_samples": self.total_samples
        }

        parameters = self.get_parameters(GetParametersIns(config={})).parameters
        return FitRes(
            parameters=parameters,
            num_examples=num_examples,
            metrics=metrics,  # Ensure metrics are passed here
            status=Status(code=Code.OK, message="Success")
        )

    def clear_evaluation_logs(self):
        # Delete the client's CSV file in evaluation_logs directory
        filename = f"evaluation_logs/client_{self.client_id}.csv"
        if os.path.exists(filename):
            os.remove(filename)
            logger.info(f"Previous evaluation log {filename} removed.")
        else:
            logger.info(f"No previous evaluation log to remove for client {self.client_id}.")

    def sample_evaluation_data(self, num_samples=5):
        # Sample a few samples from the eval dataset
        self.evaluation_samples = self.eval_dataset.shuffle(seed=SEED).select(range(num_samples))
        logger.info(f"Sampled {num_samples} evaluation samples for client {self.client_id}")

    def sample_and_evaluate(self, round_num):
        # Use the pre-sampled evaluation data
        sampled_dataset = self.evaluation_samples

        # Initialize lists to store results
        results = []

        # Iterate over samples
        for sample in sampled_dataset:
            query_tensor = sample["input_ids"].unsqueeze(0).to(self.device)
            query = sample["query"]

            # Generate response
            generation_kwargs = {
                "min_length": -1,
                "top_k": 0.0,
                "top_p": 1.0,
                "do_sample": True,
                "pad_token_id": self.tokenizer.eos_token_id,
                "max_new_tokens": 32,
            }
            response_tensor = self.model.generate(
                query_tensor,
                **generation_kwargs
            )
            generated_tokens = response_tensor[0]
            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            # Extract the response by removing the query from the generated text
            if generated_text.startswith(query):
                response = generated_text[len(query):]
            else:
                response = generated_text  # Fallback in case the query is not at the start

            # Compute rewards
            combined_reward, sentiment_reward, intrinsic_reward = self.compute_rewards([query], [response])
            combined_reward = combined_reward[0].item()
            sentiment_reward = sentiment_reward[0].item()
            intrinsic_reward = intrinsic_reward[0].item()

            # Store the result
            results.append({
                "client_id": self.client_id,
                "round": round_num,
                "query": query,
                "response": response,
                "sentiment_reward": sentiment_reward,
                "intrinsic_reward": intrinsic_reward,
                "combined_reward": combined_reward
            })

        # Save results to a single CSV file per client
        os.makedirs("evaluation_logs", exist_ok=True)
        filename = f"evaluation_logs/client_{self.client_id}.csv"
        fieldnames = ["client_id", "round", "query", "response", "sentiment_reward", "intrinsic_reward", "combined_reward"]

        # Always open in append mode since we cleared the file at the start
        with open(filename, mode='a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            # Write header only if the file does not exist or is empty
            if os.stat(filename).st_size == 0:
                writer.writeheader()
            for row in results:
                writer.writerow(row)

        logger.info(f"Evaluation samples appended for client {self.client_id} to {filename}")

    def save_model(self, round_num):
        # Create the directory if it doesn't exist
        save_dir = "trained_models"
        os.makedirs(save_dir, exist_ok=True)

        # Define the model save path
        model_save_path = os.path.join(save_dir, f"client_{self.client_id}_round_{round_num}")

        # Save the model and tokenizer
        self.model.save_pretrained(model_save_path)
        self.tokenizer.save_pretrained(model_save_path)
        logger.info(f"Model saved for client {self.client_id} at {model_save_path}")

    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        ndarrays = []
        for name, param in self.model.named_parameters():
            if name in self.model.state_dict():
                ndarrays.append(param.detach().cpu().numpy())

        parameters = ndarrays_to_parameters(ndarrays)
        return GetParametersRes(parameters=parameters, status=Status(code=Code.OK, message="Success"))

    def get_parameters_as_ndarrays(self) -> List[np.ndarray]:
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: Parameters) -> None:
        params_dict = zip(self.model.state_dict().keys(), parameters.tensors)
        state_dict = OrderedDict()
        for k, v in params_dict:
            if k in self.model.state_dict():
                try:
                    state_dict[k] = torch.tensor(np.frombuffer(v, dtype=np.float32).reshape(self.model.state_dict()[k].shape))
                except ValueError:
                    logger.info(f"Ignoring parameter {k} due to shape mismatch")
        self.model.load_state_dict(state_dict, strict=False)

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        self.set_parameters(ins.parameters)

        generation_kwargs = {
            "min_length": -1,
            "top_k": 0.0,
            "top_p": 1.0,
            "do_sample": True,
            "pad_token_id": self.tokenizer.eos_token_id,
            "max_new_tokens": 32,
        }

        total_rewards = []
        total_losses = []
        num_samples = 0

        eval_dataloader = torch.utils.data.DataLoader(
            self.eval_dataset,
            batch_size=ppo_config.batch_size,
            collate_fn=self.collator
        )

        self.model.eval()

        for batch in tqdm(eval_dataloader, desc=f"Client {self.client_id} - Evaluation"):
            query_tensors = batch["input_ids"].to(self.device)
            query_tensor_list = [tensor for tensor in query_tensors]

            # Generate responses
            response_tensors = self.trainer.generate(
                query_tensor_list,
                return_prompt=False,
                **generation_kwargs
            )

            # Decode queries and responses
            decoded_queries = self.tokenizer.batch_decode(query_tensors, clean_up_tokenization_spaces=True)
            decoded_responses = self.tokenizer.batch_decode(response_tensors, clean_up_tokenization_spaces=True)

            # Compute rewards
            combined_rewards, _, _ = self.compute_rewards(decoded_queries, decoded_responses)
            rewards = combined_rewards
            rewards = torch.stack(rewards).to(self.device)
            avg_reward = rewards.mean().item()
            total_rewards.extend([r.item() for r in rewards])

            # Approximate loss as negative average reward
            loss = -avg_reward
            total_losses.append(loss)
            num_samples += len(rewards)

            if self.verbose:
                logger.info(f"Batch Evaluation: Average Reward: {avg_reward:.4f}, Loss: {loss:.4f}")

        # Compute average reward and loss
        average_reward = sum(total_rewards) / num_samples if num_samples > 0 else 0.0
        avg_loss = sum(total_losses) / len(total_losses) if total_losses else 0.0

        logger.info(f"Evaluation completed for client {self.client_id}. Average Reward: {average_reward:.4f}, Average Loss: {avg_loss:.4f}")

        return EvaluateRes(
            loss=avg_loss,
            num_examples=num_samples,
            metrics={
                "client_id": self.client_id,
                "average_reward": average_reward,
                "avg_loss": avg_loss,
                "num_examples": num_samples
            },
            status=Status(code=Code.OK, message="Success")
        )

    def set_parameters_from_ndarrays(self, params: NDArrays) -> None:
        params_dict = zip(self.model.state_dict().keys(), params)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=False)
