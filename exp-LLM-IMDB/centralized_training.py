# centralized_training.py

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import json
from datasets import load_dataset
from transformers import AutoTokenizer, pipeline
from trl import PPOTrainer, AutoModelForCausalLMWithValueHead, PPOConfig
from tqdm import tqdm
import logging

from config import TRAINER_CONFIG, DATASET_DIVISION, SEED

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# Define PPO configuration
ppo_config = PPOConfig(
    model_name="gpt2",
    batch_size=TRAINER_CONFIG['BATCH_SIZE'],
    mini_batch_size=TRAINER_CONFIG['MINI_BATCH_SIZE'],
    learning_rate=TRAINER_CONFIG['LEARNING_RATE'],
    gradient_accumulation_steps=1,
    seed=SEED,
    query_dataset="imdb",
    dataset_num_proc=4,
)

class CentralizedRLHFTrainer:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.train_dataset = None
        self.eval_dataset = None
        self.trainer = None
        self.sentiment_pipe = None
        self.rewards_over_samples = []
        self.losses_over_samples = []
        self.step = 0
        self.total_samples = 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        self.verbose = False  # Set to True for detailed logging

        self.initialize_components()

    def initialize_components(self):
        logger.info("Initializing components for centralized training")
        self.model = AutoModelForCausalLMWithValueHead.from_pretrained(ppo_config.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(ppo_config.model_name, padding_side='left', clean_up_tokenization_spaces=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.tokenizer.eos_token_id
        self.model.config.padding_side = 'left'

        # Build dataset
        self.build_dataset(ppo_config.query_dataset, ppo_config.dataset_num_proc)
        logger.info(f"Training dataset size: {len(self.train_dataset)}")
        logger.info(f"Evaluation dataset size: {len(self.eval_dataset)}")

        # Move model to device
        self.model = self.model.to(self.device)
        logger.info(f"Model moved to {self.device}")

        # Initialize sentiment analysis pipeline
        self.sentiment_pipe = pipeline("sentiment-analysis", model="lvwerra/distilbert-imdb", device=0 if torch.cuda.is_available() else -1)
        logger.info("Sentiment analysis pipeline initialized")

        # Initialize PPO Trainer
        self.trainer = PPOTrainer(
            config=ppo_config,
            model=self.model,
            ref_model=None,
            tokenizer=self.tokenizer,
            dataset=self.train_dataset,
            data_collator=self.collator
        )

    def build_dataset(self, query_dataset, dataset_num_proc, input_min_text_length=2, input_max_text_length=8):
        logger.info("Building dataset")
        full_ds = load_dataset(query_dataset, split="train")
        full_ds = full_ds.rename_columns({"text": "review"})
        full_ds = full_ds.filter(lambda x: len(x["review"]) > 200, num_proc=dataset_num_proc)

        # Limit dataset size for testing
        max_total_samples = len(full_ds) // DATASET_DIVISION
        full_ds = full_ds.select(range(min(len(full_ds), max_total_samples)))

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

        ds = full_ds.map(tokenize, num_proc=dataset_num_proc)
        ds.set_format(type="torch", columns=["input_ids", "attention_mask", "query"])
        logger.info(f"Dataset built successfully with {len(ds)} samples")

        # Split into train and eval datasets
        split_ds = ds.train_test_split(test_size=0.2, seed=ppo_config.seed)
        self.train_dataset = split_ds['train']
        self.eval_dataset = split_ds['test']

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

            # Pass the inputs through the model
            lm_outputs = self.model(**lm_inputs)
            lm_logits = lm_outputs[0]

            # Get the actual tokens in the responses
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
        lambda_lm = 0.5
        combined_rewards = lambda_lm * sentiment_rewards + (1 - lambda_lm) * intrinsic_rewards_norm

        # Ensure rewards are lists of tensors
        combined_rewards = [r for r in combined_rewards]

        # Log combined reward for debugging
        if self.verbose:
            logger.info(f"Combined Reward for Batch: {torch.stack(combined_rewards).mean().item():.4f}")

        return combined_rewards

    def train(self):
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

        for epoch in range(TRAINER_CONFIG["num_epochs"]):
            logger.info(f"Starting epoch {epoch + 1}/{TRAINER_CONFIG['num_epochs']}")
            for batch_idx, batch in enumerate(tqdm(self.trainer.dataloader, desc=f"Epoch {epoch + 1} Training")):
                query_tensors = batch["input_ids"].to(self.device)
                query_tensor_list = [tensor for tensor in query_tensors]

                response_tensors = self.trainer.generate(
                    query_tensor_list,
                    return_prompt=False,
                    **generation_kwargs
                )

                decoded_queries = self.tokenizer.batch_decode(query_tensors, clean_up_tokenization_spaces=True)
                decoded_responses = self.tokenizer.batch_decode(response_tensors, clean_up_tokenization_spaces=True)

                combined_rewards = self.compute_rewards(decoded_queries, decoded_responses)
                rewards = combined_rewards
                avg_reward = sum([r.item() for r in rewards]) / len(rewards)
                total_rewards.extend([r.item() for r in rewards])

                stats = self.trainer.step(query_tensor_list, response_tensors, rewards)

                loss = stats.get("ppo/loss/total", 0.0)
                total_losses.append(loss)

                # Increment the total sample count
                batch_size = len(query_tensors)
                self.total_samples += batch_size

                # Collect metrics
                self.rewards_over_samples.append((self.total_samples, avg_reward))
                self.losses_over_samples.append((self.total_samples, loss))

                # Increment the total step count
                self.step += 1

                if self.verbose:
                    logger.info(f"Batch {batch_idx + 1}:")
                    logger.info(f"  Average reward for the batch: {avg_reward:.4f}")
                    logger.info(f"  Loss: {loss:.4f}")

        # Plot metrics
        self.plot_metrics()

        # Save metrics to a JSON file for combined plotting
        metrics_data = {
            "rounds": list(range(1, len(self.rewards_over_samples) + 1)),
            "avg_rewards": [r for _, r in self.rewards_over_samples],
            "avg_losses": [l for _, l in self.losses_over_samples],
            "total_steps": [self.step] * len(self.rewards_over_samples),
            "total_samples": [s for s, _ in self.rewards_over_samples],
        }
        os.makedirs("metrics", exist_ok=True)
        with open("metrics/metrics_centralized.json", "w") as f:
            json.dump(metrics_data, f)
        logger.info("Centralized training metrics saved.")

    def plot_metrics(self):
        if not self.rewards_over_samples or not self.losses_over_samples:
            logger.info("No metrics to plot for centralized training.")
            return

        samples_rewards, avg_rewards = zip(*self.rewards_over_samples)
        samples_losses, losses = zip(*self.losses_over_samples)

        plt.figure(figsize=(12, 6))

        # Rewards subplot
        plt.subplot(2, 1, 1)
        plt.plot(samples_rewards, avg_rewards, marker='o', label='Average Reward')
        plt.title("Centralized Training - Average Reward over Samples")
        plt.xlabel("Total Samples")
        plt.ylabel("Average Reward")
        plt.legend()

        # Losses subplot
        plt.subplot(2, 1, 2)
        plt.plot(samples_losses, losses, marker='o', label='Loss')
        plt.title("Centralized Training - Loss over Samples")
        plt.xlabel("Total Samples")
        plt.ylabel("Loss")
        plt.legend()

        plt.tight_layout()
        os.makedirs("training_logs", exist_ok=True)
        plt.savefig("training_logs/ppo_training_centralized.png")
        plt.close()

        logger.info("Centralized training metrics plotted and saved.")

if __name__ == "__main__":
    trainer = CentralizedRLHFTrainer()
    trainer.train()