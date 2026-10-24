"""Train a PPO agent on CartPole-v1 with Stable-Baselines3.

Usage:
    python train.py [--timesteps 50000] [--seed 42]

Outputs (created in this folder):
    models/ppo_cartpole.zip   - trained model
    results/rewards.png       - episode-reward learning curve
    results/training_log.csv  - per-episode rewards (from Monitor)
"""

import argparse
import csv
import os

import gymnasium as gym
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor


def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO on CartPole-v1")
    parser.add_argument("--timesteps", type=int, default=50_000,
                        help="Total training timesteps (default: 50000)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    return parser.parse_args()


def plot_rewards(log_path, plot_path):
    """Plot episode rewards from a Monitor CSV file."""
    episodes, rewards = [], []
    # Monitor CSV files start with '#' comment lines; skip them.
    with open(log_path) as f:
        reader = csv.DictReader(line for line in f
                                if not line.startswith("#"))
    for i, row in enumerate(reader, start=1):
        episodes.append(i)
        rewards.append(float(row["r"]))
    plt.figure(figsize=(8, 4))
    plt.plot(episodes, rewards, linewidth=0.8)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("CartPole-v1: PPO training rewards")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    print(f"Saved learning curve to {plot_path}")


def main():
    args = parse_args()
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    env = Monitor(gym.make("CartPole-v1"), "results/monitor")
    model = PPO("MlpPolicy", env, verbose=1, seed=args.seed)
    model.learn(total_timesteps=args.timesteps)
    model.save("models/ppo_cartpole")
    print("Saved trained model to models/ppo_cartpole.zip")

    env.close()
    plot_rewards("results/monitor.monitor.csv", "results/rewards.png")


if __name__ == "__main__":
    main()
