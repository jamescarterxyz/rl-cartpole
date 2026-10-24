"""Evaluate a trained PPO agent on CartPole-v1 and save a demo GIF.

Usage:
    python evaluate.py [--episodes 10]

Outputs:
    results/eval.txt  - mean/std reward over the evaluation episodes
    results/demo.gif  - the trained agent playing CartPole
"""

import argparse
import os

import gymnasium as gym
import numpy as np
from PIL import Image
from stable_baselines3 import PPO


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate PPO on CartPole-v1")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Evaluation episodes (default: 10)")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs("results", exist_ok=True)

    model = PPO.load("models/ppo_cartpole")
    env = gym.make("CartPole-v1", render_mode="rgb_array")

    rewards, frames = [], []
    for ep in range(args.episodes):
        obs, _ = env.reset()
        done, total = False, 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total += reward
            if ep == 0:  # record only the first episode for the GIF
                frames.append(Image.fromarray(env.render()))
        rewards.append(total)
        print(f"Episode {ep + 1}: reward {total}")

    mean, std = float(np.mean(rewards)), float(np.std(rewards))
    with open("results/eval.txt", "w") as f:
        f.write(f"Episodes: {args.episodes}\nMean reward: {mean:.1f}\n"
                f"Std reward: {std:.1f}\n")
    print(f"Mean reward over {args.episodes} episodes: {mean:.1f} +/- {std:.1f}")

    if frames:
        frames[0].save("results/demo.gif", save_all=True,
                       append_images=frames[1:], duration=50, loop=0)
        print(f"Saved demo GIF ({len(frames)} frames) to results/demo.gif")
    env.close()


if __name__ == "__main__":
    main()
