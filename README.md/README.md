# RL CartPole 🤖

Train a reinforcement learning agent to balance a pole on a cart, using
**PPO** from [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)
and the classic **CartPole-v1** environment from
[Gymnasium](https://github.com/Farama-Foundation/Gymnasium).

## What is this?

Reinforcement learning in one sentence: an agent learns by trial and error —
it gets a reward for every step the pole stays up, and learns a policy that
keeps it balanced as long as possible. CartPole is the "Hello, World!" of RL:
simple to understand, fast to train, and a perfect first RL project.

## Quick start

```bash
pip install -r requirements.txt
python train.py        # train PPO for 50,000 timesteps
python evaluate.py     # test the trained agent, save a demo GIF
```

## What you get

| File | Description |
|---|---|
| `train.py` | Trains PPO on CartPole-v1, saves the model and a learning curve |
| `evaluate.py` | Runs the trained agent, prints mean reward, saves a demo GIF |
| `models/ppo_cartpole.zip` | Trained model (created by `train.py`) |
| `results/rewards.png` | Episode-reward learning curve |
| `results/demo.gif` | The trained agent playing CartPole |
| `results/eval.txt` | Evaluation summary (mean/std reward) |

## Results

My training run (PPO, 50,000 timesteps, seed 42) solved CartPole completely:

- **Mean reward: 500.0 / 500 over 10 evaluation episodes** (perfect score,
  zero variance — the pole never dropped)
- CartPole-v1 is considered "solved" at an average of 475+ over 100 episodes,
  so this agent is well past the bar.

Running `train.py` and `evaluate.py` yourself regenerates the learning curve
(`results/rewards.png`), the demo GIF (`results/demo.gif`) and the summary
(`results/eval.txt`).

## Try it yourself

```bash
# Longer training (better agent, takes a bit longer)
python train.py --timesteps 100000

# Evaluate over more episodes
python evaluate.py --episodes 20
```

## Built with

- Python, Gymnasium, Stable-Baselines3 (PPO), Matplotlib, Pillow
