import stable_baselines3
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
import torch
import torch.nn as nn
import torch.optim as optim
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecEnvWrapper
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy

from balto import Balto
env = Balto(3, render_mode='human')
model = PPO(MlpPolicy, env, verbose=1)

# Use a separate environement for evaluation
from stable_baselines3.common.monitor import Monitor
eval_env = Monitor(Balto(3, render_mode='human'))

# Random Agent, before training
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=100)

print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

# Train the agent for 10000 steps
model.learn(total_timesteps=10)

# Evaluate the trained agent
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=100)

print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
