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

class ForwardModel(nn.Module):
    """
    A simple forward dynamics model: predicts next state features from current state and action.
    For demonstration, we'll assume a simple MLP that takes state and action as input.
    """
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(ForwardModel, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, state_dim)  # Predict next state representation

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)
class CuriosityEnvWrapper(VecEnvWrapper):
    """
    VecEnv wrapper that adds curiosity-driven intrinsic rewards.
    """
    def __init__(self, venv, state_dim, action_dim, learning_rate=1e-3):
        super().__init__(venv)
        self.forward_model = ForwardModel(state_dim, action_dim)
        self.optimizer = optim.Adam(self.forward_model.parameters(), lr=learning_rate)

        # store previous states and actions to train the model
        self.last_obs = None

    def reset(self):
        obs = self.venv.reset()
        self.last_obs = obs
        return obs

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()

        # convert to tensors
        obs_tensor = torch.tensor(obs, dtype=torch.float32)
        last_obs_tensor = torch.tensor(self.last_obs, dtype=torch.float32)

        # override the step_async and store the actions.
        actions_one_hot = torch.zeros((len(self.last_actions), self.action_space.n))
        for i, a in enumerate(self.last_actions):
            actions_one_hot[i, a] = 1.0

        pred_next_state = self.forward_model(last_obs_tensor, actions_one_hot)
        intrinsic_reward = torch.mean((pred_next_state - obs_tensor)**2, dim=-1).detach().numpy()

        # combine intrinsic reward with extrinsic reward
        total_reward = rewards + intrinsic_reward

        self.optimizer.zero_grad()
        loss = torch.mean((pred_next_state - obs_tensor)**2)
        loss.backward()
        self.optimizer.step()

        # update last_obs
        self.last_obs = obs
        return obs, total_reward, dones, infos

    def step_async(self, actions):
        # store actions for later use
        self.last_actions = actions
        self.venv.step_async(actions)

from balto import Balto


############################
base_env = Balto(3, render_mode='human')

obs_dim = base_env.observation_space.shape[0]
act_dim = base_env.action_space.n
base_env.close()

# Create a vectorized environment
def make_env():
    return Balto(3, render_mode='human')

venv = DummyVecEnv([make_env])

# Wrap with curiosity:
curiosity_venv = CuriosityEnvWrapper(venv, state_dim=obs_dim, action_dim=act_dim)
model = PPO(MlpPolicy, curiosity_venv, verbose=0)

############################

# env = Balto(3, render_mode='human')
# model = PPO(MlpPolicy, env, verbose=0)

############################

class SkillWrapper(gym.Wrapper):
    """
    A skill wrapper that augments observations with a skill embedding.
    Each episode, a skill index is sampled and a one-hot skill vector is appended to the observation.
    """
    def __init__(self, env, skill_dim=4):
        super().__init__(env)
        self.skill_dim = skill_dim
        self.current_skill = None

        # orig_obs_space = self.env.observation_space

        # assume original obs space is a Box
        # low = np.concatenate([orig_obs_space.low, np.zeros(self.skill_dim)])
        # high = np.concatenate([orig_obs_space.high, np.ones(self.skill_dim)])
        # self.observation_space = gym.spaces.Box(low=low, high=high, dtype=orig_obs_space.dtype)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.current_skill = np.random.randint(self.skill_dim)
        obs = self._augment_obs(obs, self.current_skill)
        return obs, info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        obs = self._augment_obs(obs, self.current_skill)
        return obs, reward, done, truncated, info

    def _augment_obs(self, obs, skill_idx):
        skill_vec = np.zeros(self.skill_dim, dtype=obs.dtype)
        skill_vec[skill_idx] = 1.0
        return np.concatenate([obs, skill_vec])


def make_env_skill(env_id="CartPole-v1", skill_dim=4):
    def _init():
        env = Balto(3, render_mode='human')
        env = SkillWrapper(env, skill_dim=skill_dim)
        return env
    return _init

skill_dim = 4
venv = DummyVecEnv([make_env_skill(None, skill_dim) for _ in range(1)])

sample_env = Balto(3, render_mode='human')
# obs_dim = sample_env.observation_space.shape[0]
# combine dimensions
aug_obs_dim = 12 + skill_dim
act_dim = 12

# wrap with curiosity
curiosity_venv = CuriosityEnvWrapper(venv, state_dim=aug_obs_dim, action_dim=act_dim)


############################



# Use a separate environement for evaluation
from stable_baselines3.common.monitor import Monitor
eval_env = Monitor(Balto(3, render_mode='human'))

# Random Agent, before training
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=100)

print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

# Train the agent for 10000 steps
model.learn(total_timesteps=1000)

# Evaluate the trained agent
eval_env = Monitor(Balto(3, render_mode='human'))
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=100)

print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
