import gymnasium as gym
import random
import numpy as np
import collections
from tqdm import tqdm
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import rl_utils
from DQN import ReplayBuffer, DQN
import os, sys


if __name__ == "__main__":
    lr = 2e-3
    num_episodes = 500
    hidden_dim = 128
    gamma = 0.98
    epsilon = 0.01
    target_update = 10
    buffer_size = 10000
    minimal_size = 500
    batch_size = 64
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    env_name = "CartPole-v1"
    env = gym.make(env_name, render_mode = "human")
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    replay_buffer = ReplayBuffer(buffer_size)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon, target_update, device)
    
    agent.load(6)
    
    the_return = 0
    
    observation, info = env.reset(seed=42)
    for _ in range(1000):
        action = agent.take_action(observation)
        # action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        
        the_return += reward

        if terminated or truncated:
            env.close()
            break
    
    print("The return is:", the_return)
