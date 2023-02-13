# Training model using CartPole.py and then test here.

import numpy as np
import torch
import gym
import math
import random
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

model = torch.load("target_net.pt")
model.eval()

env = gym.make("CartPole-v1", render_mode="human")

observation, info = env.reset(seed=42)



for _ in range(1000):
    prediction = model(torch.tensor(observation))
    action = np.argmax(prediction.detach().cpu().numpy())
    print(observation, action)
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        print("reset")
        observation, info = env.reset()

env.close()
