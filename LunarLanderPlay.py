import gym
from gym.utils.play import play
from gym.utils.play import PlayPlot
import pygame

# pip install box2d
env = gym.make("LunarLander-v2", render_mode="rgb_array")
env.action_space.seed(42)


def callback(obs_t, obs_tp1, action, rew, terminated, truncated, info):
    return [rew,]


plotter = PlayPlot(callback, 30 * 5, ["reward"])
mapping = {(pygame.K_LEFT,): 3, (pygame.K_RIGHT,): 1, (pygame.K_UP,): 2, (pygame.K_DOWN,): 0}

play(env, callback=plotter.callback, keys_to_action=mapping)

observation, info = env.reset(seed=42)

for _ in range(100):
    observation, reward, terminated, truncated, info = env.step(env.action_space.sample())

    if terminated or truncated:
        observation, info = env.reset()

env.close()
