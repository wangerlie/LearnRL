import numpy as np
import gymnasium as gym
from actor_critic import ActorCritic

# Initialise the environment
env = gym.make('Ant-v4')

# Reset the environment to generate the first observation
observation, info = env.reset(seed=42)

# Set up a rendering flag
render = True

for _ in range(1000):


    # this is where you would insert your policy
    action = env.action_space.sample()

    # step (transition) through the environment with the action
    # receiving the next observation, reward and if the episode has terminated or truncated
    observation, reward, terminated, truncated, info = env.step(action)
    # If the episode has ended then we can reset to start a new episode
    if terminated or truncated:
        observation, info = env.reset(seed=42)  # Reset with the same seed for consistency

# Close the environment after use
env.close()
print(observation.shape)
print(env.action_space.shape)
