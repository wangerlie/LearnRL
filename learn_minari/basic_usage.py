# Basic Usage of Minari API
import minari

# Load Local Datasets
dataset = minari.load_dataset("D4RL/door/human-v2")
print("Observation space:", dataset.observation_space)
print("Action space:", dataset.action_space)
print("Total episodes:", dataset.total_episodes)
print("Total steps:", dataset.total_steps)

# Sampling Episodes
dataset.set_seed(seed=123)
for i in range(5):
    # sample 5 episodes from the dataset
    episodes = dataset.sample_episodes(n_episodes=5)
    # get id's from the sampled episodes
    ids = list(map(lambda ep: ep.id, episodes))
    print(f"EPISODE ID'S SAMPLE {i}: {ids}")

episodes_generator = dataset.iterate_episodes(episode_indices=[1, 2, 0])
for episode in episodes_generator:
    print(f"EPISODE ID {episode.id}")

# Filter Episodes
# The condition must be callable
# get episodes with mean reward greater than 2

filter_dataset = dataset.filter_episodes(lambda episode: episode.rewards.mean() > 2)
print(f"TOTAL EPISODES FILTER DATASET: {filter_dataset.total_episodes}")

# Split Dataset
split_datasets = minari.split_dataset(dataset, sizes=[20, 5], seed=123)
print(f"TOTAL EPISODES FIRST SPLIT: {split_datasets[0].total_episodes}")
print(f"TOTAL EPISODES SECOND SPLIT: {split_datasets[1].total_episodes}")

# Recover Environment
env = dataset.recover_environment()
env.reset()
for _ in range(5):
    obs, rew, terminated, truncated, info = env.step(env.action_space.sample())
    if terminated or truncated:
        env.reset()

# Collecting Data
from minari import DataCollector
import gymnasium as gym

env = gym.make("CartPole-v1")
env = DataCollector(env, record_infos=True)
dataset = None
total_episodes = 100
for episode_id  in range(total_episodes):
    env.reset(seed=123)
    while True:
        # random action policy
        action = env.action_space.sample()
        obs, rew, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            break
    if (episode_id + 1) % 10 == 0:
        # Update local Minari dataset every 10 episodes.
        # This works as a checkpoint to not lose the already collected data
        if dataset is None:
            dataset = env.create_dataset(
                dataset_id='cartpole/test-v1',
                algorithm_name="Random-Policy",
                code_permalink="https://github.com/Farama-Foundation/Minari",
                author="WEL"
            )
        else:
            env.add_to_dataset(dataset)
