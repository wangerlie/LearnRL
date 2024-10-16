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

# Filter Episodes
episodes_generator = dataset.iterate_episodes(episode_indices=[1, 2, 0])
for episode in episodes_generator:
    print(f"EPISODE ID {episode.id}")