import minari
import numpy as np

def atleast_2d(x):
    while x.ndim < 2:
        x = np.expand_dims(x, axis=-1)
    return x

class WrapMinariDataset:

    def __init__(self, minari_dataset):
        self._dict = {
            'observations': [],
            'achieved_goal': [],
            'desired_goal': [],
            'actions': [],
            'rewards': [],
            'terminations': [],
            'truncations': [],
            'infos': [],
            'path_lengths': [],

        }
        self.minari_dataset = minari_dataset
        self._dict['path_lengths'] = np.zeros(self.minari_dataset.total_episodes, dtype=np.int32)
        self._get_episode_data()

    def _get_episode_data(self):
        for i,episode in enumerate(self.minari_dataset.iterate_episodes()):
            self._dict['observations'].append(atleast_2d(episode.observations['observation']))
            self._dict['achieved_goal'].append(atleast_2d(episode.observations['achieved_goal']))
            self._dict['desired_goal'].append(atleast_2d(episode.observations['desired_goal']))
            self._dict['actions'].append(atleast_2d(episode.actions))
            self._dict['rewards'].append(atleast_2d(episode.rewards))
            self._dict['terminations'].append(atleast_2d(episode.terminations))
            self._dict['truncations'].append(atleast_2d(episode.truncations))
            ## record path length
            self._dict['path_lengths'][i] = len(self._dict['observations'][i])-1
            self._dict['infos']=episode.infos
        # Convert lists to numpy arrays with the desired shape
        for key in ['observations', 'achieved_goal', 'desired_goal', 'actions', 'rewards', 'terminations', 'truncations']:
            self._dict[key] = np.array(self._dict[key])


    @property
    def total_episodes(self):
        return self.minari_dataset.total_episodes
    
    @property
    def total_steps(self):
        return self.minari_dataset.total_steps
    
    def __getitem__(self, key):
        return self._dict[key]

    def __setitem__(self, key, val):
        self._dict[key] = val
        self._add_attributes()
    
    def _add_attributes(self):
        '''
            can access fields with `buffer.observations`
            instead of `buffer['observations']`
        '''
        for key, val in self._dict.items():
            setattr(self, key, val)
    
    def items(self):
        return {k: v for k, v in self._dict.items()
                if k != 'path_lengths' and k!='infos'}.items()

    def penalize_early_termination(self):
        pass

    def truncate_path(self, path_index, step):
        old = self._dict['path_lengths'][path_index]
        new = min(step, old)
        self._dict['path_lengths'][path_index] = new


if __name__ == '__main__':
    import os
    import sys
    import torch
    import minari

    # Load the dataset

    minari_datasets_root_path = '/home/wangerlie/drl/minari/datasets'
    # TODO: add this to the config
    minari_dataset_name = 'D4RL/antmaze/large-diverse-v1'
    minari_dataset_path = os.path.join(minari_datasets_root_path, minari_dataset_name)
    minari_dataset = minari.load_dataset(minari_dataset_path)
    wrapped_dataset = WrapMinariDataset(minari_dataset)
    print(wrapped_dataset.observations[0].shape)
    