import os
import sys
import numpy as np

import torch
from torch.utils.data import DataLoader

import minari

from custom_dataset import SequenceDataset
# Load the dataset

minari_datasets_root_path = '/home/wangerlie/drl/minari/datasets'
# TODO: add this to the config
minari_dataset_name = 'D4RL/antmaze/large-diverse-v1'
minari_dataset_path = os.path.join(minari_datasets_root_path, minari_dataset_name)
minari_dataset = minari.load_dataset(minari_dataset_path)
sequence_data = SequenceDataset(minari_dataset)

minari_dataloader = DataLoader(sequence_data, batch_size=32, shuffle=True)
for batch in minari_dataloader:
    print(batch.trajectories.shape)
    print(batch.trajectories)
    break

