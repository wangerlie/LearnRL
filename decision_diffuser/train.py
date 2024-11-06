import os
import sys
import numpy as np
import torch
import minari
from torch.utils.data import DataLoader

from rl_dataloader.custom_dataset import SequenceDataset
from models import GaussianDiffusion
from models import TemporalUnet



def to_device(x, device):
    if torch.is_tensor(x):
        x = x.to(device)
        if x.dtype != torch.float32:
            x = x.float()
        return x
    elif type(x) is dict:
        return {k: to_device(v, device) for k, v in x.items()}
    else:
        print(f'Unrecognized type in `to_device`: {type(x)}')

def batch_to_device(batch, device='cuda:0'):
    vals = [
        to_device(getattr(batch, field), device)
        for field in batch._fields
    ]
    return type(batch)(*vals)

minari_datasets_root_path = '/home/wangerlie/drl/minari/datasets'
# TODO: add this to the config
minari_dataset_name = 'D4RL/antmaze/large-diverse-v1'
minari_dataset_path = os.path.join(minari_datasets_root_path, minari_dataset_name)
minari_dataset = minari.load_dataset(minari_dataset_path)
sequence_data = SequenceDataset(minari_dataset)
print(len(sequence_data))

minari_dataloader = DataLoader(sequence_data, batch_size=256, shuffle=True)

observation_dim = sequence_data.observation_dim
action_dim = sequence_data.action_dim
horizon=64
model = TemporalUnet(horizon=horizon,transition_dim=observation_dim+action_dim,cond_dim=observation_dim)
diffuser = GaussianDiffusion(model, horizon,observation_dim, action_dim)
n_train_steps = 1000
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
diffuser.to(device)

optimizer = torch.optim.Adam(diffuser.parameters(), lr=1e-3)

for step in range(n_train_steps):
    for i,batch in enumerate(minari_dataloader):
        batch = batch_to_device(batch,device)
        trajectories,condition = batch
        loss,info = diffuser.loss(trajectories,condition)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if i % 100 == 0:
            print(f'step: {step}, loss: {loss.item()}')
        





