import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
from torchvision import transforms

IMG_SIZE = 64

def load_transformed_dataset():
    data_transforms = [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),  # Scales data into [0,1]
        transforms.Lambda(lambda t: (t * 2) - 1),  # Scale between [-1, 1]
    ]
    data_transform = transforms.Compose(data_transforms)

    train = torchvision.datasets.CIFAR10(
        root="/home/wangerlie/Datasets", download=True,train=True,transform=data_transform
    )

    test = torchvision.datasets.CIFAR10(
        root="/home/wangerlie/Datasets", download=True, train=False,transform=data_transform
    )
    return torch.utils.data.ConcatDataset([train, test])

