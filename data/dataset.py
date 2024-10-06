import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from data.augmentations import ACBA, get_transform

"""
### For loading augmented dataset when we saved augmented images

class ChestXrayDataset(Dataset):
    def __init__(self, data_dir, label_file, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.labels = pd.read_csv(label_file, sep=' ', header=None)
        self.image_files = self.labels.iloc[:, 0].values
        self.targets = self.labels.iloc[:, 1:].values.astype(np.float32)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        target = torch.tensor(self.targets[idx])

        if self.transform:
            image = self.transform(image)

        return image, target
"""



class ChestXrayDataset(Dataset):
    def __init__(self, data_dir, label_file, transform=None, acba=None):
        self.data_dir = data_dir
        self.transform = transform
        self.acba = acba
        self.labels = pd.read_csv(label_file, sep=' ', header=None)
        self.image_files = self.labels.iloc[:, 0].values
        self.targets = self.labels.iloc[:, 1:].values.astype(np.float32)

        if self.acba:
            self.acba.initialize(self.targets)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        target = torch.tensor(self.targets[idx])

        if self.acba:
            image = self.acba.apply_augmentations(image, self.targets[idx])

        if self.transform:
            image = self.transform(image)

        return image, target


def get_dataloader(data_dir, label_file, batch_size, num_workers, transform, is_train=True, shuffle=True,
                   distributed=False):
    acba = ACBA() if is_train else None
    dataset = ChestXrayDataset(data_dir, label_file, transform, acba)

    if distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=shuffle)
        shuffle = False  # DistributedSampler handles shuffling
    else:
        sampler = None

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True,
                      sampler=sampler)

