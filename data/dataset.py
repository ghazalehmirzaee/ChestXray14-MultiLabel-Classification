import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import random


class AdaptiveClassBalanceAugmentation:
    def __init__(self, alpha=0.5, AF_max=3.0):
        self.alpha = alpha
        self.AF_max = AF_max
        self.class_counts = None
        self.co_occurrence_matrix = None

    def fit(self, labels):
        self.class_counts = labels.sum(axis=0)
        self.co_occurrence_matrix = labels.T @ labels

    def get_augmentation_factor(self, label):
        N = len(label)
        AF = np.minimum(1 + self.alpha * (np.max(self.class_counts) / (self.class_counts + 1e-5) - 1), self.AF_max)
        CA = np.sum(self.co_occurrence_matrix[label == 1] / (self.class_counts + 1e-5))
        AM = max(1, np.max(AF[label == 1] * (1 + CA)))
        return AM

    def __call__(self, image, label):
        AM = self.get_augmentation_factor(label)
        if random.random() < AM - 1:
            # Apply augmentation
            image = self.apply_augmentation(image)
        return image

    def apply_augmentation(self, image):
        # Implement various augmentation techniques here
        augmentations = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        ]
        aug = transforms.Compose(random.sample(augmentations, k=random.randint(1, len(augmentations))))
        return aug(image)


class ChestXrayDataset(Dataset):
    def __init__(self, image_dir, label_file, transform=None, use_acba=True):
        self.image_dir = image_dir
        self.labels = pd.read_csv(label_file, sep=' ', header=None)
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.use_acba = use_acba
        if use_acba:
            self.acba = AdaptiveClassBalanceAugmentation()
            self.acba.fit(self.labels.iloc[:, 1:].values)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_name = self.labels.iloc[idx, 0]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        labels = torch.tensor(self.labels.iloc[idx, 1:].values.astype(float))

        if self.use_acba:
            image = self.acba(image, labels.numpy())

        image = self.transform(image)
        return image, labels


def get_data_loaders(data_dir, batch_size, num_workers, use_acba=True):
    train_dataset = ChestXrayDataset(
        os.path.join(data_dir, 'categorized_images', 'train'),
        os.path.join(data_dir, 'labels', 'train_list.txt'),
        use_acba=use_acba
    )
    val_dataset = ChestXrayDataset(
        os.path.join(data_dir, 'categorized_images', 'val'),
        os.path.join(data_dir, 'labels', 'val_list.txt'),
        use_acba=False
    )
    test_dataset = ChestXrayDataset(
        os.path.join(data_dir, 'categorized_images', 'test'),
        os.path.join(data_dir, 'labels', 'test_list.txt'),
        use_acba=False
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader

