import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

class ChestXrayDataset(Dataset):
    def __init__(self, image_dir, label_file, transform=None):
        self.image_dir = image_dir
        self.labels = pd.read_csv(label_file, sep=' ', header=None)
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_name = self.labels.iloc[idx, 0]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)

        labels = torch.tensor(self.labels.iloc[idx, 1:].values.astype(float))
        return image, labels

def get_data_loaders(data_dir, batch_size, num_workers):
    train_dataset = ChestXrayDataset(
        os.path.join(data_dir, 'categorized_images', 'train'),
        os.path.join(data_dir, 'labels', 'train_list.txt')
    )
    val_dataset = ChestXrayDataset(
        os.path.join(data_dir, 'categorized_images', 'val'),
        os.path.join(data_dir, 'labels', 'val_list.txt')
    )
    test_dataset = ChestXrayDataset(
        os.path.join(data_dir, 'categorized_images', 'test'),
        os.path.join(data_dir, 'labels', 'test_list.txt')
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader

