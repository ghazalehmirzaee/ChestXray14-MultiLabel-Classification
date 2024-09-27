import os
import argparse
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
from tqdm import tqdm

from data.dataset import get_data_loaders
from models.efficient_net import EfficientNetModel
from utils.losses import FocalWeightedCrossEntropyLoss
from utils.metrics import calculate_metrics

def train(args):
    # Initialize wandb
    wandb.init(project="chest-xray-classification", config=args)

    # Set up data loaders
    train_loader, val_loader, _ = get_data_loaders(
        args.data_dir, args.batch_size, args.num_workers
    )

    # Set up model, loss function, and optimizer
    model = EfficientNetModel(num_classes=14).to(args.device)
    criterion = FocalWeightedCrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_loss = float('inf')
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        train_preds, train_targets = [], []

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}"):
            images, labels = images.to(args.device), labels.to(args.device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_preds.extend(outputs.detach().cpu().numpy())
            train_targets.extend(labels.cpu().numpy())

        train_loss /= len(train_loader)
        train_metrics = calculate_metrics(train_preds, train_targets)

        # Validation
        model.eval()
        val_loss = 0.0
        val_preds, val_targets = [], []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(args.device), labels.to(args.device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                val_preds.extend(outputs.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())

        val_loss /= len(val_loader)
        val_metrics = calculate_metrics(val_preds, val_targets)

        # Log metrics to wandb
        wandb.log({
            "train_loss": train_loss,
            "val_loss": val_loss,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()}
        })

        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"Train mAP: {train_metrics['mAP']:.4f}, Val mAP: {val_metrics['mAP']:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(wandb.run.dir, "best_model.pth"))

        scheduler.step()

    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Chest X-ray Classification Model")
    parser.add_argument("--data_dir", type=str, default="/NewRaidData/ghazal/data/ChestX-ray14", help="Path to dataset")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for training")

    args = parser.parse_args()
    train(args)

