import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm
import wandb
from models.efficientnet import EfficientNetWithAttention
from models.simclr import SimCLR, simclr_loss
from models.binary_classifiers import EnsembleBinaryClassifiers
from models.correlation_learning import CorrelationLearningModule
from models.meta_learner import MetaLearner
from data.dataset import get_dataloader
from data.augmentations import get_transform
from utils.loss import FWCELoss
from utils.metrics import calculate_metrics
import numpy as np



def train_simclr(config):
    wandb.init(project=config['wandb']['project'], entity=config['wandb']['entity'], config=config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize model
    backbone = EfficientNetWithAttention(config['model']['efficientnet_version'])
    model = SimCLR(backbone).to(device)

    # Initialize optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=config['training']['lr'],
                           weight_decay=config['training']['weight_decay'])
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    # Initialize data loaders
    train_transform = get_transform(is_train=True)
    train_loader = get_dataloader(config['data']['train_dir'], config['data']['train_labels'],
                                  config['training']['batch_size'], config['training']['num_workers'],
                                  train_transform, is_train=True)

    # Training loop
    for epoch in range(config['training']['epochs']['pretraining']):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config['training']['epochs']['pretraining']}"):
            images, _ = batch
            images = torch.cat([images, images], dim=0)  # Create two views
            images = images.to(device)

            features = model(images)
            loss = simclr_loss(features, temperature=config['training']['simclr_temperature'])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()

        avg_loss = total_loss / len(train_loader)
        wandb.log({"simclr_loss": avg_loss, "epoch": epoch})

    # Save the pre-trained model
    torch.save(model.backbone.state_dict(), "simclr_pretrained.pth")
    wandb.finish()


def train_classifiers(config):
    wandb.init(project=config['wandb']['project'], entity=config['wandb']['entity'], config=config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model
    backbone = EfficientNetWithAttention(config['model']['efficientnet_version'])
    backbone.load_state_dict(torch.load("simclr_pretrained.pth"))
    backbone = backbone.to(device)  # Move backbone to GPU

    classifiers = EnsembleBinaryClassifiers(backbone.num_features, config['model']['num_classes']).to(device)
    correlation_module = CorrelationLearningModule(config['model']['num_classes']).to(device)
    meta_learner = MetaLearner(config['model']['num_classes'], hidden_dim=64,
                               num_classes=config['model']['num_classes']).to(device)

    # Initialize optimizer and scheduler
    params = list(backbone.parameters()) + list(classifiers.parameters()) + list(
        correlation_module.parameters()) + list(meta_learner.parameters())
    optimizer = optim.Adam(params, lr=config['training']['lr'], weight_decay=config['training']['weight_decay'])
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    # Initialize loss function
    criterion = FWCELoss(class_frequencies=get_class_frequencies(config['data']['train_labels']))

    # Initialize data loaders
    train_transform = get_transform(is_train=True)
    train_loader = get_dataloader(config['data']['train_dir'], config['data']['train_labels'],
                                  config['training']['batch_size'], config['training']['num_workers'],
                                  train_transform, is_train=True)
    val_transform = get_transform(is_train=False)
    val_loader = get_dataloader(config['data']['val_dir'], config['data']['val_labels'],
                                config['training']['batch_size'], config['training']['num_workers'],
                                val_transform, is_train=False, shuffle=False)

    # Training loop
    best_val_loss = float('inf')
    for epoch in range(config['training']['epochs']['finetuning']):
        backbone.train()
        classifiers.train()
        correlation_module.train()
        meta_learner.train()

        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config['training']['epochs']['finetuning']}"):
            images, labels = batch
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            features = backbone(images)
            initial_predictions = classifiers(features)
            correlation_adjusted = correlation_module(initial_predictions)
            final_predictions = meta_learner(correlation_adjusted)

            loss = criterion(final_predictions, labels, params)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()

        # Validation
        val_loss, val_metrics = evaluate(backbone, classifiers, correlation_module, meta_learner, criterion, val_loader,
                                         device)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'backbone': backbone.state_dict(),
                'classifiers': classifiers.state_dict(),
                'correlation_module': correlation_module.state_dict(),
                'meta_learner': meta_learner.state_dict()
            }, "best_model.pth")

            # Log metrics
            wandb.log({
                "train_loss": total_loss / len(train_loader),
                "val_loss": val_loss,
                "val_auc_roc": np.mean(val_metrics['auc_roc']),
                "val_mean_ap": val_metrics['mean_ap'],
                "epoch": epoch
            })

        wandb.finish()

def evaluate(backbone, classifiers, correlation_module, meta_learner, criterion, dataloader, device):
    backbone.eval()
    classifiers.eval()
    correlation_module.eval()
    meta_learner.eval()

    total_loss = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            images, labels = batch
            images, labels = images.to(device), labels.to(device)

            features = backbone(images)
            initial_predictions = classifiers(features)
            correlation_adjusted = correlation_module(initial_predictions)
            final_predictions = meta_learner(correlation_adjusted)

            loss = criterion(final_predictions, labels, [])
            total_loss += loss.item()

            all_predictions.append(final_predictions.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_predictions = np.concatenate(all_predictions)
    all_labels = np.concatenate(all_labels)

    metrics = calculate_metrics(all_labels, (all_predictions > 0.5).astype(int), all_predictions)

    return total_loss / len(dataloader), metrics

def get_class_frequencies(label_file):
    import pandas as pd
    labels = pd.read_csv(label_file, sep=' ', header=None).iloc[:, 1:].values
    return {i: labels[:, i].sum() for i in range(labels.shape[1])}

if __name__ == "__main__":
    import yaml

    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    train_simclr(config)
    train_classifiers(config)
