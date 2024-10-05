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
from utils.training_utils import EarlyStopping
import numpy as np
import os


def train_simclr(config):
    wandb.init(project=config['wandb']['project'], entity=config['wandb']['entity'], config=config,
               name="SimCLR_Pretraining")

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
    val_loader = get_dataloader(config['data']['val_dir'], config['data']['val_labels'],
                                config['training']['batch_size'], config['training']['num_workers'],
                                train_transform, is_train=False)

    # Initialize early stopping
    early_stopping = EarlyStopping(patience=10, verbose=True, path='simclr_checkpoint.pt')

    # Training loop
    for epoch in range(config['training']['epochs']['pretraining']):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader,
                          desc=f"SimCLR Pretraining Epoch {epoch + 1}/{config['training']['epochs']['pretraining']}"):
            images, _ = batch
            images = torch.cat([images, images], dim=0)  # Create two views
            images = images.to(device)

            features = model(images)
            loss = simclr_loss(features, temperature=config['training']['simclr_temperature'])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                images, _ = batch
                images = torch.cat([images, images], dim=0)
                images = images.to(device)
                features = model(images)
                loss = simclr_loss(features, temperature=config['training']['simclr_temperature'])
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        scheduler.step()

        wandb.log({
            "simclr_train_loss": avg_train_loss,
            "simclr_val_loss": avg_val_loss,
            "epoch": epoch
        })

        # Early stopping
        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    # Load the best model
    model.load_state_dict(torch.load('simclr_checkpoint.pt'))

    # Save the pre-trained model
    torch.save(model.backbone.state_dict(), "simclr_pretrained.pth")
    wandb.save("simclr_pretrained.pth")
    wandb.finish()


def train_classifiers(config, use_correlation=True):
    wandb.init(project=config['wandb']['project'], entity=config['wandb']['entity'], config=config,
               name=f"Classifiers_{'With' if use_correlation else 'Without'}_Correlation")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model
    backbone = EfficientNetWithAttention(config['model']['efficientnet_version'])

    # Load the pretrained SimCLR model
    pretrained_path = "simclr_pretrained.pth"
    if os.path.exists(pretrained_path):
        backbone.load_state_dict(torch.load(pretrained_path))
        print(f"Loaded pretrained SimCLR model from {pretrained_path}")
    else:
        print(f"Pretrained SimCLR model not found at {pretrained_path}. Initializing from scratch.")

    backbone = backbone.to(device)

    classifiers = EnsembleBinaryClassifiers(backbone.num_features, config['model']['num_classes']).to(device)

    if use_correlation:
        correlation_module = CorrelationLearningModule(config['model']['num_classes']).to(device)
        meta_learner = MetaLearner(config['model']['num_classes'], hidden_dim=64,
                                   num_classes=config['model']['num_classes']).to(device)
    else:
        correlation_module = None
        meta_learner = None

    # Initialize optimizer and scheduler
    params = list(backbone.parameters()) + list(classifiers.parameters())
    if use_correlation:
        params += list(correlation_module.parameters()) + list(meta_learner.parameters())

    lr = config['training']['lr']
    weight_decay = config['training']['weight_decay']
    optimizer = optim.Adam(params, lr=lr, weight_decay=weight_decay)
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

    # Initialize early stopping
    early_stopping = EarlyStopping(patience=10, verbose=True,
                                   path=f'classifiers_{"with" if use_correlation else "without"}_correlation_checkpoint.pt')

    # Training loop
    for epoch in range(config['training']['epochs']['finetuning']):
        backbone.train()
        classifiers.train()
        if use_correlation:
            correlation_module.train()
            meta_learner.train()

        total_loss = 0
        for batch in tqdm(train_loader,
                          desc=f"Classifiers {'With' if use_correlation else 'Without'} Correlation - Epoch {epoch + 1}/{config['training']['epochs']['finetuning']}"):
            images, labels = batch
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            features = backbone(images)
            initial_predictions = classifiers(features)

            if use_correlation:
                correlation_adjusted = correlation_module(initial_predictions)
                final_predictions = meta_learner(correlation_adjusted)
            else:
                final_predictions = initial_predictions

            loss = criterion(final_predictions, labels, params)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()

        # Validation
        val_loss, val_metrics = evaluate(backbone, classifiers, correlation_module, meta_learner, criterion, val_loader,
                                         device, use_correlation)

        # Log metrics
        wandb.log({
            "train_loss": total_loss / len(train_loader),
            "val_loss": val_loss,
            "val_auc_roc": np.mean(val_metrics['auc_roc']),
            "val_mean_ap": val_metrics['mean_ap'],
            "epoch": epoch
        })

        # Early stopping
        early_stopping(val_loss, classifiers)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    # Load the best model
    classifiers.load_state_dict(
        torch.load(f'classifiers_{"with" if use_correlation else "without"}_correlation_checkpoint.pt'))

    # Save the final model
    torch.save({
        'backbone': backbone.state_dict(),
        'classifiers': classifiers.state_dict(),
        'correlation_module': correlation_module.state_dict() if use_correlation else None,
        'meta_learner': meta_learner.state_dict() if use_correlation else None
    }, f"best_model_{'with' if use_correlation else 'without'}_correlation.pth")

    wandb.save(f"best_model_{'with' if use_correlation else 'without'}_correlation.pth")
    wandb.finish()


def evaluate(backbone, classifiers, correlation_module, meta_learner, criterion, dataloader, device, use_correlation):
    backbone.eval()
    classifiers.eval()
    if use_correlation:
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

            if use_correlation:
                correlation_adjusted = correlation_module(initial_predictions)
                final_predictions = meta_learner(correlation_adjusted)
            else:
                final_predictions = initial_predictions

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
