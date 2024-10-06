import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm
import wandb
import os
import numpy as np
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


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def train_simclr(rank, world_size, config):
    setup(rank, world_size)

    if rank == 0:
        wandb.init(project=config['wandb']['project'], entity=config['wandb']['entity'], config=config,
                   name="SimCLR_Pretraining")

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    backbone = EfficientNetWithAttention(config['model']['efficientnet_version'])
    model = SimCLR(backbone).to(device)
    model = DDP(model, device_ids=[rank])

    optimizer = optim.Adam(model.parameters(), lr=config['training']['lr'],
                           weight_decay=config['training']['weight_decay'])
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    scaler = GradScaler()

    train_transform = get_transform(is_train=True)
    train_loader = get_dataloader(config['data']['train_dir'], config['data']['train_labels'],
                                  config['training']['batch_size'], config['training']['num_workers'],
                                  train_transform, is_train=True, distributed=True)
    val_loader = get_dataloader(config['data']['val_dir'], config['data']['val_labels'],
                                config['training']['batch_size'], config['training']['num_workers'],
                                train_transform, is_train=False, distributed=True)

    early_stopping = EarlyStopping(patience=10, verbose=True, path=f'simclr_checkpoint_rank{rank}.pt')

    for epoch in range(config['training']['epochs']['pretraining']):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader,
                          desc=f"SimCLR Pretraining Epoch {epoch + 1}/{config['training']['epochs']['pretraining']}",
                          disable=rank != 0):
            images, _ = batch
            images = torch.cat([images, images], dim=0).to(device)

            with autocast():
                features = model(images)
                loss = simclr_loss(features, temperature=config['training']['simclr_temperature'])

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        val_loss = validate_simclr(model, val_loader, device, config)

        scheduler.step()

        if rank == 0:
            wandb.log({
                "simclr_train_loss": avg_train_loss,
                "simclr_val_loss": val_loss,
                "learning_rate": optimizer.param_groups[0]['lr'],
                "epoch": epoch
            })

            if (epoch + 1) % 10 == 0:
                torch.save(model.module.state_dict(), f"simclr_checkpoint_epoch{epoch + 1}.pt")

        early_stopping(val_loss, model.module, optimizer)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    if rank == 0:
        model.module.load_state_dict(torch.load('simclr_checkpoint_rank0.pt'))
        torch.save(model.module.backbone.state_dict(), "simclr_pretrained.pth")
        wandb.save("simclr_pretrained.pth")

        analysis = early_stopping.analyze_progress()
        wandb.log(analysis)
        wandb.log({"training_progress_plot": wandb.Image('training_progress.png')})

        wandb.finish()

    cleanup()


def validate_simclr(model, val_loader, device, config):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            images, _ = batch
            images = torch.cat([images, images], dim=0).to(device)
            with autocast():
                features = model(images)
                loss = simclr_loss(features, temperature=config['training']['simclr_temperature'])
            total_loss += loss.item()
    return total_loss / len(val_loader)


def train_classifiers(rank, world_size, config, use_correlation=True):
    setup(rank, world_size)

    if rank == 0:
        wandb.init(project=config['wandb']['project'], entity=config['wandb']['entity'], config=config,
                   name=f"Classifiers_{'With' if use_correlation else 'Without'}_Correlation")

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    backbone = EfficientNetWithAttention(config['model']['efficientnet_version'])
    pretrained_path = "simclr_pretrained.pth"
    if os.path.exists(pretrained_path):
        backbone.load_state_dict(torch.load(pretrained_path, map_location=device))
        print(f"Loaded pretrained SimCLR model from {pretrained_path}")
    else:
        print(f"Pretrained SimCLR model not found at {pretrained_path}. Initializing from scratch.")

    backbone = DDP(backbone.to(device), device_ids=[rank])
    classifiers = DDP(
        EnsembleBinaryClassifiers(backbone.module.num_features, config['model']['num_classes']).to(device),
        device_ids=[rank])

    if use_correlation:
        correlation_module = DDP(CorrelationLearningModule(config['model']['num_classes']).to(device),
                                 device_ids=[rank])
        meta_learner = DDP(MetaLearner(config['model']['num_classes'], hidden_dim=64,
                                       num_classes=config['model']['num_classes']).to(device), device_ids=[rank])
    else:
        correlation_module = None
        meta_learner = None

    params = list(backbone.parameters()) + list(classifiers.parameters())
    if use_correlation:
        params += list(correlation_module.parameters()) + list(meta_learner.parameters())

    optimizer = optim.Adam(params, lr=config['training']['lr'], weight_decay=config['training']['weight_decay'])
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    scaler = GradScaler()

    criterion = FWCELoss(class_frequencies=get_class_frequencies(config['data']['train_labels']))

    train_transform = get_transform(is_train=True)

    train_loader = get_dataloader(config['data']['train_dir'], config['data']['train_labels'],
                                  config['training']['batch_size'], config['training']['num_workers'],
                                  train_transform, is_train=True, distributed=True)

    val_transform = get_transform(is_train=False)

    val_loader = get_dataloader(config['data']['val_dir'], config['data']['val_labels'],
                                config['training']['batch_size'], config['training']['num_workers'],
                                val_transform, is_train=False, shuffle=False, distributed=True)

    early_stopping = EarlyStopping(patience=10, verbose=True,
                                   path=f'classifiers_{"with" if use_correlation else "without"}_correlation_checkpoint_rank{rank}.pt')

    for epoch in range(config['training']['epochs']['finetuning']):
        train_loss = train_epoch(backbone, classifiers, correlation_module, meta_learner, criterion, optimizer, scaler,
                                 train_loader, device, use_correlation, rank)
        val_loss, val_metrics = evaluate(backbone, classifiers, correlation_module, meta_learner, criterion, val_loader,
                                         device, use_correlation)

        scheduler.step()

        if rank == 0:
            wandb.log({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_auc_roc": np.mean(val_metrics['per_disease']['auc_roc']),
                "val_mean_ap": np.mean(val_metrics['per_disease']['ap']),
                "learning_rate": optimizer.param_groups[0]['lr'],
                "epoch": epoch
            })

            if (epoch + 1) % 10 == 0:
                torch.save({
                    'backbone': backbone.module.state_dict(),
                    'classifiers': classifiers.module.state_dict(),
                    'correlation_module': correlation_module.module.state_dict() if use_correlation else None,
                    'meta_learner': meta_learner.module.state_dict() if use_correlation else None
                }, f"classifiers_checkpoint_epoch{epoch + 1}.pt")

        early_stopping(val_loss, classifiers.module, optimizer)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    if rank == 0:
        classifiers.module.load_state_dict(
            torch.load(f'classifiers_{"with" if use_correlation else "without"}_correlation_checkpoint_rank0.pt'))

        torch.save({
            'backbone': backbone.module.state_dict(),
            'classifiers': classifiers.module.state_dict(),
            'correlation_module': correlation_module.module.state_dict() if use_correlation else None,
            'meta_learner': meta_learner.module.state_dict() if use_correlation else None
        }, f"best_model_{'with' if use_correlation else 'without'}_correlation.pth")

        wandb.save(f"best_model_{'with' if use_correlation else 'without'}_correlation.pth")

        analysis = early_stopping.analyze_progress()
        wandb.log(analysis)
        wandb.log({"training_progress_plot": wandb.Image('training_progress.png')})

        wandb.finish()

    cleanup()


def train_epoch(backbone, classifiers, correlation_module, meta_learner, criterion, optimizer, scaler, train_loader,
                device, use_correlation, rank):
    backbone.train()
    classifiers.train()
    if use_correlation:
        correlation_module.train()
        meta_learner.train()

    total_loss = 0
    for batch in tqdm(train_loader, disable=rank != 0):
        images, labels = batch
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        with autocast():
            features = backbone(images)
            initial_predictions = classifiers(features)

            if use_correlation:
                correlation_adjusted = correlation_module(initial_predictions)
                final_predictions = meta_learner(correlation_adjusted)
            else:
                final_predictions = initial_predictions

            loss = criterion(final_predictions, labels, list(backbone.parameters()) + list(classifiers.parameters()))

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(backbone.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(classifiers.parameters(), max_norm=1.0)
        if use_correlation:
            torch.nn.utils.clip_grad_norm_(correlation_module.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(meta_learner.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    return total_loss / len(train_loader)


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

            with autocast():
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


if __name__ == "__main__":
    import yaml
    from torch.multiprocessing import spawn

    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    world_size = torch.cuda.device_count()
    spawn(train_simclr, args=(world_size, config), nprocs=world_size, join=True)
    spawn(train_classifiers, args=(world_size, config, True), nprocs=world_size, join=True)

