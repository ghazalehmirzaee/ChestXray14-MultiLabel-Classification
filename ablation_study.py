import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import wandb
import yaml
from models.efficientnet import EfficientNetWithAttention
from models.binary_classifiers import EnsembleBinaryClassifiers
from models.correlation_learning import CorrelationLearningModule
from models.meta_learner import MetaLearner
from data.dataset import get_dataloader
from data.augmentations import get_transform
from utils.metrics import calculate_metrics
from train import train_classifiers
from torch.multiprocessing import spawn


def run_ablation_study(config):
    world_size = torch.cuda.device_count()

    # Model A: Without correlation learning
    print("Training Model A (Without Correlation Learning)")
    wandb.init(project=config['wandb']['project'], entity=config['wandb']['entity'], name="Model_Without_Correlation", config=config)
    spawn(train_classifiers, args=(world_size, config, False), nprocs=world_size, join=True)
    metrics_a = evaluate_model(config, "best_model_without_correlation.pth", use_correlation=False)
    log_metrics(metrics_a, "Without_Correlation", config['model']['disease_names'])
    wandb.finish()

    # Model B: With correlation learning
    print("Training Model B (With Correlation Learning)")
    wandb.init(project=config['wandb']['project'], entity=config['wandb']['entity'], name="Model_With_Correlation", config=config)
    spawn(train_classifiers, args=(world_size, config, True), nprocs=world_size, join=True)
    metrics_b = evaluate_model(config, "best_model_with_correlation.pth", use_correlation=True)
    log_metrics(metrics_b, "With_Correlation", config['model']['disease_names'])
    wandb.finish()

    # Compare results
    compare_metrics(metrics_a, metrics_b, config['model']['disease_names'])


def evaluate_model(config, model_path, use_correlation):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    backbone = EfficientNetWithAttention(config['model']['efficientnet_version'])
    classifiers = EnsembleBinaryClassifiers(backbone.num_features, config['model']['num_classes'])

    if use_correlation:
        correlation_module = CorrelationLearningModule(config['model']['num_classes'])
        meta_learner = MetaLearner(config['model']['num_classes'], hidden_dim=64,
                                   num_classes=config['model']['num_classes'])

    checkpoint = torch.load(model_path)
    backbone.load_state_dict(checkpoint['backbone'])
    classifiers.load_state_dict(checkpoint['classifiers'])

    if use_correlation:
        correlation_module.load_state_dict(checkpoint['correlation_module'])
        meta_learner.load_state_dict(checkpoint['meta_learner'])

    backbone.to(device)
    classifiers.to(device)

    if use_correlation:
        correlation_module.to(device)
        meta_learner.to(device)

    # Prepare data
    test_transform = get_transform(is_train=False)
    test_loader = get_dataloader(config['data']['test_dir'], config['data']['test_labels'],
                                 config['training']['batch_size'], config['training']['num_workers'], test_transform,
                                 shuffle=False)

    # Evaluation
    backbone.eval()
    classifiers.eval()

    if use_correlation:
        correlation_module.eval()
        meta_learner.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            images, labels = batch
            images, labels = images.to(device), labels.to(device)

            features = backbone(images)
            initial_predictions = classifiers(features)

            if use_correlation:
                correlation_adjusted = correlation_module(initial_predictions)
                final_predictions = meta_learner(correlation_adjusted)
            else:
                final_predictions = initial_predictions

            all_predictions.append(final_predictions.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_predictions = np.concatenate(all_predictions)
    all_labels = np.concatenate(all_labels)

    return calculate_metrics(all_labels, (all_predictions > 0.5).astype(int), all_predictions)


def log_metrics(metrics, model_name, disease_names):
    if wandb.run is None:
        print("WandB is not initialized. Skipping logging.")
        return

    # Log overall metrics
    wandb.log({f"{model_name}/overall_{k}": v for k, v in metrics['overall'].items()})

    # Log per-disease metrics
    for i, disease in enumerate(disease_names):
        wandb.log({
            f"{model_name}/{disease}/AUC-ROC": metrics['per_disease']['auc_roc'][i],
            f"{model_name}/{disease}/AP": metrics['per_disease']['ap'][i],
            f"{model_name}/{disease}/F1": metrics['per_disease']['f1'][i],
            f"{model_name}/{disease}/Specificity": metrics['per_disease']['specificity'][i],
            f"{model_name}/{disease}/Sensitivity": metrics['per_disease']['sensitivity'][i],
            f"{model_name}/{disease}/AUPRC": metrics['per_disease']['auprc'][i],
        })

        # Log confusion matrix as a table
        cm = metrics['per_disease']['cm'][i]
        cm_table = wandb.Table(data=cm.tolist(), columns=["Predicted Negative", "Predicted Positive"])
        wandb.log({f"{model_name}/{disease}/Confusion_Matrix": cm_table})


def compare_metrics(metrics_a, metrics_b, disease_names):
    print("Ablation Study Results:")
    print("Model A: Without Correlation Learning")
    print("Model B: With Correlation Learning")

    print("\nOverall Metrics:")
    for metric, value_a in metrics_a['overall'].items():
        value_b = metrics_b['overall'][metric]
        print(f"{metric}:")
        print(f"  Model A: {value_a:.4f}")
        print(f"  Model B: {value_b:.4f}")
        print(f"  Improvement: {((value_b - value_a) / value_a * 100):.2f}%")

    print("\nPer-disease Metrics:")
    for i, disease in enumerate(disease_names):
        print(f"\n{disease}:")
        for metric in ['auc_roc', 'ap', 'f1', 'specificity', 'sensitivity', 'auprc']:
            value_a = metrics_a['per_disease'][metric][i]
            value_b = metrics_b['per_disease'][metric][i]
            print(f"  {metric}:")
            print(f"    Model A: {value_a:.4f}")
            print(f"    Model B: {value_b:.4f}")
            print(f"    Improvement: {((value_b - value_a) / value_a * 100):.2f}%")


if __name__ == "__main__":
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Convert relevant string values to float
    config['training']['lr'] = float(config['training']['lr'])
    config['training']['weight_decay'] = float(config['training']['weight_decay'])
    config['training']['simclr_temperature'] = float(config['training']['simclr_temperature'])

    run_ablation_study(config)

