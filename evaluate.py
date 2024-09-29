import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from models.efficientnet import EfficientNetWithAttention
from models.binary_classifiers import EnsembleBinaryClassifiers
from models.correlation_learning import CorrelationLearningModule
from models.meta_learner import MetaLearner
from data.dataset import get_dataloader
from data.augmentations import get_transform
from utils.metrics import calculate_metrics
from utils.visualization import plot_confusion_matrix, generate_gradcam, plot_gradcam


def evaluate_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    backbone = EfficientNetWithAttention(config['model']['efficientnet_version'])
    classifiers = EnsembleBinaryClassifiers(backbone.num_features, config['model']['num_classes'])
    correlation_module = CorrelationLearningModule(config['model']['num_classes'])
    meta_learner = MetaLearner(config['model']['num_classes'], hidden_dim=64,
                               num_classes=config['model']['num_classes'])

    checkpoint = torch.load("best_model.pth")
    backbone.load_state_dict(checkpoint['backbone'])
    classifiers.load_state_dict(checkpoint['classifiers'])
    correlation_module.load_state_dict(checkpoint['correlation_module'])
    meta_learner.load_state_dict(checkpoint['meta_learner'])

    backbone.to(device)
    classifiers.to(device)
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
            correlation_adjusted = correlation_module(initial_predictions)
            final_predictions = meta_learner(correlation_adjusted)

            all_predictions.append(final_predictions.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_predictions = np.concatenate(all_predictions)
    all_labels = np.concatenate(all_labels)

    # Calculate metrics
    metrics = calculate_metrics(all_labels, (all_predictions > 0.5).astype(int), all_predictions)

    # Print results
    print("Overall Metrics:")
    print("Micro F1: {:.4f}".format(metrics['micro_f1']))
    print("Macro F1: {:.4f}".format(metrics['macro_f1']))
    print("Weighted F1: {:.4f}".format(metrics['weighted_f1']))
    print("Mean AP: {:.4f}".format(metrics['mean_ap']))
    print("Subset Accuracy: {:.4f}".format(metrics['subset_accuracy']))
    print("Label Ranking Average Precision: {:.4f}".format(metrics['lrap']))

    print("\nPer-class Metrics:")
    for i, disease in enumerate(config['model']['disease_names']):
        print("{}:".format(disease))
        print("  AUC-ROC: {:.4f}".format(metrics['auc_roc'][i]))
        print("  AP: {:.4f}".format(metrics['ap'][i]))
        print("  F1: {:.4f}".format(metrics['f1'][i]))

    # Plot confusion matrices
    for i, disease in enumerate(config['model']['disease_names']):
        cm = metrics['cm_per_class'][i]
        plt = plot_confusion_matrix(cm, ['Negative', 'Positive'])
        plt.savefig("confusion_matrix_{}.png".format(disease))
        plt.close()

    # Generate GradCAM visualizations
    for i, (image, label) in enumerate(test_loader):
        if i >= 5:  # Generate for first 5 images
            break
        image = image.to(device)
        for j, disease in enumerate(config['model']['disease_names']):
            if label[0, j] == 1:
                heatmap = generate_gradcam(backbone, image, j)
                plt = plot_gradcam(image[0].cpu().numpy().transpose(1, 2, 0), heatmap)
                plt.savefig("gradcam_{}_{}.png".format(i, disease))
                plt.close()


if __name__ == "__main__":
    import yaml

    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    evaluate_model(config)

